//! Greedy TDT decoding that runs entirely on the GPU.
//!
//! One decoding step is five kernels (joint + argmax, bookkeeping, LSTM layer 0, LSTM layer 1, prediction
//! projection + state commit) operating on device-resident state; the host only enqueues steps (as a CUDA graph
//! of `CHUNK` steps) and polls a "rows still active" flag through pinned memory, one chunk behind.

use crate::model::Token;
use crate::ops::gpu;
use anyhow::{bail, Result};
use candle_core::cuda_backend::cudarc::driver::{
    result, sys, CudaEvent, CudaFunction, CudaStream, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use candle_core::{CudaDevice, DType, Device, Tensor};
use std::sync::Arc;

const CHUNK: usize = 8;
const GROUP: usize = 8; // DEC_G in fluid.cu
const JOINT_ROWS_PER_WARP: usize = 4; // DEC_JR in fluid.cu
const K: usize = 640; // DEC_K in fluid.cu

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DecState {
    embed: u64,
    wih0: u64,
    whh0: u64,
    b0: u64,
    wih1: u64,
    whh1: u64,
    b1: u64,
    pred_w: u64,
    pred_b: u64,
    head_w: u64,
    head_b: u64,
    ep: u64,
    lens: u64,
    h0: u64,
    c0: u64,
    h1: u64,
    c1: u64,
    h0n: u64,
    c0n: u64,
    h1n: u64,
    c1n: u64,
    dp: u64,
    pos: u64,
    emitted: u64,
    last_tok: u64,
    update: u64,
    best: u64,
    dur: u64,
    out_tok: u64,
    out_frame: u64,
    count: u64,
    flags: u64,
    durations: u64,
    pmax: u64,
    psum: u64,
    out_prob: u64,
    b: i32,
    t: i32,
    p: i32,
    j: i32,
    v: i32,
    nd: i32,
    cap: i32,
    blank: i32,
    max_symbols: i32,
    jblocks: i32,
}

unsafe impl DeviceRepr for DecState {}

pub struct GpuDecoder {
    dev: CudaDevice,
    embed: Tensor,
    wih: [Tensor; 2],
    whh: [Tensor; 2],
    bias: [Tensor; 2],
    pred_w: Tensor,
    pred_b: Tensor,
    head_w: Tensor,
    head_b: Tensor,
    enc_w: Tensor,
    enc_b: Tensor,
    durations: Tensor,
    n_durations: usize,
    blank: usize,
    max_symbols: usize,
    hidden: usize,
    joint: usize,
    sms: u32,
    pinned: std::sync::Mutex<PinnedHostSlice<u32>>,
}

/// Tensors needed by the decoder, all already on the device (fp16 weights, fp32 biases).
pub struct DecoderWeights {
    pub embed: Tensor,
    pub wih: [Tensor; 2],
    pub whh: [Tensor; 2],
    pub bias: [Tensor; 2],
    pub pred_w: Tensor,
    pub pred_b: Tensor,
    pub head_w: Tensor,
    pub head_b: Tensor,
    pub enc_w: Tensor,
    pub enc_b: Tensor,
}

struct Funcs {
    joint: CudaFunction,
    update: CudaFunction,
    lstm: CudaFunction,
    pred: CudaFunction,
    group: usize,
}

impl GpuDecoder {
    pub fn new(w: DecoderWeights, durations: &[usize], blank: usize, max_symbols: usize) -> Result<Self> {
        let dev = gpu::dev(&w.embed)?;
        let hidden = w.embed.dim(1)?;
        let joint = w.pred_w.dim(0)?;
        if hidden != K || joint != K {
            bail!("the GPU decoder kernels are specialised for 640-wide prediction/joint networks");
        }
        let d: Vec<u32> = durations.iter().map(|&x| x as u32).collect();
        let durations_t = Tensor::from_vec(d, durations.len(), &Device::Cuda(dev.clone()))?;
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        let sms = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map(|v| v as u32)
            .unwrap_or(64);
        let pinned = unsafe { ctx.alloc_pinned::<u32>(4)? };
        Ok(Self {
            dev,
            embed: w.embed,
            wih: w.wih,
            whh: w.whh,
            bias: w.bias,
            pred_w: w.pred_w,
            pred_b: w.pred_b,
            head_w: w.head_w,
            head_b: w.head_b,
            enc_w: w.enc_w,
            enc_b: w.enc_b,
            durations: durations_t,
            n_durations: durations.len(),
            blank,
            max_symbols,
            hidden,
            joint,
            sms,
            pinned: std::sync::Mutex::new(pinned),
        })
    }

    fn funcs(&self, group: usize) -> Result<Funcs> {
        let (j, l, p) = match group {
            1 => ("dec_joint_g1", "dec_lstm_g1", "dec_pred_g1"),
            2 => ("dec_joint_g2", "dec_lstm_g2", "dec_pred_g2"),
            4 => ("dec_joint_g4", "dec_lstm_g4", "dec_pred_g4"),
            _ => ("dec_joint_g8", "dec_lstm_g8", "dec_pred_g8"),
        };
        Ok(Funcs {
            joint: gpu::func(&self.dev, j)?,
            update: gpu::func(&self.dev, "dec_update")?,
            lstm: gpu::func(&self.dev, l)?,
            pred: gpu::func(&self.dev, p)?,
            group,
        })
    }

    fn launch_pred_step(&self, f: &Funcs, stream: &Arc<CudaStream>, st: &DecState) -> Result<()> {
        let (p, j) = (self.hidden, self.joint);
        for layer in 0..2i32 {
            let mut b = stream.launch_builder(&f.lstm);
            b.arg(st).arg(&layer);
            // one warp per hidden unit, sized to fill every SM exactly once
            let wpb = p.div_ceil(self.sms as usize).min(8);
            let cfg = LaunchConfig {
                grid_dim: (p.div_ceil(wpb) as u32, 1, 1),
                block_dim: ((wpb * 32) as u32, 1, 1),
                shared_mem_bytes: (f.group * 2 * p * 4) as u32,
            };
            unsafe { b.launch(cfg) }?;
        }
        let mut b = stream.launch_builder(&f.pred);
        b.arg(st);
        let wpb = j.div_ceil(self.sms as usize).min(8);
        let cfg = LaunchConfig {
            grid_dim: (j.div_ceil(wpb) as u32, 1, 1),
            block_dim: ((wpb * 32) as u32, 1, 1),
            shared_mem_bytes: (f.group * p * 4) as u32,
        };
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// (warps per block, blocks) of the joint kernel
    fn joint_grid(&self) -> (usize, usize) {
        let rows = self.blank + 1 + self.n_durations;
        let wpb = rows.div_ceil(self.sms as usize * JOINT_ROWS_PER_WARP).min(8);
        (wpb, rows.div_ceil(wpb * JOINT_ROWS_PER_WARP))
    }

    fn launch_step(&self, f: &Funcs, stream: &Arc<CudaStream>, st: &DecState) -> Result<()> {
        let j = self.joint;
        let mut b = stream.launch_builder(&f.joint);
        b.arg(st);
        let (wpb, blocks) = self.joint_grid();
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: ((wpb * 32) as u32, 1, 1),
            // hidden rows, then per-warp packed argmax (u64) and log-sum-exp partials (2 x f32)
            shared_mem_bytes: (f.group * j * 4 + wpb * f.group * 16) as u32,
        };
        unsafe { b.launch(cfg) }?;
        let mut b = stream.launch_builder(&f.update);
        b.arg(st);
        let smem = (2 * st.b as usize * 4) as u32;
        unsafe { b.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem }) }?;
        self.launch_pred_step(f, stream, st)
    }

    /// enc: [B, T, d_model] (fp16), lens: valid frames per row.
    pub fn decode(&self, enc: &Tensor, lens: &[usize]) -> Result<Vec<Vec<Token>>> {
        let (bsz, t, _) = enc.dims3()?;
        if bsz == 0 {
            return Ok(vec![]);
        }
        let device = Device::Cuda(self.dev.clone());
        let stream = self.dev.cuda_stream();
        let (p, j) = (self.hidden, self.joint);
        let v = self.blank + 1;
        // encoder half of the joint for all frames at once
        let ep = crate::model::linear_b(&enc.to_dtype(DType::F16)?, &self.enc_w, &self.enc_b)?.contiguous()?;
        let cap = t * 2 + 16;
        let zf = |n: usize| Tensor::zeros(n, DType::F32, &device);
        let zu = |n: usize| Tensor::zeros(n, DType::U32, &device);
        let (h0, c0, h1, c1) = (zf(bsz * p)?, zf(bsz * p)?, zf(bsz * p)?, zf(bsz * p)?);
        let (h0n, c0n, h1n, c1n) = (zf(bsz * p)?, zf(bsz * p)?, zf(bsz * p)?, zf(bsz * p)?);
        let dp = zf(bsz * j)?;
        let dur = zf(bsz * self.n_durations)?;
        let (pos, emitted, count) = (zu(bsz)?, zu(bsz)?, zu(bsz)?);
        let best = zu(2 * bsz)?; // u64 per row
        let (out_tok, out_frame) = (zu(bsz * cap)?, zu(bsz * cap)?);
        let out_prob = zf(bsz * cap)?;
        let jblocks = self.joint_grid().1;
        let (pmax, psum) = (zf(bsz * jblocks)?, zf(bsz * jblocks)?);
        let lens_u: Vec<u32> = lens.iter().map(|&l| l.min(t) as u32).collect();
        let n_active = lens_u.iter().filter(|&&l| l > 0).count() as u32;
        let lens_t = Tensor::from_vec(lens_u, bsz, &device)?;
        let last_tok = Tensor::from_vec(vec![self.blank as u32; bsz], bsz, &device)?;
        let update = Tensor::from_vec(vec![1u32; bsz], bsz, &device)?;
        let flags = Tensor::from_vec(vec![1u32, n_active], 2, &device)?;
        let st = DecState {
            embed: gpu::ptr(&self.embed)?,
            wih0: gpu::ptr(&self.wih[0])?,
            whh0: gpu::ptr(&self.whh[0])?,
            b0: gpu::ptr(&self.bias[0])?,
            wih1: gpu::ptr(&self.wih[1])?,
            whh1: gpu::ptr(&self.whh[1])?,
            b1: gpu::ptr(&self.bias[1])?,
            pred_w: gpu::ptr(&self.pred_w)?,
            pred_b: gpu::ptr(&self.pred_b)?,
            head_w: gpu::ptr(&self.head_w)?,
            head_b: gpu::ptr(&self.head_b)?,
            ep: gpu::ptr(&ep)?,
            lens: gpu::ptr(&lens_t)?,
            h0: gpu::ptr(&h0)?,
            c0: gpu::ptr(&c0)?,
            h1: gpu::ptr(&h1)?,
            c1: gpu::ptr(&c1)?,
            h0n: gpu::ptr(&h0n)?,
            c0n: gpu::ptr(&c0n)?,
            h1n: gpu::ptr(&h1n)?,
            c1n: gpu::ptr(&c1n)?,
            dp: gpu::ptr(&dp)?,
            pos: gpu::ptr(&pos)?,
            emitted: gpu::ptr(&emitted)?,
            last_tok: gpu::ptr(&last_tok)?,
            update: gpu::ptr(&update)?,
            best: gpu::ptr(&best)?,
            dur: gpu::ptr(&dur)?,
            out_tok: gpu::ptr(&out_tok)?,
            out_frame: gpu::ptr(&out_frame)?,
            pmax: gpu::ptr(&pmax)?,
            psum: gpu::ptr(&psum)?,
            out_prob: gpu::ptr(&out_prob)?,
            jblocks: jblocks as i32,
            count: gpu::ptr(&count)?,
            flags: gpu::ptr(&flags)?,
            durations: gpu::ptr(&self.durations)?,
            b: bsz as i32,
            t: t as i32,
            p: p as i32,
            j: j as i32,
            v: v as i32,
            nd: self.n_durations as i32,
            cap: cap as i32,
            blank: self.blank as i32,
            max_symbols: self.max_symbols as i32,
        };
        let f = self.funcs(bsz.next_power_of_two().min(GROUP))?;
        // initial prediction-network step on <blank>
        self.launch_pred_step(&f, &stream, &st)?;

        // capture CHUNK steps into a graph
        let graph = gpu::capture_graph(&stream, || (0..CHUNK).try_for_each(|_| self.launch_step(&f, &stream, &st)))?;

        let ctx = stream.context();
        let events: [CudaEvent; 2] = [ctx.new_event(None)?, ctx.new_event(None)?];
        let mut pinned = self.pinned.lock().unwrap();
        let host: &mut [u32] = pinned.as_mut_slice()?;
        let max_chunks = (t * (self.max_symbols + 1)).div_ceil(CHUNK) + 2;
        let flag_ptr = st.flags + 4;
        for k in 0..max_chunks {
            graph.launch()?;
            let slot = k % 2;
            unsafe { result::memcpy_dtoh_async(&mut host[slot..slot + 1], flag_ptr, stream.cu_stream()) }?;
            events[slot].record(&stream)?;
            if k > 0 {
                let prev = 1 - slot;
                events[prev].synchronize()?;
                if host[prev] == 0 {
                    break;
                }
            }
        }
        drop(pinned);
        let counts = count.to_vec1::<u32>()?;
        let toks = out_tok.to_vec1::<u32>()?;
        let frames = out_frame.to_vec1::<u32>()?;
        let probs = out_prob.to_vec1::<f32>()?;
        Ok((0..bsz)
            .map(|b| {
                (0..counts[b] as usize)
                    .map(|i| Token { id: toks[b * cap + i], frame: frames[b * cap + i], prob: probs[b * cap + i] })
                    .collect()
            })
            .collect())
    }
}
