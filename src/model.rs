#![cfg_attr(not(feature = "cuda"), allow(unused_variables, unused_mut))]
//! FastConformer encoder + LSTM prediction network + TDT joint, loaded from the exported safetensors.

use crate::config::Config;
use crate::ops;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;

/// x: [..., K] @ w: [N, K]^T -> [..., N]
fn linear(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims().to_vec();
    let k = *dims.last().unwrap();
    let rows = x.elem_count() / k;
    let x2 = x.reshape((rows, k))?;
    // the CPU gemm parallelises skinny problems better with the weight on the left
    let y = if !x.device().is_cuda() && rows <= 64 { w.matmul(&x2.t()?)?.t()?.contiguous()? } else { x2.matmul(&w.t()?)? };
    let mut out = dims;
    *out.last_mut().unwrap() = w.dim(0)?;
    y.reshape(out)
}

pub(crate) fn linear_b(x: &Tensor, w: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    linear(x, w)?.broadcast_add(b)
}

struct Weights {
    map: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
}

impl Weights {
    fn get(&self, name: &str) -> Result<Tensor> {
        let t = self.map.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok(t.to_device(&self.device)?.to_dtype(self.dtype)?)
    }
    fn get_dt(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let t = self.map.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok(t.to_device(&self.device)?.to_dtype(dtype)?)
    }
}

struct Layer {
    ff1_norm: (Tensor, Tensor),
    ff1_w1: Tensor,
    ff1_w2: Tensor,
    att_norm: (Tensor, Tensor),
    wqkv: Tensor,
    pos_u: Tensor,
    pos_v: Tensor,
    wo: Tensor,
    conv_norm: (Tensor, Tensor),
    pw1: Tensor,
    dw_w: Tensor,
    dw_b: Tensor,
    pw2: Tensor,
    ff2_norm: (Tensor, Tensor),
    ff2_w1: Tensor,
    ff2_w2: Tensor,
    out_norm: (Tensor, Tensor),
}

pub struct Encoder {
    cfg: Config,
    device: Device,
    dtype: DType,
    conv0: (Tensor, Tensor),
    conv2: (Tensor, Tensor),
    conv3: (Tensor, Tensor),
    conv5: (Tensor, Tensor),
    conv6: (Tensor, Tensor),
    proj: (Tensor, Tensor),
    pos_all: Tensor,
    layers: Vec<Layer>,
    #[cfg(feature = "cuda")]
    gpu: Option<crate::gpu_encoder::GpuEncoder>,
}

pub struct Decoder {
    blank: usize,
    durations: Vec<usize>,
    max_symbols: usize,
    embed: Tensor,
    lstm: Vec<(Tensor, Tensor)>,
    hidden: usize,
    enc_w: Tensor,
    enc_b: Tensor,
    pred_w: Tensor,
    pred_b: Tensor,
    head_w: Tensor,
    head_b: Tensor,
    #[cfg(feature = "cuda")]
    gpu: Option<crate::gpu_decode::GpuDecoder>,
}

pub fn load(dir: &std::path::Path, cfg: &Config, device: &Device, opts: &crate::LoadOptions) -> Result<(Encoder, Decoder)> {
    let map = crate::weights::load(&dir.join("model.safetensors"))?;
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
    let w = Weights { map, device: device.clone(), dtype };
    Ok((Encoder::new(&w, cfg, opts)?, Decoder::new(&w, cfg, opts)?))
}

impl Encoder {
    fn new(w: &Weights, cfg: &Config, opts: &crate::LoadOptions) -> Result<Self> {
        let c = cfg.subsampling_channels;
        let conv = |i: usize, k: usize| -> Result<(Tensor, Tensor)> {
            let wt = w.get(&format!("encoder.pre_encode.conv.{i}.weight"))?;
            let wt = if k == 9 { wt.reshape((c, 9))? } else { wt.reshape((c, c))? };
            Ok((wt, w.get(&format!("encoder.pre_encode.conv.{i}.bias"))?))
        };
        // Core ML flattens [C, F] as c * F + f; our channel-last layout flattens as f * C + c.
        let pw = w.get("encoder.pre_encode.out.weight")?;
        let (dm, cf) = pw.dims2()?;
        let pw = pw.reshape((dm, c, cf / c))?.transpose(1, 2)?.contiguous()?.reshape((dm, cf))?;
        let mut layers = Vec::new();
        let mut pos_ws = Vec::new();
        for i in 0..cfg.n_layers {
            let p = |n: &str| format!("encoder.layers.{i}.{n}");
            let norm = |n: &str| -> Result<(Tensor, Tensor)> { Ok((w.get(&p(&format!("{n}.weight")))?, w.get(&p(&format!("{n}.bias")))?)) };
            let wqkv = Tensor::cat(
                &[w.get(&p("self_attn.linear_q.weight"))?, w.get(&p("self_attn.linear_k.weight"))?, w.get(&p("self_attn.linear_v.weight"))?],
                0,
            )?;
            pos_ws.push(w.get(&p("self_attn.linear_pos.weight"))?);
            let dw = w.get(&p("conv.depthwise_conv.weight"))?;
            let k = dw.dim(D::Minus1)?;
            layers.push(Layer {
                ff1_norm: norm("norm_feed_forward1")?,
                ff1_w1: w.get(&p("feed_forward1.linear1.weight"))?,
                ff1_w2: w.get(&p("feed_forward1.linear2.weight"))?,
                att_norm: norm("norm_self_att")?,
                wqkv,
                pos_u: w.get(&p("self_attn.pos_bias_u"))?.contiguous()?,
                pos_v: w.get(&p("self_attn.pos_bias_v"))?.contiguous()?,
                wo: w.get(&p("self_attn.linear_out.weight"))?,
                conv_norm: norm("norm_conv")?,
                pw1: w.get(&p("conv.pointwise_conv1.weight"))?,
                dw_w: dw.reshape((cfg.d_model, k))?,
                dw_b: w.get(&p("conv.depthwise_conv.bias"))?,
                pw2: w.get(&p("conv.pointwise_conv2.weight"))?,
                ff2_norm: norm("norm_feed_forward2")?,
                ff2_w1: w.get(&p("feed_forward2.linear1.weight"))?,
                ff2_w2: w.get(&p("feed_forward2.linear2.weight"))?,
                out_norm: norm("norm_out")?,
            });
        }
        let mut enc = Self {
            cfg: cfg.clone(),
            device: w.device.clone(),
            dtype: w.dtype,
            conv0: conv(0, 9)?,
            conv2: conv(2, 9)?,
            conv3: conv(3, 1)?,
            conv5: conv(5, 9)?,
            conv6: conv(6, 1)?,
            proj: (pw, w.get("encoder.pre_encode.out.bias")?),
            pos_all: Tensor::cat(&pos_ws, 0)?,
            layers,
            #[cfg(feature = "cuda")]
            gpu: None,
        };
        #[cfg(feature = "cuda")]
        if enc.device.is_cuda() && opts.fused_cuda {
            enc.gpu = Some(enc.build_gpu(opts)?);
        }
        Ok(enc)
    }

    #[cfg(feature = "cuda")]
    fn build_gpu(&self, opts: &crate::LoadOptions) -> Result<crate::gpu_encoder::GpuEncoder> {
        use crate::gpu_encoder::{EncoderPtrs, LayerPtrs, W};
        use crate::ops::gpu::ptr;
        let mut keep = Vec::new();
        let mut p = |t: &Tensor| -> Result<u64> {
            let t = t.contiguous()?;
            let v = ptr(&t)?;
            keep.push(t);
            Ok(v)
        };
        macro_rules! w {
            ($t:expr) => {
                W { dense: p(&$t)?, n: $t.dim(0)?, k: $t.dim(1)? }
            };
        }
        let conv = [&self.conv0, &self.conv2, &self.conv3, &self.conv5, &self.conv6];
        let mut convp = [(0u64, 0u64); 5];
        for (i, (wt, b)) in conv.iter().enumerate() {
            convp[i] = (p(wt)?, p(b)?);
        }
        let proj = (p(&self.proj.0)?, p(&self.proj.1)?);
        let mut layers = Vec::new();
        for l in &self.layers {
            layers.push(LayerPtrs {
                ff1_norm: (p(&l.ff1_norm.0)?, p(&l.ff1_norm.1)?),
                ff1_w1: w!(l.ff1_w1),
                ff1_w2: w!(l.ff1_w2),
                att_norm: (p(&l.att_norm.0)?, p(&l.att_norm.1)?),
                wqkv: w!(l.wqkv),
                pos_u: p(&l.pos_u)?,
                pos_v: p(&l.pos_v)?,
                wo: w!(l.wo),
                conv_norm: (p(&l.conv_norm.0)?, p(&l.conv_norm.1)?),
                pw1: w!(l.pw1),
                dw_w: p(&l.dw_w)?,
                dw_b: p(&l.dw_b)?,
                pw2: w!(l.pw2),
                ff2_norm: (p(&l.ff2_norm.0)?, p(&l.ff2_norm.1)?),
                ff2_w1: w!(l.ff2_w1),
                ff2_w2: w!(l.ff2_w2),
                out_norm: (p(&l.out_norm.0)?, p(&l.out_norm.1)?),
            });
        }
        let pos_all = p(&self.pos_all)?;
        let c = &self.cfg;
        crate::gpu_encoder::GpuEncoder::new(
            &self.device,
            opts.fast_gemm,
            opts.autotune,
            EncoderPtrs { conv: convp, proj, pos_all, layers },
            keep,
            self.pos_all.clone(),
            c.d_model,
            c.n_heads,
            c.ff_dim,
            c.n_layers,
            c.subsampling_channels,
            c.conv_kernel,
            c.n_mels,
            c.layer_norm_eps as f32,
            c.ff_residual_factor as f32,
        )
    }

    /// Encode host features `mel` ([B, tm, n_mels], zero padded) with valid lengths `mel_lens`.
    pub fn forward_host(&self, mel: &[f32], b: usize, tm: usize, mel_lens: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let nm = self.cfg.n_mels;
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            use crate::gpu_encoder::FRAME_BUCKET;
            let tp = tm.div_ceil(FRAME_BUCKET) * FRAME_BUCKET;
            let mut buf = vec![half::f16::ZERO; b * tp * nm];
            for i in 0..b {
                let src = &mel[i * tm * nm..(i + 1) * tm * nm];
                for (d, s) in buf[i * tp * nm..i * tp * nm + tm * nm].iter_mut().zip(src) {
                    *d = half::f16::from_f32(*s);
                }
            }
            return g.forward(&buf, b, tp, mel_lens);
        }
        let mel = Tensor::from_vec(mel.to_vec(), (b, tm, nm), &self.device)?;
        self.forward(&mel, mel_lens)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Relative sinusoidal positions T-1 .. -(T-1): [2T-1, d]
    fn rel_pos(&self, t: usize) -> Result<Tensor> {
        let d = self.cfg.d_model;
        let n = 2 * t - 1;
        let mut pe = vec![0f32; n * d];
        for (r, row) in pe.chunks_mut(d).enumerate() {
            let pos = (t as f64 - 1.0) - r as f64;
            for i in 0..d / 2 {
                let div = (-(2.0 * i as f64) * (10000f64.ln() / d as f64)).exp();
                row[2 * i] = (pos * div).sin() as f32;
                row[2 * i + 1] = (pos * div).cos() as f32;
            }
        }
        Ok(Tensor::from_vec(pe, (n, d), &self.device)?.to_dtype(self.dtype)?)
    }

    fn lens_tensor(&self, lens: &[usize]) -> Result<Tensor> {
        let v: Vec<u32> = lens.iter().map(|&l| l as u32).collect();
        Ok(Tensor::from_vec(v, lens.len(), &self.device)?)
    }

    /// mel: [B, Tm, n_mels] (zero padded), mel_lens: valid frames per item.
    /// Returns ([B, T, d_model], valid encoder frames per item).
    pub fn forward(&self, mel: &Tensor, mel_lens: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let cfg = &self.cfg;
        let half = |v: &[usize]| v.iter().map(|&l| (l.max(1) - 1) / 2 + 1).collect::<Vec<_>>();
        let l1 = half(mel_lens);
        let l2 = half(&l1);
        let l3 = half(&l2);
        let (l1t, l2t, l3t) = (self.lens_tensor(&l1)?, self.lens_tensor(&l2)?, self.lens_tensor(&l3)?);
        let mel = mel.to_dtype(self.dtype)?.contiguous()?;

        // ---- subsampling (channel-last)
        let x = ops::sub_conv0(&mel, &self.conv0.0, &self.conv0.1, &l1t)?;
        let x = ops::sub_dwconv(&x, &self.conv2.0, &self.conv2.1, &l2t)?;
        let x = ops::bias_mask_relu(&linear(&x, &self.conv3.0)?, &self.conv3.1, &l2t)?;
        let x = ops::sub_dwconv(&x, &self.conv5.0, &self.conv5.1, &l3t)?;
        let x = ops::bias_mask_relu(&linear(&x, &self.conv6.0)?, &self.conv6.1, &l3t)?;
        let (b, t, f, c) = x.dims4()?;
        let mut x = linear_b(&x.reshape((b, t, f * c))?, &self.proj.0, &self.proj.1)?;

        // ---- relative positions for all layers in one GEMM: [L, H, 2T-1, dk]
        let (h, d) = (cfg.n_heads, cfg.d_model);
        let dk = d / h;
        let p_all = linear(&self.rel_pos(t)?, &self.pos_all)?
            .reshape((2 * t - 1, cfg.n_layers, h, dk))?
            .permute((1, 2, 0, 3))?
            .contiguous()?;
        let scale = 1.0 / (dk as f32).sqrt();
        let eps = cfg.layer_norm_eps as f32;
        let ffr = cfg.ff_residual_factor as f32;

        let mut hn = ops::layer_norm(&x, &self.layers[0].ff1_norm.0, &self.layers[0].ff1_norm.1, eps)?;
        for (li, l) in self.layers.iter().enumerate() {
            // feed forward 1 (half step)
            let y = linear(&ops::silu(&linear(&hn, &l.ff1_w1)?)?, &l.ff1_w2)?;
            let (x1, h1) = ops::add_layer_norm(&x, &y, ffr, &l.att_norm.0, &l.att_norm.1, eps)?;
            // relative-position multi-head attention
            let qkv = linear(&h1, &l.wqkv)?;
            let (qu, qv, k, v) = ops::qkv_split(&qkv, &l.pos_u, &l.pos_v, h)?;
            let ac = qu.matmul(&k.t()?)?; // [B, H, T, T]
            let p = p_all.get(li)?; // [H, 2T-1, dk]
            let bd = qv.reshape((h, b * t, dk))?.matmul(&p.t()?)?; // [H, B*T, 2T-1]
            let probs = ops::relpos_softmax(&ac, &bd, &l3t, scale)?;
            let o = ops::merge_heads(&probs.matmul(&v)?)?;
            let o = linear(&o, &l.wo)?;
            let (x2, h2) = ops::add_layer_norm(&x1, &o, 1.0, &l.conv_norm.0, &l.conv_norm.1, eps)?;
            // convolution module
            let g = ops::glu_mask(&linear(&h2, &l.pw1)?, &l3t)?;
            let g = ops::dwconv1d_silu(&g, &l.dw_w, &l.dw_b)?;
            let g = linear(&g, &l.pw2)?;
            let (x3, h3) = ops::add_layer_norm(&x2, &g, 1.0, &l.ff2_norm.0, &l.ff2_norm.1, eps)?;
            // feed forward 2 (half step) + output norm
            let y = linear(&ops::silu(&linear(&h3, &l.ff2_w1)?)?, &l.ff2_w2)?;
            let (_, xo) = ops::add_layer_norm(&x3, &y, ffr, &l.out_norm.0, &l.out_norm.1, eps)?;
            x = xo;
            if li + 1 < self.layers.len() {
                let nl = &self.layers[li + 1];
                hn = ops::layer_norm(&x, &nl.ff1_norm.0, &nl.ff1_norm.1, eps)?;
            }
        }
        Ok((x, l3))
    }
}

/// One emitted token: id, encoder frame index, and its softmax probability (over the vocabulary + blank).
#[derive(Debug, Clone, Copy)]
pub struct Token {
    pub id: u32,
    pub frame: u32,
    pub prob: f32,
}

impl Decoder {
    fn new(w: &Weights, cfg: &Config, opts: &crate::LoadOptions) -> Result<Self> {
        let f = DType::F32;
        let mut lstm = Vec::new();
        for l in 0..cfg.pred_layers {
            let p = |n: &str| format!("decoder.prediction.dec_rnn.lstm.{n}_l{l}");
            let wcat = Tensor::cat(&[w.get_dt(&p("weight_ih"), f)?, w.get_dt(&p("weight_hh"), f)?], 1)?;
            let b = (w.get_dt(&p("bias_ih"), f)? + w.get_dt(&p("bias_hh"), f)?)?;
            lstm.push((wcat, b));
        }
        #[cfg(feature = "cuda")]
        let gpu = if w.device.is_cuda() && opts.fused_cuda {
            let h = DType::F16;
            let p = |n: &str, l: usize| format!("decoder.prediction.dec_rnn.lstm.{n}_l{l}");
            let bias = |l: usize| -> Result<Tensor> { Ok((w.get_dt(&p("bias_ih", l), f)? + w.get_dt(&p("bias_hh", l), f)?)?) };
            Some(crate::gpu_decode::GpuDecoder::new(
                crate::gpu_decode::DecoderWeights {
                    embed: w.get_dt("decoder.prediction.embed.weight", h)?,
                    wih: [w.get_dt(&p("weight_ih", 0), h)?, w.get_dt(&p("weight_ih", 1), h)?],
                    whh: [w.get_dt(&p("weight_hh", 0), h)?, w.get_dt(&p("weight_hh", 1), h)?],
                    bias: [bias(0)?, bias(1)?],
                    pred_w: w.get_dt("joint.pred.weight", h)?,
                    pred_b: w.get_dt("joint.pred.bias", f)?,
                    head_w: w.get_dt("joint.joint_net.2.weight", h)?,
                    head_b: w.get_dt("joint.joint_net.2.bias", f)?,
                    enc_w: w.get_dt("joint.enc.weight", h)?,
                    enc_b: w.get_dt("joint.enc.bias", h)?,
                },
                &cfg.durations,
                cfg.blank_id,
                cfg.max_symbols_per_step,
            )?)
        } else {
            None
        };
        Ok(Self {
            #[cfg(feature = "cuda")]
            gpu,
            blank: cfg.blank_id,
            durations: cfg.durations.clone(),
            max_symbols: cfg.max_symbols_per_step,
            embed: w.get_dt("decoder.prediction.embed.weight", f)?,
            lstm,
            hidden: cfg.pred_hidden,
            enc_w: w.get_dt("joint.enc.weight", f)?,
            enc_b: w.get_dt("joint.enc.bias", f)?,
            pred_w: w.get_dt("joint.pred.weight", f)?,
            pred_b: w.get_dt("joint.pred.bias", f)?,
            head_w: w.get_dt("joint.joint_net.2.weight", f)?,
            head_b: w.get_dt("joint.joint_net.2.bias", f)?,
        })
    }

    /// Advance the prediction network for rows with mask[b] == 1. Returns the new joint projection.
    fn pred_step(&self, toks: &[u32], mask: &[u32], h: &mut [Tensor], c: &mut [Tensor]) -> Result<Tensor> {
        let dev = self.embed.device();
        let b = toks.len();
        let toks = Tensor::from_vec(toks.to_vec(), b, dev)?;
        let mask = Tensor::from_vec(mask.to_vec(), b, dev)?;
        let mut x = self.embed.index_select(&toks, 0)?;
        for (l, (wcat, bias)) in self.lstm.iter().enumerate() {
            let gates = linear_b(&Tensor::cat(&[&x, &h[l]], 1)?, wcat, bias)?;
            let (hn, cn) = ops::lstm_cell(&gates, &h[l], &c[l], &mask)?;
            h[l] = hn;
            c[l] = cn;
            x = h[l].clone();
        }
        Ok(linear_b(&x, &self.pred_w, &self.pred_b)?)
    }

    /// Batched greedy TDT decoding. enc: [B, T, d_model]
    pub fn decode(&self, enc: &Tensor, lens: &[usize]) -> Result<Vec<Vec<Token>>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            return g.decode(enc, lens);
        }
        let (b, t, _) = enc.dims3()?;
        let dev = self.embed.device().clone();
        let ep = linear_b(&enc.to_dtype(DType::F32)?, &self.enc_w, &self.enc_b)?.contiguous()?; // [B, T, J]
        let zeros = Tensor::zeros((b, self.hidden), DType::F32, &dev)?;
        let mut h = vec![zeros.clone(); self.lstm.len()];
        let mut c = vec![zeros; self.lstm.len()];
        let mut last = vec![self.blank as u32; b];
        let mut dp = self.pred_step(&last, &vec![1; b], &mut h, &mut c)?;
        let mut pos = vec![0usize; b];
        let mut emitted = vec![0usize; b];
        let mut out = vec![Vec::new(); b];
        let v = self.blank + 1;
        loop {
            let active: Vec<bool> = (0..b).map(|i| pos[i] < lens[i]).collect();
            if !active.iter().any(|&a| a) {
                break;
            }
            let frame: Vec<u32> = (0..b).map(|i| pos[i].min(t - 1) as u32).collect();
            let frame = Tensor::from_vec(frame, b, &dev)?;
            let z = ops::joint_hidden(&ep, &dp, &frame)?;
            let logits = linear_b(&z, &self.head_w, &self.head_b)?;
            let res = ops::tdt_argmax(&logits, v)?;
            // probability of the argmax token: 1 / sum(exp(logit - max))
            let tl = logits.narrow(1, 0, v)?;
            let probs = tl.broadcast_sub(&tl.max_keepdim(1)?)?.exp()?.sum(1)?.recip()?.to_vec1::<f32>()?;
            let mut update = vec![0u32; b];
            for i in 0..b {
                if !active[i] {
                    continue;
                }
                let (tok, di) = res[i];
                let mut dur = self.durations[di as usize];
                if tok as usize != self.blank {
                    out[i].push(Token { id: tok, frame: pos[i] as u32, prob: probs[i] });
                    last[i] = tok;
                    update[i] = 1;
                    emitted[i] += 1;
                } else if dur == 0 {
                    dur = 1;
                }
                if dur == 0 && emitted[i] >= self.max_symbols {
                    dur = 1;
                }
                if dur > 0 {
                    emitted[i] = 0;
                }
                pos[i] += dur;
            }
            if update.iter().any(|&u| u != 0) {
                dp = self.pred_step(&last, &update, &mut h, &mut c)?;
            }
        }
        Ok(out)
    }
}
