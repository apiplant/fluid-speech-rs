//! Static-shape CUDA execution of the FastConformer encoder.
//!
//! All activations live in one preallocated workspace; GEMMs go straight to cuBLAS and every other op is a
//! kernel from `kernels/fluid.cu`. The whole forward pass for a given (batch, mel frames) bucket is captured
//! once into a CUDA graph and replayed, so a forward costs one graph launch instead of ~600 kernel launches.

use crate::ops::gpu;
use anyhow::{bail, Result};
use crate::lt::Problem;
use candle_core::cuda_backend::cudarc::driver::{
    result, CudaFunction, CudaGraph, CudaStream, LaunchConfig, PushKernelArg,
};
use candle_core::{CudaDevice, DType, Device, Tensor};
use half::f16;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mel frames are padded up to a multiple of this so graphs can be reused across similar lengths.
pub const FRAME_BUCKET: usize = 64;
const MAX_GRAPHS: usize = 64;

/// A linear weight [n, k] (fp16, row-major).
#[derive(Clone, Copy)]
pub struct W {
    pub dense: u64,
    pub n: usize,
    pub k: usize,
}

pub struct LayerPtrs {
    pub ff1_norm: (u64, u64),
    pub ff1_w1: W,
    pub ff1_w2: W,
    pub att_norm: (u64, u64),
    pub wqkv: W,
    pub pos_u: u64,
    pub pos_v: u64,
    pub wo: W,
    pub conv_norm: (u64, u64),
    pub pw1: W,
    pub dw_w: u64,
    pub dw_b: u64,
    pub pw2: W,
    pub ff2_norm: (u64, u64),
    pub ff2_w1: W,
    pub ff2_w2: W,
    pub out_norm: (u64, u64),
}

pub struct EncoderPtrs {
    pub conv: [(u64, u64); 5], // pre_encode convs 0, 2, 3, 5, 6 (weight, bias)
    pub proj: (u64, u64),
    pub pos_all: u64, // [L * d, d]
    pub layers: Vec<LayerPtrs>,
}

#[derive(Clone, Copy)]
struct Dims {
    d: usize,
    h: usize,
    ff: usize,
    layers: usize,
    c: usize,
    kconv: usize,
    n_mels: usize,
    eps: f32,
    ffr: f32,
}

struct Workspace {
    buf: Option<Tensor>,
    bytes: usize,
    graphs: HashMap<(usize, usize), CudaGraph>,
    order: Vec<(usize, usize)>,
}

/// Relative positions projected through every layer's linear_pos: [2*tmax - 1, L * d], computed once.
struct PosTable {
    tmax: usize,
    table: Tensor,
}

pub struct GpuEncoder {
    dev: CudaDevice,
    stream: Arc<CudaStream>,
    gemm: crate::lt::Gemm,
    _blas_ws: Tensor,
    w: EncoderPtrs,
    /// keeps the weight tensors behind the raw pointers alive
    _keep: Vec<Tensor>,
    pos_w: Tensor,
    dims: Dims,
    ws: Mutex<Workspace>,
    pos: Mutex<Option<PosTable>>,
    funcs: HashMap<&'static str, CudaFunction>,
}

unsafe impl Send for GpuEncoder {}
unsafe impl Sync for GpuEncoder {}

/// Bump allocator over the workspace (256-byte aligned regions).
struct Alloc {
    base: u64,
    off: usize,
}

impl Alloc {
    fn take(&mut self, elems: usize, elem_size: usize) -> u64 {
        let p = self.base + self.off as u64;
        self.off += (elems * elem_size).div_ceil(256) * 256;
        p
    }
}

/// Buffer layout of one forward pass.
struct Plan {
    b: usize,
    tm: usize,
    t1: usize,
    t2: usize,
    t: usize,
    mel: u64,
    lens: u64, // [3, B] u32
    s0: u64,
    s1: u64,
    s2: u64,
    x: u64,
    xs: u64,
    hn: u64,
    ff: u64,
    qkv: u64,
    qu: u64,
    qv: u64,
    k: u64,
    v: u64,
    ac: u64,
    bd: u64,
    o: u64,
    m: u64,
    y: u64,
    pw: u64,
    g: u64,
    g2: u64,
    bytes: usize,
}

const KERNELS: &[&str] = &[
    "sub_conv01", "silu8", "sub_dwconv", "bias_mask_relu", "layer_norm", "add_layer_norm", "silu", "qkv_split",
    "relpos_softmax", "merge_heads", "glu_mask", "dwconv1d_silu", "add_bias",
];

impl GpuEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        fp16_acc: bool,
        tune: bool,
        w: EncoderPtrs,
        keep: Vec<Tensor>,
        pos_w: Tensor,
        d: usize,
        h: usize,
        ff: usize,
        layers: usize,
        c: usize,
        kconv: usize,
        n_mels: usize,
        eps: f32,
        ffr: f32,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => bail!("GpuEncoder needs a cuda device"),
        };
        let stream = dev.cuda_stream();
        // fixed GEMM workspace so nothing is allocated during graph capture
        let blas_ws = gpu::empty(&dev, 32 << 20, DType::U32)?;
        let gemm = crate::lt::Gemm::new(stream.clone(), gpu::ptr(&blas_ws)?, 128 << 20, fp16_acc, tune)?;
        let mut funcs = HashMap::new();
        for &k in KERNELS {
            funcs.insert(k, gpu::func(&dev, k)?);
        }
        Ok(Self {
            dev,
            stream,
            gemm,
            _blas_ws: blas_ws,
            w,
            _keep: keep,
            pos_w,
            dims: Dims { d, h, ff, layers, c, kconv, n_mels, eps, ffr },
            ws: Mutex::new(Workspace { buf: None, bytes: 0, graphs: HashMap::new(), order: Vec::new() }),
            pos: Mutex::new(None),
            funcs,
        })
    }

    fn plan(&self, base: u64, b: usize, tm: usize) -> Plan {
        let dm = self.dims;
        let half = |v: usize| (v - 1) / 2 + 1;
        let (t1, t2, t) = (half(tm), half(half(tm)), half(half(half(tm))));
        let (_f1, f2, f3) = (half(dm.n_mels), half(half(dm.n_mels)), half(half(half(dm.n_mels))));
        let (d, h, c) = (dm.d, dm.h, dm.c);
        let bt = b * t;
        let mut a = Alloc { base, off: 0 };
        let mel = a.take(b * tm * dm.n_mels, 2);
        let lens = a.take(3 * b, 4);
        let s0 = a.take(b * t2 * f2 * c, 2);
        let s1 = a.take(b * t2 * f2 * c, 2);
        let s2 = a.take(b * t * f3 * c, 2);
        let x = a.take(bt * d, 2);
        let xs = a.take(bt * d, 2);
        let hn = a.take(bt * d, 2);
        let ff = a.take(bt * dm.ff, 2);
        let qkv = a.take(bt * 3 * d, 2);
        let qu = a.take(bt * d, 2);
        let qv = a.take(bt * d, 2);
        let k = a.take(bt * d, 2);
        let v = a.take(bt * d, 2);
        let ac = a.take(b * h * t * t, 2);
        let bd = a.take(h * bt * (2 * t - 1), 2);
        let o = a.take(bt * d, 2);
        let m = a.take(bt * d, 2);
        let y = a.take(bt * d, 2);
        let pw = a.take(bt * 2 * d, 2);
        let g = a.take(bt * d, 2);
        let g2 = a.take(bt * d, 2);
        Plan {
            b, tm, t1, t2, t, mel, lens, s0, s1, s2, x, xs, hn, ff, qkv, qu, qv, k, v, ac, bd, o, m, y, pw, g, g2,
            bytes: a.off,
        }
    }

    fn launch(&self, name: &'static str, grid: usize, block: u32, smem: usize, args: &[gpu::A]) -> Result<()> {
        let f = &self.funcs[name];
        let mut bld = self.stream.launch_builder(f);
        for a in args {
            match a {
                gpu::A::P(p) => bld.arg(p),
                gpu::A::I(i) => bld.arg(i),
                gpu::A::L(l) => bld.arg(l),
                gpu::A::F(x) => bld.arg(x),
            };
        }
        let cfg =
            LaunchConfig { grid_dim: (grid.max(1) as u32, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem as u32 };
        unsafe { bld.launch(cfg) }?;
        Ok(())
    }

    /// Row-major Y[rows, w.n] = X[rows, w.k] . W^T for a model weight.
    fn wgemm(&self, cap: bool, rows: usize, x: u64, w: &W, y: u64) -> Result<()> {
        self.gemm_nt(cap, rows, w.n, w.k, x, w.k, 0, w.dense, w.k, 0, y, w.n, 0, 1)
    }

    /// Row-major Y[m, n] = X[m, k] . W[n, k]^T (optionally batched with element strides).
    #[allow(clippy::too_many_arguments)]
    fn gemm_nt(
        &self, cap: bool, m: usize, n: usize, k: usize, x: u64, ldx: usize, sx: usize, w: u64, ldw: usize, sw: usize,
        y: u64, ldy: usize, sy: usize, batch: usize,
    ) -> Result<()> {
        let p = Problem { ta: true, m: n, n: m, k, lda: ldw, ldb: ldx, ldc: ldy, sa: sw, sb: sx, sc: sy, batch };
        self.gemm.gemm(p, w, x, y, cap)
    }

    /// Row-major Y[m, n] = X[m, k] . V[k, n] (batched).
    #[allow(clippy::too_many_arguments)]
    fn gemm_nn(
        &self, cap: bool, m: usize, n: usize, k: usize, x: u64, ldx: usize, sx: usize, v: u64, ldv: usize, sv: usize,
        y: u64, ldy: usize, sy: usize, batch: usize,
    ) -> Result<()> {
        let p = Problem { ta: false, m: n, n: m, k, lda: ldv, ldb: ldx, ldc: ldy, sa: sv, sb: sx, sc: sy, batch };
        self.gemm.gemm(p, v, x, y, cap)
    }

    /// Ensure the projected relative-position table covers `t` frames; returns (table ptr, tmax).
    fn pos_table(&self, t: usize) -> Result<(u64, usize)> {
        let mut g = self.pos.lock().unwrap();
        if g.as_ref().is_none_or(|p| p.tmax < t) {
            let tmax = t.max(512).next_power_of_two();
            let d = self.dims.d;
            let n = 2 * tmax - 1;
            let mut pe = vec![0f32; n * d];
            for (r, row) in pe.chunks_mut(d).enumerate() {
                let pos = (tmax as f64 - 1.0) - r as f64;
                for i in 0..d / 2 {
                    let div = (-(2.0 * i as f64) * (10000f64.ln() / d as f64)).exp();
                    row[2 * i] = (pos * div).sin() as f32;
                    row[2 * i + 1] = (pos * div).cos() as f32;
                }
            }
            let dev = Device::Cuda(self.dev.clone());
            let pe = Tensor::from_vec(pe, (n, d), &dev)?.to_dtype(DType::F16)?;
            let table = pe.matmul(&self.pos_w.t()?)?.contiguous()?; // [n, L*d]
            *g = Some(PosTable { tmax, table });
        }
        let p = g.as_ref().unwrap();
        Ok((gpu::ptr(&p.table)?, p.tmax))
    }

    /// Enqueue the whole forward pass for plan `p` (no host synchronisation, no allocation).
    fn record(&self, p: &Plan, pos: u64, tmax: usize, cap: bool) -> Result<()> {
        use gpu::A::{F, I, L, P};
        let dm = self.dims;
        let (d, h, c, b, t) = (dm.d, dm.h, dm.c, p.b, p.t);
        let dk = d / h;
        let bt = b * t;
        let nm = dm.n_mels;
        let half = |v: usize| (v - 1) / 2 + 1;
        let (f1, f2, f3) = (half(nm), half(half(nm)), half(half(half(nm))));
        let (l1, l2, l3) = (p.lens, p.lens + 4 * b as u64, p.lens + 8 * b as u64);
        let g = |n: usize| n.div_ceil(256);
        let w = &self.w;
        // ---- subsampling
        let n = b * p.t2 * f2 * c;
        self.launch("sub_conv01", b * p.t2 * f2, c as u32, 0, &[
            P(p.mel), P(w.conv[0].0), P(w.conv[0].1), P(w.conv[1].0), P(w.conv[1].1), P(l1), P(l2), P(p.s1),
            I(b as i32), I(p.tm as i32), I(nm as i32), I(p.t1 as i32), I(f1 as i32), I(p.t2 as i32), I(f2 as i32),
            I(c as i32),
        ])?;
        self.gemm_nt(cap, b * p.t2 * f2, c, c, p.s1, c, 0, w.conv[2].0, c, 0, p.s0, c, 0, 1)?;
        self.launch("bias_mask_relu", g(n), 256, 0, &[
            P(p.s0), P(w.conv[2].1), P(l2), P(p.s0), I(b as i32), I(p.t2 as i32), I(f2 as i32), I(c as i32),
        ])?;
        let n = b * t * f3 * c;
        self.launch("sub_dwconv", g(n), 256, 0, &[
            P(p.s0), P(w.conv[3].0), P(w.conv[3].1), P(l3), P(p.s2),
            I(b as i32), I(p.t2 as i32), I(f2 as i32), I(t as i32), I(f3 as i32), I(c as i32),
        ])?;
        self.gemm_nt(cap, b * t * f3, c, c, p.s2, c, 0, w.conv[4].0, c, 0, p.s1, c, 0, 1)?;
        self.launch("bias_mask_relu", g(n), 256, 0, &[
            P(p.s1), P(w.conv[4].1), P(l3), P(p.s1), I(b as i32), I(t as i32), I(f3 as i32), I(c as i32),
        ])?;
        self.gemm_nt(cap, bt, d, f3 * c, p.s1, f3 * c, 0, w.proj.0, f3 * c, 0, p.x, d, 0, 1)?;
        self.launch("add_bias", g(bt * d), 256, 0, &[P(p.x), P(w.proj.1), L((bt * d) as i64), I(d as i32)])?;

        // ---- conformer layers
        let rows = bt;
        let ln = |x: u64, gb: (u64, u64), out: u64| -> Result<()> {
            self.launch("layer_norm", rows, 256, 0, &[P(x), P(gb.0), P(gb.1), P(out), I(d as i32), F(dm.eps)])
        };
        // x <- x + alpha * y ; out <- LN(x)   (xo may alias x)
        let add_ln = |x: u64, y: u64, alpha: f32, gb: (u64, u64), xo: u64, out: u64| -> Result<()> {
            self.launch("add_layer_norm", rows, 256, d * 4, &[
                P(x), P(y), F(alpha), P(gb.0), P(gb.1), P(xo), P(out), I(d as i32), F(dm.eps),
            ])
        };
        let ffn = |w1: &W, w2: &W| -> Result<()> {
            self.wgemm(cap, bt, p.hn, w1, p.ff)?;
            self.launch("silu8", g(bt * dm.ff / 8), 256, 0, &[P(p.ff), P(p.ff), L((bt * dm.ff / 8) as i64)])?;
            self.wgemm(cap, bt, p.ff, w2, p.y)
        };
        let np = 2 * t - 1;
        let ld_pos = dm.layers * d;
        let scale = 1.0 / (dk as f32).sqrt();
        ln(p.x, w.layers[0].ff1_norm, p.hn)?;
        for (li, l) in w.layers.iter().enumerate() {
            ffn(&l.ff1_w1, &l.ff1_w2)?;
            add_ln(p.x, p.y, dm.ffr, l.att_norm, p.x, p.hn)?;
            // attention
            self.wgemm(cap, bt, p.hn, &l.wqkv, p.qkv)?;
            self.launch("qkv_split", g(bt * d), 256, 0, &[
                P(p.qkv), P(l.pos_u), P(l.pos_v), P(p.qu), P(p.qv), P(p.k), P(p.v),
                I(b as i32), I(t as i32), I(h as i32), I(dk as i32),
            ])?;
            // ac[b,h] = qu[b,h] . k[b,h]^T
            self.gemm_nt(cap, t, t, dk, p.qu, dk, t * dk, p.k, dk, t * dk, p.ac, t, t * t, b * h)?;
            // bd[h] = qv[h] . P_l[h]^T, P_l[h] read in place from the shared table (rows tmax-t .. tmax+t-2)
            let p0 = pos + (((tmax - t) * ld_pos + li * d) * 2) as u64;
            self.gemm_nt(cap, bt, np, dk, p.qv, dk, bt * dk, p0, ld_pos, dk, p.bd, np, bt * np, h)?;
            let block = if t >= 512 { 256 } else { 128 };
            self.launch("relpos_softmax", b * h * t, block, t * 4, &[
                P(p.ac), P(p.bd), P(l3), P(p.ac), I(b as i32), I(h as i32), I(t as i32), F(scale),
            ])?;
            // o[b,h] = probs[b,h] . v[b,h]
            self.gemm_nn(cap, t, dk, t, p.ac, t, t * t, p.v, dk, t * dk, p.o, dk, t * dk, b * h)?;
            self.launch("merge_heads", g(bt * d), 256, 0, &[P(p.o), P(p.m), I(b as i32), I(h as i32), I(t as i32), I(dk as i32)])?;
            self.wgemm(cap, bt, p.m, &l.wo, p.y)?;
            add_ln(p.x, p.y, 1.0, l.conv_norm, p.x, p.hn)?;
            // convolution module
            self.wgemm(cap, bt, p.hn, &l.pw1, p.pw)?;
            self.launch("glu_mask", g(bt * d), 256, 0, &[P(p.pw), P(l3), P(p.g), I(b as i32), I(t as i32), I(d as i32)])?;
            self.launch("dwconv1d_silu", g(bt * d), 256, 0, &[
                P(p.g), P(l.dw_w), P(l.dw_b), P(p.g2), I(b as i32), I(t as i32), I(d as i32), I(dm.kconv as i32),
            ])?;
            self.wgemm(cap, bt, p.g2, &l.pw2, p.y)?;
            add_ln(p.x, p.y, 1.0, l.ff2_norm, p.x, p.hn)?;
            ffn(&l.ff2_w1, &l.ff2_w2)?;
            // x <- LN_out(x + ffr * y)
            add_ln(p.x, p.y, dm.ffr, l.out_norm, p.xs, p.x)?;
            if li + 1 < w.layers.len() {
                ln(p.x, w.layers[li + 1].ff1_norm, p.hn)?;
            }
        }
        Ok(())
    }

    /// mel: host features [B, Tm, n_mels] already padded to `tm` (a multiple of FRAME_BUCKET).
    /// Returns a fresh [B, T, d] fp16 tensor.
    pub fn forward(&self, mel: &[f16], b: usize, tm: usize, mel_lens: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let half = |v: &[usize]| v.iter().map(|&l| (l.max(1) - 1) / 2 + 1).collect::<Vec<_>>();
        let l1 = half(mel_lens);
        let l2 = half(&l1);
        let l3 = half(&l2);
        let probe = self.plan(0, b, tm);
        let (pos, tmax) = self.pos_table(probe.t)?;
        let mut ws = self.ws.lock().unwrap();
        if ws.bytes < probe.bytes {
            ws.graphs.clear();
            ws.order.clear();
            ws.buf = None;
            let bytes = probe.bytes.next_power_of_two().max(64 << 20);
            ws.buf = Some(gpu::empty(&self.dev, bytes / 4, DType::U32)?);
            ws.bytes = bytes;
        }
        let base = gpu::ptr(ws.buf.as_ref().unwrap())?;
        let p = self.plan(base, b, tm);
        // inputs
        let lens: Vec<u32> = l1.iter().chain(&l2).chain(&l3).map(|&v| v as u32).collect();
        unsafe {
            result::memcpy_htod_async(p.mel, mel, self.stream.cu_stream())?;
            result::memcpy_htod_async(p.lens, &lens, self.stream.cu_stream())?;
        }
        let key = (b, tm);
        if !ws.graphs.contains_key(&key) {
            if self.gemm.tune {
                // direct warm-up pass: autotunes every GEMM shape not seen before (synchronises)
                self.record(&p, pos, tmax, false)?;
            }
            let graph = gpu::capture_graph(&self.stream, || self.record(&p, pos, tmax, true))?;
            if ws.order.len() >= MAX_GRAPHS {
                let old = ws.order.remove(0);
                ws.graphs.remove(&old);
            }
            ws.graphs.insert(key, graph);
            ws.order.push(key);
        }
        ws.graphs[&key].launch()?;
        // copy the output out of the workspace
        let out = gpu::empty(&self.dev, (b, p.t, self.dims.d), DType::F16)?;
        unsafe {
            result::memcpy_dtod_async(gpu::ptr(&out)?, p.x, b * p.t * self.dims.d * 2, self.stream.cu_stream())?;
        }
        Ok((out, l3))
    }
}

