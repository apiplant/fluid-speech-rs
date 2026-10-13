#![cfg_attr(not(feature = "cuda"), allow(unused_variables, unused_mut))]
//! Fused operations used by the encoder and the TDT decoder.
//!
//! Each op has a CUDA implementation (kernels in `kernels/fluid.cu`, fp16 activations / fp32 math for the
//! encoder, fp32 for the decoder) and a CPU implementation (fp32, rayon-parallel).

use candle_core::{bail, DType, Device, Result, Tensor, D};
use rayon::prelude::*;

fn is_cuda(t: &Tensor) -> bool {
    matches!(t.device(), Device::Cuda(_))
}

fn cpu_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
}

fn cpu_u32(t: &Tensor) -> Result<Vec<u32>> {
    t.flatten_all()?.to_vec1::<u32>()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(feature = "cuda")]
pub(crate) mod gpu {
    use candle_core::cuda_backend::cudarc::driver::{
        CudaFunction, CudaModule, DevicePtr, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{bail, CudaDevice, CudaStorage, DType, Result, Shape, Storage, Tensor};
    use half::f16;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fluid.ptx"));

    struct Kernels {
        module: Arc<CudaModule>,
        funcs: HashMap<&'static str, CudaFunction>,
    }

    /// Loaded kernel module per GPU ordinal.
    static KERNELS: OnceLock<Mutex<HashMap<usize, Kernels>>> = OnceLock::new();

    pub fn func(dev: &CudaDevice, name: &'static str) -> Result<CudaFunction> {
        let cell = KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut g = cell.lock().unwrap();
        let ctx = dev.cuda_stream().context().clone();
        let k = match g.entry(ctx.ordinal()) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let module = ctx.load_module(PTX.into()).w()?;
                e.insert(Kernels { module, funcs: HashMap::new() })
            }
        };
        if let Some(f) = k.funcs.get(name) {
            return Ok(f.clone());
        }
        let f = k.module.load_function(name).w()?;
        k.funcs.insert(name, f.clone());
        Ok(f)
    }

    /// Raw device pointer of a contiguous cuda tensor (start offset included).
    pub fn ptr(t: &Tensor) -> Result<u64> {
        let (s, l) = t.storage_and_layout();
        if !l.is_contiguous() {
            bail!("fluid kernels need contiguous inputs, got {:?}", l);
        }
        let cs = match &*s {
            Storage::Cuda(c) => c,
            _ => bail!("expected a cuda tensor"),
        };
        let stream = cs.device.cuda_stream();
        let (base, size) = match &cs.slice {
            CudaStorageSlice::F16(x) => (x.device_ptr(&stream).0, 2),
            CudaStorageSlice::F32(x) => (x.device_ptr(&stream).0, 4),
            CudaStorageSlice::U32(x) => (x.device_ptr(&stream).0, 4),
            CudaStorageSlice::I64(x) => (x.device_ptr(&stream).0, 8),
            CudaStorageSlice::U8(x) => (x.device_ptr(&stream).0, 1),
            _ => bail!("unsupported dtype {:?}", t.dtype()),
        };
        Ok(base + (l.start_offset() * size) as u64)
    }

    /// Uninitialised contiguous tensor.
    pub fn empty<S: Into<Shape>>(dev: &CudaDevice, shape: S, dtype: DType) -> Result<Tensor> {
        let shape = shape.into();
        let n = shape.elem_count();
        let slice = unsafe {
            match dtype {
                DType::F16 => CudaStorageSlice::F16(dev.alloc::<f16>(n)?),
                DType::F32 => CudaStorageSlice::F32(dev.alloc::<f32>(n)?),
                DType::U32 => CudaStorageSlice::U32(dev.alloc::<u32>(n)?),
                DType::U8 => CudaStorageSlice::U8(dev.alloc::<u8>(n)?),
                _ => bail!("unsupported dtype {dtype:?}"),
            }
        };
        let storage = Storage::Cuda(CudaStorage { slice, device: dev.clone() });
        Ok(Tensor::from_storage(storage, shape, BackpropOp::none(), false))
    }

    pub enum A {
        P(u64),
        I(i32),
        L(i64),
        F(f32),
    }

    pub fn launch(dev: &CudaDevice, name: &'static str, grid: u32, block: u32, smem: u32, args: &[A]) -> Result<()> {
        let f = func(dev, name)?;
        let stream = dev.cuda_stream();
        let mut b = stream.launch_builder(&f);
        for a in args {
            match a {
                A::P(p) => b.arg(p),
                A::I(i) => b.arg(i),
                A::L(l) => b.arg(l),
                A::F(x) => b.arg(x),
            };
        }
        let cfg = LaunchConfig { grid_dim: (grid.max(1), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem };
        unsafe { b.launch(cfg) }.w()?;
        Ok(())
    }

    /// Capture the work enqueued by `record` on `stream` into a CUDA graph. Capturing executes nothing, so a
    /// capture invalidated from outside (another thread's implicitly synchronising call, e.g. lazy module
    /// loading by another engine) is simply retried.
    pub fn capture_graph(
        stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
        mut record: impl FnMut() -> anyhow::Result<()>,
    ) -> anyhow::Result<candle_core::cuda_backend::cudarc::driver::CudaGraph> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let mut last = None;
        for _ in 0..16 {
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let rec = record();
            let graph = stream.end_capture(sys::CUgraphInstantiate_flags(0));
            match (rec, graph) {
                (Ok(()), Ok(Some(g))) => return Ok(g),
                (Ok(()), Ok(None)) => anyhow::bail!("captured an empty graph"),
                (Err(e), _) => last = Some(e),
                (_, Err(e)) => last = Some(e.into()),
            }
            std::thread::yield_now();
        }
        Err(last.unwrap().context("CUDA graph capture kept failing"))
    }

    pub fn grid(n: usize, block: u32) -> u32 {
        n.div_ceil(block as usize) as u32
    }

    pub fn dev(t: &Tensor) -> Result<CudaDevice> {
        match t.device() {
            candle_core::Device::Cuda(d) => Ok(d.clone()),
            _ => bail!("expected cuda device"),
        }
    }
}

#[cfg(feature = "cuda")]
use gpu::A;

macro_rules! cuda_only {
    () => {
        bail!("built without the `cuda` feature")
    };
}

// ------------------------------------------------------------------------------------------------ subsampling

/// conv0 (1 -> C channels, 3x3 / stride 2 / pad 1) + bias + time mask + relu.
/// x: [B, T, F], w: [C, 9], bias: [C], lens: [B] (output lengths) -> [B, To, Fo, C]
pub fn sub_conv0(x: &Tensor, w: &Tensor, bias: &Tensor, lens: &Tensor) -> Result<Tensor> {
    let (b, t, f) = x.dims3()?;
    let c = w.dim(0)?;
    let (to, fo) = ((t - 1) / 2 + 1, (f - 1) / 2 + 1);
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, (b, to, fo, c), x.dtype())?;
            let n = b * to * fo * c;
            gpu::launch(&dev, "sub_conv0", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(w)?), A::P(gpu::ptr(bias)?), A::P(gpu::ptr(lens)?), A::P(gpu::ptr(&out)?),
                A::I(b as i32), A::I(t as i32), A::I(f as i32), A::I(to as i32), A::I(fo as i32), A::I(c as i32),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (xv, wv, bv, lv) = (cpu_vec(x)?, cpu_vec(w)?, cpu_vec(bias)?, cpu_u32(lens)?);
    let mut out = vec![0f32; b * to * fo * c];
    out.par_chunks_mut(fo * c).enumerate().for_each(|(r, row)| {
        let (bi, ti) = (r / to, r % to);
        if ti >= lv[bi] as usize {
            return;
        }
        for fi in 0..fo {
            for ci in 0..c {
                let mut acc = bv[ci];
                for i in 0..3 {
                    let tt = (2 * ti + i) as isize - 1;
                    if tt < 0 || tt >= t as isize {
                        continue;
                    }
                    for j in 0..3 {
                        let ff = (2 * fi + j) as isize - 1;
                        if ff < 0 || ff >= f as isize {
                            continue;
                        }
                        acc += wv[ci * 9 + i * 3 + j] * xv[(bi * t + tt as usize) * f + ff as usize];
                    }
                }
                row[fi * c + ci] = acc.max(0.0);
            }
        }
    });
    Tensor::from_vec(out, (b, to, fo, c), x.device())
}

/// depthwise 3x3 / stride 2 / pad 1 + bias + time mask. x: [B, Ti, Fi, C] -> [B, To, Fo, C]
pub fn sub_dwconv(x: &Tensor, w: &Tensor, bias: &Tensor, lens: &Tensor) -> Result<Tensor> {
    let (b, ti_, fi_, c) = x.dims4()?;
    let (to, fo) = ((ti_ - 1) / 2 + 1, (fi_ - 1) / 2 + 1);
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, (b, to, fo, c), x.dtype())?;
            let n = b * to * fo * c;
            gpu::launch(&dev, "sub_dwconv", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(w)?), A::P(gpu::ptr(bias)?), A::P(gpu::ptr(lens)?), A::P(gpu::ptr(&out)?),
                A::I(b as i32), A::I(ti_ as i32), A::I(fi_ as i32), A::I(to as i32), A::I(fo as i32), A::I(c as i32),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (xv, wv, bv, lv) = (cpu_vec(x)?, cpu_vec(w)?, cpu_vec(bias)?, cpu_u32(lens)?);
    let mut out = vec![0f32; b * to * fo * c];
    out.par_chunks_mut(fo * c).enumerate().for_each(|(r, row)| {
        let (bi, t) = (r / to, r % to);
        if t >= lv[bi] as usize {
            return;
        }
        for fo_i in 0..fo {
            let o = &mut row[fo_i * c..(fo_i + 1) * c];
            o.copy_from_slice(&bv);
            for i in 0..3 {
                let tt = (2 * t + i) as isize - 1;
                if tt < 0 || tt >= ti_ as isize {
                    continue;
                }
                for j in 0..3 {
                    let ff = (2 * fo_i + j) as isize - 1;
                    if ff < 0 || ff >= fi_ as isize {
                        continue;
                    }
                    let src = &xv[((bi * ti_ + tt as usize) * fi_ + ff as usize) * c..][..c];
                    for ci in 0..c {
                        o[ci] += wv[ci * 9 + i * 3 + j] * src[ci];
                    }
                }
            }
        }
    });
    Tensor::from_vec(out, (b, to, fo, c), x.device())
}

/// relu((x + bias[c]) * mask[b, t]) for x: [B, T, F, C]
pub fn bias_mask_relu(x: &Tensor, bias: &Tensor, lens: &Tensor) -> Result<Tensor> {
    let (b, t, f, c) = x.dims4()?;
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, (b, t, f, c), x.dtype())?;
            let n = b * t * f * c;
            gpu::launch(&dev, "bias_mask_relu", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(bias)?), A::P(gpu::ptr(lens)?), A::P(gpu::ptr(&out)?),
                A::I(b as i32), A::I(t as i32), A::I(f as i32), A::I(c as i32),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (mut xv, bv, lv) = (cpu_vec(x)?, cpu_vec(bias)?, cpu_u32(lens)?);
    xv.par_chunks_mut(f * c).enumerate().for_each(|(r, row)| {
        let valid = (r % t) < lv[r / t] as usize;
        for (i, v) in row.iter_mut().enumerate() {
            *v = if valid { (*v + bv[i % c]).max(0.0) } else { 0.0 };
        }
    });
    Tensor::from_vec(xv, (b, t, f, c), x.device())
}

// ------------------------------------------------------------------------------------------------ norms

fn ln_row(x: &[f32], g: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let d = x.len() as f32;
    let mean = x.iter().sum::<f32>() / d;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d;
    let rstd = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * rstd * g[i] + b[i];
    }
}

/// LayerNorm over the last dim.
pub fn layer_norm(x: &Tensor, g: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let rows = x.elem_count() / d;
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, x.shape(), x.dtype())?;
            gpu::launch(&dev, "layer_norm", rows as u32, 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(g)?), A::P(gpu::ptr(b)?), A::P(gpu::ptr(&out)?), A::I(d as i32), A::F(eps),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (xv, gv, bv) = (cpu_vec(x)?, cpu_vec(g)?, cpu_vec(b)?);
    let mut out = vec![0f32; xv.len()];
    out.par_chunks_mut(d).zip(xv.par_chunks(d)).for_each(|(o, xr)| ln_row(xr, &gv, &bv, eps, o));
    Tensor::from_vec(out, x.shape(), x.device())
}

/// xo = x + alpha * y; returns (xo, LayerNorm(xo)).
pub fn add_layer_norm(x: &Tensor, y: &Tensor, alpha: f32, g: &Tensor, b: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    let d = x.dim(D::Minus1)?;
    let rows = x.elem_count() / d;
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let xo = gpu::empty(&dev, x.shape(), x.dtype())?;
            let out = gpu::empty(&dev, x.shape(), x.dtype())?;
            gpu::launch(&dev, "add_layer_norm", rows as u32, 256, (d * 4) as u32, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(y)?), A::F(alpha), A::P(gpu::ptr(g)?), A::P(gpu::ptr(b)?),
                A::P(gpu::ptr(&xo)?), A::P(gpu::ptr(&out)?), A::I(d as i32), A::F(eps),
            ])?;
            return Ok((xo, out));
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (mut xv, yv, gv, bv) = (cpu_vec(x)?, cpu_vec(y)?, cpu_vec(g)?, cpu_vec(b)?);
    let mut out = vec![0f32; xv.len()];
    out.par_chunks_mut(d).zip(xv.par_chunks_mut(d)).zip(yv.par_chunks(d)).for_each(|((o, xr), yr)| {
        for i in 0..d {
            xr[i] += alpha * yr[i];
        }
        ln_row(xr, &gv, &bv, eps, o);
    });
    Ok((Tensor::from_vec(xv, x.shape(), x.device())?, Tensor::from_vec(out, x.shape(), x.device())?))
}

/// x + alpha * y
pub fn axpy(x: &Tensor, y: &Tensor, alpha: f32) -> Result<Tensor> {
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, x.shape(), x.dtype())?;
            let n = x.elem_count();
            gpu::launch(&dev, "axpy", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(y)?), A::F(alpha), A::P(gpu::ptr(&out)?), A::L(n as i64),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    x + (y * alpha as f64)?
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, x.shape(), x.dtype())?;
            let n = x.elem_count();
            gpu::launch(&dev, "silu", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(&out)?), A::L(n as i64),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    x.silu()
}

// ------------------------------------------------------------------------------------------------ attention

/// qkv: [B, T, 3*H*dk] -> (q+u: [B, H, T, dk], q+v: [H, B, T, dk], k: [B, H, T, dk], v: [B, H, T, dk])
pub fn qkv_split(qkv: &Tensor, pu: &Tensor, pv: &Tensor, h: usize) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (b, t, d3) = qkv.dims3()?;
    let dk = d3 / 3 / h;
    if is_cuda(qkv) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(qkv)?;
            let dt = qkv.dtype();
            let qu = gpu::empty(&dev, (b, h, t, dk), dt)?;
            let qv = gpu::empty(&dev, (h, b, t, dk), dt)?;
            let k = gpu::empty(&dev, (b, h, t, dk), dt)?;
            let v = gpu::empty(&dev, (b, h, t, dk), dt)?;
            let n = b * t * h * dk;
            gpu::launch(&dev, "qkv_split", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(qkv)?), A::P(gpu::ptr(pu)?), A::P(gpu::ptr(pv)?), A::P(gpu::ptr(&qu)?),
                A::P(gpu::ptr(&qv)?), A::P(gpu::ptr(&k)?), A::P(gpu::ptr(&v)?),
                A::I(b as i32), A::I(t as i32), A::I(h as i32), A::I(dk as i32),
            ])?;
            return Ok((qu, qv, k, v));
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let x = qkv.reshape((b, t, 3, h, dk))?;
    let q = x.narrow(2, 0, 1)?.squeeze(2)?; // [B, T, H, dk]
    let k = x.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
    let v = x.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
    let qu = q.broadcast_add(pu)?.transpose(1, 2)?.contiguous()?;
    let qv = q.broadcast_add(pv)?.permute((2, 0, 1, 3))?.contiguous()?;
    Ok((qu, qv, k, v))
}

/// softmax over j of (ac[b,h,i,j] + bd[h,b,i,j+T-1-i]) * scale with keys j >= lens[b] masked.
pub fn relpos_softmax(ac: &Tensor, bd: &Tensor, lens: &Tensor, scale: f32) -> Result<Tensor> {
    let (b, h, t, _) = ac.dims4()?;
    if is_cuda(ac) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(ac)?;
            let out = gpu::empty(&dev, (b, h, t, t), ac.dtype())?;
            let block = if t >= 512 { 256 } else { 128 };
            gpu::launch(&dev, "relpos_softmax", (b * h * t) as u32, block, (t * 4) as u32, &[
                A::P(gpu::ptr(ac)?), A::P(gpu::ptr(bd)?), A::P(gpu::ptr(lens)?), A::P(gpu::ptr(&out)?),
                A::I(b as i32), A::I(h as i32), A::I(t as i32), A::F(scale),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (acv, bdv, lv) = (cpu_vec(ac)?, cpu_vec(bd)?, cpu_u32(lens)?);
    let p = 2 * t - 1;
    let mut out = vec![0f32; b * h * t * t];
    out.par_chunks_mut(t).enumerate().for_each(|(r, row)| {
        let i = r % t;
        let hh = (r / t) % h;
        let bb = r / t / h;
        let len = lv[bb] as usize;
        let bdr = &bdv[((hh * b + bb) * t + i) * p + (t - 1 - i)..];
        let acr = &acv[r * t..(r + 1) * t];
        let mut m = f32::NEG_INFINITY;
        for j in 0..t {
            let s = (acr[j] + bdr[j]) * scale + if j >= len { -10000.0 } else { 0.0 };
            row[j] = s;
            m = m.max(s);
        }
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        row.iter_mut().for_each(|v| *v *= inv);
    });
    Tensor::from_vec(out, (b, h, t, t), ac.device())
}

/// [B, H, T, dk] -> [B, T, H*dk]
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, t, dk) = x.dims4()?;
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, (b, t, h * dk), x.dtype())?;
            let n = b * t * h * dk;
            gpu::launch(&dev, "merge_heads", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(&out)?), A::I(b as i32), A::I(h as i32), A::I(t as i32), A::I(dk as i32),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    x.transpose(1, 2)?.contiguous()?.reshape((b, t, h * dk))
}

// ------------------------------------------------------------------------------------------------ conv module

/// GLU over the last dim + time mask: x [B, T, 2D] -> [B, T, D]
pub fn glu_mask(x: &Tensor, lens: &Tensor) -> Result<Tensor> {
    let (b, t, d2) = x.dims3()?;
    let d = d2 / 2;
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, (b, t, d), x.dtype())?;
            let n = b * t * d;
            gpu::launch(&dev, "glu_mask", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(lens)?), A::P(gpu::ptr(&out)?), A::I(b as i32), A::I(t as i32), A::I(d as i32),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (xv, lv) = (cpu_vec(x)?, cpu_u32(lens)?);
    let mut out = vec![0f32; b * t * d];
    out.par_chunks_mut(d).enumerate().for_each(|(r, o)| {
        if (r % t) >= lv[r / t] as usize {
            return;
        }
        let row = &xv[r * 2 * d..(r + 1) * 2 * d];
        for c in 0..d {
            o[c] = row[c] * sigmoid(row[d + c]);
        }
    });
    Tensor::from_vec(out, (b, t, d), x.device())
}

/// depthwise conv1d over time ("same" padding) + bias + silu. x: [B, T, D], w: [D, K], bias: [D]
pub fn dwconv1d_silu(x: &Tensor, w: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (b, t, d) = x.dims3()?;
    let k = w.dim(1)?;
    if is_cuda(x) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(x)?;
            let out = gpu::empty(&dev, (b, t, d), x.dtype())?;
            let n = b * t * d;
            gpu::launch(&dev, "dwconv1d_silu", gpu::grid(n, 256), 256, 0, &[
                A::P(gpu::ptr(x)?), A::P(gpu::ptr(w)?), A::P(gpu::ptr(bias)?), A::P(gpu::ptr(&out)?),
                A::I(b as i32), A::I(t as i32), A::I(d as i32), A::I(k as i32),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (xv, wv, bv) = (cpu_vec(x)?, cpu_vec(w)?, cpu_vec(bias)?);
    let p = k / 2;
    let mut out = vec![0f32; b * t * d];
    out.par_chunks_mut(d).enumerate().for_each(|(r, o)| {
        let (bi, ti) = (r / t, r % t);
        o.copy_from_slice(&bv);
        for kk in 0..k {
            let tt = ti as isize + kk as isize - p as isize;
            if tt < 0 || tt >= t as isize {
                continue;
            }
            let src = &xv[(bi * t + tt as usize) * d..][..d];
            for c in 0..d {
                o[c] += wv[c * k + kk] * src[c];
            }
        }
        for v in o.iter_mut() {
            *v *= sigmoid(*v);
        }
    });
    Tensor::from_vec(out, (b, t, d), x.device())
}

// ------------------------------------------------------------------------------------------------ TDT decoding

/// z[b] = relu(ep[b, frame[b]] + dp[b]). ep: [B, T, J], dp: [B, J], frame: [B] u32
pub fn joint_hidden(ep: &Tensor, dp: &Tensor, frame: &Tensor) -> Result<Tensor> {
    let (b, t, j) = ep.dims3()?;
    if is_cuda(ep) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(ep)?;
            let out = gpu::empty(&dev, (b, j), DType::F32)?;
            gpu::launch(&dev, "joint_hidden", gpu::grid(b * j, 256), 256, 0, &[
                A::P(gpu::ptr(ep)?), A::P(gpu::ptr(dp)?), A::P(gpu::ptr(frame)?), A::P(gpu::ptr(&out)?),
                A::I(b as i32), A::I(t as i32), A::I(j as i32),
            ])?;
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let fv = cpu_u32(frame)?;
    let mut rows = Vec::with_capacity(b);
    for (bi, &f) in fv.iter().enumerate() {
        rows.push(ep.get(bi)?.get(f as usize)?);
    }
    let _ = (t, j);
    Tensor::stack(&rows, 0)?.add(dp)?.relu()
}

/// argmax over tokens [0, V) and durations [V, V+ND) per row -> host Vec of (token, duration index)
pub fn tdt_argmax(logits: &Tensor, v: usize) -> Result<Vec<(u32, u32)>> {
    let (b, n) = logits.dims2()?;
    let nd = n - v;
    if is_cuda(logits) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(logits)?;
            let out = gpu::empty(&dev, (b, 2), DType::U32)?;
            gpu::launch(&dev, "tdt_argmax", b as u32, 256, 0, &[
                A::P(gpu::ptr(logits)?), A::P(gpu::ptr(&out)?), A::I(v as i32), A::I(nd as i32),
            ])?;
            let r = out.flatten_all()?.to_vec1::<u32>()?;
            return Ok(r.chunks(2).map(|c| (c[0], c[1])).collect());
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let lv = cpu_vec(logits)?;
    Ok(lv
        .chunks(n)
        .map(|row| {
            let am = |s: &[f32]| {
                let mut bi = 0;
                for (i, x) in s.iter().enumerate() {
                    if *x > s[bi] {
                        bi = i;
                    }
                }
                bi as u32
            };
            (am(&row[..v]), am(&row[v..]))
        })
        .collect())
}

/// LSTM cell (gate order i, f, g, o). gates: [B, 4H]; rows with mask[b] == 0 keep (h, c).
pub fn lstm_cell(gates: &Tensor, h: &Tensor, c: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, hd) = h.dims2()?;
    if is_cuda(h) {
        #[cfg(feature = "cuda")]
        {
            let dev = gpu::dev(h)?;
            let ho = gpu::empty(&dev, (b, hd), DType::F32)?;
            let co = gpu::empty(&dev, (b, hd), DType::F32)?;
            gpu::launch(&dev, "lstm_cell", gpu::grid(b * hd, 256), 256, 0, &[
                A::P(gpu::ptr(gates)?), A::P(gpu::ptr(h)?), A::P(gpu::ptr(c)?), A::P(gpu::ptr(mask)?),
                A::P(gpu::ptr(&ho)?), A::P(gpu::ptr(&co)?), A::I(b as i32), A::I(hd as i32),
            ])?;
            return Ok((ho, co));
        }
        #[allow(unreachable_code)]
        {
            cuda_only!()
        }
    }
    let (gv, mut hv, mut cv, mv) = (cpu_vec(gates)?, cpu_vec(h)?, cpu_vec(c)?, cpu_u32(mask)?);
    for bi in 0..b {
        if mv[bi] == 0 {
            continue;
        }
        let g = &gv[bi * 4 * hd..(bi + 1) * 4 * hd];
        for j in 0..hd {
            let (i_, f_, g_, o_) = (sigmoid(g[j]), sigmoid(g[hd + j]), g[2 * hd + j].tanh(), sigmoid(g[3 * hd + j]));
            let cn = f_ * cv[bi * hd + j] + i_ * g_;
            cv[bi * hd + j] = cn;
            hv[bi * hd + j] = o_ * cn.tanh();
        }
    }
    Ok((Tensor::from_vec(hv, (b, hd), h.device())?, Tensor::from_vec(cv, (b, hd), h.device())?))
}
