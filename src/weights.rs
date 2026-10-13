//! Reading `model.safetensors`, expanding the compact quantised encoding to fp16.
//!
//! Weights may be stored dense or in the encoding written by `tools/export_safetensors.py`:
//! * `W.q4` (u8 [N, K/2], low nibble = even k) + `W.lut` (f16 [N/16, 16]) + `W.scale` (f16 [N]):
//!   `W[n, k] = f16(lut[n / 16][idx(n, k)] * scale[n])` (palettized 2/3/4-bit linears)
//! * `W.q8` (i8, any shape) + `W.qscale` (f16, one per leading row): `W = f16(q8 * qscale[row])`
//!
//! Products are computed in fp32 and rounded once, which reproduces the dense export bit for bit.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use half::f16;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

fn f16s(bytes: &[u8]) -> Vec<f16> {
    bytes.chunks_exact(2).map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]]))).collect()
}

/// All tensors of the file as CPU tensors (quantised weights expanded to fp16).
pub fn load(path: &Path) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    let get = |n: &str| st.tensor(n).with_context(|| format!("missing {n}"));
    let mut out = HashMap::new();
    for name in &names {
        let view = st.tensor(name)?;
        let shape = view.shape().to_vec();
        if let Some(base) = name.strip_suffix(".q4") {
            let (n, half_k) = (shape[0], shape[1]);
            let lut = f16s(get(&format!("{base}.lut"))?.data());
            let scale = f16s(get(&format!("{base}.scale"))?.data());
            let q = view.data();
            let mut w = vec![f16::ZERO; n * half_k * 2];
            w.par_chunks_mut(half_k * 2).enumerate().for_each(|(r, row)| {
                let l = &lut[(r / 16) * 16..(r / 16) * 16 + 16];
                let s = scale[r].to_f32();
                for (j, &b) in q[r * half_k..(r + 1) * half_k].iter().enumerate() {
                    row[2 * j] = f16::from_f32(l[(b & 15) as usize].to_f32() * s);
                    row[2 * j + 1] = f16::from_f32(l[(b >> 4) as usize].to_f32() * s);
                }
            });
            out.insert(base.to_string(), Tensor::from_vec(w, (n, half_k * 2), &Device::Cpu)?);
        } else if let Some(base) = name.strip_suffix(".q8") {
            let scale = f16s(get(&format!("{base}.qscale"))?.data());
            let rows = shape[0];
            let per = view.data().len() / rows;
            let q = view.data();
            let mut w = vec![f16::ZERO; rows * per];
            w.par_chunks_mut(per).enumerate().for_each(|(r, row)| {
                let s = scale[r].to_f32();
                for (o, &b) in row.iter_mut().zip(&q[r * per..(r + 1) * per]) {
                    *o = f16::from_f32(b as i8 as f32 * s);
                }
            });
            out.insert(base.to_string(), Tensor::from_vec(w, shape, &Device::Cpu)?);
        } else if [".lut", ".scale", ".qscale"].iter().any(|s| name.ends_with(s))
            && (names.contains(&format!("{}.q4", name.rsplit_once('.').unwrap().0))
                || names.contains(&format!("{}.q8", name.rsplit_once('.').unwrap().0)))
        {
            continue;
        } else {
            let dtype = match view.dtype() {
                Dtype::F16 => DType::F16,
                Dtype::BF16 => DType::BF16,
                Dtype::F32 => DType::F32,
                d => bail!("unsupported dtype {d:?} for {name}"),
            };
            out.insert(name.clone(), Tensor::from_raw_buffer(view.data(), dtype, &shape, &Device::Cpu)?);
        }
    }
    Ok(out)
}
