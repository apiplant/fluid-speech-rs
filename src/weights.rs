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
    // Transformers checkpoints use Parakeet module names and keep BatchNorm separate. Import those weights
    // into the engine's internal NeMo-style layout so no Python conversion step is needed.
    if st.names().contains(&"decoder.embedding.weight") {
        return load_transformers(&st);
    }
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

fn tensor_from_view(view: &safetensors::tensor::TensorView<'_>, shape: &[usize]) -> Result<Tensor> {
    let dtype = match view.dtype() {
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::F32 => DType::F32,
        d => bail!("unsupported Transformers tensor dtype {d:?}"),
    };
    Ok(Tensor::from_raw_buffer(view.data(), dtype, shape, &Device::Cpu)?)
}

fn add_view(st: &SafeTensors<'_>, out: &mut HashMap<String, Tensor>, src: &str, dst: &str) -> Result<()> {
    let view = st.tensor(src).with_context(|| format!("missing Transformers tensor {src}"))?;
    out.insert(dst.to_string(), tensor_from_view(&view, view.shape())?);
    Ok(())
}

fn add_reshaped_view(st: &SafeTensors<'_>, out: &mut HashMap<String, Tensor>, src: &str, dst: &str, shape: &[usize]) -> Result<()> {
    let view = st.tensor(src).with_context(|| format!("missing Transformers tensor {src}"))?;
    let count: usize = view.shape().iter().product();
    let target_count: usize = shape.iter().product();
    if count != target_count {
        bail!("cannot reshape Transformers tensor {src} from {:?} to {shape:?}", view.shape());
    }
    out.insert(dst.to_string(), tensor_from_view(&view, shape)?);
    Ok(())
}

fn read_f32(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let view = st.tensor(name).with_context(|| format!("missing Transformers tensor {name}"))?;
    if view.dtype() != Dtype::F32 {
        bail!("expected float32 in Transformers tensor {name}, found {:?}", view.dtype());
    }
    let values = view.data().chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((values, view.shape().to_vec()))
}

fn load_transformers(st: &SafeTensors<'_>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();

    // NeMo's 2-D subsampling kernels are represented as 4-D Conv2d tensors in Transformers.
    for i in [0, 2, 3, 5, 6] {
        let prefix = format!("encoder.subsampling.layers.{i}");
        let target = format!("encoder.pre_encode.conv.{i}");
        let view = st.tensor(&format!("{prefix}.weight"))
            .with_context(|| format!("missing Transformers tensor {prefix}.weight"))?;
        let shape = view.shape();
        let flat_shape = vec![shape[0], shape[1..].iter().product()];
        add_reshaped_view(st, &mut out, &format!("{prefix}.weight"), &format!("{target}.weight"), &flat_shape)?;
        add_view(st, &mut out, &format!("{prefix}.bias"), &format!("{target}.bias"))?;
    }
    add_view(st, &mut out, "encoder.subsampling.linear.weight", "encoder.pre_encode.out.weight")?;
    add_view(st, &mut out, "encoder.subsampling.linear.bias", "encoder.pre_encode.out.bias")?;

    let n_layers = st.names().iter().filter_map(|name| {
        let rest = name.strip_prefix("encoder.layers.")?;
        rest.split('.').next()?.parse::<usize>().ok()
    }).max().map_or(0, |n| n + 1);
    let layer_norms = ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"];
    let feed_forwards = ["feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2"];
    for i in 0..n_layers {
        let p = format!("encoder.layers.{i}");
        for name in layer_norms {
            for suffix in ["weight", "bias"] {
                let key = format!("{p}.{name}.{suffix}");
                add_view(st, &mut out, &key, &key)?;
            }
        }
        for name in feed_forwards {
            let key = format!("{p}.{name}.weight");
            add_view(st, &mut out, &key, &key)?;
        }
        for (src, dst) in [
            ("self_attn.q_proj.weight", "self_attn.linear_q.weight"),
            ("self_attn.k_proj.weight", "self_attn.linear_k.weight"),
            ("self_attn.v_proj.weight", "self_attn.linear_v.weight"),
            ("self_attn.o_proj.weight", "self_attn.linear_out.weight"),
            ("self_attn.relative_k_proj.weight", "self_attn.linear_pos.weight"),
            ("self_attn.bias_u", "self_attn.pos_bias_u"),
            ("self_attn.bias_v", "self_attn.pos_bias_v"),
        ] {
            add_view(st, &mut out, &format!("{p}.{src}"), &format!("{p}.{dst}"))?;
        }
        for name in ["conv.pointwise_conv1.weight", "conv.pointwise_conv2.weight"] {
            let src = format!("{p}.{name}");
            let view = st.tensor(&src).with_context(|| format!("missing Transformers tensor {src}"))?;
            let shape = view.shape();
            let flat_shape = vec![shape[0], shape[1..].iter().product()];
            add_reshaped_view(st, &mut out, &src, &src, &flat_shape)?;
        }

        // Fold inference BatchNorm into the depthwise convolution. The engine uses this fused form, as does
        // the equivalent Core ML export used by Fluid Speech Mini.
        let (mut dw, shape) = read_f32(st, &format!("{p}.conv.depthwise_conv.weight"))?;
        let (gamma, _) = read_f32(st, &format!("{p}.conv.norm.weight"))?;
        let (beta, _) = read_f32(st, &format!("{p}.conv.norm.bias"))?;
        let (mean, _) = read_f32(st, &format!("{p}.conv.norm.running_mean"))?;
        let (var, _) = read_f32(st, &format!("{p}.conv.norm.running_var"))?;
        let channels = shape[0];
        let kernel = shape.iter().skip(1).product::<usize>();
        let mut bias = vec![0.0f32; channels];
        for c in 0..channels {
            let scale = gamma[c] / (var[c] + 1e-5).sqrt();
            for x in &mut dw[c * kernel..(c + 1) * kernel] {
                *x *= scale;
            }
            bias[c] = beta[c] - mean[c] * scale;
        }
        out.insert(format!("{p}.conv.depthwise_conv.weight"), Tensor::from_vec(dw, (channels, kernel), &Device::Cpu)?);
        out.insert(format!("{p}.conv.depthwise_conv.bias"), Tensor::from_vec(bias, channels, &Device::Cpu)?);
    }

    for (src, dst) in [
        ("decoder.embedding.weight", "decoder.prediction.embed.weight"),
        ("encoder_projector.weight", "joint.enc.weight"),
        ("encoder_projector.bias", "joint.enc.bias"),
        ("decoder.decoder_projector.weight", "joint.pred.weight"),
        ("decoder.decoder_projector.bias", "joint.pred.bias"),
        ("joint.head.weight", "joint.joint_net.2.weight"),
        ("joint.head.bias", "joint.joint_net.2.bias"),
    ] {
        add_view(st, &mut out, src, dst)?;
    }
    let n_pred_layers = (0..).take_while(|layer| st.names().contains(&format!("decoder.lstm.weight_ih_l{layer}").as_str())).count();
    for layer in 0..n_pred_layers {
        for suffix in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
            let src = format!("decoder.lstm.{suffix}_l{layer}");
            let dst = format!("decoder.prediction.dec_rnn.lstm.{suffix}_l{layer}");
            add_view(st, &mut out, &src, &dst)?;
        }
    }
    Ok(out)
}
