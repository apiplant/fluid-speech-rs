"""Convert the Fluid Speech Mini Core ML package into a single safetensors file + config.

Tensor names follow NeMo's Parakeet TDT layout (pointwise convs are stored as 2-D [out, in] matrices, the depthwise
conv has its batch norm already folded in).

By default weights keep their Core ML quantisation (~320 MB), and loaders expand them to fp16:
  W.q4 (u8 [N, K/2], low nibble = even k), W.lut (f16 [N/16, 16]), W.scale (f16 [N])
      -> W[n, k] = f16(lut[n // 16, idx(n, k)] * scale[n])          (palettized 2/3/4-bit encoder linears)
  W.q8 (i8, original shape), W.qscale (f16 [rows])
      -> W = f16(q8 * qscale[row])                                    (int8 blockwise weights)
The products are computed in fp32 and rounded once. With --dense every weight is written expanded (fp16, ~1.25 GB),
identical to what loaders reconstruct.

usage: export_safetensors.py <coreml dir> <out dir> [--dense]
"""
import json
import os
import re
import shutil
import sys

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(__file__))
import mil  # noqa: E402

LINEAR_NAMES = {
    "ff1_1": "feed_forward1.linear1", "ff1_2": "feed_forward1.linear2",
    "ff2_1": "feed_forward2.linear1", "ff2_2": "feed_forward2.linear2",
    "q": "self_attn.linear_q", "k": "self_attn.linear_k", "v": "self_attn.linear_v",
    "o": "self_attn.linear_out", "pos": "self_attn.linear_pos",
    "pw1": "conv.pointwise_conv1", "pw2": "conv.pointwise_conv2",
}


def consumers(func):
    out = {}
    for op in func.ops:
        for v in op.args.values():
            refs = [v] if v[0] == "var" else (v[1] if v[0] == "tuple" else [])
            for r in refs:
                if r[0] == "var":
                    out.setdefault(r[1], []).append(op)
    return out


def arg(op, k):
    v = op.args.get(k)
    return v[1] if v is not None and v[0] == "var" else None


def producers(model):
    return {op.outputs[0][0]: op for op in model.main.ops if op.op.startswith("constexpr_")}


def q8_form(model, op):
    """(int8 data, per-row fp16 scale) of a constexpr_blockwise_shift_scale with one scale per leading row."""
    assert op.op == "constexpr_blockwise_shift_scale" and "offset" not in op.args, op.op
    data = model.literal(op.args["data"])
    scale = model.literal(op.args["scale"])
    assert data.dtype == np.int8 and scale.shape[0] == data.shape[0] and scale.size == data.shape[0], scale.shape
    return data, scale.reshape(-1).astype(np.float16)


def export_encoder(model, tensors, quant):
    c = model.constants()
    prod = producers(model)
    cons = consumers(model.main)
    layer = -1
    sub_convs = []
    pos_u = pos_v = 0
    eps = set()
    for op in model.main.ops:
        if op.op == "layer_norm":
            g = arg(op, "gamma")
            m = re.match(r"enc_layers_(\d+)_(norm_\w+?)_weight_to_fp16", g)
            layer = int(m.group(1))
            pre = f"encoder.layers.{layer}.{m.group(2)}"
            tensors[pre + ".weight"] = c[g]
            tensors[pre + ".bias"] = c[arg(op, "beta")]
            eps.add(float(c[arg(op, "epsilon")]))
        elif op.op == "linear":
            w = arg(op, "weight")
            out = op.outputs[0][0]
            m = re.match(r"blocks_(\d+)_(\w+?)_weight_to_fp16_palettized", w)
            if m:
                i, short = int(m.group(1)), m.group(2)
                name = f"encoder.layers.{i}.{LINEAR_NAMES[short]}"
            elif w.startswith("enc_subsampling_linear"):
                name = "encoder.pre_encode.out"
            else:
                raise ValueError(w)
            W = c[w].astype(np.float32)
            b = c[arg(op, "bias")].astype(np.float32) if arg(op, "bias") else np.zeros(W.shape[0], np.float32)
            scale = None
            for nxt in cons.get(out, []):
                if nxt.op == "mul" and arg(nxt, "x") == out and arg(nxt, "y") in c:
                    scale = c[arg(nxt, "y")].astype(np.float32)
            if scale is not None and m:
                W = W * scale[:, None]
                b = b * scale
                lop = prod[w]
                idx = model.literal(lop.args["indices"]).astype(np.uint8)
                lut = model.literal(lop.args["lut"])
                g, _, npal, _ = lut.shape
                assert idx.shape[0] // g == 16 and lut.shape[1] == 1 and lut.shape[3] == 1
                lut16 = np.zeros((g, 16), np.float16)
                lut16[:, :npal] = lut[:, 0, :, 0]
                quant[name + ".weight"] = {"q4": idx[:, 0::2] | (idx[:, 1::2] << 4), "lut": lut16,
                                           "scale": scale.astype(np.float16)}
            elif w in prod:
                d8, s8 = q8_form(model, prod[w])
                quant[name + ".weight"] = {"q8": d8, "qscale": s8}
            tensors[name + ".weight"] = W
            if np.any(b != 0):
                tensors[name + ".bias"] = b
        elif op.op == "conv":
            w = arg(op, "weight")
            if w.startswith("enc_subsampling_layers_"):
                idx = int(re.match(r"enc_subsampling_layers_(\d+)_", w).group(1))
                tensors[f"encoder.pre_encode.conv.{idx}.weight"] = c[w]
                d8, s8 = q8_form(model, prod[w])
                quant[f"encoder.pre_encode.conv.{idx}.weight"] = {"q8": d8, "qscale": s8}
                tensors[f"encoder.pre_encode.conv.{idx}.bias"] = c[arg(op, "bias")]
                sub_convs.append(idx)
            else:
                tensors[f"encoder.layers.{layer}.conv.depthwise_conv.weight"] = c[w]
                d8, s8 = q8_form(model, prod[w])
                quant[f"encoder.layers.{layer}.conv.depthwise_conv.weight"] = {"q8": d8, "qscale": s8}
                tensors[f"encoder.layers.{layer}.conv.depthwise_conv.bias"] = c[arg(op, "bias")]
        elif op.op == "add" and op.outputs[0][0].startswith(("q_u_", "q_v_")):
            y = arg(op, "y")
            kind = "u" if op.outputs[0][0].startswith("q_u_") else "v"
            i = pos_u if kind == "u" else pos_v
            tensors[f"encoder.layers.{i}.self_attn.pos_bias_{kind}"] = c[y].reshape(8, 128)
            if kind == "u":
                pos_u += 1
            else:
                pos_v += 1
    # constants of the block (residual factor, attention scale) for the config
    return sorted(sub_convs), eps


def export_decoder(dec, joint, tensors, quant):
    c = dec.constants()
    dp, jp = producers(dec), producers(joint)

    def q8(model, prod, var, name, fix=lambda x: x):
        d8, s8 = q8_form(model, prod[var])
        quant[name] = {"q8": fix(d8), "qscale": fix(s8)}

    lstm_ops = [op for op in dec.main.ops if op.op == "lstm"]
    tensors["decoder.prediction.embed.weight"] = c["embedding_weight_to_fp16_quantized"]
    q8(dec, dp, "embedding_weight_to_fp16_quantized", "decoder.prediction.embed.weight")
    H = 640

    def reorder(w):  # Core ML gates i, f, o, g  ->  PyTorch i, f, g, o
        i, f, o, g = np.split(w, 4, axis=0)
        return np.concatenate([i, f, g, o], axis=0)

    for li, op in enumerate(lstm_ops):
        pre = "decoder.prediction.dec_rnn.lstm"
        tensors[f"{pre}.weight_ih_l{li}"] = reorder(c[arg(op, "weight_ih")])
        tensors[f"{pre}.weight_hh_l{li}"] = reorder(c[arg(op, "weight_hh")])
        q8(dec, dp, arg(op, "weight_ih"), f"{pre}.weight_ih_l{li}", reorder)
        q8(dec, dp, arg(op, "weight_hh"), f"{pre}.weight_hh_l{li}", reorder)
        tensors[f"{pre}.bias_ih_l{li}"] = reorder(c[arg(op, "bias")])
        tensors[f"{pre}.bias_hh_l{li}"] = np.zeros(4 * H, np.float32)
    j = joint.constants()
    tensors["joint.enc.weight"] = j["encoder_projector_weight_to_fp16_quantized"]
    tensors["joint.enc.bias"] = j["encoder_projector_bias_to_fp16"]
    tensors["joint.pred.weight"] = j["decoder_projector_weight_to_fp16_quantized"]
    tensors["joint.pred.bias"] = j["decoder_projector_bias_to_fp16"]
    tensors["joint.joint_net.2.weight"] = j["head_weight_to_fp16_quantized"]
    tensors["joint.joint_net.2.bias"] = j["head_bias_to_fp16"]
    q8(joint, jp, "encoder_projector_weight_to_fp16_quantized", "joint.enc.weight")
    q8(joint, jp, "decoder_projector_weight_to_fp16_quantized", "joint.pred.weight")
    q8(joint, jp, "head_weight_to_fp16_quantized", "joint.joint_net.2.weight")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    dense = "--dense" in sys.argv[3:]
    os.makedirs(dst, exist_ok=True)
    enc = mil.Model(os.path.join(src, "Encoder.mlmodelc"))
    dec = mil.Model(os.path.join(src, "Decoder.mlmodelc"))
    joint = mil.Model(os.path.join(src, "JointDecision.mlmodelc"))
    tensors, quant = {}, {}
    sub_convs, eps = export_encoder(enc, tensors, quant)
    export_decoder(dec, joint, tensors, quant)
    n_layers = 1 + max(int(k.split(".")[2]) for k in tensors if k.startswith("encoder.layers."))
    total = sum(v.size for v in tensors.values())
    out = {}
    for k, v in tensors.items():
        if k in quant and not dense:
            for suffix, t in quant[k].items():
                out[f"{k}.{suffix}"] = np.ascontiguousarray(t)
        else:
            out[k] = np.ascontiguousarray(v.astype(np.float16))
    save_file(out, os.path.join(dst, "model.safetensors"), metadata={
        "format": "pt", "encoding": "dense-fp16" if dense else "fluid-q4lut-q8", "source": "FluidVoice-Labs/fluid-speech-mini-coreml",
        "credit": "Fluid Speech Mini by FluidVoice. Built on NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra.",
        "license": "CC-BY-4.0"})
    cfg = {
        "model_type": "parakeet_tdt",
        "sample_rate": 16000, "n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 128,
        "preemphasis": 0.97, "log_zero_guard": 2.0 ** -24, "norm_eps": 1e-5, "mel_fmin": 0.0, "mel_fmax": 8000.0,
        "d_model": 1024, "n_heads": 8, "n_layers": n_layers, "ff_dim": 4096, "conv_kernel": 9,
        "subsampling_channels": 256, "subsampling_factor": 8, "subsampling_convs": sub_convs,
        "layer_norm_eps": sorted(eps)[0], "ff_residual_factor": 0.5,
        "pred_hidden": 640, "pred_layers": 2, "joint_hidden": 640,
        "vocab_size": 8192, "blank_id": 8192, "durations": [0, 1, 2, 3, 4], "max_symbols_per_step": 10,
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    shutil.copy(os.path.join(src, "parakeet_vocab.json"), os.path.join(dst, "vocab.json"))
    card = os.path.join(os.path.dirname(__file__), "MODEL_CARD.md")
    if not dense and os.path.exists(card):
        shutil.copy(card, os.path.join(dst, "README.md"))
    for f in ("LICENSE", "NOTICE"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print(f"{len(out)} tensors, {sum(v.nbytes for v in out.values()) / 1e6:.1f} MB, {total / 1e6:.1f}M params, layers={n_layers}, convs={sub_convs}, eps={eps}")


if __name__ == "__main__":
    main()
