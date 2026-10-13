"""Plain numpy implementation of the exported safetensors model (architecture blueprint for the Rust engine).

usage: np_model.py <model dir> <audio.wav> [--ref ref.npz]
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from safetensors.numpy import load_file


def hz_to_mel(f):  # slaney
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    log_region = f >= 1000.0
    return np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f = m * (200.0 / 3)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)


def mel_filters(sr, n_fft, n_mels, fmin, fmax):
    fftfreqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mpts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mpts)
    ramps = mpts[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mpts[2:n_mels + 2] - mpts[:n_mels])
    return (w * enorm[:, None]).astype(np.float32)


def features(wav, cfg):
    n_fft, hop, win = cfg["n_fft"], cfg["hop_length"], cfg["win_length"]
    x = np.concatenate([wav[:1], wav[1:] - cfg["preemphasis"] * wav[:-1]]).astype(np.float32)
    window = np.zeros(n_fft)
    left = (n_fft - win) // 2
    window[left:left + win] = np.hanning(win)  # symmetric hann, as torch.hann_window(periodic=False)
    padded = np.pad(x, n_fft // 2, mode="reflect")
    frames = 1 + len(x) // hop
    idx = np.arange(n_fft)[None] + hop * np.arange(frames)[:, None]
    spec = np.fft.rfft(padded[idx] * window, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    fb = mel_filters(cfg["sample_rate"], n_fft, cfg["n_mels"], cfg["mel_fmin"], cfg["mel_fmax"])
    mel = np.log(power @ fb.T + cfg["log_zero_guard"])  # [T, n_mels]
    mean = mel.mean(axis=0)
    std = np.sqrt(((mel - mean) ** 2).sum(axis=0) / (frames - 1)) + cfg["norm_eps"]
    return ((mel - mean) / std).astype(np.float32)


def silu(x):
    return x / (1 + np.exp(-x))


def ln(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def conv2d(x, w, b, stride, pad, groups):
    """x [C, H, W]; w [Co, Cg, kh, kw]."""
    C, H, W = x.shape
    Co, Cg, kh, kw = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh = (H + 2 * pad - kh) // stride + 1
    ow = (W + 2 * pad - kw) // stride + 1
    out = np.zeros((Co, oh, ow), np.float32)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, i:i + stride * (oh - 1) + 1:stride, j:j + stride * (ow - 1) + 1:stride]
            if groups == 1:
                out += np.einsum("chw,oc->ohw", patch, w[:, :, i, j])
            else:
                out += patch * w[:, 0, i, j][:, None, None]
    return out + b[:, None, None]


class NpModel:
    def __init__(self, d):
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.w = {k: v.astype(np.float32) for k, v in load_file(os.path.join(d, "model.safetensors")).items()}
        self.vocab = json.load(open(os.path.join(d, "vocab.json")))

    def lin(self, x, name):
        y = x @ self.w[name + ".weight"].T
        b = self.w.get(name + ".bias")
        return y + b if b is not None else y

    def encode(self, feats):
        w, cfg = self.w, self.cfg
        x = feats[None]  # [1, T, 128] as [C=1, H=T, W=128]
        x = np.maximum(conv2d(x, w["encoder.pre_encode.conv.0.weight"], w["encoder.pre_encode.conv.0.bias"], 2, 1, 1), 0)
        x = conv2d(x, w["encoder.pre_encode.conv.2.weight"], w["encoder.pre_encode.conv.2.bias"], 2, 1, 256)
        x = np.maximum(conv2d(x, w["encoder.pre_encode.conv.3.weight"], w["encoder.pre_encode.conv.3.bias"], 1, 0, 1), 0)
        x = conv2d(x, w["encoder.pre_encode.conv.5.weight"], w["encoder.pre_encode.conv.5.bias"], 2, 1, 256)
        x = np.maximum(conv2d(x, w["encoder.pre_encode.conv.6.weight"], w["encoder.pre_encode.conv.6.bias"], 1, 0, 1), 0)
        C, T, F = x.shape
        x = self.lin(x.transpose(1, 0, 2).reshape(T, C * F), "encoder.pre_encode.out")
        d, H = cfg["d_model"], cfg["n_heads"]
        dk = d // H
        pos = np.arange(T - 1, -T, -1, dtype=np.float64)[:, None]
        div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe = np.zeros((2 * T - 1, d))
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        pe = pe.astype(np.float32)
        eps = cfg["layer_norm_eps"]
        for i in range(cfg["n_layers"]):
            p = f"encoder.layers.{i}."
            g = lambda n: (w[p + n + ".weight"], w[p + n + ".bias"])  # noqa: E731
            h = ln(x, *g("norm_feed_forward1"), eps)
            x = x + 0.5 * self.lin(silu(self.lin(h, p + "feed_forward1.linear1")), p + "feed_forward1.linear2")
            h = ln(x, *g("norm_self_att"), eps)
            q = self.lin(h, p + "self_attn.linear_q").reshape(T, H, dk)
            k = self.lin(h, p + "self_attn.linear_k").reshape(T, H, dk).transpose(1, 0, 2)
            v = self.lin(h, p + "self_attn.linear_v").reshape(T, H, dk).transpose(1, 0, 2)
            pp = self.lin(pe, p + "self_attn.linear_pos").reshape(2 * T - 1, H, dk).transpose(1, 0, 2)
            qu = (q + w[p + "self_attn.pos_bias_u"]).transpose(1, 0, 2)
            qv = (q + w[p + "self_attn.pos_bias_v"]).transpose(1, 0, 2)
            ac = qu @ k.transpose(0, 2, 1)  # [H, T, T]
            bd = qv @ pp.transpose(0, 2, 1)  # [H, T, 2T-1]
            # rel shift: out[i, j] = bd[i, j + T - 1 - i]
            ii = np.arange(T)[:, None]
            jj = np.arange(T)[None, :]
            bd = bd[:, ii, jj + T - 1 - ii]
            s = (ac + bd) / np.sqrt(dk)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o = (s @ v).transpose(1, 0, 2).reshape(T, d)
            x = x + self.lin(o, p + "self_attn.linear_out")
            h = ln(x, *g("norm_conv"), eps)
            h = self.lin(h, p + "conv.pointwise_conv1")
            h = h[:, :d] * (1 / (1 + np.exp(-h[:, d:])))
            dw = w[p + "conv.depthwise_conv.weight"][:, 0, :]  # [d, 9]
            K = dw.shape[1]
            hp = np.pad(h, ((K // 2, K // 2), (0, 0)))
            h = sum(hp[j:j + T] * dw[:, j] for j in range(K)) + w[p + "conv.depthwise_conv.bias"]
            h = silu(h)
            x = x + self.lin(h, p + "conv.pointwise_conv2")
            h = ln(x, *g("norm_feed_forward2"), eps)
            x = x + 0.5 * self.lin(silu(self.lin(h, p + "feed_forward2.linear1")), p + "feed_forward2.linear2")
            x = ln(x, *g("norm_out"), eps)
        return x

    def lstm_step(self, tok, h, c):
        w = self.w
        x = w["decoder.prediction.embed.weight"][tok]
        H = 640
        hs, cs = [], []
        for l in range(2):
            p = "decoder.prediction.dec_rnn.lstm."
            gates = (w[p + f"weight_ih_l{l}"] @ x + w[p + f"bias_ih_l{l}"] + w[p + f"weight_hh_l{l}"] @ h[l]
                     + w[p + f"bias_hh_l{l}"])
            i, f, g, o = (gates[k * H:(k + 1) * H] for k in range(4))
            sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
            cn = sig(f) * c[l] + sig(i) * np.tanh(g)
            hn = sig(o) * np.tanh(cn)
            hs.append(hn)
            cs.append(cn)
            x = hn
        return x, np.stack(hs), np.stack(cs)

    def decode(self, enc):
        cfg, w = self.cfg, self.w
        blank, durs = cfg["blank_id"], cfg["durations"]
        ep = self.lin(enc, "joint.enc")
        h = np.zeros((2, 640), np.float32)
        c = np.zeros((2, 640), np.float32)
        dout, h, c = self.lstm_step(blank, h, c)
        dp = self.lin(dout, "joint.pred")
        tokens, t, emitted, T = [], 0, 0, enc.shape[0]
        while t < T:
            z = np.maximum(ep[t] + dp, 0)
            logits = self.lin(z, "joint.joint_net.2")
            tok = int(logits[:blank + 1].argmax())
            dur = durs[int(logits[blank + 1:].argmax())]
            if tok != blank:
                tokens.append(tok)
                dout, h, c = self.lstm_step(tok, h, c)
                dp = self.lin(dout, "joint.pred")
                emitted += 1
            if tok == blank and dur == 0:
                dur = 1
            if dur == 0 and emitted >= cfg["max_symbols_per_step"]:
                dur = 1
            if dur:
                emitted = 0
            t += dur
        return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("audio")
    ap.add_argument("--ref")
    a = ap.parse_args()
    m = NpModel(a.model_dir)
    wav, sr = sf.read(a.audio, dtype="float32")
    feats = features(wav, m.cfg)
    if a.ref:
        r = np.load(a.ref)
        rm = r["mel"][0].T[: feats.shape[0]]
        print("mel max abs diff vs coreml:", np.abs(rm - feats).max(), "mean", np.abs(rm - feats).mean())
    enc = m.encode(feats)
    if a.ref:
        re_ = r["encoder"][0].T[: enc.shape[0]]
        print("enc shape", enc.shape, "ref valid", int(r["encoder_length"]))
        print("enc max abs diff:", np.abs(re_ - enc).max(), "rel:", np.linalg.norm(re_ - enc) / np.linalg.norm(re_))
    toks = m.decode(enc)
    print(toks)
    print("".join(m.vocab[str(i)] for i in toks).replace("▁", " ").strip())


if __name__ == "__main__":
    main()
