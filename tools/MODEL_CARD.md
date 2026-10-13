---
license: cc-by-4.0
language:
- en
library_name: candle
pipeline_tag: automatic-speech-recognition
base_model:
- FluidVoice-Labs/fluid-speech-mini-coreml
- nvidia/parakeet-tdt-0.6b-v3
- moondream/parakeet-ultra
tags:
- speech-recognition
- parakeet
- tdt
- fastconformer
- safetensors
- candle
- rust
---

# Fluid Speech Mini (safetensors)

**Fluid Speech Mini by FluidVoice**, converted from the Core ML package
[FluidVoice-Labs/fluid-speech-mini-coreml](https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-coreml) to a
single safetensors file (327 MB), so the model can run outside Apple platforms. It is English speech-to-text with a
Parakeet TDT 0.6B architecture: a FastConformer encoder (24 layers, d_model 1024), a 2-layer LSTM prediction network
and a TDT joint (8,192 tokens + blank, durations 0–4).

The weights keep the original Core ML quantisation:
- **2/3/4-bit palettized** encoder linears, stored as 4-bit indices with one 16-entry LUT per 16 output rows.
- **int8** for the rest (subsampling, depthwise convs, decoder, joint).

Expanding to fp16 gives 627M parameters / 1.25 GB, bit-identical to a direct fp16 export of the Core ML weights.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | All weights; NeMo-style names (`encoder.layers.N.self_attn.linear_q.weight`, `decoder.prediction.dec_rnn.lstm.weight_ih_l0`, `joint.joint_net.2.weight`, …) |
| `config.json` | Front-end and architecture hyper-parameters |
| `vocab.json` | 8,192 SentencePiece pieces (id → piece; a leading space marks a word start) |
| `LICENSE`, `NOTICE` | CC BY 4.0 and the original attributions |

## Weight encoding

A weight `W` is stored in one of three forms:

| Stored tensors | Meaning |
|---|---|
| `W` | fp16, as is (norms, biases, position biases) |
| `W.q4` (u8 `[N, K/2]`, low nibble = even k), `W.lut` (f16 `[N/16, 16]`), `W.scale` (f16 `[N]`) | `W[n, k] = fp16(lut[n // 16, idx(n, k)] * scale[n])` |
| `W.q8` (i8, original shape), `W.qscale` (f16, one per leading row) | `W = fp16(q8 * qscale[row])` |

Products are computed in fp32 and rounded once. To expand everything to plain fp16 tensors:

```python
import numpy as np
from safetensors.numpy import load_file

def load_dense(path="model.safetensors"):
    q, out = load_file(path), {}
    for k, v in q.items():
        if k.endswith(".q4"):
            b = k[:-3]
            idx = np.empty((v.shape[0], v.shape[1] * 2), np.uint8)
            idx[:, 0::2], idx[:, 1::2] = v & 15, v >> 4
            lut, s = q[b + ".lut"].astype(np.float32), q[b + ".scale"].astype(np.float32)
            out[b] = (lut[np.arange(v.shape[0])[:, None] // 16, idx] * s[:, None]).astype(np.float16)
        elif k.endswith(".q8"):
            b = k[:-3]
            s = q[b + ".qscale"].astype(np.float32).reshape((-1,) + (1,) * (v.ndim - 1))
            out[b] = (v.astype(np.float32) * s).astype(np.float16)
        elif not any(k.endswith(x) for x in (".lut", ".scale", ".qscale")):
            out[k] = v
    return out
```

Layout notes:
- Pointwise convs are 2-D `[out, in]`.
- The depthwise convs already contain their batch norm.
- LSTM gates use PyTorch order (i, f, g, o); the combined bias is in `bias_ih_l*` and `bias_hh_l*` is zero.
- The relative positional encoding is the standard sinusoidal one and is not stored.
- `encoder.pre_encode.out.weight` takes the subsampling output flattened as `channel * 16 + freq`.

## Front end and decoding

- **Features:** preemphasis 0.97; reflect-padded STFT (n_fft 512, symmetric Hann window 400, hop 160); power spectrum; Slaney mel with 128 bins over 0–8 kHz; `log(x + 2^-24)`; per-feature mean/std normalisation over the clip (unbiased std + 1e-5).
- **Decoding:** greedy TDT. Start the prediction network from the blank token. At each encoder frame (80 ms), the joint gives a token and a duration (0–4 frames). A blank with duration 0 advances 1 frame, and at most 10 symbols are emitted per frame.

## Accuracy

The conversion was validated against the Core ML graph, which was executed in numpy. An independent implementation
using these weights matches the official ONNX build on LibriSpeech test-clean: 1.96% vs 1.95% WER on a 328-clip
subset. On the full set it scores 2.07% with a simple normaliser and 1.82% with the Whisper English normaliser.

## Attribution

CC BY 4.0. Please credit **Fluid Speech Mini by FluidVoice**, built on NVIDIA Parakeet TDT 0.6B v3 and Moondream
Parakeet Ultra (both CC BY 4.0); the decoder and joint network are Parakeet Ultra's. This repository changes only the
file format. See `NOTICE`.
