# fluid-speech-rs

[Fluid Speech Mini](https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-coreml) (FluidVoice's English dictation
model, a Parakeet TDT 0.6B variant) ported from its Core ML package to **safetensors**, running on a custom
**Rust + candle** inference engine with hand-written CUDA kernels.

On an RTX 4090 it transcribes an 11 s clip in **5.3 ms** end to end and all of LibriSpeech test-clean (5.4 h) in
**4.2 s** (4,600× real time; 5,800× with `--fast`). Its transcripts match the official ONNX build.

## Layout

| Path | What it is |
|---|---|
| `tools/mil.py` | Parser and numpy interpreter for compiled Core ML programs (`model.mil` + `weight.bin`), including palettized (2/3/4-bit LUT) and int8 blockwise weights |
| `tools/coreml_ref.py` | Runs the whole Core ML package in numpy: the reference for the port |
| `tools/export_safetensors.py` | Core ML → `model/model.safetensors` (327 MB, original quantisation; `--dense` for 1.25 GB fp16) + `config.json` + `vocab.json` + model card |
| `tools/np_model.py` | Plain numpy model over the safetensors file: the architecture spec the Rust engine follows |
| `src/` | The engine: mel front end, FastConformer encoder, LSTM prediction net, TDT joint, greedy TDT decoding with token confidences |
| `kernels/fluid.cu` | CUDA kernels (built to PTX by `build.rs` with `nvcc`) |
| `src/main.rs`, `src/bin/fluid-dictate.rs` | The `fluid-speech` CLI (files, `--mic`, manifests) and the `fluid-dictate` daemon |
| `src/mic.rs`, `src/dictate/` | Microphone capture + VAD; hotkeys, typing, clipboard and notifications (Linux evdev/uinput, macOS Quartz) |
| `tests/transcribe.rs` | End-to-end tests (CPU and CUDA, incl. concurrent engines) |

## Quick start

```sh
# build (CUDA needs nvcc on PATH; the GPU arch is detected, or set CUDA_COMPUTE_CAP=89)
cargo build --release --features cuda      # or without --features cuda for CPU only
# Linux: cpal links ALSA, so the build needs alsa-lib / libasound2-dev

./target/release/fluid-speech samples/jfk.wav
```

The model is looked for in `--model` / `$FLUID_MODEL`, then `./model`, `../model` and next to the binary. Failing
that, it is downloaded once (327 MB) from
[apiplant/fluid-speech-mini-safetensors](https://huggingface.co/apiplant/fluid-speech-mini-safetensors) into
`~/.cache/fluid-speech-rs/` (`$XDG_CACHE_HOME`). To build it from the Core ML package instead:

```sh
hf download FluidVoice-Labs/fluid-speech-mini-coreml --local-dir hf_coreml
python tools/export_safetensors.py hf_coreml model     # numpy, safetensors, soundfile
```

## Transcription: `fluid-speech`

```bash
fluid-speech recording.wav                          # CUDA if built with it and a GPU is present, else CPU
fluid-speech talk.mp3 --device cpu --format srt > talk.srt
fluid-speech a.flac b.m4a --format json             # words with start/end times and confidence
fluid-speech hour.flac --bench                      # timings + real-time factor on stderr
fluid-speech --manifest clips.jsonl --out hyps.jsonl  # batch: {"id","audio"} lines in, {"id","hyp"} lines out

fluid-speech --mic                                  # live from the default input; Ctrl-C to stop
fluid-speech --list-mics
fluid-speech --mic --mic-device wave --format json  # one JSON line per utterance
```

Files can be in any format symphonia decodes (wav, flac, mp3, ogg/vorbis, aac/m4a, ...). They are mixed to mono and
resampled to 16 kHz.

**Options**

- `--device auto|cpu|cuda`.
- `--max-pass 40`: clips up to 40 s go through the encoder in one pass. Longer audio is cut at quiet points into pieces of about 30 s; all pieces and files are batched together.
- `--batch-size 16`.
- `--threads N`: defaults to the physical core count.
- CUDA only:
  - `--fast`: fp16 accumulation in the encoder GEMMs.
  - `--tune`: cuBLASLt autotuning.
  - `--reference-ops`: plain candle ops instead of the fused engine.
- `--warmup` and `--repeat N` (with `--bench`): for benchmarking.

**Microphone mode**

`--mic` captures the chosen input and splits it into utterances with an energy VAD on the 200 Hz high-passed signal.
The speech threshold is 4× a running noise-floor estimate, or `--vad-threshold` if you set one.

- **While you speak**, the utterance is re-transcribed about every half second and shown dimmed on stderr.
- **After a pause** of `--silence-ms` (700 ms by default), the final text goes to stdout: plain text, one JSON line per utterance (`--format json`, word times from the session start), or SRT cues.
- **Noise filtering:** one- or two-word utterances below `--min-confidence` (0.9) are dropped, because knocks and breaths tend to decode as "Yeah.". Confidence is the geometric mean of the tokens' softmax probabilities.
- **Ctrl-C** flushes the utterance in progress and exits.

On Linux the microphone is opened through PulseAudio's protocol, which PipeWire also serves, so `--mic-device` picks
the sound server's named sources. Plain ALSA is the fallback.

## Dictation: `fluid-dictate`

Push-button dictation on Linux (Wayland or X11) and macOS: press a hotkey, speak, and press it again. The transcript
is typed into whatever window has focus. The model stays loaded on the GPU, so a sentence comes back in about 10–25 ms.

```bash
cargo install --path . --features cuda     # puts fluid-speech and fluid-dictate in ~/.cargo/bin
fluid-dictate setup                        # press your combo twice; saved to ~/.config/fluid-speech/dictate.json
fluid-dictate setup --hold                 # same, but push-to-talk (record while held)
fluid-dictate                              # run the daemon (keep it running: autostart / systemd user unit)
```

Instead of a saved hotkey, you can:

- pass one on the command line: `--key f9`, `--key ctrl+alt+d`, `--key super+space`, `--hold`.
- run the daemon with `--no-hotkey` and bind `fluid-dictate toggle` in your desktop's shortcut settings. `start` and `stop` also exist; the daemon listens on `$XDG_RUNTIME_DIR/fluid-dictate.sock`.

**Linux**

- **Hotkeys** are read from `/dev/input/event*` (evdev), so they work under any compositor. Keyboards plugged in later are picked up. Left and right modifiers count as the same key. The hotkey is not grabbed, so pick a combo that does nothing else (F13–F24, Pause, an unused Super+letter, a mouse side button).
- **Text** is typed through a virtual keyboard on `/dev/uinput`, using the US layout. Characters it cannot type are pasted (`wl-copy`, then Ctrl+V).
  - `--output paste` always pastes (use it with non-US layouts).
  - `--output clipboard` only copies; `--output stdout` prints.
- **Both need the `input` group**: `sudo usermod -aG input $USER`, then log in again.

**macOS**

- The hotkey comes from a Quartz event tap, and text is typed with posted key events, so any character works.
- Grant **Input Monitoring** and **Accessibility** to the app that runs it.
- `cmd` means Command and `alt` means Option; the paste shortcut is `cmd+v`.

**Behaviour**

- Recordings with under 0.25 s of voiced audio are discarded, as are decodes that come out as one or two low-confidence words.
- Other options: `--mic-device`, `--quiet` (no notifications), `--no-space` (no trailing space), `--max-secs 300`, `--fast`.

Example systemd user unit (`~/.config/systemd/user/fluid-dictate.service`):

```ini
[Unit]
Description=Fluid Speech dictation
After=graphical-session.target

[Service]
ExecStart=%h/.cargo/bin/fluid-dictate
Restart=on-failure

[Install]
WantedBy=graphical-session.target
```

## Use as a library

Everything beyond inference is behind features: `mic` (cpal, ctrlc: `fluid_speech::mic`), `cli` (the `fluid-speech`
binary) and `dictate` (the `fluid-dictate` binary; evdev / Quartz). `dictate` implies `cli`, which implies `mic`, and
the defaults enable `dictate`. For inference only:

```toml
fluid-speech = { path = "...", default-features = false }                     # CPU
# fluid-speech = { path = "...", default-features = false, features = ["cuda"] }
```

```rust
use fluid_speech::{audio, text, DeviceArg, Engine, LoadOptions};

let engine = Engine::open(None, DeviceArg::Auto, &LoadOptions::default())?; // finds or downloads the model
let samples = audio::load("talk.wav".as_ref())?;      // any symphonia format -> 16 kHz mono f32
let t = engine.transcribe(&samples)?;                 // long audio is cut at pauses and batched
println!("{}", t.text);                               // t.words: word, start, end, confidence
let many = engine.transcribe_many(&[&a, &b])?;        // several clips in one batch
print!("{}", text::srt(&t.words));
```

`Engine` is `Sync`: share one across threads, or create several. Each runs on its own CUDA stream.

## The port

The Core ML package has four programs: Preprocessor, Encoder, Decoder and JointDecision. There is no PyTorch
checkpoint, so `tools/mil.py` reads the compiled MIL text and blob file directly:

- **Blob storage v2**: 64-byte headers with sentinel `0xDEADBEEF` pointing at raw data. Sub-byte types (`uint2/3/4`) are bit-packed little-endian.
- **`constexpr_lut_to_dense`**: the encoder's linears are palettized at mixed 2/3/4 bits, with one LUT per 16 output rows. Core ML multiplies the linear output by a per-channel scale afterwards; the exporter folds that scale into the weight.
- **`constexpr_blockwise_shift_scale`**: int8 weights with per-row scales (subsampling, decoder, joint, depthwise convs). Batch norm is already folded into the depthwise conv.
- **LSTM gate order**: Core ML uses (i, f, o, g). It is reordered to PyTorch's (i, f, g, o), and the two biases are combined.
- **Positional embedding**: the graph carries a 375-row constant. It matches the standard sinusoidal formula to int8 rounding, so the engine generates it for any length.

Tensor names follow NeMo's Parakeet layout (`encoder.layers.N.self_attn.linear_q.weight`, `decoder.prediction.dec_rnn.lstm.weight_ih_l0`,
`joint.joint_net.2.weight`, …). Pointwise convs are stored as 2-D `[out, in]`.

The file keeps the Core ML quantisation, so it is **327 MB**. It is published as
[apiplant/fluid-speech-mini-safetensors](https://huggingface.co/apiplant/fluid-speech-mini-safetensors).

- **Palettized linears** are stored as `W.q4` (4-bit indices) + `W.lut` (16 entries per 16 rows) + `W.scale`.
- **int8 weights** are stored as `W.q8` + `W.qscale`.
- **At load**, `src/weights.rs` expands everything to fp16 in about 0.1 s. The result is bit-identical to `--dense` (627M parameters, 1.25 GB), and transcripts are identical too.

**Validation chain**

1. The numpy Core ML interpreter transcribes `jfk.wav` perfectly.
2. The numpy model built from the safetensors file matches the interpreter's encoder to 0.15% when given the same mel.
3. The Rust CPU encoder matches numpy to 1e-6 relative error; the CUDA fp16 encoder is within 0.13% (fp16 noise, the same level as Core ML's own fp16 run).
4. On a 328-clip LibriSpeech subset the official ONNX build scores 1.95% WER and this engine 1.96% (one word apart).

The front end follows NeMo / the ONNX build: preemphasis, reflect-padded STFT, Slaney mel, log, per-feature
normalisation. It is computed in f32 rather than through the Core ML preprocessor's int8 filterbank.

## Engine

**CPU path.** Candle tensors (f32) plus rayon kernels for the fused ops. Every op also exists in this form, and the
CUDA path is validated against it.

**CUDA path.** Everything below was profiled and kept only when it measured faster.

- **Static-graph encoder** (`src/gpu_encoder.rs`). One workspace per (batch, length bucket) with all buffers bump-allocated. GEMMs go straight to cuBLAS; everything else is a custom kernel. The forward pass for a bucket (≈600 kernels) is captured once into a CUDA graph and replayed, which cut host overhead from 4.1 ms to about zero per call. Lengths are padded to 64-frame buckets; masking keeps padded results identical.
- **Fused kernels** (`kernels/fluid.cu`):
  - conv0 + depthwise conv of the subsampling stage, fused. This avoids materialising the largest activation in the model (~790 MB for a 16 × 30 s batch).
  - Rel-pos attention scores: shift + scale + key mask + softmax in one kernel.
  - Residual add + LayerNorm, GLU + mask, depthwise conv1d + SiLU, vectorised SiLU, q/k/v split with the u/v position biases.
- **Positional projection** for all 24 layers is precomputed once as a 50 MB table. Each layer's attention GEMM reads it in place through strided-batched cuBLAS.
- **Device-side TDT decoding** (`src/gpu_decode.rs`). The greedy loop never returns to the host per step. One step is five kernels: joint + argmax via packed `atomicMax`, bookkeeping, LSTM layer 0, LSTM layer 1, and prediction projection + state commit. Decoder weights are fp16 and cached in registers.
  - Token confidences come for free: each joint block also emits a partial log-sum-exp of its logits, and the bookkeeping kernel merges them warp-parallel into the softmax probability of the emitted token.
  - Steps are replayed as a CUDA graph of 8; the host polls a "rows still active" flag through pinned memory, one chunk behind.
  - Decode went from 3.7 ms to 1.0 ms for an 11 s clip.
- **Batching**:
  - Long audio is cut at quiet points into pieces of about 30 s (single pass up to 40 s).
  - Pieces and clips are grouped by length and encoded/decoded together.
  - Mel features for the next group are computed on CPU threads while the GPU runs the current one.
- **Concurrency**: the engine runs on its own non-blocking stream; candle's default per-thread stream is blocking. A graph capture invalidated by another thread's implicitly synchronising call (such as lazy module loading) is retried. A test runs four engines in parallel threads.

## Results

RTX 4090 + Ryzen 9 7950X, CUDA 13.4, release build. Latency is the median of 100 runs and includes mel, encoder and
decoder (audio already in memory).

| Input | Default | `--fast` |
|---|---|---|
| 2 s clip | 3.2 ms | 3.4 ms |
| 4 s clip | 3.4 ms | 3.7 ms |
| 8 s clip | 4.2 ms | 4.3 ms |
| 11 s clip (jfk.wav) | 5.3 ms | 4.9 ms |
| 13.1 min recording | 185 ms (4,270×) | 153 ms (5,160×) |
| LibriSpeech test-clean, 2,620 clips / 5.4 h | 4.24 s (4,590×) | 3.35 s (5,800×, warm) |

| Accuracy, LibriSpeech test-clean | WER |
|---|---|
| this engine, simple normaliser | 2.07% (`--fast`: 2.07%) |
| this engine, Whisper English normaliser | 1.82% |
| official ONNX build, 328-clip subset vs this engine | 1.95% vs 1.96% |

The model card's 1.45% uses the Open ASR leaderboard's evaluation pipeline. On identical clips this engine and the
official ONNX build agree.

Other measurements:

- **GPU memory**: about 2.1 GB peak for single clips and 3.3 GB for batches of 16 × ~35 s. That is fp16 weights 1.25 GB (expanded at load), CUDA context, workspaces and the positional table.
- **CPU path** (7950X, 32 threads): RTFx 16 on a single 11 s clip; 328 clips (40 min) in 76 s, against 88 s for the official ONNX build with 16 threads on the same clips.

## Things that were tried and dropped

- **Palettized (4-bit LUT) GEMM kernel.** It decodes the original Core ML palettes on the fly with `mma.sync`, reading 4× fewer weight bytes than fp16. It was exact, but on this GPU it never beat cuBLAS on dense fp16 at these sizes. Each GEMM pays several microseconds of serialized prologue/epilogue latency, and GeForce cards run fp32-accumulate tensor math at half rate. Hiding the latency needs Hopper's programmatic dependent launch.
- **cuBLASLt autotuning.** About 5% on single clips and nothing in batch mode, so it is off by default (`--tune`).
- **fp16 accumulation.** About 20% faster with the same WER on test-clean, but it is less faithful numerically, so it is opt-in (`--fast`).

## Tests

```sh
cargo test --release                  # CPU
cargo test --release --features cuda  # CPU + CUDA, incl. batching, long audio, concurrent engines
python tools/prep_librispeech.py      # LibriSpeech test-clean manifest for --manifest / tools/wer.py
```

## License and attribution

The engine code is MIT OR Apache-2.0. The model is CC BY 4.0: **Fluid Speech Mini by FluidVoice**, built on NVIDIA
Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra. The exporter copies the model's LICENSE and NOTICE into `model/`.
