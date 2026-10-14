/** Everything the home page says about the project, in one place. */

export interface Card {
  name: string;
  href: string;
  tagline: string;
  body: string;
}

export interface Feature {
  title: string;
  body: string;
}

export const SITE = {
  pkg: "fluid-speech-rs",
  hasCuda: true,
  bins: ["fluid-speech", "fluid-dictate"],
  badges: ["Rust", "candle", "CUDA", "Apache-2.0"],
  hero: { pre: "Fluid Speech Mini, ", accent: "in Rust", post: "." },
  lead: "Transcribe files or the microphone, or dictate into any window on Linux or macOS. FluidVoice's Parakeet TDT dictation model, ported from Core ML to a 327 MB safetensors file and run by a custom candle + CUDA engine. No Python, no server.",
  heroNote: "An 11 s clip in 5.3 ms on an RTX 4090; LibriSpeech test-clean at 4,600x real time and 2.07 % WER.",
  demo: undefined as { href: string; label: string } | undefined,
  cargo: `cargo install fluid-speech-rs
cargo install fluid-speech-rs --features cuda   # with the CUDA engine (needs nvcc)`,
  terminal: {
    title: "fluid-speech",
    command: "fluid-speech jfk.wav --format json",
    outputLang: "json",
    output: `[
  {
    "file": "jfk.wav",
    "duration": 11.0,
    "text": "And so, my fellow Americans, ask not what your country can do for you, ...",
    "words": [
      { "word": "And", "start": 0.24, "end": 0.56, "confidence": 0.9998 },
      { "word": "so,", "start": 0.56, "end": 1.04, "confidence": 0.9945 },
      { "word": "my", "start": 1.04, "end": 1.28, "confidence": 1.0 },
      { "word": "fellow", "start": 1.28, "end": 1.76, "confidence": 1.0 },
      ...
    ]
  }
]`,
  },
  cardsTitle: "Two binaries, one crate",
  cardsLead: "A transcriber for files and microphones, and a push-to-talk dictation daemon, on the same engine.",
  cards: [
    {
      name: "fluid-speech",
      href: "/docs/cli",
      tagline: "Files and microphone to text",
      body: "Any format symphonia decodes (wav, flac, mp3, ogg, m4a). Plain text, word-timed JSON or SRT subtitles. Batch manifests, and a live microphone mode with an energy VAD.",
    },
    {
      name: "fluid-dictate",
      href: "/docs/dictate",
      tagline: "Press a key, speak, text appears",
      body: "A daemon for Linux (Wayland or X11) and macOS that keeps the model on the GPU and types the transcript into the focused window, 10 to 25 ms after you stop. Toggle or push-to-talk hotkeys.",
    },
  ] as Card[],
  featuresTitle: "A Core ML model, everywhere else",
  featuresLead: "Fluid Speech Mini shipped only as a Core ML package. The port reads its compiled program directly and keeps its quantisation.",
  features: [
    {
      title: "327 MB, original quantisation",
      body: "2/3/4-bit palettized encoder linears and int8 elsewhere, stored as-is and expanded to fp16 in 0.1 s at load, bit-identical to a dense export.",
    },
    {
      title: "Accurate",
      body: "2.07 % WER on LibriSpeech test-clean. On a 328-clip subset it is one word apart from the official ONNX build (1.96 % vs 1.95 %).",
    },
    {
      title: "Hand-written CUDA",
      body: "Fused subsampling, attention, LayerNorm and conv kernels; the whole encoder captured once per length bucket into a CUDA graph and replayed.",
    },
    {
      title: "Decoding on the GPU",
      body: "The greedy TDT loop runs device-side, five kernels a step, with token confidences from a fused log-sum-exp. No host round trip per token.",
    },
    {
      title: "Word timestamps",
      body: "Every word carries start and end times and a confidence. Export as JSON or ready-to-use SRT cues.",
    },
    {
      title: "Live microphone",
      body: "An energy VAD with an adaptive noise floor splits speech into utterances; the one in progress updates twice a second.",
    },
    {
      title: "Batched long audio",
      body: "Long recordings are cut at quiet points into ~30 s pieces, grouped by length and encoded together: 13 minutes in 185 ms.",
    },
    {
      title: "CPU or CUDA",
      body: "f32 candle + rayon on CPU, fp16 on CUDA. Every fused op has a CPU twin the GPU path is validated against.",
    },
    {
      title: "Thread-safe engine",
      body: "Engine is Sync; each instance runs on its own CUDA stream, so several can transcribe in parallel.",
    },
  ] as Feature[],
  lib: {
    lead: "The inference engine is a library: load the model, hand it 16 kHz mono samples, get timed words back. Everything beyond inference (file decoding, download, microphone, CLI, dictation) sits behind features.",
    add: `cargo add fluid-speech-rs --no-default-features`,
    caption: "src/main.rs",
    snippet: `use fluid_speech::{text, DeviceArg, Engine, LoadOptions};

let engine = Engine::open(Some("model".into()), DeviceArg::Auto, &LoadOptions::default())?;
let t = engine.transcribe(&samples)?;                 // 16 kHz mono f32
println!("{}", t.text);
for w in &t.words {
    println!("{:6.2}-{:6.2}  {:.2}  {}", w.start, w.end, w.confidence, w.word);
}
print!("{}", text::srt(&t.words));`,
  },
};
