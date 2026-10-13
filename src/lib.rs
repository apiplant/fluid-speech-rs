//! Fluid Speech Mini (Parakeet TDT 0.6B architecture) inference on candle, plus the audio, microphone (`mic`
//! feature) and dictation (`dictate` feature) utilities shared by the `fluid-speech` and `fluid-dictate` binaries.
//!
//! ```no_run
//! use fluid_speech::{audio, DeviceArg, Engine, LoadOptions};
//! // `None`: ./model, ../model, next to the binary, or downloaded from Hugging Face into ~/.cache on first use
//! let engine = Engine::open(None, DeviceArg::Auto, &LoadOptions::default())?;
//! let wav = audio::load("clip.wav".as_ref())?; // any format symphonia decodes -> 16 kHz mono
//! let t = engine.transcribe(&wav)?;
//! println!("{}", t.text); // t.words: word, start, end, confidence
//! # anyhow::Ok(())
//! ```

pub mod audio;
pub mod config;
#[cfg(feature = "dictate")]
pub mod dictate;
pub mod download;
#[cfg(feature = "cuda")]
pub mod gpu_decode;
#[cfg(feature = "cuda")]
pub mod gpu_encoder;
#[cfg(feature = "cuda")]
pub mod lt;
pub mod mel;
#[cfg(feature = "mic")]
pub mod mic;
pub mod model;
pub mod ops;
pub mod text;
pub mod vocab;
pub mod weights;

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub use text::Word;

pub use config::Config;
pub use model::Token;

#[derive(Debug, Clone, Default)]
pub struct Timing {
    pub mel: f64,
    pub encoder: f64,
    pub decoder: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub text: String,
    /// words with start/end seconds and confidence
    pub words: Vec<Word>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum DeviceArg {
    /// CUDA when built with it and a GPU is present, else CPU
    Auto,
    Cpu,
    Cuda,
}

/// One rayon worker per physical core: SMT siblings slow the CPU GEMMs down rather than up.
pub fn init_threads(threads: Option<usize>) -> usize {
    let n = threads.unwrap_or_else(num_cpus::get_physical);
    rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
    // candle's CPU matmuls size their pool from this
    unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    n
}

fn is_model_dir(p: &Path) -> bool {
    p.join("model.safetensors").is_file() && p.join("config.json").is_file() && p.join("vocab.json").is_file()
}

/// The model to use when none is given: `./model` or `../model`, a `model` dir next to the binary (or up to three
/// levels above it, i.e. the repo root for `target/release/...`), else the Hugging Face copy in the cache
/// directory, downloaded on first use.
pub fn default_model() -> Result<PathBuf> {
    let mut roots = vec![PathBuf::from("."), PathBuf::from("..")];
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())) {
        roots.extend(dir.ancestors().take(4).map(|p| p.to_path_buf()));
    }
    match roots.iter().map(|r| r.join("model")).find(|p| is_model_dir(p)) {
        Some(found) => Ok(found),
        None => download::download().context("no model found; pass --model or set FLUID_MODEL, or allow the download"),
    }
}

/// Engine construction options.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// fp16 accumulation in the encoder GEMMs: ~20% faster on GeForce GPUs, WER within noise (2.08% vs 2.07%
    /// on LibriSpeech test-clean)
    pub fast_gemm: bool,
    /// time cuBLASLt algorithms per GEMM shape on first use (small, shape-dependent gains)
    pub autotune: bool,
    /// use the static-graph CUDA encoder and the device-side decoder (false = plain candle ops, for reference)
    pub fused_cuda: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { fast_gemm: false, autotune: false, fused_cuda: true }
    }
}

/// Transcription options.
pub struct Options {
    /// clips up to this length go through the encoder in one pass
    pub max_single_pass_s: f64,
    /// target piece length when splitting longer audio at quiet points
    pub target_piece_s: f64,
    /// max encoder batch size (pieces encoded together)
    pub batch_size: usize,
    /// cap on mel frames per encoder batch (padded B * T), bounds activation memory
    pub max_batch_frames: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { max_single_pass_s: 40.0, target_piece_s: 30.0, batch_size: 16, max_batch_frames: 16 * 3001 }
    }
}

pub struct Engine {
    pub cfg: Config,
    pub device: Device,
    pub mel: mel::MelFrontend,
    pub encoder: model::Encoder,
    pub decoder: model::Decoder,
    pub vocab: vocab::Vocab,
    pub opts: Options,
    /// running on a CUDA device
    pub cuda: bool,
    pub timing: std::sync::Mutex<Timing>,
}

impl Engine {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_with(dir, device, &LoadOptions::default())
    }

    /// Load `model` (a directory with model.safetensors, config.json, vocab.json; `None` = [`default_model`]) on
    /// the chosen device.
    pub fn open(model: Option<PathBuf>, device: DeviceArg, opts: &LoadOptions) -> Result<Self> {
        let dir = match model {
            Some(p) if p.is_file() => p.parent().map(Path::to_path_buf).unwrap_or_default(),
            Some(p) => p,
            None => default_model()?,
        };
        let cuda = match device {
            DeviceArg::Cpu => false,
            DeviceArg::Cuda => true,
            DeviceArg::Auto => cfg!(feature = "cuda") && candle_core::utils::cuda_is_available(),
        };
        let dev = if cuda {
            Device::new_cuda(0).context("CUDA device (build with --features cuda)")?
        } else {
            Device::Cpu
        };
        Self::load_with(&dir, &dev, opts)
    }

    pub fn load_with(dir: &Path, device: &Device, opts: &LoadOptions) -> Result<Self> {
        // CUDA: run on a dedicated non-blocking stream. candle's default per-thread stream is a blocking stream,
        // so legacy-stream work from other threads (e.g. another engine being created) would implicitly sync with
        // it and invalidate an in-flight CUDA graph capture.
        #[cfg(feature = "cuda")]
        let device = &match device {
            Device::Cuda(_) => {
                let ordinal = match device.location() {
                    candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                    _ => 0,
                };
                Device::Cuda(candle_core::CudaDevice::new_with_stream(ordinal)?)
            }
            d => d.clone(),
        };
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = device {
            // Everything runs on the device's single stream, so cross-stream event tracking is pure overhead.
            unsafe { d.disable_event_tracking() };
            if opts.fast_gemm {
                candle_core::cuda::set_gemm_reduced_precision_f16(true);
            }
        }
        let cfg = Config::load(&dir.join("config.json"))?;
        let (encoder, decoder) = model::load(dir, &cfg, device, opts)
            .with_context(|| format!("loading the model from {}", dir.display()))?;
        Ok(Self {
            mel: mel::MelFrontend::new(&cfg),
            vocab: vocab::Vocab::load(&dir.join("vocab.json"))?,
            cfg,
            device: device.clone(),
            encoder,
            decoder,
            opts: Options::default(),
            cuda: device.is_cuda(),
            timing: Default::default(),
        })
    }

    fn sync(&self) -> Result<()> {
        Ok(self.device.synchronize()?)
    }

    /// Seconds per encoder frame.
    pub fn frame_seconds(&self) -> f64 {
        (self.cfg.hop_length * self.cfg.subsampling_factor) as f64 / self.cfg.sample_rate as f64
    }

    /// Transcribe one clip of 16 kHz mono audio (any length).
    pub fn transcribe(&self, wav: &[f32]) -> Result<Transcript> {
        Ok(self.transcribe_many(&[wav])?.remove(0))
    }

    /// Transcribe several clips; long clips are split at quiet points and all pieces are batched.
    pub fn transcribe_many(&self, clips: &[&[f32]]) -> Result<Vec<Transcript>> {
        let sr = self.cfg.sample_rate;
        // (clip index, start sample, samples)
        let mut pieces: Vec<(usize, usize, &[f32])> = Vec::new();
        for (ci, wav) in clips.iter().enumerate() {
            for (start, p) in self.split(wav) {
                if p.len() >= self.cfg.hop_length * 2 {
                    pieces.push((ci, start, p));
                }
            }
        }
        let results = self.run_pieces(&pieces.iter().map(|p| p.2).collect::<Vec<_>>())?;
        let mut out: Vec<Vec<(u32, f32, f32)>> = vec![Vec::new(); clips.len()];
        let fs = self.frame_seconds();
        for ((ci, start, _), toks) in pieces.iter().zip(results) {
            let off = *start as f64 / sr as f64;
            out[*ci].extend(toks.iter().map(|t| (t.id, (off + t.frame as f64 * fs) as f32, t.prob)));
        }
        Ok(out.into_iter().map(|toks| self.make_transcript(toks)).collect())
    }

    fn make_transcript(&self, tokens: Vec<(u32, f32, f32)>) -> Transcript {
        let ids: Vec<u32> = tokens.iter().map(|t| t.0).collect();
        let text = self.vocab.decode(&ids);
        let words = text::words(&self.vocab, &tokens, self.frame_seconds() as f32);
        Transcript { text, words }
    }

    /// Split at quiet points into pieces of about `target_piece_s` when longer than `max_single_pass_s`.
    pub fn split<'a>(&self, wav: &'a [f32]) -> Vec<(usize, &'a [f32])> {
        let sr = self.cfg.sample_rate as f64;
        let max = (self.opts.max_single_pass_s * sr) as usize;
        if wav.len() <= max {
            return vec![(0, wav)];
        }
        let hop = self.cfg.sample_rate / 50;
        let energy: Vec<f64> =
            wav.chunks(hop).map(|c| (c.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hop as f64).sqrt()).collect();
        let smooth: Vec<f64> = (0..energy.len())
            .map(|i| {
                let (a, b) = (i.saturating_sub(5), (i + 5).min(energy.len()));
                energy[a..b].iter().sum::<f64>() / 10.0
            })
            .collect();
        let target = self.opts.target_piece_s;
        let search = (target / 3.0).min(10.0);
        let mut pieces = Vec::new();
        let mut start = 0usize;
        while wav.len() - start > max {
            let lo = (start + ((target - search / 2.0) * sr) as usize) / hop;
            let hi = ((start + ((target + search / 2.0) * sr) as usize) / hop).min(smooth.len());
            let rel = smooth[lo..hi].iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |m| m.0);
            let cut = (lo + rel) * hop;
            pieces.push((start, &wav[start..cut]));
            start = cut;
        }
        pieces.push((start, &wav[start..]));
        pieces
    }

    /// Encode + decode a list of pieces (each at most `max_single_pass_s`), batched by length.
    pub fn run_pieces(&self, pieces: &[&[f32]]) -> Result<Vec<Vec<Token>>> {
        let nm = self.mel.n_mels();
        // group by length (longest first) so batches need little padding
        let mut order: Vec<usize> = (0..pieces.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(pieces[i].len()));
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut i = 0;
        while i < order.len() {
            let tmax = self.mel.frames(pieces[order[i]].len());
            let fit = (self.opts.max_batch_frames / tmax.max(1)).clamp(1, self.opts.batch_size);
            groups.push(order[i..(i + fit).min(order.len())].to_vec());
            i += fit;
        }
        // padded features [B, tmax, n_mels] + valid lengths for one group
        let features = |group: &[usize]| -> (Vec<f32>, usize, Vec<usize>) {
            use rayon::prelude::*;
            let t0 = Instant::now();
            let feats: Vec<Vec<f32>> = group.par_iter().map(|&g| self.mel.compute(pieces[g])).collect();
            let tmax = feats.iter().map(|f| f.len() / nm).max().unwrap_or(1);
            let mut buf = vec![0f32; group.len() * tmax * nm];
            for (k, f) in feats.iter().enumerate() {
                buf[k * tmax * nm..k * tmax * nm + f.len()].copy_from_slice(f);
            }
            self.timing.lock().unwrap().mel += t0.elapsed().as_secs_f64();
            (buf, tmax, feats.iter().map(|f| f.len() / nm).collect())
        };
        let mut results = vec![Vec::new(); pieces.len()];
        let mut next = groups.first().map(|g| features(g));
        for (gi, group) in groups.iter().enumerate() {
            let (buf, tmax, lens) = next.take().expect("features for every group");
            // features of the next group are computed on the CPU while this group runs on the device
            let toks = std::thread::scope(|sc| -> Result<_> {
                let pending = groups.get(gi + 1).map(|g| sc.spawn(|| features(g)));
                let t1 = Instant::now();
                let (enc, enc_lens) = self.encoder.forward_host(&buf, group.len(), tmax, &lens)?;
                self.sync()?;
                let t2 = Instant::now();
                let toks = self.decoder.decode(&enc, &enc_lens)?;
                let t3 = Instant::now();
                {
                    let mut tm = self.timing.lock().unwrap();
                    tm.encoder += (t2 - t1).as_secs_f64();
                    tm.decoder += (t3 - t2).as_secs_f64();
                }
                next = pending.map(|h| h.join().expect("feature thread panicked"));
                Ok(toks)
            })?;
            for (k, &g) in group.iter().enumerate() {
                results[g] = toks[k].clone();
            }
        }
        Ok(results)
    }
}
