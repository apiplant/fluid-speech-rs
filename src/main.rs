use anyhow::Result;
use clap::{Parser, ValueEnum};
use fluid_speech::text::{self, Word};
use fluid_speech::{audio, mic, DeviceArg, Engine, LoadOptions};
use rayon::prelude::*;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum Format {
    Text,
    Json,
    Srt,
}

/// Transcribe English speech with Fluid Speech Mini (Parakeet TDT 0.6B) on CPU or CUDA.
#[derive(Parser, Debug)]
#[command(name = "fluid-speech", version)]
struct Args {
    /// Audio files (wav, flac, mp3, ogg/vorbis, m4a/aac, ...); any sample rate, mixed to mono
    #[arg(required_unless_present_any = ["mic", "list_mics", "manifest"])]
    files: Vec<PathBuf>,

    /// Transcribe the microphone live: the utterance in progress shows on stderr, finished ones print to stdout
    #[arg(long, conflicts_with_all = ["files", "manifest"])]
    mic: bool,

    /// Input device for --mic: any part of a name from --list-mics (default: the system default input)
    #[arg(long)]
    mic_device: Option<String>,

    /// List audio input devices (* marks the default) and exit
    #[arg(long)]
    list_mics: bool,

    /// --mic: pause (ms) that ends an utterance
    #[arg(long, default_value_t = 700)]
    silence_ms: u32,

    /// --mic: fixed RMS level (of the 200 Hz high-passed signal) counted as speech (default: adaptive, 4x the
    /// measured noise floor)
    #[arg(long)]
    vad_threshold: Option<f32>,

    /// --mic: drop one- or two-word utterances below this confidence (noise often decodes as "Yeah." / "Okay.");
    /// 0 keeps everything. RUST_LOG=fluid_speech=debug shows what gets dropped
    #[arg(long, default_value_t = 0.9)]
    min_confidence: f32,

    /// Model directory (model.safetensors, config.json, vocab.json). Default: ./model, ../model, next to the
    /// binary, else downloaded from Hugging Face into ~/.cache/fluid-speech-rs on first use
    #[arg(short, long, env = "FLUID_MODEL")]
    model: Option<PathBuf>,

    #[arg(short, long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Clips up to this many seconds are encoded in one pass; longer audio is cut at quiet points into ~30 s pieces
    /// (in --mic mode: the longest single utterance)
    #[arg(long, default_value_t = 40.0)]
    max_pass: f32,

    /// Pieces per encoder batch
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    #[arg(short, long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// CPU threads (default: number of physical cores)
    #[arg(long)]
    threads: Option<usize>,

    /// Print load / decode timings and real-time factor to stderr
    #[arg(long)]
    bench: bool,

    /// Run one untimed pass first (CUDA graph capture, cuBLAS initialisation)
    #[arg(long)]
    warmup: bool,

    /// Repeat the transcription N times and report the median (with --bench)
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// JSONL manifest ({"id": ..., "audio": ...} per line); writes {"id", "hyp"} lines to stdout or --out
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Output JSONL for --manifest
    #[arg(long)]
    out: Option<PathBuf>,

    /// CUDA: fp16 accumulation in encoder GEMMs (~20% faster on GeForce, WER within noise)
    #[arg(long)]
    fast: bool,

    /// CUDA: autotune cuBLASLt algorithms per GEMM shape on first use
    #[arg(long)]
    tune: bool,

    /// CUDA: run the encoder/decoder with plain candle ops instead of the fused graph engine (for comparison)
    #[arg(long)]
    reference_ops: bool,
}

#[derive(Serialize)]
struct FileResult {
    file: String,
    duration: f32,
    text: String,
    words: Vec<Word>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,pulseaudio=off")).init();
    let a = Args::parse();
    if a.list_mics {
        return mic::list();
    }
    let threads = fluid_speech::init_threads(a.threads);
    let t0 = Instant::now();
    let opts = LoadOptions { fast_gemm: a.fast, autotune: a.tune, fused_cuda: !a.reference_ops };
    let mut engine = Engine::open(a.model.clone(), a.device, &opts)?;
    engine.opts.batch_size = a.batch_size;
    engine.opts.max_single_pass_s = a.max_pass as f64;
    let load_s = t0.elapsed().as_secs_f32();

    if a.mic {
        let mopts = mic::MicOptions {
            device: a.mic_device.clone(),
            vad_threshold: a.vad_threshold,
            silence_ms: a.silence_ms,
            max_secs: a.max_pass,
            json: a.format == Format::Json,
            srt: a.format == Format::Srt,
            min_confidence: a.min_confidence,
        };
        let transcribe = |samples: &[f32]| -> Result<Vec<Word>> { Ok(engine.transcribe(samples)?.words) };
        return mic::run(&mopts, &transcribe);
    }
    if let Some(m) = &a.manifest {
        return run_manifest(&engine, m, a.out.as_deref(), a.bench, a.warmup, load_s);
    }

    let t1 = Instant::now();
    let audios: Vec<Vec<f32>> = a.files.par_iter().map(|p| audio::load(p)).collect::<Result<_>>()?;
    let audio_s = t1.elapsed().as_secs_f32();
    let clips: Vec<&[f32]> = audios.iter().map(|v| v.as_slice()).collect();

    if a.warmup {
        if let Some(c) = clips.first() {
            engine.transcribe(&c[..c.len().min(10 * audio::SAMPLE_RATE)])?;
        }
    }

    let mut runs = Vec::new();
    let mut transcripts = Vec::new();
    for _ in 0..a.repeat.max(1) {
        *engine.timing.lock().unwrap() = Default::default();
        let t2 = Instant::now();
        transcripts = engine.transcribe_many(&clips)?;
        runs.push((t2.elapsed().as_secs_f32(), engine.timing.lock().unwrap().clone()));
    }

    let results: Vec<FileResult> = a
        .files
        .iter()
        .zip(&audios)
        .zip(transcripts)
        .map(|((p, s), t)| FileResult {
            file: p.display().to_string(),
            duration: s.len() as f32 / audio::SAMPLE_RATE as f32,
            text: t.text,
            words: t.words,
        })
        .collect();

    match a.format {
        Format::Json => println!("{}", serde_json::to_string_pretty(&results)?),
        Format::Text => {
            for r in &results {
                if results.len() > 1 {
                    println!("{}: {}", r.file, r.text);
                } else {
                    println!("{}", r.text);
                }
            }
        }
        Format::Srt => {
            for r in &results {
                if results.len() > 1 {
                    println!("# {}", r.file);
                }
                print!("{}", text::srt(&r.words));
            }
        }
    }

    if a.bench {
        let total: f32 = results.iter().map(|r| r.duration).sum();
        runs.sort_by(|x, y| x.0.total_cmp(&y.0));
        let (infer_s, tm) = &runs[runs.len() / 2];
        let dev = if engine.cuda { "cuda" } else { "cpu" };
        eprintln!(
            "device={dev} threads={threads} | load {load_s:.2}s | audio decode {audio_s:.2}s | {total:.1}s of audio \
             transcribed in {:.1} ms (mel {:.1} enc {:.1} dec {:.1}) = {:.0}x realtime{}",
            infer_s * 1e3,
            tm.mel * 1e3,
            tm.encoder * 1e3,
            tm.decoder * 1e3,
            total / infer_s.max(1e-6),
            if runs.len() > 1 { format!(" (median of {})", runs.len()) } else { String::new() }
        );
    }
    Ok(())
}

fn run_manifest(engine: &Engine, manifest: &Path, out: Option<&Path>, bench: bool, warmup: bool, load_s: f32) -> Result<()> {
    use std::io::Write;
    const GROUP: usize = 512;
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(manifest)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let t = Instant::now();
    let wavs: Vec<Vec<f32>> =
        rows.par_iter().map(|r| audio::load(r["audio"].as_str().unwrap_or_default().as_ref())).collect::<Result<_>>()?;
    let load_audio_s = t.elapsed().as_secs_f64();
    let total: f64 = wavs.iter().map(|w| w.len() as f64).sum::<f64>() / audio::SAMPLE_RATE as f64;
    let mut sink: Box<dyn Write> = match out {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::stdout().lock()),
    };
    if warmup {
        // a full untimed pass captures every graph shape
        for w in wavs.chunks(GROUP) {
            engine.transcribe_many(&w.iter().map(|w| w.as_slice()).collect::<Vec<_>>())?;
        }
        *engine.timing.lock().unwrap() = Default::default();
    }
    let t = Instant::now();
    for (rows, wavs) in rows.chunks(GROUP).zip(wavs.chunks(GROUP)) {
        let refs: Vec<&[f32]> = wavs.iter().map(|w| w.as_slice()).collect();
        for (r, tr) in rows.iter().zip(engine.transcribe_many(&refs)?) {
            writeln!(sink, "{}", serde_json::json!({"id": r["id"], "hyp": tr.text}))?;
        }
    }
    sink.flush()?;
    let el = t.elapsed().as_secs_f64();
    if bench {
        let tm = engine.timing.lock().unwrap();
        eprintln!(
            "{} clips, {:.1} min audio | model load {load_s:.2}s, audio load {load_audio_s:.2}s | transcribe {el:.2}s \
             (mel {:.2} enc {:.2} dec {:.2}) | {:.0}x realtime",
            rows.len(),
            total / 60.0,
            tm.mel,
            tm.encoder,
            tm.decoder,
            total / el
        );
    }
    Ok(())
}
