//! Live microphone transcription: capture with cpal, cut utterances with an adaptive energy VAD, show the
//! utterance in progress on stderr and print each finished one to stdout.

use crate::audio;
use crate::text::{self, Word};
use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct MicOptions {
    pub device: Option<String>,
    /// Fixed RMS speech threshold; adaptive (4x the noise floor) when None
    pub vad_threshold: Option<f32>,
    /// Pause that ends an utterance
    pub silence_ms: u32,
    /// Longest utterance before a forced cut
    pub max_secs: f32,
    pub json: bool,
    pub srt: bool,
    /// Utterances of one or two words below this confidence are dropped (0 disables the check)
    pub min_confidence: f32,
}

/// Frames of 20 ms that must be voiced in a row before speech counts as started: clicks and taps are shorter.
const ONSET_FRAMES: usize = 3;
/// Utterances with less voiced audio than this are dropped without decoding.
pub const MIN_VOICED_SECS: f32 = 0.25;

/// Energy voice-activity detector on 20 ms frames of the signal high-passed at 200 Hz, so desk knocks, mic bumps,
/// hum and rumble (which the model likes to read as "Yeah.") don't count as speech.
pub struct Vad {
    hp: [f32; 5], // b0 b1 b2 a1 a2
    z: [f32; 2],
    noise: f32,
    fixed: Option<f32>,
    run: usize,
}

impl Vad {
    pub fn new(rate: usize, fixed: Option<f32>) -> Self {
        // RBJ biquad high-pass, 200 Hz, Q = 1/sqrt(2)
        let w = 2.0 * std::f32::consts::PI * 200.0 / rate as f32;
        let alpha = w.sin() / std::f32::consts::SQRT_2;
        let (c, a0) = (w.cos(), 1.0 + alpha);
        let hp = [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0, -2.0 * c / a0, (1.0 - alpha) / a0];
        Self { hp, z: [0.0; 2], noise: f32::INFINITY, fixed, run: 0 }
    }

    fn energy(&mut self, x: &[f32]) -> f32 {
        let [b0, b1, b2, a1, a2] = self.hp;
        let mut acc = 0f32;
        for &v in x {
            // transposed direct form II
            let y = b0 * v + self.z[0];
            self.z[0] = b1 * v - a1 * y + self.z[1];
            self.z[1] = b2 * v - a2 * y;
            acc += y * y;
        }
        (acc / x.len().max(1) as f32).sqrt()
    }

    /// Feed one frame; returns how many frames just became confirmed speech (0, 1, or ONSET_FRAMES at onset).
    pub fn frame(&mut self, x: &[f32]) -> usize {
        let e = self.energy(x);
        // noise floor: follow quiet frames down fast, drift up slowly
        self.noise = if e < self.noise { e } else { self.noise * 1.0005 + 1e-7 };
        let thr = self.fixed.unwrap_or((self.noise * 4.0).max(0.003));
        log::trace!("vad energy {e:.5} threshold {thr:.5} run {}", self.run);
        if e > thr {
            self.run += 1;
        } else {
            self.run = 0;
        }
        match self.run {
            r if r == ONSET_FRAMES => ONSET_FRAMES,
            r if r > ONSET_FRAMES => 1,
            _ => 0,
        }
    }
}

/// Seconds of confirmed speech in a whole recording (noise floor taken from its quietest frames).
pub fn voiced_secs(samples: &[f32], rate: usize) -> f32 {
    let win = rate / 50;
    let mut probe = Vad::new(rate, None);
    let mut es: Vec<f32> = samples.chunks_exact(win).map(|f| probe.energy(f)).collect();
    if es.is_empty() {
        return 0.0;
    }
    es.sort_by(f32::total_cmp);
    let floor = es[es.len() / 10];
    let mut vad = Vad::new(rate, Some((floor * 4.0).max(0.003)));
    let frames: usize = samples.chunks_exact(win).map(|f| vad.frame(f)).sum();
    frames as f32 * 0.02
}

/// Drop decodes that look like noise: one or two words at low confidence.
pub fn plausible(words: &[Word], min_confidence: f32) -> bool {
    !words.is_empty() && (words.len() > 2 || text::confidence(words) >= min_confidence)
}

/// PulseAudio (served by PipeWire on most desktops) when it is running, so named devices are the sound server's
/// sources rather than raw ALSA hardware it holds exclusively; otherwise the platform default host.
fn host() -> cpal::Host {
    #[cfg(target_os = "linux")]
    if let Ok(h) = cpal::host_from_id(cpal::HostId::PulseAudio) {
        return h;
    }
    cpal::default_host()
}

fn describe(d: &cpal::Device) -> String {
    let id = d.id().map(|i| i.to_string()).unwrap_or_else(|_| "?".into());
    match d.description() {
        Ok(desc) => format!("{id} ({desc})"),
        Err(_) => id,
    }
}

pub fn list() -> Result<()> {
    let host = host();
    let default = host.default_input_device();
    let default_id = default.as_ref().and_then(|d| d.id().ok()).map(|i| i.to_string());
    let mut out = std::io::stdout().lock();
    // the system default is often not among the enumerated devices; list it first
    if let Some(d) = &default {
        writeln!(out, "* {}", describe(d))?;
    }
    for d in host.input_devices()? {
        if d.id().ok().map(|i| i.to_string()) != default_id {
            writeln!(out, "  {}", describe(&d))?;
        }
    }
    Ok(())
}

fn pick_device(name: Option<&str>) -> Result<cpal::Device> {
    let host = host();
    match name {
        None => host.default_input_device().context("no default input device"),
        Some(n) => {
            let n = n.to_lowercase();
            // real inputs before loopback monitors of outputs that happen to share the name
            let mut hits: Vec<(bool, cpal::Device)> = host
                .input_devices()?
                .filter_map(|d| {
                    let desc = describe(&d).to_lowercase();
                    desc.contains(&n).then(|| (desc.contains("monitor"), d))
                })
                .collect();
            hits.sort_by_key(|(monitor, _)| *monitor);
            hits.into_iter().next().map(|(_, d)| d).with_context(|| format!("no input device matching {n:?} (see --list-mics)"))
        }
    }
}

/// Mono f32 samples at the device rate, appended by the capture callback, and how many have arrived in total.
type Shared = Arc<(Mutex<Vec<f32>>, std::sync::atomic::AtomicUsize)>;

fn build_stream<T>(device: &cpal::Device, config: &cpal::StreamConfig, buf: Shared) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let ch = config.channels as usize;
    let stream = device.build_input_stream(
        *config,
        move |data: &[T], _: &_| {
            buf.1.fetch_add(data.len() / ch.max(1), Ordering::Relaxed);
            let mut b = buf.0.lock().unwrap();
            if ch == 1 {
                b.extend(data.iter().map(|s| f32::from_sample(*s)));
            } else {
                let inv = 1.0 / ch as f32;
                b.extend(data.chunks_exact(ch).map(|f| f.iter().map(|s| f32::from_sample(*s)).sum::<f32>() * inv));
            }
        },
        |e| eprintln!("\nmicrophone error: {e}"),
        None,
    )?;
    Ok(stream)
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// An open input stream appending mono f32 samples (at `rate`) to a shared buffer. Capture stops on drop.
pub struct Capture {
    stream: cpal::Stream,
    buf: Shared,
    started: Instant,
    pub rate: usize,
    pub name: String,
}

impl Capture {
    /// Open `device` (any part of a name from `list`, or the default input) and start capturing.
    pub fn open(device: Option<&str>) -> Result<Self> {
        let device = pick_device(device)?;
        let default = device.default_input_config()?;
        // sound servers convert for free: ask for f32 at the default rate when offered
        let supported = device
            .supported_input_configs()?
            .filter(|c| c.sample_format() == SampleFormat::F32)
            .find_map(|c| c.try_with_sample_rate(default.sample_rate()))
            .unwrap_or(default);
        let rate = supported.sample_rate() as usize;
        #[allow(unused_mut)] // only adjusted on Linux
        let mut config: cpal::StreamConfig = supported.config();
        // small periods: the sound server otherwise hands audio over in ~1/3 s bursts, and whatever is still in
        // flight when recording stops would be lost. Linux only: CoreAudio already uses small periods, and forcing
        // a fixed size there makes the device overrun
        #[cfg(target_os = "linux")]
        if let cpal::SupportedBufferSize::Range { min, max } = supported.buffer_size() {
            config.buffer_size = cpal::BufferSize::Fixed(((rate / 100) as u32).clamp(*min, *max));
        }
        let buf: Shared = Arc::default();
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build_stream::<f32>(&device, &config, buf.clone())?,
            SampleFormat::I16 => build_stream::<i16>(&device, &config, buf.clone())?,
            SampleFormat::I32 => build_stream::<i32>(&device, &config, buf.clone())?,
            SampleFormat::I24 => build_stream::<cpal::I24>(&device, &config, buf.clone())?,
            SampleFormat::U32 => build_stream::<u32>(&device, &config, buf.clone())?,
            SampleFormat::F64 => build_stream::<f64>(&device, &config, buf.clone())?,
            SampleFormat::U16 => build_stream::<u16>(&device, &config, buf.clone())?,
            SampleFormat::U8 => build_stream::<u8>(&device, &config, buf.clone())?,
            SampleFormat::I8 => build_stream::<i8>(&device, &config, buf.clone())?,
            f => bail!("unsupported microphone sample format {f:?}"),
        };
        stream.play()?;
        Ok(Self { stream, buf, started: Instant::now(), rate, name: describe(&device) })
    }

    /// Samples captured since the last call, at `self.rate`.
    pub fn take(&self) -> Vec<f32> {
        std::mem::take(&mut *self.buf.0.lock().unwrap())
    }

    /// Stop capturing and return what is left in the buffer. Waits (up to a second) until the audio up to now
    /// plus `tail` has actually arrived, so the last word isn't cut off.
    pub fn finish(self, tail: Duration) -> Vec<f32> {
        let want = ((self.started.elapsed() + tail).as_secs_f64() * self.rate as f64) as usize;
        let deadline = Instant::now() + tail + Duration::from_secs(1);
        while self.buf.1.load(Ordering::Relaxed) < want && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(self.stream);
        std::mem::take(&mut *self.buf.0.lock().unwrap())
    }
}

/// `transcribe` maps 16 kHz mono samples to words (times relative to the samples' start).
pub fn run(opts: &MicOptions, transcribe: &dyn Fn(&[f32]) -> Result<Vec<Word>>) -> Result<()> {
    // first call pays for CUDA/cuBLAS initialisation; do it before listening
    transcribe(&vec![0.0; audio::SAMPLE_RATE])?;
    let cap = Capture::open(opts.device.as_deref())?;
    let rate = cap.rate;

    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;

    eprintln!("Listening on {} at {rate} Hz. Ctrl-C to stop.", cap.name);

    let tty = std::io::stderr().is_terminal();
    let win = rate / 50; // 20 ms VAD frames
    let preroll = rate * 3 / 10;
    let silence = rate * opts.silence_ms as usize / 1000;
    let max_len = (opts.max_secs * rate as f32) as usize;

    let mut seg: Vec<f32> = Vec::new(); // current utterance, device rate
    let mut seg_start = 0usize; // its position in the session, device samples
    let mut vad_pos = 0usize; // next seg index to run the VAD on
    let mut speech = false;
    let mut last_speech = 0usize;
    let mut vad = Vad::new(rate, opts.vad_threshold);
    let mut voiced = 0usize; // confirmed speech frames in the current utterance
    let mut last_partial = Instant::now();
    let mut partial_every = Duration::from_millis(500);
    let mut cue = 0usize;

    let to16k = |x: &[f32]| audio::resample(x.to_vec(), rate);
    let clear = || {
        if tty {
            eprint!("\r\x1b[2K");
        }
    };

    let finish = |x: &[f32], start: usize, cue: &mut usize, voiced: usize| -> Result<()> {
        let offset = start as f32 / rate as f32;
        if (voiced as f32) * 0.02 < MIN_VOICED_SECS {
            clear();
            log::debug!("dropped {:.2}s with {:.2}s of speech", x.len() as f32 / rate as f32, voiced as f32 * 0.02);
            return Ok(());
        }
        let mut words = transcribe(&to16k(x)?)?;
        if !plausible(&words, opts.min_confidence) {
            clear();
            if !words.is_empty() {
                log::debug!("dropped {:?} (confidence {:.2})", text::join(&words), text::confidence(&words));
            }
            return Ok(());
        }
        words.iter_mut().for_each(|w| {
            w.start += offset;
            w.end += offset;
        });
        clear();
        if words.is_empty() {
            return Ok(());
        }
        let line = text::join(&words);
        let (a, b) = (words[0].start, words.last().unwrap().end);
        let mut out = std::io::stdout().lock();
        if opts.json {
            let v = serde_json::json!({ "start": a, "end": b, "text": line, "words": words });
            writeln!(out, "{v}")?;
        } else if opts.srt {
            *cue += 1;
            writeln!(out, "{cue}\n{} --> {}\n{line}\n", crate::text::srt_time(a), crate::text::srt_time(b))?;
        } else {
            writeln!(out, "{line}")?;
        }
        out.flush()?;
        Ok(())
    };

    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(50));
        seg.extend(cap.take());

        while vad_pos + win <= seg.len() {
            let n = vad.frame(&seg[vad_pos..vad_pos + win]);
            if n > 0 {
                speech = true;
                voiced += n;
                last_speech = vad_pos + win;
            }
            vad_pos += win;
        }

        if !speech {
            if seg.len() > preroll {
                let drop = seg.len() - preroll;
                seg.drain(..drop);
                seg_start += drop;
                vad_pos -= drop.min(vad_pos);
            }
            continue;
        }

        if seg.len() - last_speech >= silence {
            finish(&seg[..last_speech.min(seg.len())], seg_start, &mut cue, voiced)?;
            seg_start += seg.len();
            seg.clear();
            vad_pos = 0;
            speech = false;
            voiced = 0;
        } else if seg.len() >= max_len {
            // cut at the quietest 20 ms in the last quarter, keep the rest for the next utterance
            let from = seg.len() * 3 / 4;
            let cut = (from..seg.len() - win)
                .step_by(win / 2)
                .min_by(|&a, &b| rms(&seg[a..a + win]).total_cmp(&rms(&seg[b..b + win])))
                .map(|p| p + win / 2)
                .unwrap_or(seg.len());
            finish(&seg[..cut], seg_start, &mut cue, voiced)?;
            voiced = (seg.len() - cut) / win; // mid-speech: count the carried-over tail as voiced
            seg.drain(..cut);
            seg_start += cut;
            vad_pos = vad_pos.saturating_sub(cut);
            last_speech = last_speech.saturating_sub(cut);
        } else if tty && voiced as f32 * 0.02 >= MIN_VOICED_SECS && last_partial.elapsed() >= partial_every {
            let t = Instant::now();
            let words = transcribe(&to16k(&seg)?)?;
            // keep partial updates from eating the machine on slow CPUs
            partial_every = Duration::from_millis(500).max(t.elapsed() * 2);
            last_partial = Instant::now();
            let line = if plausible(&words, opts.min_confidence) { text::join(&words) } else { String::new() };
            let width = 120usize;
            let shown: String = if line.chars().count() > width {
                let skip = line.chars().count() - width;
                format!("…{}", line.chars().skip(skip + 1).collect::<String>())
            } else {
                line
            };
            eprint!("\r\x1b[2K\x1b[2m{shown}\x1b[0m");
        }
    }

    seg.extend(cap.finish(Duration::ZERO));
    if speech && !seg.is_empty() {
        finish(&seg, seg_start, &mut cue, voiced)?;
    }
    clear();
    Ok(())
}
