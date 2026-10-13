//! Audio loading (any format symphonia decodes), mono mixdown and resampling to 16 kHz.

use anyhow::{bail, Context, Result};
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler};
use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub const SAMPLE_RATE: usize = 16_000;

/// Decode `path` to 16 kHz mono f32.
pub fn load(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .with_context(|| format!("unsupported audio format: {}", path.display()))?;
    let track = format.default_track(TrackType::Audio).context("no audio track")?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .context("track has no audio codec parameters")?;
    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(params, &AudioDecoderOptions::default())?;

    let mut mono: Vec<f32> = Vec::new();
    let mut rate = 0usize;
    let mut buf: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = decoded.spec();
        rate = spec.rate() as usize;
        let channels = spec.channels().count().max(1);
        buf.resize(decoded.samples_interleaved(), 0.0);
        decoded.copy_to_slice_interleaved(&mut buf);
        if channels == 1 {
            mono.extend_from_slice(&buf);
        } else {
            let inv = 1.0 / channels as f32;
            mono.extend(buf.chunks_exact(channels).map(|f| f.iter().sum::<f32>() * inv));
        }
    }
    if rate == 0 {
        bail!("{}: no audio decoded", path.display());
    }
    resample(mono, rate)
}

/// Resample mono audio from `rate` to 16 kHz.
pub fn resample(samples: Vec<f32>, rate: usize) -> Result<Vec<f32>> {
    if rate == SAMPLE_RATE || samples.is_empty() {
        return Ok(samples);
    }
    let mut rs = Fft::<f32>::new(rate, SAMPLE_RATE, 1024, 1, FixedSync::Input)?;
    let n = samples.len();
    let input = [samples];
    let adapter = SequentialSliceOfVecs::new(&input, 1, n)?;
    let out = rs.process_all(&adapter, n, None)?;
    Ok(out.take_data())
}
