//! NeMo / Parakeet log-mel front end: preemphasis, reflect-padded STFT (512-point FFT, 400-sample symmetric
//! Hann window, hop 160), power spectrum, Slaney mel filterbank (128 bins, 0-8 kHz), log(x + 2^-24), and
//! per-feature normalisation over the clip.

use crate::config::Config;
use rayon::prelude::*;
use realfft::RealFftPlanner;

pub struct MelFrontend {
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    preemph: f32,
    log_guard: f32,
    norm_eps: f32,
    window: Vec<f32>,
    /// sparse filterbank: per mel bin (first fft bin, weights)
    filters: Vec<(usize, Vec<f32>)>,
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
}

fn hz_to_mel(f: f64) -> f64 {
    let lin = f / (200.0 / 3.0);
    if f >= 1000.0 {
        15.0 + (f / 1000.0).ln() / (6.4f64.ln() / 27.0)
    } else {
        lin
    }
}

fn mel_to_hz(m: f64) -> f64 {
    if m >= 15.0 {
        1000.0 * ((6.4f64.ln() / 27.0) * (m - 15.0)).exp()
    } else {
        m * (200.0 / 3.0)
    }
}

/// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney")
pub fn mel_filterbank(sr: f64, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<Vec<f32>> {
    let nbins = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..nbins).map(|i| i as f64 * sr / n_fft as f64).collect();
    let (mmin, mmax) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let pts: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f64 / (n_mels + 1) as f64)).collect();
    (0..n_mels)
        .map(|m| {
            let enorm = 2.0 / (pts[m + 2] - pts[m]);
            fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - pts[m]) / (pts[m + 1] - pts[m]);
                    let upper = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
                    (lower.min(upper).max(0.0) * enorm) as f32
                })
                .collect()
        })
        .collect()
}

impl MelFrontend {
    pub fn new(cfg: &Config) -> Self {
        let n_fft = cfg.n_fft;
        let mut window = vec![0f32; n_fft];
        let left = (n_fft - cfg.win_length) / 2;
        for i in 0..cfg.win_length {
            let v = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (cfg.win_length - 1) as f64).cos();
            window[left + i] = v as f32;
        }
        let fb = mel_filterbank(cfg.sample_rate as f64, n_fft, cfg.n_mels, cfg.mel_fmin, cfg.mel_fmax);
        let filters = fb
            .into_iter()
            .map(|row| {
                let first = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&w| w != 0.0).unwrap_or(0);
                (first, row[first..=last.max(first)].to_vec())
            })
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n_fft);
        Self {
            n_fft,
            hop: cfg.hop_length,
            n_mels: cfg.n_mels,
            preemph: cfg.preemphasis as f32,
            log_guard: cfg.log_zero_guard as f32,
            norm_eps: cfg.norm_eps as f32,
            window,
            filters,
            fft,
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Number of feature frames for `n` samples.
    pub fn frames(&self, n: usize) -> usize {
        n / self.hop + 1
    }

    /// Normalised log-mel features, time-major: [frames, n_mels].
    pub fn compute(&self, wav: &[f32]) -> Vec<f32> {
        let n = wav.len();
        let pad = self.n_fft / 2;
        // preemphasis + reflect padding
        let mut x = vec![0f32; n + 2 * pad];
        {
            let core = &mut x[pad..pad + n];
            if n > 0 {
                core[0] = wav[0];
            }
            for i in 1..n {
                core[i] = wav[i] - self.preemph * wav[i - 1];
            }
        }
        for i in 0..pad {
            // reflect (no edge repeat); clamp for very short clips
            let l = (pad - i).min(n.saturating_sub(1));
            x[i] = x[pad + l];
            let r = (i + 1).min(n.saturating_sub(1));
            x[pad + n + i] = x[pad + n - 1 - r];
        }
        let frames = self.frames(n);
        let nm = self.n_mels;
        let mut mel = vec![0f32; frames * nm];
        let nbins = self.n_fft / 2 + 1;
        mel.par_chunks_mut(nm * 64).enumerate().for_each(|(ci, chunk)| {
            let mut inp = self.fft.make_input_vec();
            let mut spec = self.fft.make_output_vec();
            let mut scratch = self.fft.make_scratch_vec();
            let mut power = vec![0f32; nbins];
            for (k, out) in chunk.chunks_mut(nm).enumerate() {
                let f = ci * 64 + k;
                let s = &x[f * self.hop..f * self.hop + self.n_fft];
                for i in 0..self.n_fft {
                    inp[i] = s[i] * self.window[i];
                }
                self.fft.process_with_scratch(&mut inp, &mut spec, &mut scratch).unwrap();
                for (p, c) in power.iter_mut().zip(spec.iter()) {
                    *p = c.re * c.re + c.im * c.im;
                }
                for (m, (first, w)) in self.filters.iter().enumerate() {
                    let mut acc = 0f32;
                    for (i, wi) in w.iter().enumerate() {
                        acc += wi * power[first + i];
                    }
                    out[m] = (acc + self.log_guard).ln();
                }
            }
        });
        // per-feature normalisation (unbiased std)
        let mut mean = vec![0f64; nm];
        for row in mel.chunks(nm) {
            for (m, v) in row.iter().enumerate() {
                mean[m] += *v as f64;
            }
        }
        mean.iter_mut().for_each(|m| *m /= frames as f64);
        let mut var = vec![0f64; nm];
        for row in mel.chunks(nm) {
            for (m, v) in row.iter().enumerate() {
                let d = *v as f64 - mean[m];
                var[m] += d * d;
            }
        }
        let denom = (frames.max(2) - 1) as f64;
        let scale: Vec<f32> = var.iter().map(|v| (1.0 / ((v / denom).sqrt() + self.norm_eps as f64)) as f32).collect();
        let mean: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
        mel.par_chunks_mut(nm).for_each(|row| {
            for m in 0..nm {
                row[m] = (row[m] - mean[m]) * scale[m];
            }
        });
        mel
    }
}
