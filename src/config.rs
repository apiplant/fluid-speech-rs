use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: f64,
    pub log_zero_guard: f64,
    pub norm_eps: f64,
    pub mel_fmin: f64,
    pub mel_fmax: f64,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub ff_dim: usize,
    pub conv_kernel: usize,
    pub subsampling_channels: usize,
    pub subsampling_factor: usize,
    pub layer_norm_eps: f64,
    pub ff_residual_factor: f64,
    pub pred_hidden: usize,
    pub pred_layers: usize,
    pub joint_hidden: usize,
    pub vocab_size: usize,
    pub blank_id: usize,
    pub durations: Vec<usize>,
    pub max_symbols_per_step: usize,
}

impl Config {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Encoder frames produced for `mel_frames` input frames (three stride-2 convs, kernel 3, pad 1).
    pub fn encoder_frames(&self, mel_frames: usize) -> usize {
        let mut t = mel_frames;
        for _ in 0..3 {
            t = (t - 1) / 2 + 1;
        }
        t
    }
}
