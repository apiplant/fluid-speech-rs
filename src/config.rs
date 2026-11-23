use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;

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
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        // Fluid Speech Pro is published in Transformers' Parakeet format. Keep the engine's compact config
        // format as-is, while translating the upstream nested config when encoder_config is present.
        let Some(enc) = json.get("encoder_config") else {
            return Ok(serde_json::from_value(json)?);
        };
        let processor_path = path.with_file_name("processor_config.json");
        let processor: Value = if processor_path.is_file() {
            serde_json::from_str(&std::fs::read_to_string(processor_path)?)?
        } else {
            Value::Null
        };
        let feature = processor.get("feature_extractor").cloned().unwrap_or(Value::Null);
        let num = |v: &Value, key: &str, default: usize| v.get(key).and_then(Value::as_u64).map(|n| n as usize).unwrap_or(default);
        let fnum = |v: &Value, key: &str, default: f64| v.get(key).and_then(Value::as_f64).unwrap_or(default);
        let blank_id = num(&json, "blank_token_id", 8192);
        Ok(Self {
            sample_rate: num(&feature, "sampling_rate", 16_000),
            n_fft: num(&feature, "n_fft", 512),
            win_length: num(&feature, "win_length", 400),
            hop_length: num(&feature, "hop_length", 160),
            n_mels: num(enc, "num_mel_bins", 128),
            preemphasis: fnum(&feature, "preemphasis", 0.97),
            log_zero_guard: 2.0f64.powi(-24),
            norm_eps: 1e-5,
            mel_fmin: 0.0,
            mel_fmax: num(&feature, "sampling_rate", 16_000) as f64 / 2.0,
            d_model: num(enc, "hidden_size", 1024),
            n_heads: num(enc, "num_attention_heads", 8),
            n_layers: num(enc, "num_hidden_layers", 24),
            ff_dim: num(enc, "intermediate_size", 4096),
            conv_kernel: num(enc, "conv_kernel_size", 9),
            subsampling_channels: num(enc, "subsampling_conv_channels", 256),
            subsampling_factor: num(enc, "subsampling_factor", 8),
            layer_norm_eps: fnum(enc, "layer_norm_eps", 1e-5),
            ff_residual_factor: 0.5,
            pred_hidden: num(&json, "decoder_hidden_size", 640),
            pred_layers: num(&json, "num_decoder_layers", 2),
            joint_hidden: num(&json, "decoder_hidden_size", 640),
            // Upstream vocab_size includes the blank class; this field is the non-blank token count.
            vocab_size: blank_id,
            blank_id,
            durations: json.get("durations").and_then(Value::as_array)
                .map(|xs| xs.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
                .filter(|xs: &Vec<usize>| !xs.is_empty()).unwrap_or_else(|| vec![0, 1, 2, 3, 4]),
            max_symbols_per_step: num(&json, "max_symbols_per_step", 10),
        })
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
