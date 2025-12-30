use directories::BaseDirs;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use log::warn;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InferenceDevice {
    CPU,
    CUDA,
}

impl Default for InferenceDevice {
    fn default() -> Self {
        InferenceDevice::CPU
    }
}

impl InferenceDevice {
    pub fn is_gpu(self) -> bool {
        matches!(self, InferenceDevice::CUDA)
    }

    pub fn from_i32(value: i32) -> Self {
        match value {
            1 => InferenceDevice::CUDA,
            _ => InferenceDevice::CPU,
        }
    }
}

impl fmt::Display for InferenceDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            InferenceDevice::CPU => "cpu",
            InferenceDevice::CUDA => "cuda",
        };
        write!(f, "{label}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_path: String,
    pub threads: u8,
    pub language: String,
    pub inference_device: InferenceDevice,
    pub gpu_device: i32,
    #[serde(alias = "cuda_flash_attn")]
    pub flash_attn: bool,

    // Translation settings (builtin Google Translate)
    pub from_lang: String,
    pub to_lang: String,

    pub local_chunk_size_ms: u64,
    pub network_chunk_size_ms: u64,
    pub wav_chunk_size_ms: u64,
    pub show_progress: bool,
    pub start_at_zero: bool,
    pub save_srt: bool,

    pub ffmpeg_timeout_ms: u64,
    pub ffprobe_timeout_ms: u64,
    pub whisper_timeout_ms: u64,
    pub translate_timeout_ms: u64,
    pub translate_concurrency: usize,

    // Delay handling features (always enabled)
    pub catchup_threshold_ms: u64,
    pub lookahead_chunks: usize,
    pub lookahead_limit_ms: u64,

    // Network cache settings
    pub demuxer_max_bytes: Option<i64>,
    pub min_network_chunk_ms: u64,

    // Auto-start settings
    pub auto_start: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: "ggml-base.bin".to_string(),
            threads: 8,
            language: "en".to_string(),
            inference_device: InferenceDevice::CPU,
            gpu_device: 0,
            flash_attn: false,

            // Translation defaults (builtin Google Translate)
            from_lang: "en".to_string(),
            to_lang: "zh".to_string(),

            local_chunk_size_ms: 15_000,
            network_chunk_size_ms: 15_000,
            wav_chunk_size_ms: 16_000,
            show_progress: true,
            start_at_zero: true,
            save_srt: true,

            ffmpeg_timeout_ms: 30_000,
            ffprobe_timeout_ms: 10_000,
            whisper_timeout_ms: 120_000,
            translate_timeout_ms: 30_000,
            translate_concurrency: 4,

            // Delay handling defaults (always enabled)
            catchup_threshold_ms: 30_000,  // 30 seconds behind -> catch up
            lookahead_chunks: 2,  // Pre-process 2 chunks ahead
            lookahead_limit_ms: 60_000,  // Maximum 60 seconds ahead of playback

            // Network cache defaults
            demuxer_max_bytes: None,  // Use mpv's default (150MB) if not specified
            min_network_chunk_ms: 5_000,

            // Auto-start defaults
            auto_start: false,  // Disabled by default, use Ctrl+. to toggle manually
        }
    }
}

impl Config {
    pub fn default_config_path() -> Option<PathBuf> {
        let base = BaseDirs::new()?;
        Some(base.config_dir().join("mpv").join("whispersubs.toml"))
    }

    pub fn config_path_from_env() -> Option<PathBuf> {
        std::env::var_os("WHISPERSUBS_CONFIG").map(PathBuf::from)
    }

    pub fn load() -> Self {
        let config_path = Self::config_path_from_env().or_else(Self::default_config_path);

        let mut figment = Figment::from(Serialized::defaults(Config::default()));

        if let Some(path) = config_path.as_ref() {
            figment = figment.merge(Toml::file(path));
        }

        // Env should take precedence over file/defaults.
        figment = figment.merge(Env::prefixed("WHISPERSUBS_"));

        match figment.extract::<Config>() {
            Ok(mut cfg) => {
                // Backward-compatible env var for old config name.
                if !cfg.flash_attn {
                    if let Ok(value) = std::env::var("WHISPERSUBS_CUDA_FLASH_ATTN") {
                        if let Some(parsed) = parse_env_bool(&value) {
                            cfg.flash_attn = parsed;
                        }
                    }
                }
                cfg
            }
            Err(err) => {
                // Logging might not be initialized yet; fall back silently.
                warn!("Failed to load config, using defaults: {err}");
                Config::default()
            }
        }
    }
}

fn parse_env_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
