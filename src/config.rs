use directories::BaseDirs;
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use tracing::warn;

/// Wire protocol an STT source speaks. The plugin is a pure remote client:
/// both protocols are compiled in, and a source's `protocol` picks which one
/// its requests use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SttProtocol {
    Ferrum,
    OpenAi,
}

impl std::fmt::Display for SttProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            SttProtocol::Ferrum => "ferrum",
            SttProtocol::OpenAi => "openai",
        };
        write!(f, "{label}")
    }
}

/// Wire protocol a translation source speaks. Every protocol is compiled in
/// (no feature exclusivity); a source's `protocol` picks which one its
/// requests use. Mirrors `SttProtocol` for the STT side.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TranslateSourceProtocol {
    /// Built-in free source: Google's web translate endpoint.
    Google,
    /// Built-in free source: Microsoft Edge's translate endpoint.
    Edge,
    /// Built-in free source: translate.alibaba.com.
    Alibaba,
    /// External DeepL-compatible service.
    DeepL,
    /// External LibreTranslate-compatible service.
    LibreTranslate,
}

impl std::fmt::Display for TranslateSourceProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            TranslateSourceProtocol::Google => "google",
            TranslateSourceProtocol::Edge => "edge",
            TranslateSourceProtocol::Alibaba => "alibaba",
            TranslateSourceProtocol::DeepL => "deepl",
            TranslateSourceProtocol::LibreTranslate => "libretranslate",
        };
        write!(f, "{label}")
    }
}

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
    pub stt: SttConfig,
    pub translate: TranslateConfig,
    pub chunk: ChunkConfig,
    pub timeout: TimeoutConfig,
    pub playback: PlaybackConfig,
    pub prefetch: PrefetchConfig,
    pub network: NetworkConfig,
    pub log: LogConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            stt: SttConfig::default(),
            translate: TranslateConfig::default(),
            chunk: ChunkConfig::default(),
            timeout: TimeoutConfig::default(),
            playback: PlaybackConfig::default(),
            prefetch: PrefetchConfig::default(),
            network: NetworkConfig::default(),
            log: LogConfig::default(),
        }
    }
}

/// Logging configuration. Every key has a usable default, so an existing config
/// file keeps working without a `[log]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// `EnvFilter` directives, e.g. `info` or
    /// `mpv_stt_plugin_rs::stt=trace,warn`. The `MPV_STT_PLUGIN_RS_LOG`
    /// environment variable overrides this.
    pub level: String,
    /// Directives for the log file when it should differ from `level`; the
    /// file defaults to `debug` so a GUI-launched player leaves enough behind
    /// to diagnose a failure after the fact. Empty = same as `level`.
    pub file_level: String,
    /// `auto` = `<config dir>/mpv_stt_plugin_rs.log`, `""` = no file, or an
    /// explicit path.
    pub file: String,
    /// Rotated files to keep (daily rotation).
    pub file_max_files: usize,
    /// `compact` (one line per event) | `full` (event plus span open/close with
    /// timings) | `json`.
    pub format: String,
    /// Colorize: `""` = only when the sink is a terminal, else `true`/`false`.
    pub ansi: String,
    /// Draw `warn` and above on mpv's OSD.
    pub osd: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file_level: "debug".to_string(),
            file: "auto".to_string(),
            file_max_files: 5,
            format: "compact".to_string(),
            ansi: String::new(),
            osd: true,
        }
    }
}

/// `[stt]`: which declared source is active, and the sources themselves.
///
/// Sources are declared as a map so a name is written exactly once
/// (`[stt.sources.groq]`), and the same protocol can be declared any number of
/// times with different servers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    /// Name of the source to use. Empty = the only declared one, which is why
    /// a single-source config needs no `source` line at all.
    pub source: String,
    /// Declared sources, keyed by name. `BTreeMap` so error messages and log
    /// lines list them in a stable order.
    pub sources: BTreeMap<String, SttSourceConfig>,
}

impl SttConfig {
    /// The name of the source the selector resolves to, or the reason it does
    /// not resolve. Kept here (not in `stt`) so both the runner and the tests
    /// see the same wording.
    pub fn active_source_name(&self) -> Result<&str, String> {
        let declared = || {
            if self.sources.is_empty() {
                "(none)".to_string()
            } else {
                self.sources.keys().cloned().collect::<Vec<_>>().join(", ")
            }
        };
        if !self.source.is_empty() {
            return self
                .sources
                .contains_key(&self.source)
                .then_some(self.source.as_str())
                .ok_or_else(|| {
                    format!(
                        "STT has no source named {:?}; declared: {}",
                        self.source,
                        declared()
                    )
                });
        }
        match self.sources.len() {
            0 => Err("STT has no source declared; add [stt.sources.<name>]".to_string()),
            1 => Ok(self.sources.keys().next().map(String::as_str).unwrap_or("")),
            n => Err(format!(
                "STT has {n} sources but no [stt] source selected; declared: {}",
                declared()
            )),
        }
    }
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            source: String::new(),
            sources: BTreeMap::new(),
        }
    }
}

/// Strategy used when an STT chunk needs an automatic retry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SttRetryStrategy {
    /// Double the delay after each failure, retaining the existing defaults.
    #[default]
    Exponential,
    /// Retry every failed chunk after the same configured interval.
    Fixed,
}

/// Automatic retry timing for one STT source. `max_retry` on the source still
/// controls short retries inside an individual HTTP request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SttRetryConfig {
    pub strategy: SttRetryStrategy,
    /// Delay for ordinary failures, and the fixed delay when strategy = fixed.
    pub interval_secs: u64,
    /// Exponential base delay for HTTP 429 responses.
    pub rate_limit_interval_secs: u64,
}

impl Default for SttRetryConfig {
    fn default() -> Self {
        Self {
            strategy: SttRetryStrategy::Exponential,
            interval_secs: 2,
            rate_limit_interval_secs: 15,
        }
    }
}

/// One declared STT source. The fields are the union of what both protocols
/// read; `protocol` decides which of them the request actually uses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SttSourceConfig {
    /// Which wire protocol this source speaks. Required: STT ships no built-in
    /// source, so nothing here can be inferred from the name alone.
    pub protocol: Option<SttProtocol>,
    /// Base URL of the transcription server, e.g. `http://127.0.0.1:8000`
    /// (OpenAI-compatible) or `http://127.0.0.1:9000` (ferrum).
    pub server_addr: Option<String>,
    /// Model id. `openai` sends it as the multipart `model` field, `ferrum` as
    /// the `x-model` header. Must be one the server offers — subtitle-gateway:
    /// "sensevoice" / "fun-asr-mlt-nano", OpenAI: "whisper-1", Groq:
    /// "whisper-large-v3" / "whisper-large-v3-turbo".
    pub model: Option<String>,
    /// Optional language hint (e.g. "ja", "zh", "en"); omitted = server
    /// auto-detects. `openai` sends it as the multipart `language` field,
    /// `ferrum` as the `x-language` header.
    pub language: Option<String>,
    /// Optional API key. `openai` sends `Authorization: Bearer {key}` for
    /// servers that require auth; omitted for a local gateway that needs none.
    pub api_key: Option<String>,
    /// Additional HTTP headers for the `openai` protocol, declared under
    /// `[stt.sources.<name>.headers]`. Custom values override generated headers
    /// with the same name. Ignored by the `ferrum` protocol.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Automatic retry timing for failed transcription chunks.
    #[serde(default)]
    pub retry: SttRetryConfig,
    pub timeout_ms: Option<u64>,
    pub max_retry: Option<usize>,
    /// `ferrum` only: Opus compression to reduce network payload size.
    pub use_opus: Option<bool>,
    /// `ferrum` only: AES-GCM encryption of the request payload.
    pub enable_encryption: Option<bool>,
    /// `ferrum` only: passphrase for `enable_encryption`.
    pub encryption_key: Option<String>,
    /// `ferrum` only: shared secret for the `x-auth-token` header.
    pub auth_secret: Option<String>,
}

impl SttSourceConfig {
    /// The fields the OpenAI protocol reads, with the shipped defaults filled
    /// in for anything the source left out.
    pub fn openai(&self) -> crate::stt::SttOpenAiConfig {
        crate::stt::SttOpenAiConfig {
            server_addr: self
                .server_addr
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:8000".to_string()),
            model: self
                .model
                .clone()
                .unwrap_or_else(|| "sensevoice".to_string()),
            language: self.language.clone(),
            api_key: self.api_key.clone(),
            headers: self.headers.clone(),
            timeout_ms: self.timeout_ms.unwrap_or(120_000),
            max_retry: self.max_retry.unwrap_or(3),
        }
    }

    /// The fields the ferrum protocol reads, with the shipped defaults filled
    /// in. A ferrum source carries the auth/encryption knobs verbatim: the
    /// protocol itself talks raw-body POST, Opus and AES-GCM, so nothing is
    /// inferred from the name.
    pub fn ferrum(&self) -> crate::stt::SttFerrumConfig {
        crate::stt::SttFerrumConfig {
            server_addr: self
                .server_addr
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:9000".to_string()),
            model: self
                .model
                .clone()
                .unwrap_or_else(|| "sensevoice".to_string()),
            language: self.language.clone(),
            timeout_ms: self.timeout_ms.unwrap_or(120_000),
            max_retry: self.max_retry.unwrap_or(3),
            use_opus: self.use_opus.unwrap_or(true),
            enable_encryption: self.enable_encryption.unwrap_or(false),
            encryption_key: self.encryption_key.clone().unwrap_or_default(),
            auth_secret: self.auth_secret.clone().unwrap_or_default(),
        }
    }
}

/// `[translate]`: which source is active, plus the sources the user declares.
/// The built-in free sources need no declaration; everything else does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateConfig {
    pub from_lang: String,
    pub to_lang: String,
    pub concurrency: usize,
    /// Name of the source to use, or `auto` to walk the built-in free sources
    /// (`google_free` → `edge_free` → `alibaba_free`) until one answers.
    pub source: String,
    /// Declared sources, keyed by name. Naming one of the built-in free
    /// sources here (e.g. `[translate.sources.edge_free]`) overrides just the
    /// fields written; the rest stay at the built-in values. `auto` is
    /// reserved and cannot be declared.
    pub sources: BTreeMap<String, TranslateSourceConfig>,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self {
            from_lang: "en".to_string(),
            to_lang: "zh".to_string(),
            concurrency: 4,
            source: "auto".to_string(),
            sources: BTreeMap::new(),
        }
    }
}

/// One declared translation source. Every field is optional so that overriding
/// a built-in free source is a one-liner: `[translate.sources.edge_free]` with
/// only `api_key` keeps the built-in host. A name outside that table has to
/// state `protocol` and `server_addr` both.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranslateSourceConfig {
    /// Wire protocol. Required unless the name is one of the built-in free
    /// ones, which already imply a protocol.
    pub protocol: Option<TranslateSourceProtocol>,
    /// Base URL, e.g. `https://api-free.deepl.com` or
    /// `http://127.0.0.1:5000`. Required unless the name is one of the
    /// built-in free ones, which already carry their endpoint.
    pub server_addr: Option<String>,
    /// Optional API key: `deepl` sends it as `Authorization: DeepL-Auth-Key`,
    /// `libretranslate` in the body, `edge` as `Ocp-Apim-Subscription-Key`.
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    pub local_ms: u64,
    pub network_ms: u64,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            local_ms: 15_000,
            network_ms: 15_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    pub ffmpeg_ms: u64,
    pub ffprobe_ms: u64,
    pub stt_ms: u64,
    pub translate_ms: u64,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            ffmpeg_ms: 30_000,
            ffprobe_ms: 10_000,
            stt_ms: 120_000,
            translate_ms: 30_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackConfig {
    pub show_progress: bool,
    pub save_srt: bool,
    pub auto_start: bool,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            show_progress: true,
            save_srt: true,
            auto_start: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchConfig {
    pub lookahead_chunks: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            lookahead_chunks: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub demuxer_max_bytes: Option<i64>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            demuxer_max_bytes: None,
        }
    }
}

/// Whether an `MPV_STT_PLUGIN_RS_*` key (prefix already stripped) names a
/// config field.
///
/// `MPV_STT_PLUGIN_RS_LOG` is the exception: it holds an `EnvFilter` directive
/// string that `logging::LogSettings` reads straight from the environment.
/// Letting the config provider see it would put `mpv_stt_plugin_rs::stt=trace`
/// into `log.level`, where it is not a level at all.
fn is_config_env_key(key: &str) -> bool {
    !key.eq_ignore_ascii_case("log")
}

impl Config {
    pub fn default_config_path() -> Option<PathBuf> {
        let base = BaseDirs::new()?;
        Some(base.config_dir().join("mpv").join("mpv_stt_plugin_rs.toml"))
    }

    pub fn config_path_from_env() -> Option<PathBuf> {
        std::env::var_os("MPV_STT_PLUGIN_RS_CONFIG").map(PathBuf::from)
    }

    pub fn load() -> Self {
        let env_path = Self::config_path_from_env();
        let config_path = env_path.clone().or_else(Self::default_config_path);

        let mut figment = Figment::from(Serialized::defaults(Config::default()));

        if let Some(path) = config_path.as_ref() {
            figment = figment.merge(Toml::file(path));
        }

        // Env overrides, with `_` mapping to a nesting level so
        // `MPV_STT_PLUGIN_RS_LOG_FILE=off` sets `log.file`. (Flat keys such as
        // `MPV_STT_PLUGIN_RS_...` for the top-level sections still resolve, since
        // Figment tries the unsplit key first.)
        //
        // `MPV_STT_PLUGIN_RS_LOG` is the log filter, not a config key: it is read
        // directly by `logging::LogSettings`, and letting the splitter see it
        // would turn the filter text into `log.level`.
        figment = figment.merge(
            Env::prefixed("MPV_STT_PLUGIN_RS_")
                // `filter` runs before `split`, so this sees the key with the
                // prefix already removed (`LOG`, not `log.level`).
                .filter(|key| is_config_env_key(key.as_str()))
                .split("_"),
        );

        match figment.extract::<Config>() {
            Ok(cfg) => cfg,
            Err(err) => {
                // Logging might not be initialized yet; fall back silently.
                warn!(error = %err, "failed to load config, using defaults");
                Config::default()
            }
        }
    }

    /// Directory `log.file = "auto"` writes into: next to the config file, so a
    /// user who overrides the config path also moves the log. `None` when no
    /// config directory is resolvable.
    pub fn log_dir(&self, config_path: Option<&PathBuf>) -> Option<PathBuf> {
        if let Some(path) = config_path {
            return path.parent().map(PathBuf::from);
        }
        Self::default_config_path().and_then(|p| p.parent().map(PathBuf::from))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MPV_STT_PLUGIN_RS_LOG` is the `EnvFilter` directive string, read by
    /// `logging::LogSettings`. It must not also land in `log.level`, or the
    /// filter text (`mpv_stt_plugin_rs::stt=trace,warn`) would be parsed as a
    /// level and the documented override would apply to nothing.
    #[test]
    fn the_log_filter_env_var_is_not_a_config_key() {
        // `Env::lowercase` means the provider hands over `LOG` for
        // `MPV_STT_PLUGIN_RS_LOG`.
        assert!(!is_config_env_key("LOG"));
        assert!(!is_config_env_key("log"));

        // Every sibling key still resolves. The splitter turns `LOG_FILE` into
        // `log.file`, which is the nesting level the `[log]` section expects.
        assert!(is_config_env_key("LOG_FILE"));
        assert!(is_config_env_key("LOG_LEVEL"));
        assert_eq!("LOG_FILE".replace('_', ".").to_lowercase(), "log.file");
    }
}
