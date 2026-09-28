pub mod audio;
pub mod common;
pub mod config;
pub mod crypto;
pub mod ffi;
pub mod logging;
pub mod plugin;
pub mod process;
pub mod srt;
pub mod stt;
pub mod subtitle_manager;
pub mod translate;

pub use crate::common::{MpvSttError, Result};
pub use crate::crypto::{AuthToken, EncryptionKey};
pub use crate::logging::{LogFormat, LogSettings, install_panic_hook};
pub use crate::srt::{SrtFile, SubtitleEntry};
pub use audio::AudioExtractor;
pub use config::{
    BackendKind, Config, InferenceDevice, LogConfig, TranslateAlibabaConfig, TranslateBackendKind,
    TranslateEdgeConfig, TranslateGoogleConfig, TranslateLibreTranslateConfig,
};
#[cfg(feature = "stt_ferrum")]
pub use stt::SttFerrumConfig;
#[cfg(feature = "stt_openai")]
pub use stt::SttOpenAiConfig;
pub use stt::{SttBackend, SttRunner};
pub use subtitle_manager::SubtitleManager;
pub use translate::{Translator, TranslatorConfig};
