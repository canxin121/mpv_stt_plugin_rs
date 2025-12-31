pub mod audio;
pub mod config;
pub mod crypto;
pub mod error;
pub mod ffi;
pub mod plugin;
pub mod process;
pub mod srt;
pub mod stt;
pub mod subtitle_manager;
pub mod translate;

// Re-export main types for convenience
pub use audio::AudioExtractor;
pub use config::{Config, InferenceDevice};
pub use crypto::{AuthToken, EncryptionKey};
pub use error::{MpvSttPluginRsError, Result};
pub use srt::{SrtFile, SubtitleEntry};
#[cfg(any(feature = "stt_local_cpu", feature = "stt_local_cuda"))]
pub use stt::LocalModelConfig;
#[cfg(feature = "stt_remote_udp")]
pub use stt::RemoteSttConfig;
pub use stt::{ActiveBackend as SttActiveBackend, SttRunner};
pub use subtitle_manager::SubtitleManager;
pub use translate::{Translator, TranslatorConfig};
