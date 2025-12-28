pub mod audio;
pub mod config;
pub mod error;
pub mod ffi;
pub mod plugin;
pub mod process;
pub mod srt;
pub mod subtitle_manager;
pub mod translate;
pub mod whisper;

// Re-export main types for convenience
pub use audio::AudioExtractor;
pub use config::{Config, InferenceDevice};
pub use error::{Result, WhisperSubsError};
pub use srt::{SrtFile, SubtitleEntry};
pub use subtitle_manager::SubtitleManager;
pub use translate::{Translator, TranslatorConfig};
pub use whisper::{WhisperConfig, WhisperRunner};
