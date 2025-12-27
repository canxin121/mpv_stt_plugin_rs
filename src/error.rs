use thiserror::Error;

#[derive(Error, Debug)]
pub enum WhisperSubsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Process execution failed: {0}")]
    ProcessFailed(String),

    #[error("Process timed out: {0}")]
    ProcessTimeout(String),

    #[error("Invalid SRT format: {0}")]
    InvalidSrt(String),

    #[error("Translation failed: {0}")]
    TranslationFailed(String),

    #[error("Audio extraction failed: {0}")]
    AudioExtractionFailed(String),

    #[error("WAV error: {0}")]
    Wav(#[from] hound::Error),

    #[error("Whisper execution failed: {0}")]
    WhisperFailed(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),
}

pub type Result<T> = std::result::Result<T, WhisperSubsError>;
