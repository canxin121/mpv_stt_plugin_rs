use thiserror::Error;

/// Everything that can go wrong in the plugin.
///
/// Variants keep the underlying error as `#[source]` where one exists, so
/// `logging::err_chain` can still name the root cause (a DNS failure, a TLS
/// error, a malformed response) after the error has travelled up several
/// layers. Variants carry their facts in separate fields — the HTTP status, the
/// endpoint, the body — so a log line can filter on them instead of only
/// printing a sentence.
#[derive(Error, Debug)]
pub enum MpvSttError {
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

    #[error("Translation request to {url} failed")]
    TranslationRequest {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    /// The server answered, but not with a success status.
    #[error("{server} returned HTTP {status}: {body}")]
    HttpStatus {
        /// Human-readable identification of what was called, e.g.
        /// `POST /v1/audio/transcriptions`.
        server: String,
        status: u16,
        body: String,
    },

    /// A response body that was expected to be JSON (or a known shape) was not.
    #[error("Malformed response from {server}: {context}")]
    MalformedResponse {
        server: String,
        context: String,
    },

    #[error("Audio extraction failed: {0}")]
    AudioExtractionFailed(String),

    #[error("Audio extraction cancelled")]
    AudioExtractionCancelled,

    #[error("WAV error: {0}")]
    Wav(String),

    #[error("STT execution failed: {0}")]
    SttFailed(String),

    #[error("STT execution cancelled")]
    SttCancelled,

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Encryption/Decryption error: {0}")]
    CryptoError(String),
}

pub type Result<T> = std::result::Result<T, MpvSttError>;

impl From<hound::Error> for MpvSttError {
    fn from(err: hound::Error) -> Self {
        MpvSttError::Wav(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::err_chain;

    #[test]
    fn http_status_names_both_the_caller_and_the_server() {
        let err = MpvSttError::HttpStatus {
            server: "POST /v1/audio/transcriptions".to_string(),
            status: 404,
            body: r#"{"error":{"code":"model_not_found"}}"#.to_string(),
        };
        let text = err.to_string();
        assert!(text.contains("404"), "{text}");
        assert!(text.contains("model_not_found"), "{text}");
        assert!(text.contains("/v1/audio/transcriptions"), "{text}");
    }

    #[test]
    fn request_errors_expose_the_transport_cause() {
        // `reqwest::Error` cannot be constructed directly, so drive the chain
        // through a deliberately unroutable endpoint and unwrap the source.
        let err = reqwest::blocking::Client::new()
            .get("http://127.0.0.1:1/unreachable")
            .send()
            .unwrap_err();
        let wrapped = MpvSttError::TranslationRequest {
            url: "http://127.0.0.1:1/v1/translate".to_string(),
            source: err,
        };

        let chain = err_chain(&wrapped);
        assert!(chain.contains("127.0.0.1:1/v1/translate"), "{chain}");
        // The cause is what makes this error useful; it must survive wrapping.
        assert!(chain.contains("<-"), "source chain was lost: {chain}");
    }
}
