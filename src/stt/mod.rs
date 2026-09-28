use crate::common::Result;
use crate::config::{SttConfig, SttProtocol};
use std::path::Path;
use std::sync::{Arc, atomic::AtomicU64};
use tracing::debug;

/// Common trait for all speech-to-text backends.
pub trait SttBackend: Send {
    fn protocol(&self) -> SttProtocol;

    fn transcribe<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()>;

    /// Request cancellation of in-flight work.
    fn cancel_inflight(&self);

    /// Shared generation used by an external event loop to cancel a backend
    /// while `transcribe` is running on a worker thread.
    fn cancellation_generation(&self) -> Arc<AtomicU64>;

    /// Optional notice about the effective device used (for UI).
    fn take_device_notice(&mut self) -> Option<SttDeviceNotice>;
}

#[derive(Debug, Clone)]
pub struct SttDeviceNotice {
    pub requested: crate::config::InferenceDevice,
    pub effective: crate::config::InferenceDevice,
    pub reason: String,
    pub gpu_device: i32,
}

// Backend modules. Both remote protocols are compiled in (the plugin is a pure
// remote client); which one a request uses is decided by the active source's
// `protocol`.
#[cfg(feature = "stt_ferrum")]
mod ferrum;

#[cfg(feature = "stt_openai")]
mod openai;

// Config exports
#[cfg(feature = "stt_ferrum")]
pub use ferrum::SttFerrumConfig;

#[cfg(feature = "stt_openai")]
pub use openai::SttOpenAiConfig;

/// The STT backend of the selected source. Both remote protocols are compiled
/// in; which one runs is decided from `[stt] source` at startup.
pub enum SttRunner {
    #[cfg(feature = "stt_ferrum")]
    Ferrum(ferrum::FerrumBackend),
    #[cfg(feature = "stt_openai")]
    OpenAi(openai::OpenAiBackend),
}

impl SttRunner {
    /// Build the backend of the selected source. `cfg.source` names it; an
    /// empty `source` means the only declared one. A selector that resolves to
    /// nothing is an error rather than a fallback: silently transcribing with
    /// an unintended source is worse than not starting.
    pub fn from_config(cfg: &SttConfig) -> Result<Self> {
        let name = cfg.active_source_name().map_err(|e| {
            crate::common::MpvSttError::SttFailed(format!("{e}; protocols: openai, ferrum"))
        })?;
        let source = cfg
            .sources
            .get(name)
            .expect("the resolved name is always a declared source");
        let protocol = source.protocol.ok_or_else(|| {
            crate::common::MpvSttError::SttFailed(format!(
                "STT source {name:?} has no protocol; set protocol = \"openai\" or \"ferrum\""
            ))
        })?;

        match protocol {
            SttProtocol::Ferrum => {
                #[cfg(feature = "stt_ferrum")]
                {
                    let ferrum_cfg = source.ferrum();
                    debug!(
                        source = %name,
                        protocol = %protocol,
                        server = %ferrum_cfg.server_addr,
                        model = %ferrum_cfg.model,
                        language = ferrum_cfg.language.as_deref().unwrap_or("auto"),
                        opus = ferrum_cfg.use_opus,
                        encrypted = ferrum_cfg.enable_encryption,
                        "selected the STT source"
                    );
                    Ok(SttRunner::Ferrum(ferrum::FerrumBackend::new(ferrum_cfg)?))
                }
                #[cfg(not(feature = "stt_ferrum"))]
                {
                    let _ = source;
                    Err(crate::common::MpvSttError::SttFailed(
                        "stt_ferrum feature not enabled".to_string(),
                    ))
                }
            }
            SttProtocol::OpenAi => {
                #[cfg(feature = "stt_openai")]
                {
                    let openai_cfg = source.openai();
                    debug!(
                        source = %name,
                        protocol = %protocol,
                        server = %openai_cfg.server_addr,
                        model = %openai_cfg.model,
                        language = openai_cfg.language.as_deref().unwrap_or("auto"),
                        authenticated = openai_cfg.api_key.is_some(),
                        "selected the STT source"
                    );
                    Ok(SttRunner::OpenAi(openai::OpenAiBackend::new(openai_cfg)?))
                }
                #[cfg(not(feature = "stt_openai"))]
                {
                    let _ = source;
                    Err(crate::common::MpvSttError::SttFailed(
                        "stt_openai feature not enabled".to_string(),
                    ))
                }
            }
        }
    }
}

impl SttBackend for SttRunner {
    fn protocol(&self) -> SttProtocol {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.protocol(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.protocol(),
        }
    }

    fn transcribe<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.transcribe(audio_path, output_prefix, duration_ms),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.transcribe(audio_path, output_prefix, duration_ms),
        }
    }

    fn cancel_inflight(&self) {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.cancel_inflight(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.cancel_inflight(),
        }
    }

    fn cancellation_generation(&self) -> Arc<AtomicU64> {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.cancellation_generation(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.cancellation_generation(),
        }
    }

    fn take_device_notice(&mut self) -> Option<SttDeviceNotice> {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.take_device_notice(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.take_device_notice(),
        }
    }
}
