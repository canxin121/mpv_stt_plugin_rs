use crate::common::{MpvSttError, Result};
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

    /// Transcribe only while the generation captured when the job was queued is
    /// still current. Backends with cancellable preflight work should override
    /// this so a cancellation between dequeue and backend entry is preserved.
    fn transcribe_with_generation<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
        expected_generation: u64,
    ) -> Result<()> {
        if self
            .cancellation_generation()
            .load(std::sync::atomic::Ordering::Acquire)
            != expected_generation
        {
            return Err(MpvSttError::SttCancelled);
        }
        self.transcribe(audio_path, output_prefix, duration_ms)
    }

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

// Backend modules. Remote protocols are compiled in according to Cargo
// features; the active source's `protocol` selects the implementation.
#[cfg(feature = "stt_ferrum")]
mod ferrum;

#[cfg(feature = "stt_openai")]
mod openai;

#[cfg(feature = "stt_cloudflare")]
mod cloudflare;

#[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
mod segments;

// Config exports
#[cfg(feature = "stt_ferrum")]
pub use ferrum::SttFerrumConfig;

#[cfg(feature = "stt_openai")]
pub use openai::SttOpenAiConfig;

#[cfg(feature = "stt_cloudflare")]
pub use cloudflare::SttCloudflareConfig;

/// The STT backend of the selected source. Enabled remote protocols are
/// compiled in; which one runs is decided from `[stt] source` at startup.
///
/// The ferrum backend is boxed: it carries the whole resolved config inline
/// (~1.2 KB, against ~120 bytes for the OpenAI one), and one `SttRunner` lives
/// for the lifetime of a session, so the indirection costs nothing and keeps
/// the enum from sizing every move to its largest arm.
pub enum SttRunner {
    #[cfg(feature = "stt_ferrum")]
    Ferrum(Box<ferrum::FerrumBackend>),
    #[cfg(feature = "stt_openai")]
    OpenAi(openai::OpenAiBackend),
    #[cfg(feature = "stt_cloudflare")]
    Cloudflare(cloudflare::CloudflareBackend),
}

impl SttRunner {
    /// Build the backend of the selected source. `cfg.source` names it; an
    /// empty `source` means the only declared one. A selector that resolves to
    /// nothing is an error rather than a fallback: silently transcribing with
    /// an unintended source is worse than not starting.
    pub fn from_config(cfg: &SttConfig) -> Result<Self> {
        let name = cfg.active_source_name().map_err(|e| {
            crate::common::MpvSttError::SttFailed(format!(
                "{e}; protocols: openai, ferrum, cloudflare"
            ))
        })?;
        let source = cfg
            .sources
            .get(name)
            .expect("the resolved name is always a declared source");
        let protocol = source.protocol.ok_or_else(|| {
            crate::common::MpvSttError::SttFailed(format!(
                "STT source {name:?} has no protocol; set protocol = \"openai\", \"ferrum\", or \"cloudflare\""
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
                    Ok(SttRunner::Ferrum(Box::new(ferrum::FerrumBackend::new(
                        ferrum_cfg,
                    )?)))
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
                        authenticated = openai_cfg.api_key.is_some()
                            || openai_cfg.headers.keys().any(|name| {
                                name.eq_ignore_ascii_case("authorization")
                                    || name.eq_ignore_ascii_case("x-api-key")
                                    || name.eq_ignore_ascii_case("api-key")
                                    || name.eq_ignore_ascii_case("x-portkey-api-key")
                            }),
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
            SttProtocol::Cloudflare => {
                #[cfg(feature = "stt_cloudflare")]
                {
                    let cloudflare_cfg = source.cloudflare();
                    debug!(
                        source = %name,
                        protocol = %protocol,
                        model = %cloudflare_cfg.model,
                        language = cloudflare_cfg.language.as_deref().unwrap_or("auto"),
                        authenticated = cloudflare_cfg.api_key.is_some(),
                        "selected the STT source"
                    );
                    Ok(SttRunner::Cloudflare(cloudflare::CloudflareBackend::new(
                        cloudflare_cfg,
                    )?))
                }
                #[cfg(not(feature = "stt_cloudflare"))]
                {
                    let _ = source;
                    Err(crate::common::MpvSttError::SttFailed(
                        "stt_cloudflare feature not enabled".to_string(),
                    ))
                }
            }
        }
    }

    #[cfg(all(test, feature = "stt_cloudflare"))]
    pub(crate) fn cloudflare_for_test(config: SttCloudflareConfig, api_root: &str) -> Result<Self> {
        Ok(Self::Cloudflare(
            cloudflare::CloudflareBackend::build_for_test(config, api_root)?,
        ))
    }
}

impl SttBackend for SttRunner {
    fn protocol(&self) -> SttProtocol {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.protocol(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.protocol(),
            #[cfg(feature = "stt_cloudflare")]
            SttRunner::Cloudflare(b) => b.protocol(),
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
            #[cfg(feature = "stt_cloudflare")]
            SttRunner::Cloudflare(b) => b.transcribe(audio_path, output_prefix, duration_ms),
        }
    }

    fn transcribe_with_generation<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
        expected_generation: u64,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.transcribe_with_generation(
                audio_path,
                output_prefix,
                duration_ms,
                expected_generation,
            ),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.transcribe_with_generation(
                audio_path,
                output_prefix,
                duration_ms,
                expected_generation,
            ),
            #[cfg(feature = "stt_cloudflare")]
            SttRunner::Cloudflare(b) => b.transcribe_with_generation(
                audio_path,
                output_prefix,
                duration_ms,
                expected_generation,
            ),
        }
    }

    fn cancel_inflight(&self) {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.cancel_inflight(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.cancel_inflight(),
            #[cfg(feature = "stt_cloudflare")]
            SttRunner::Cloudflare(b) => b.cancel_inflight(),
        }
    }

    fn cancellation_generation(&self) -> Arc<AtomicU64> {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.cancellation_generation(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.cancellation_generation(),
            #[cfg(feature = "stt_cloudflare")]
            SttRunner::Cloudflare(b) => b.cancellation_generation(),
        }
    }

    fn take_device_notice(&mut self) -> Option<SttDeviceNotice> {
        match self {
            #[cfg(feature = "stt_ferrum")]
            SttRunner::Ferrum(b) => b.take_device_notice(),
            #[cfg(feature = "stt_openai")]
            SttRunner::OpenAi(b) => b.take_device_notice(),
            #[cfg(feature = "stt_cloudflare")]
            SttRunner::Cloudflare(b) => b.take_device_notice(),
        }
    }
}
