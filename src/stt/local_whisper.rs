use super::{BackendKind, SttBackend, SttDeviceNotice};
use crate::config::InferenceDevice;
use crate::error::{Result, WhisperSubsError};
use crate::srt::{SrtFile, SubtitleEntry};
use hound::{SampleFormat, WavReader};
use log::{debug, info, trace};
use srtlib::Timestamp;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
};

const EXPECTED_SAMPLE_RATE: u32 = 16_000;
const EXPECTED_CHANNELS: u16 = 1;

#[cfg(feature = "stt_local_cuda")]
const FEATURE_DEVICE: InferenceDevice = InferenceDevice::CUDA;
#[cfg(feature = "stt_local_cpu")]
const FEATURE_DEVICE: InferenceDevice = InferenceDevice::CPU;

pub struct LocalModelConfig {
    pub model_path: String,
    pub threads: u8,
    pub language: String,
    pub gpu_device: i32,
    pub flash_attn: bool,
    pub timeout_ms: u64,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            threads: 8,
            language: "auto".to_string(),
            gpu_device: 0,
            flash_attn: false,
            timeout_ms: 120_000,
        }
    }
}

impl LocalModelConfig {
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            ..Default::default()
        }
    }

    pub fn with_threads(mut self, threads: u8) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_language(mut self, language: String) -> Self {
        self.language = language;
        self
    }

    pub fn with_gpu_device(mut self, device: i32) -> Self {
        self.gpu_device = device;
        self
    }

    pub fn with_flash_attn(mut self, enabled: bool) -> Self {
        self.flash_attn = enabled;
        self
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
}

pub struct LocalWhisperBackend {
    config: LocalModelConfig,
    ctx: Option<WhisperContext>,
    active_device: Option<InferenceDevice>,
    pending_device_notice: Option<SttDeviceNotice>,
    cancel_generation: Arc<AtomicU64>,
}

impl SttBackend for LocalWhisperBackend {
    fn kind(&self) -> BackendKind {
        #[cfg(feature = "stt_local_cuda")]
        return BackendKind::LocalModelCuda;

        #[cfg(feature = "stt_local_cpu")]
        return BackendKind::LocalModelCpu;
    }

    fn transcribe<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        self.transcribe_impl(audio_path, output_prefix, duration_ms)
    }

    fn cancel_inflight(&self) {
        self.cancel_generation.fetch_add(1, Ordering::Relaxed);
    }

    fn take_device_notice(&mut self) -> Option<SttDeviceNotice> {
        self.pending_device_notice.take()
    }
}

impl LocalWhisperBackend {
    pub fn new(config: LocalModelConfig) -> Self {
        Self {
            config,
            ctx: None,
            active_device: None,
            pending_device_notice: None,
            cancel_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    fn build_context_params_for_device(
        &self,
        effective_device: InferenceDevice,
        reason: &str,
    ) -> WhisperContextParameters<'static> {
        let mut params = WhisperContextParameters::default();
        let use_gpu = effective_device.is_gpu();

        info!(
            "STT inference device: {} (reason: {}, gpu_device: {})",
            effective_device, reason, self.config.gpu_device
        );

        params
            .use_gpu(use_gpu)
            .gpu_device(self.config.gpu_device)
            .flash_attn(self.config.flash_attn);

        params
    }

    fn ensure_context_for_device(&mut self, device: InferenceDevice, reason: &str) -> Result<()> {
        if self.ctx.is_some() && self.active_device == Some(device) {
            return Ok(());
        }

        let params = self.build_context_params_for_device(device, reason);
        let ctx = WhisperContext::new_with_params(&self.config.model_path, params)
            .map_err(|e| stt_error("Failed to load model", e))?;

        self.ctx = Some(ctx);
        self.active_device = Some(device);
        self.pending_device_notice = Some(SttDeviceNotice {
            requested: FEATURE_DEVICE,
            effective: device,
            reason: reason.to_string(),
            gpu_device: self.config.gpu_device,
        });
        Ok(())
    }

    fn ensure_context(&mut self) -> Result<()> {
        if self.ctx.is_some() {
            return Ok(());
        }

        if self.config.model_path.trim().is_empty() {
            return Err(WhisperSubsError::SttFailed(
                "Model path is empty".to_string(),
            ));
        }

        self.ensure_context_for_device(FEATURE_DEVICE, "feature-selected local_model")?;

        Ok(())
    }

    fn load_audio_samples<P: AsRef<Path>>(&self, audio_path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(audio_path.as_ref())
            .map_err(|e| WhisperSubsError::SttFailed(format!("Failed to read WAV: {}", e)))?;
        let spec = reader.spec();

        if spec.channels != EXPECTED_CHANNELS
            || spec.sample_rate != EXPECTED_SAMPLE_RATE
            || spec.bits_per_sample != 16
            || spec.sample_format != SampleFormat::Int
        {
            return Err(WhisperSubsError::SttFailed(format!(
                "Unexpected WAV format: channels={}, sample_rate={}, bits_per_sample={}, format={:?}",
                spec.channels, spec.sample_rate, spec.bits_per_sample, spec.sample_format
            )));
        }

        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<Vec<i16>, _>>()
            .map_err(|e| {
                WhisperSubsError::SttFailed(format!("Failed to read WAV samples: {}", e))
            })?;

        let mut float_samples = vec![0f32; samples.len()];
        whisper_rs::convert_integer_to_float_audio(&samples, &mut float_samples)
            .map_err(|e| stt_error("Failed to convert audio", e))?;

        Ok(float_samples)
    }

    fn build_params(&self) -> Result<FullParams<'static>> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(&self.config.language);
        params.set_n_threads(self.config.threads.into());
        params.set_suppress_blank(true);
        params.set_max_len(0);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_duration_ms(self.config.timeout_ms as i32);
        Ok(params)
    }

    fn transcribe_impl<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        self.ensure_context()?;

        let audio_str = audio_path
            .as_ref()
            .to_str()
            .ok_or_else(|| WhisperSubsError::InvalidPath("Invalid audio path".to_string()))?;

        trace!(
            "Running local model STT on {} (duration: {}ms, language: {})",
            audio_str, duration_ms, self.config.language
        );

        let samples = self.load_audio_samples(audio_path)?;

        let params = self.build_params()?;
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            WhisperSubsError::SttFailed("STT context not initialized".to_string())
        })?;

        // Abort handling: stash current generation.
        let run_generation = self.cancel_generation.load(Ordering::Relaxed);
        let abort_ctx = AbortCtx {
            gen_ptr: &self.cancel_generation,
            expected: run_generation,
        };
        let abort_ptr = &abort_ctx as *const _ as *mut c_void;
        let timeout_ms = self.config.timeout_ms;

        params.set_abort_callback(Some(stt_abort_callback));
        params.set_abort_callback_user_data(abort_ptr);
        params.set_duration_ms(timeout_ms as i32);
        params.set_translate(false);

        let mut state = ctx
            .create_state()
            .map_err(|e| stt_error("Failed to create state", e))?;

        state.full(params, &samples).map_err(|err| match err {
            WhisperError::Abort => WhisperSubsError::SttCancelled,
            _ => stt_error("STT inference failed", err),
        })?;

        if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
            return Err(WhisperSubsError::SttCancelled);
        }

        let segments = collect_segments(&state)?;
        let mut srt_file = SrtFile::new();
        for (idx, seg) in segments.into_iter().enumerate() {
            srt_file.append_entry(SubtitleEntry {
                index: idx as u32 + 1,
                start_time: Timestamp::from_milliseconds(seg.start_ms as i64),
                end_time: Timestamp::from_milliseconds(seg.end_ms as i64),
                text: seg.text,
            });
        }

        let output_path = PathBuf::from(output_prefix.as_ref()).with_extension("srt");
        srt_file.save(&output_path)?;

        debug!("Local model STT completed successfully");
        Ok(())
    }
}

#[derive(Debug)]
struct SegmentData {
    start_ms: f64,
    end_ms: f64,
    text: String,
}

#[repr(C)]
struct AbortCtx {
    gen_ptr: *const AtomicU64,
    expected: u64,
}

unsafe extern "C" fn stt_abort_callback(user_data: *mut c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    let ctx = &*(user_data as *const AbortCtx);
    let gen_ptr = ctx.gen_ptr;
    if gen_ptr.is_null() {
        return false;
    }
    let current = (*gen_ptr).load(Ordering::Relaxed);
    current != ctx.expected
}

fn collect_segments(state: &whisper_rs::WhisperState) -> Result<Vec<SegmentData>> {
    let num_segments = state.full_n_segments().unwrap_or(0);
    let mut segments = Vec::with_capacity(num_segments as usize);

    for i in 0..num_segments {
        let start = state.full_get_segment_t0(i).unwrap_or(0) as f64;
        let end = state.full_get_segment_t1(i).unwrap_or(0) as f64;
        let text = state
            .full_get_segment_text(i)
            .map_err(|e| stt_error("Failed to read segment text", e))?
            .trim()
            .to_string();

        segments.push(SegmentData {
            start_ms: start,
            end_ms: end,
            text,
        });
    }

    Ok(segments)
}

fn stt_error(context: &str, err: WhisperError) -> WhisperSubsError {
    WhisperSubsError::SttFailed(format!("{}: {}", context, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = LocalModelConfig::new("/path/to/model".to_string())
            .with_threads(4)
            .with_language("ja".to_string())
            .with_gpu_device(1)
            .with_flash_attn(true)
            .with_timeout_ms(42_000);

        assert_eq!(config.model_path, "/path/to/model");
        assert_eq!(config.threads, 4);
        assert_eq!(config.language, "ja");
        assert_eq!(config.gpu_device, 1);
        assert!(config.flash_attn);
        assert_eq!(config.timeout_ms, 42_000);
    }
}
