use crate::error::{Result, WhisperSubsError};
use crate::srt::{SrtFile, SubtitleEntry};
use hound::{SampleFormat, WavReader};
use log::{debug, info, trace, warn};
use srtlib::Timestamp;
use std::path::{Path, PathBuf};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
};
use crate::config::InferenceDevice;

const EXPECTED_SAMPLE_RATE: u32 = 16_000;
const EXPECTED_CHANNELS: u16 = 1;

pub struct WhisperConfig {
    pub model_path: String,
    pub threads: u8,
    pub language: String,
    pub inference_device: InferenceDevice,
    pub gpu_device: i32,
    pub flash_attn: bool,
    pub timeout_ms: u64,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            threads: 8,
            language: "auto".to_string(),
            inference_device: InferenceDevice::CPU,
            gpu_device: 0,
            flash_attn: false,
            timeout_ms: 120_000,
        }
    }
}

impl WhisperConfig {
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

    pub fn with_inference_device(mut self, device: InferenceDevice) -> Self {
        self.inference_device = device;
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

pub struct WhisperRunner {
    config: WhisperConfig,
    ctx: Option<WhisperContext>,
    active_device: Option<InferenceDevice>,
}

impl WhisperRunner {
    pub fn new(config: WhisperConfig) -> Self {
        Self {
            config,
            ctx: None,
            active_device: None,
        }
    }

    fn select_effective_device(&self) -> (InferenceDevice, String) {
        let requested_device = self.config.inference_device;
        let (effective_device, reason) = match requested_device {
            InferenceDevice::CUDA => {
                if cfg!(feature = "whisper-cuda") {
                    (
                        InferenceDevice::CUDA,
                        "requested cuda and whisper-cuda feature enabled".to_string(),
                    )
                } else {
                    warn!(
                        "inference_device=CUDA but whisper-cuda feature is disabled; falling back to CPU"
                    );
                    (
                        InferenceDevice::CPU,
                        "whisper-cuda feature disabled".to_string(),
                    )
                }
            }
            InferenceDevice::CPU => (InferenceDevice::CPU, "requested cpu".to_string()),
        };

        if effective_device != requested_device {
            warn!(
                "Whisper inference device fallback: requested={}, effective={}, reason={}",
                requested_device, effective_device, reason
            );
        }

        (effective_device, reason)
    }

    fn build_context_params_for_device(
        &self,
        effective_device: InferenceDevice,
        reason: &str,
    ) -> WhisperContextParameters<'static> {
        let mut params = WhisperContextParameters::default();
        let use_gpu = effective_device.is_gpu();

        info!(
            "Whisper inference device: {} (reason: {}, gpu_device: {})",
            effective_device, reason, self.config.gpu_device
        );

        params
            .use_gpu(use_gpu)
            .gpu_device(self.config.gpu_device)
            .flash_attn(self.config.flash_attn);

        params
    }

    fn ensure_context_for_device(
        &mut self,
        device: InferenceDevice,
        reason: &str,
    ) -> Result<()> {
        if self.ctx.is_some() && self.active_device == Some(device) {
            return Ok(());
        }

        let params = self.build_context_params_for_device(device, reason);
        let ctx = WhisperContext::new_with_params(&self.config.model_path, params)
            .map_err(|e| whisper_error("Failed to load model", e))?;

        self.ctx = Some(ctx);
        self.active_device = Some(device);
        Ok(())
    }

    fn ensure_context(&mut self) -> Result<()> {
        if self.ctx.is_some() {
            return Ok(());
        }

        if self.config.model_path.trim().is_empty() {
            return Err(WhisperSubsError::WhisperFailed(
                "Model path is empty".to_string(),
            ));
        }

        let (device, reason) = self.select_effective_device();
        if let Err(err) = self.ensure_context_for_device(device, &reason) {
            if device.is_gpu() {
                warn!(
                    "Whisper context init failed on {} ({}); falling back to CPU",
                    device, err
                );
                self.ctx = None;
                self.active_device = None;
                return self.ensure_context_for_device(
                    InferenceDevice::CPU,
                    "fallback after gpu init failure",
                );
            }
            return Err(err);
        }

        Ok(())
    }

    fn load_audio_samples<P: AsRef<Path>>(&self, audio_path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(audio_path.as_ref()).map_err(|e| {
            WhisperSubsError::WhisperFailed(format!("Failed to read WAV: {}", e))
        })?;
        let spec = reader.spec();

        if spec.channels != EXPECTED_CHANNELS
            || spec.sample_rate != EXPECTED_SAMPLE_RATE
            || spec.bits_per_sample != 16
            || spec.sample_format != SampleFormat::Int
        {
            return Err(WhisperSubsError::WhisperFailed(format!(
                "Unexpected WAV format: channels={}, sample_rate={}, bits_per_sample={}, format={:?}",
                spec.channels, spec.sample_rate, spec.bits_per_sample, spec.sample_format
            )));
        }

        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<Vec<i16>, _>>()
            .map_err(|e| {
                WhisperSubsError::WhisperFailed(format!("Failed to read WAV samples: {}", e))
            })?;

        let mut float_samples = vec![0.0f32; samples.len()];
        whisper_rs::convert_integer_to_float_audio(&samples, &mut float_samples)
            .map_err(|e| whisper_error("Failed to convert audio", e))?;

        Ok(float_samples)
    }

    fn build_params<'a>(&'a self, duration_ms: u64) -> FullParams<'a, 'a> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
        params.set_n_threads(self.config.threads as i32);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        params.set_no_timestamps(false);
        params.set_translate(false);

        if self.config.language.trim().eq_ignore_ascii_case("auto") {
            params.set_detect_language(true);
            params.set_language(None);
        } else {
            params.set_detect_language(false);
            params.set_language(Some(self.config.language.as_str()));
        }

        let duration_ms_i32 = i32::try_from(duration_ms).unwrap_or(i32::MAX);
        params.set_offset_ms(0);
        params.set_duration_ms(duration_ms_i32);

        params
    }

    fn write_srt<P: AsRef<Path>>(&self, output_prefix: P, segments: &[SegmentData]) -> Result<()> {
        let mut srt_file = SrtFile::new();
        for (idx, segment) in segments.iter().enumerate() {
            srt_file.append_entry(SubtitleEntry {
                index: (idx + 1) as u32,
                start_time: Timestamp::from_milliseconds(segment.start_ms),
                end_time: Timestamp::from_milliseconds(segment.end_ms),
                text: segment.text.clone(),
            });
        }

        let output_path = PathBuf::from(output_prefix.as_ref()).with_extension("srt");
        srt_file.save(&output_path)?;
        Ok(())
    }

    /// Run whisper on audio file and generate SRT subtitles
    pub fn transcribe<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        let audio_str = audio_path
            .as_ref()
            .to_str()
            .ok_or_else(|| WhisperSubsError::InvalidPath("Invalid audio path".to_string()))?;

        trace!(
            "Running Whisper on {} (duration: {}ms, language: {})",
            audio_str,
            duration_ms,
            self.config.language
        );

        self.ensure_context()?;

        let audio = self.load_audio_samples(&audio_path)?;
        if audio.is_empty() {
            return Err(WhisperSubsError::WhisperFailed(
                "Audio buffer is empty".to_string(),
            ));
        }

        let segments = match self.run_inference(&audio, duration_ms) {
            Ok(segments) => Ok(segments),
            Err(err) => {
                if self.active_device.map_or(false, |device| device.is_gpu()) {
                    let failed_device = self.active_device.unwrap_or(InferenceDevice::CPU);
                    warn!(
                        "Whisper inference failed on {} ({}); retrying on CPU",
                        failed_device, err
                    );
                    self.ctx = None;
                    self.active_device = None;
                    self.ensure_context_for_device(
                        InferenceDevice::CPU,
                        "fallback after gpu inference failure",
                    )?;
                    self.run_inference(&audio, duration_ms)
                } else {
                    Err(err)
                }
            }
        }?;
        self.write_srt(output_prefix, &segments)?;

        debug!("Whisper transcription completed successfully");
        Ok(())
    }
}

impl WhisperRunner {
    fn run_inference(&self, audio: &[f32], duration_ms: u64) -> Result<Vec<SegmentData>> {
        let params = self.build_params(duration_ms);

        let ctx = self.ctx.as_ref().ok_or_else(|| {
            WhisperSubsError::WhisperFailed("Whisper context not initialized".to_string())
        })?;
        let mut state = ctx
            .create_state()
            .map_err(|e| whisper_error("Failed to create state", e))?;

        state
            .full(params, audio)
            .map_err(|e| whisper_error("Whisper inference failed", e))?;

        collect_segments(&state)
    }
}

struct SegmentData {
    start_ms: u32,
    end_ms: u32,
    text: String,
}

fn collect_segments(state: &whisper_rs::WhisperState) -> Result<Vec<SegmentData>> {
    let mut segments = Vec::new();
    for segment in state.as_iter() {
        let start_ms = timestamp_to_millis(segment.start_timestamp());
        let end_ms = timestamp_to_millis(segment.end_timestamp());
        if end_ms <= start_ms {
            continue;
        }
        let text = segment
            .to_str_lossy()
            .map_err(|e| whisper_error("Failed to read segment text", e))?
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }
        segments.push(SegmentData {
            start_ms,
            end_ms,
            text,
        });
    }
    Ok(segments)
}

fn timestamp_to_millis(timestamp_cs: i64) -> u32 {
    let millis = timestamp_cs.saturating_mul(10);
    if millis < 0 {
        0
    } else if millis > u32::MAX as i64 {
        u32::MAX
    } else {
        millis as u32
    }
}

fn whisper_error(context: &str, err: WhisperError) -> WhisperSubsError {
    WhisperSubsError::WhisperFailed(format!("{}: {}", context, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_config_builder() {
        let config = WhisperConfig::new("/path/to/model".to_string())
            .with_threads(4)
            .with_language("en".to_string())
            .with_inference_device(InferenceDevice::CUDA)
            .with_gpu_device(1)
            .with_flash_attn(true);

        assert_eq!(config.model_path, "/path/to/model");
        assert_eq!(config.threads, 4);
        assert_eq!(config.language, "en");
        assert_eq!(config.inference_device, InferenceDevice::CUDA);
        assert_eq!(config.gpu_device, 1);
        assert!(config.flash_attn);
    }
}
