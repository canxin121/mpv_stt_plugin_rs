use crate::error::{Result, WhisperSubsError};
use crate::srt::{SrtFile, SubtitleEntry};
use hound::{SampleFormat, WavReader};
use log::{debug, info, trace, warn};
use srtlib::Timestamp;
#[cfg(feature = "whisper-opencl")]
use std::ffi::CStr;
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

#[derive(Debug, Clone)]
pub struct WhisperDeviceNotice {
    pub requested: InferenceDevice,
    pub effective: InferenceDevice,
    pub reason: String,
    pub gpu_device: i32,
    pub backend_info: Option<BackendDeviceInfo>,
}

#[derive(Debug, Clone)]
pub struct BackendDeviceInfo {
    pub name: String,
    pub dev_type: BackendDeviceType,
}

#[derive(Debug, Clone, Copy)]
pub enum BackendDeviceType {
    CPU,
    GPU,
    IGPU,
    ACCEL,
    Unknown(i32),
}

impl BackendDeviceType {
    fn as_str(self) -> &'static str {
        match self {
            BackendDeviceType::CPU => "CPU",
            BackendDeviceType::GPU => "GPU",
            BackendDeviceType::IGPU => "iGPU",
            BackendDeviceType::ACCEL => "ACCEL",
            BackendDeviceType::Unknown(_) => "unknown",
        }
    }
}

impl std::fmt::Display for BackendDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.name.is_empty() {
            write!(f, "{}", self.dev_type.as_str())
        } else {
            write!(f, "{}: {}", self.dev_type.as_str(), self.name)
        }
    }
}

#[cfg(feature = "whisper-opencl")]
fn query_ggml_backend_device_info(gpu_device: i32) -> Option<BackendDeviceInfo> {
    unsafe {
        let count = whisper_rs_sys::ggml_backend_dev_count();
        let mut gpu_index = 0i32;
        for idx in 0..count {
            let dev = whisper_rs_sys::ggml_backend_dev_get(idx);
            if dev.is_null() {
                continue;
            }
            let dev_type = whisper_rs_sys::ggml_backend_dev_type(dev);
            let is_gpu = dev_type == whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                || dev_type
                    == whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU;
            if !is_gpu {
                continue;
            }
            if gpu_index == gpu_device {
                let name_ptr = whisper_rs_sys::ggml_backend_dev_name(dev);
                let name = if name_ptr.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                };
                let dev_type = match dev_type {
                    whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU => {
                        BackendDeviceType::CPU
                    }
                    whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => {
                        BackendDeviceType::GPU
                    }
                    whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU => {
                        BackendDeviceType::IGPU
                    }
                    whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL => {
                        BackendDeviceType::ACCEL
                    }
                    other => BackendDeviceType::Unknown(other as i32),
                };
                return Some(BackendDeviceInfo { name, dev_type });
            }
            gpu_index += 1;
        }
    }
    None
}

#[cfg(feature = "whisper-opencl")]
mod opencl_query {
    use super::{BackendDeviceInfo, BackendDeviceType};
    use std::ffi::{c_void, CStr};
    #[allow(non_camel_case_types)]
    type cl_platform_id = *mut c_void;
    #[allow(non_camel_case_types)]
    type cl_device_id = *mut c_void;
    #[allow(non_camel_case_types)]
    type cl_uint = u32;
    #[allow(non_camel_case_types)]
    type cl_int = i32;
    #[allow(non_camel_case_types)]
    type cl_device_type = u64;

    const CL_SUCCESS: cl_int = 0;
    const CL_DEVICE_TYPE_DEFAULT: cl_device_type = 1 << 0;
    const CL_DEVICE_TYPE_CPU: cl_device_type = 1 << 1;
    const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
    const CL_DEVICE_TYPE_ACCELERATOR: cl_device_type = 1 << 3;
    const CL_DEVICE_TYPE_ALL: cl_device_type = 0xFFFF_FFFF_FFFF_FFFF;

    const CL_PLATFORM_NAME: cl_uint = 0x0902;
    const CL_DEVICE_NAME: cl_uint = 0x102B;
    const CL_DEVICE_TYPE: cl_uint = 0x1000;

    #[link(name = "OpenCL")]
    unsafe extern "C" {
        fn clGetPlatformIDs(
            num_entries: cl_uint,
            platforms: *mut cl_platform_id,
            num_platforms: *mut cl_uint,
        ) -> cl_int;
        fn clGetPlatformInfo(
            platform: cl_platform_id,
            param_name: cl_uint,
            param_value_size: usize,
            param_value: *mut c_void,
            param_value_size_ret: *mut usize,
        ) -> cl_int;
        fn clGetDeviceIDs(
            platform: cl_platform_id,
            device_type: cl_device_type,
            num_entries: cl_uint,
            devices: *mut cl_device_id,
            num_devices: *mut cl_uint,
        ) -> cl_int;
        fn clGetDeviceInfo(
            device: cl_device_id,
            param_name: cl_uint,
            param_value_size: usize,
            param_value: *mut c_void,
            param_value_size_ret: *mut usize,
        ) -> cl_int;
    }

    fn get_string(
        platform: Option<cl_platform_id>,
        device: Option<cl_device_id>,
        param_name: cl_uint,
    ) -> Option<String> {
        let mut size: usize = 0;
        let status = if let Some(p) = platform {
            unsafe { clGetPlatformInfo(p, param_name, 0, std::ptr::null_mut(), &mut size) }
        } else if let Some(d) = device {
            unsafe { clGetDeviceInfo(d, param_name, 0, std::ptr::null_mut(), &mut size) }
        } else {
            return None;
        };
        if status != CL_SUCCESS || size == 0 {
            return None;
        }
        let mut buf = vec![0u8; size];
        let status = if let Some(p) = platform {
            unsafe {
                clGetPlatformInfo(
                    p,
                    param_name,
                    size,
                    buf.as_mut_ptr().cast(),
                    std::ptr::null_mut(),
                )
            }
        } else if let Some(d) = device {
            unsafe {
                clGetDeviceInfo(
                    d,
                    param_name,
                    size,
                    buf.as_mut_ptr().cast(),
                    std::ptr::null_mut(),
                )
            }
        } else {
            return None;
        };
        if status != CL_SUCCESS {
            return None;
        }
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr().cast()) };
        Some(cstr.to_string_lossy().into_owned())
    }

    fn get_device_type(device: cl_device_id) -> Option<cl_device_type> {
        let mut dev_type: cl_device_type = 0;
        let status = unsafe {
            clGetDeviceInfo(
                device,
                CL_DEVICE_TYPE,
                std::mem::size_of::<cl_device_type>(),
                (&mut dev_type as *mut cl_device_type).cast(),
                std::ptr::null_mut(),
            )
        };
        if status == CL_SUCCESS {
            Some(dev_type)
        } else {
            None
        }
    }

    pub fn query(gpu_device: i32) -> Option<BackendDeviceInfo> {
        let mut n_platforms: cl_uint = 0;
        if unsafe { clGetPlatformIDs(0, std::ptr::null_mut(), &mut n_platforms) } != CL_SUCCESS {
            return None;
        }
        if n_platforms == 0 {
            return None;
        }
        let mut platforms = vec![std::ptr::null_mut(); n_platforms as usize];
        if unsafe { clGetPlatformIDs(n_platforms, platforms.as_mut_ptr(), std::ptr::null_mut()) }
            != CL_SUCCESS
        {
            return None;
        }

            let mut selected_platform: cl_platform_id = std::ptr::null_mut();
            let mut selected_devices: Vec<cl_device_id> = Vec::new();

        for platform in &platforms {
            let mut n_devices: cl_uint = 0;
            let status = unsafe {
                clGetDeviceIDs(
                    *platform,
                    CL_DEVICE_TYPE_ALL,
                    0,
                    std::ptr::null_mut(),
                    &mut n_devices,
                )
            };
            if status != CL_SUCCESS || n_devices == 0 {
                continue;
            }
            let mut devices = vec![std::ptr::null_mut(); n_devices as usize];
            if unsafe {
                clGetDeviceIDs(
                    *platform,
                    CL_DEVICE_TYPE_ALL,
                    n_devices,
                    devices.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            } != CL_SUCCESS
            {
                continue;
            }

            let mut has_gpu = false;
            for dev in &devices {
                if let Some(dev_type) = get_device_type(*dev) {
                    if (dev_type & CL_DEVICE_TYPE_GPU) != 0 {
                        has_gpu = true;
                        break;
                    }
                }
            }
            if selected_platform.is_null() || has_gpu {
                selected_platform = *platform;
                selected_devices = devices;
                if has_gpu {
                    break;
                }
            }
        }

        if selected_platform.is_null() || selected_devices.is_empty() {
            return None;
        }

        let mut default_index: usize = 0;
        for (idx, dev) in selected_devices.iter().enumerate() {
            if let Some(dev_type) = get_device_type(*dev) {
                if (dev_type & CL_DEVICE_TYPE_GPU) != 0 {
                    default_index = idx;
                    break;
                }
            }
        }

        if default_index != 0 {
            selected_devices.swap(0, default_index);
        }

        let idx = if gpu_device >= 0 {
            gpu_device as usize
        } else {
            0
        };
        let chosen = *selected_devices.get(idx).unwrap_or(&selected_devices[0]);
        let name = get_string(None, Some(chosen), CL_DEVICE_NAME).unwrap_or_default();
        let dev_type = match get_device_type(chosen).unwrap_or(CL_DEVICE_TYPE_DEFAULT) {
            t if (t & CL_DEVICE_TYPE_GPU) != 0 => BackendDeviceType::GPU,
            t if (t & CL_DEVICE_TYPE_CPU) != 0 => BackendDeviceType::CPU,
            t if (t & CL_DEVICE_TYPE_ACCELERATOR) != 0 => BackendDeviceType::ACCEL,
            _ => BackendDeviceType::Unknown(0),
        };

        let _ = get_string(Some(selected_platform), None, CL_PLATFORM_NAME);

        Some(BackendDeviceInfo { name, dev_type })
    }
}

#[cfg(feature = "whisper-opencl")]
fn query_opencl_backend_device_info(gpu_device: i32) -> Option<BackendDeviceInfo> {
    query_ggml_backend_device_info(gpu_device).or_else(|| opencl_query::query(gpu_device))
}

#[cfg(not(feature = "whisper-opencl"))]
fn query_opencl_backend_device_info(_gpu_device: i32) -> Option<BackendDeviceInfo> {
    None
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
    pending_device_notice: Option<WhisperDeviceNotice>,
}

impl WhisperRunner {
    pub fn new(config: WhisperConfig) -> Self {
        Self {
            config,
            ctx: None,
            active_device: None,
            pending_device_notice: None,
        }
    }

    pub fn take_device_notice(&mut self) -> Option<WhisperDeviceNotice> {
        self.pending_device_notice.take()
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
            InferenceDevice::OPENCL => {
                if cfg!(feature = "whisper-opencl") {
                    (
                        InferenceDevice::OPENCL,
                        "requested opencl and whisper-opencl feature enabled".to_string(),
                    )
                } else {
                    warn!(
                        "inference_device=OPENCL but whisper-opencl feature is disabled; falling back to CPU"
                    );
                    (
                        InferenceDevice::CPU,
                        "whisper-opencl feature disabled".to_string(),
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
        let backend_info = if device == InferenceDevice::OPENCL {
            query_opencl_backend_device_info(self.config.gpu_device)
        } else {
            None
        };
        self.pending_device_notice = Some(WhisperDeviceNotice {
            requested: self.config.inference_device,
            effective: device,
            reason: reason.to_string(),
            gpu_device: self.config.gpu_device,
            backend_info,
        });
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
        self.ensure_backend_info();
        self.write_srt(output_prefix, &segments)?;

        debug!("Whisper transcription completed successfully");
        Ok(())
    }
}

impl WhisperRunner {
    fn ensure_backend_info(&mut self) {
        if self.active_device != Some(InferenceDevice::OPENCL) {
            return;
        }
        if let Some(notice) = self.pending_device_notice.as_mut() {
            if notice.backend_info.is_none() {
                notice.backend_info = query_opencl_backend_device_info(self.config.gpu_device);
            }
        }
    }

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
