use super::{SttBackend, SttDeviceNotice};
use crate::common::{MpvSttError, Result};
use crate::config::SttProtocol;
use crate::crypto::{AuthToken, EncryptionKey};
use crate::srt::SrtFile;
use libc;
use opusic_sys as opus;
use reqwest::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, trace, warn};

/// The ferrum protocol's own fields, as resolved from a named source.
#[derive(Debug, Clone, Default)]
pub struct SttFerrumConfig {
    pub server_addr: String,
    /// Model id sent via the `x-model` header, e.g. "sensevoice" or "fun-asr-mlt-nano".
    pub model: String,
    /// Optional language hint sent via the `x-language` header (e.g. "ja",
    /// "zh", "en"); `None` = server auto-detects.
    pub language: Option<String>,
    pub timeout_ms: u64,
    pub max_retry: usize,
    /// Enable Opus compression to reduce network payload size.
    pub use_opus: bool,
    pub enable_encryption: bool,
    pub encryption_key: String,
    pub auth_secret: String,
}

const HEADER_REQUEST_ID: &str = "x-request-id";
const HEADER_DURATION_MS: &str = "x-duration-ms";
const HEADER_AUTH_TOKEN: &str = "x-auth-token";
const HEADER_COMPRESSION: &str = "x-compression";
const HEADER_ENCRYPTED: &str = "x-encrypted";
const HEADER_MODEL: &str = "x-model";
const HEADER_LANGUAGE: &str = "x-language";
const HEADER_QUEUE_MS: &str = "x-metric-queue-ms";
const HEADER_INFER_MS: &str = "x-metric-infer-ms";
const HEADER_WORKER_MS: &str = "x-metric-worker-ms";
const HEADER_BYTES_IN: &str = "x-bytes-in";
const HEADER_BYTES_OUT: &str = "x-bytes-out";

// HTTP payloads are raw 16 kHz mono PCM WAV bytes; advertise them truthfully.
const COMPRESSION_PCM: &str = "pcm";
const COMPRESSION_OPUS: &str = "opus";

pub struct FerrumBackend {
    config: SttFerrumConfig,
    server_url: String,
    cancel_generation: Arc<AtomicU64>,
    encryption_key: Option<EncryptionKey>,
    auth_token: AuthToken,
    client: Client,
}

impl FerrumBackend {
    pub fn new(config: SttFerrumConfig) -> Result<Self> {
        let encryption_key = if config.enable_encryption {
            if config.encryption_key.is_empty() {
                return Err(MpvSttError::SttFailed(
                    "Encryption enabled but encryption_key is empty".to_string(),
                ));
            }
            Some(EncryptionKey::from_passphrase(&config.encryption_key))
        } else {
            None
        };

        let auth_token = if !config.auth_secret.is_empty() {
            AuthToken::from_secret(&config.auth_secret)
        } else {
            AuthToken::from_secret("")
        };

        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|e| MpvSttError::SttFailed(format!("HTTP client build failed: {}", e)))?;

        let server_url = normalize_server_url(&config.server_addr);

        Ok(Self {
            config,
            server_url,
            cancel_generation: Arc::new(AtomicU64::new(0)),
            encryption_key,
            auth_token,
            client,
        })
    }

    fn transcribe_impl<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        let audio_str = audio_path
            .as_ref()
            .to_str()
            .ok_or_else(|| MpvSttError::InvalidPath("Invalid audio path".to_string()))?;

        trace!(
            server = %self.server_url,
            model = %self.config.model,
            language = self.config.language.as_deref().unwrap_or("auto"),
            audio = audio_str,
            duration_ms,
            "transcribing a chunk over the ferrum protocol"
        );

        let run_generation = self.cancel_generation.load(Ordering::Relaxed);

        let audio_data = self.compress_audio(&audio_path)?;
        if audio_data.is_empty() {
            return Err(MpvSttError::MalformedResponse {
                server: "local WAV".to_string(),
                context: format!("{audio_str} compressed to nothing"),
            });
        }

        let request_id = self.generate_request_id();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| MpvSttError::SttFailed(format!("cannot build the async runtime: {e}")))?;
        let srt_data = runtime.block_on(self.send_request_with_retry(
            request_id,
            &audio_data,
            duration_ms,
            run_generation,
        ))?;

        if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
            return Err(MpvSttError::SttCancelled);
        }

        if srt_data.iter().all(|b| b.is_ascii_whitespace()) {
            debug!(
                server = %self.server_url,
                duration_ms,
                "the server returned no subtitles for this chunk"
            );
            let output_path = PathBuf::from(output_prefix.as_ref()).with_extension("srt");
            SrtFile::new().save(&output_path)?;
            return Ok(());
        }

        let srt_file = SrtFile::parse_content(&String::from_utf8_lossy(&srt_data))?;
        let output_path = PathBuf::from(output_prefix.as_ref()).with_extension("srt");
        srt_file.save(&output_path)?;

        debug!(
            entries = srt_file.entries.len(),
            duration_ms,
            path = %output_path.display(),
            "chunk transcribed"
        );
        Ok(())
    }

    fn generate_request_id(&self) -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    async fn send_request_with_retry(
        &self,
        request_id: u64,
        audio: &[u8],
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<Vec<u8>> {
        let mut last_error = None;
        let max_attempts = self.config.max_retry.max(1);

        for attempt in 0..max_attempts {
            if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
                return Err(MpvSttError::SttCancelled);
            }

            match self
                .send_request(request_id, audio, duration_ms, run_generation)
                .await
            {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt + 1 < max_attempts {
                        warn!(
                            attempt = attempt + 1,
                            of = max_attempts,
                            error = %e,
                            cause = %crate::logging::err_chain(&e),
                            "transcription request failed; retrying"
                        );
                        last_error = Some(e);
                        tokio::select! {
                            () = tokio::time::sleep(Duration::from_millis(500)) => {}
                            () = Self::wait_for_cancellation(
                                &self.cancel_generation,
                                run_generation,
                            ) => return Err(MpvSttError::SttCancelled),
                        }
                    } else {
                        last_error = Some(e);
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    async fn send_request(
        &self,
        request_id: u64,
        audio: &[u8],
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<Vec<u8>> {
        let mut payload = audio.to_vec();
        let encrypted = if let Some(key) = self.encryption_key.as_ref() {
            payload = key.encrypt(&payload)?;
            true
        } else {
            false
        };
        let payload_len = payload.len();

        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_REQUEST_ID,
            HeaderValue::from_str(&request_id.to_string()).map_err(|e| {
                MpvSttError::MalformedResponse {
                    server: "local request".to_string(),
                    context: format!("cannot encode the {HEADER_REQUEST_ID} header: {e}"),
                }
            })?,
        );
        headers.insert(
            HEADER_DURATION_MS,
            HeaderValue::from_str(&duration_ms.to_string()).map_err(|e| {
                MpvSttError::MalformedResponse {
                    server: "local request".to_string(),
                    context: format!("cannot encode the {HEADER_DURATION_MS} header: {e}"),
                }
            })?,
        );
        headers.insert(
            HEADER_AUTH_TOKEN,
            HeaderValue::from_str(&hex::encode(self.auth_token.as_bytes())).map_err(|e| {
                MpvSttError::MalformedResponse {
                    server: "local request".to_string(),
                    context: format!("cannot encode the {HEADER_AUTH_TOKEN} header: {e}"),
                }
            })?,
        );
        let compression = if self.config.use_opus {
            COMPRESSION_OPUS
        } else {
            COMPRESSION_PCM
        };
        headers.insert(HEADER_COMPRESSION, HeaderValue::from_static(compression));
        if encrypted {
            headers.insert(HEADER_ENCRYPTED, HeaderValue::from_static("1"));
        }
        headers.insert(
            HEADER_MODEL,
            HeaderValue::from_str(&self.config.model).map_err(|e| {
                MpvSttError::MalformedResponse {
                    server: "local request".to_string(),
                    context: format!("cannot encode the {HEADER_MODEL} header: {e}"),
                }
            })?,
        );
        if let Some(lang) = self.config.language.as_ref() {
            headers.insert(
                HEADER_LANGUAGE,
                HeaderValue::from_str(lang).map_err(|e| MpvSttError::MalformedResponse {
                    server: "local request".to_string(),
                    context: format!("cannot encode the {HEADER_LANGUAGE} header: {e}"),
                })?,
            );
        }

        let wall_start = Instant::now();
        let request = self
            .client
            .post(format!("{}/transcribe", self.server_url))
            .headers(headers)
            .body(payload);
        let endpoint = format!("{}/transcribe", self.server_url);
        let request_future = async {
            let response = request
                .send()
                .await
                .map_err(|e| MpvSttError::TranslationRequest {
                    url: endpoint.clone(),
                    source: e,
                })?;
            let status = response.status();
            if !status.is_success() {
                let text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown error".to_string());
                return Err(MpvSttError::HttpStatus {
                    server: endpoint.clone(),
                    status: status.as_u16(),
                    body: crate::logging::one_line(&text, 300),
                });
            }

            let response_headers = response.headers().clone();
            let data = response
                .bytes()
                .await
                .map_err(|e| MpvSttError::MalformedResponse {
                    server: endpoint.clone(),
                    context: format!("cannot read the response body: {e}"),
                })?
                .to_vec();
            Ok((response_headers, data))
        };
        let (response_headers, mut data) = tokio::select! {
            result = request_future => result?,
            () = Self::wait_for_cancellation(
                &self.cancel_generation,
                run_generation,
            ) => return Err(MpvSttError::SttCancelled),
        };
        let raw_resp_len = data.len();

        if encrypted {
            if let Some(key) = self.encryption_key.as_ref() {
                data = key.decrypt(&data)?;
            }
        }

        let wall_ms = wall_start.elapsed().as_millis() as u64;
        let server_queue_ms = parse_u64_header(&response_headers, HEADER_QUEUE_MS);
        let server_infer_ms = parse_u64_header(&response_headers, HEADER_INFER_MS);
        let server_worker_ms = parse_u64_header(&response_headers, HEADER_WORKER_MS);
        let server_bytes_in = parse_u64_header(&response_headers, HEADER_BYTES_IN);
        let server_bytes_out = parse_u64_header(&response_headers, HEADER_BYTES_OUT);
        let server_total_ms = server_queue_ms.saturating_add(server_worker_ms);
        let network_ms = wall_ms.saturating_sub(server_total_ms);
        let server_non_infer_ms = server_worker_ms.saturating_sub(server_infer_ms);

        debug!(
            request_id,
            server = %self.server_url,
            model = %self.config.model,
            duration_ms,
            wall_ms,
            network_ms,
            server_queue_ms,
            server_worker_ms,
            server_infer_ms,
            server_non_infer_ms,
            bytes_out = payload_len,
            bytes_in = server_bytes_in,
            server_bytes_out,
            response_bytes = raw_resp_len,
            "transcription response received"
        );

        Ok(data)
    }

    async fn wait_for_cancellation(cancel_generation: &AtomicU64, run_generation: u64) {
        loop {
            if cancel_generation.load(Ordering::Acquire) != run_generation {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    fn compress_audio<P: AsRef<Path>>(&self, audio_path: P) -> Result<Vec<u8>> {
        use hound::WavReader;

        let path_ref = audio_path.as_ref();
        let mut reader = WavReader::open(path_ref)
            .map_err(|e| MpvSttError::SttFailed(format!("Failed to read WAV: {}", e)))?;

        let spec = reader.spec();
        if spec.channels != 1 || spec.sample_rate != 16000 || spec.bits_per_sample != 16 {
            return Err(MpvSttError::SttFailed(format!(
                "Unsupported WAV format: {}ch {}Hz {}-bit",
                spec.channels, spec.sample_rate, spec.bits_per_sample
            )));
        }

        if !self.config.use_opus {
            let bytes = std::fs::read(path_ref)
                .map_err(|e| MpvSttError::SttFailed(format!("Failed to read WAV bytes: {}", e)))?;
            return Ok(bytes);
        }

        // Encode to Opus (mono, 16 kHz, 20 ms frames; framing: [u32_le_len][packet]...)
        let mut encoder = SimpleOpusEncoder::new()
            .map_err(|e| MpvSttError::SttFailed(format!("Opus encoder init failed: {e}")))?;

        let frame_size = SimpleOpusEncoder::FRAME_SIZE as usize; // 20 ms @ 16 kHz
        let mut pcm: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| MpvSttError::SttFailed(format!("Read WAV samples failed: {}", e)))?;

        if pcm.is_empty() {
            return Err(MpvSttError::SttFailed("Audio data is empty".to_string()));
        }

        // Pad last frame with zeros if not aligned.
        let rem = pcm.len() % frame_size;
        if rem != 0 {
            pcm.extend(std::iter::repeat(0).take(frame_size - rem));
        }

        let mut encoded = Vec::with_capacity(pcm.len() / 2);
        let mut out_buf = vec![0u8; 4000]; // generous per-frame buffer

        for chunk in pcm.chunks(frame_size) {
            let len = encoder
                .encode(chunk, &mut out_buf)
                .map_err(|e| MpvSttError::SttFailed(format!("Opus encode failed: {e}")))?;
            encoded.extend_from_slice(&(len as u32).to_le_bytes());
            encoded.extend_from_slice(&out_buf[..len]);
        }

        Ok(encoded)
    }
}

// Minimal safe wrapper around opusic-sys encoder.
struct SimpleOpusEncoder {
    enc: *mut opus::OpusEncoder,
}

impl SimpleOpusEncoder {
    const SAMPLE_RATE: i32 = 16_000;
    const CHANNELS: i32 = 1;
    // 20 ms @ 16 kHz
    const FRAME_SIZE: i32 = 320;

    fn new() -> std::result::Result<Self, String> {
        let mut err: libc::c_int = 0;
        let enc = unsafe {
            opus::opus_encoder_create(
                Self::SAMPLE_RATE,
                Self::CHANNELS,
                opus::OPUS_APPLICATION_AUDIO,
                &mut err,
            )
        };
        if enc.is_null() || err != opus::OPUS_OK {
            return Err(format!("opus_encoder_create failed: {}", opus_error(err)));
        }
        Ok(Self { enc })
    }

    fn encode(&mut self, pcm: &[i16], out: &mut [u8]) -> std::result::Result<usize, String> {
        if pcm.len() != Self::FRAME_SIZE as usize {
            return Err(format!(
                "invalid frame samples: expected {}, got {}",
                Self::FRAME_SIZE,
                pcm.len()
            ));
        }
        let ret = unsafe {
            opus::opus_encode(
                self.enc,
                pcm.as_ptr(),
                Self::FRAME_SIZE,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        if ret < 0 {
            return Err(format!("opus_encode failed: {}", opus_error(ret)));
        }
        Ok(ret as usize)
    }
}

impl Drop for SimpleOpusEncoder {
    fn drop(&mut self) {
        unsafe { opus::opus_encoder_destroy(self.enc) };
    }
}

fn opus_error(code: libc::c_int) -> String {
    unsafe {
        let cstr = opus::opus_strerror(code);
        if cstr.is_null() {
            format!("Opus error {}", code)
        } else {
            std::ffi::CStr::from_ptr(cstr)
                .to_string_lossy()
                .into_owned()
        }
    }
}

fn parse_u64_header(headers: &HeaderMap, name: &str) -> u64 {
    headers
        .get(name)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

fn normalize_server_url(raw: &str) -> String {
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{}", raw)
    }
}

impl SttBackend for FerrumBackend {
    fn protocol(&self) -> SttProtocol {
        SttProtocol::Ferrum
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

    fn cancellation_generation(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.cancel_generation)
    }

    fn take_device_notice(&mut self) -> Option<SttDeviceNotice> {
        None
    }
}
