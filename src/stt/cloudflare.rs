use super::segments::{RawSegment, Segment, normalize_segments};
use super::{SttBackend, SttDeviceNotice};
use crate::common::{MpvSttError, Result};
use crate::config::SttProtocol;
use crate::srt::{SrtFile, SubtitleEntry, Timestamp};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use reqwest::Client;
use reqwest::Url;
use reqwest::header::CONTENT_TYPE;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, trace, warn};

const API_ROOT: &str = "https://api.cloudflare.com/client/v4";
const MODEL_WHISPER: &str = "@cf/openai/whisper";
const MODEL_WHISPER_TURBO: &str = "@cf/openai/whisper-large-v3-turbo";

/// The Cloudflare Workers AI fields resolved from a named STT source.
#[derive(Debug, Clone)]
pub struct SttCloudflareConfig {
    pub account_id: String,
    pub model: String,
    pub language: Option<String>,
    pub api_key: Option<String>,
    pub timeout_ms: u64,
    pub max_retry: usize,
}

impl Default for SttCloudflareConfig {
    fn default() -> Self {
        Self {
            account_id: String::new(),
            model: MODEL_WHISPER_TURBO.to_string(),
            language: None,
            api_key: None,
            timeout_ms: 120_000,
            max_retry: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperModel {
    Whisper,
    Turbo,
}

impl WhisperModel {
    fn from_id(model: &str) -> Result<Self> {
        match model {
            MODEL_WHISPER => Ok(Self::Whisper),
            MODEL_WHISPER_TURBO => Ok(Self::Turbo),
            _ => Err(MpvSttError::SttFailed(format!(
                "unsupported Cloudflare Workers AI model {model:?}; supported models are {MODEL_WHISPER} and {MODEL_WHISPER_TURBO}"
            ))),
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::Whisper => MODEL_WHISPER,
            Self::Turbo => MODEL_WHISPER_TURBO,
        }
    }
}

/// Workers AI Whisper backend. The two documented Whisper models use different
/// request and response shapes, so they are normalized separately before the
/// common timestamp validation and SRT-writing path.
pub struct CloudflareBackend {
    endpoint: Url,
    account_id: String,
    model: WhisperModel,
    language: Option<String>,
    api_key: String,
    max_retry: usize,
    cancel_generation: Arc<AtomicU64>,
    client: Client,
}

impl CloudflareBackend {
    pub fn new(config: SttCloudflareConfig) -> Result<Self> {
        Self::build(config, API_ROOT)
    }

    #[cfg(test)]
    pub(super) fn build_for_test(config: SttCloudflareConfig, api_root: &str) -> Result<Self> {
        Self::build(config, api_root)
    }

    fn build(config: SttCloudflareConfig, api_root: &str) -> Result<Self> {
        let account_id = config.account_id.trim();
        if account_id.is_empty()
            || account_id.len() > 128
            || !account_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(MpvSttError::SttFailed(
                "Cloudflare Workers AI requires a valid account_id (letters, digits, and hyphens only)".to_string(),
            ));
        }

        let api_key = config
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .ok_or_else(|| {
                MpvSttError::SttFailed(
                    "Cloudflare Workers AI requires api_key with a Workers AI API token"
                        .to_string(),
                )
            })?;
        let model = WhisperModel::from_id(config.model.trim())?;
        let language = config
            .language
            .as_deref()
            .map(str::trim)
            .filter(|language| !language.is_empty())
            .map(str::to_string);
        if model == WhisperModel::Whisper && language.is_some() {
            return Err(MpvSttError::SttFailed(
                "Cloudflare @cf/openai/whisper does not accept a language hint; remove language or use @cf/openai/whisper-large-v3-turbo".to_string(),
            ));
        }

        let endpoint = Self::build_endpoint(api_root, account_id, model)?;
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|e| MpvSttError::SttFailed(format!("HTTP client build failed: {e}")))?;

        Ok(Self {
            endpoint,
            account_id: account_id.to_string(),
            model,
            language,
            api_key: api_key.to_string(),
            max_retry: config.max_retry.max(1),
            cancel_generation: Arc::new(AtomicU64::new(0)),
            client,
        })
    }

    fn build_endpoint(api_root: &str, account_id: &str, model: WhisperModel) -> Result<Url> {
        let mut endpoint = Url::parse(api_root)
            .map_err(|e| MpvSttError::SttFailed(format!("invalid Cloudflare API root URL: {e}")))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(MpvSttError::SttFailed(
                "Cloudflare API root must be an HTTP(S) URL without query or fragment".to_string(),
            ));
        }

        let mut path = endpoint.path_segments_mut().map_err(|_| {
            MpvSttError::SttFailed("Cloudflare API root cannot be used as a base URL".to_string())
        })?;
        path.pop_if_empty();
        path.push("accounts")
            .push(account_id)
            .push("ai")
            .push("run");
        for segment in model.id().split('/') {
            path.push(segment);
        }
        drop(path);
        Ok(endpoint)
    }

    fn generate_request_id(&self) -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    fn transcribe_impl<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        let run_generation = self.cancel_generation.load(Ordering::Acquire);
        self.transcribe_impl_with_generation(audio_path, output_prefix, duration_ms, run_generation)
    }

    fn transcribe_impl_with_generation<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<()> {
        self.transcribe_impl_with_reader(
            audio_path,
            output_prefix,
            duration_ms,
            run_generation,
            |path| std::fs::read(path),
        )
    }

    fn transcribe_impl_with_reader<P, F>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
        run_generation: u64,
        read_audio: F,
    ) -> Result<()>
    where
        P: AsRef<Path>,
        F: FnOnce(&Path) -> std::io::Result<Vec<u8>>,
    {
        if self.cancel_generation.load(Ordering::Acquire) != run_generation {
            return Err(MpvSttError::SttCancelled);
        }

        let audio_str = audio_path
            .as_ref()
            .to_str()
            .ok_or_else(|| MpvSttError::InvalidPath("Invalid audio path".to_string()))?;
        trace!(
            model = self.model.id(),
            audio = audio_str,
            duration_ms,
            "transcribing a chunk with Cloudflare Workers AI"
        );

        let audio =
            read_audio(audio_path.as_ref()).map_err(|e| MpvSttError::MalformedResponse {
                server: "local WAV".to_string(),
                context: format!("cannot read {audio_str}: {e}"),
            })?;
        if self.cancel_generation.load(Ordering::Acquire) != run_generation {
            return Err(MpvSttError::SttCancelled);
        }
        if audio.is_empty() {
            return Err(MpvSttError::MalformedResponse {
                server: "local WAV".to_string(),
                context: format!("{audio_str} is empty"),
            });
        }

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| MpvSttError::SttFailed(format!("cannot build the async runtime: {e}")))?;
        let response = runtime.block_on(self.send_with_retry(
            self.generate_request_id(),
            &audio,
            duration_ms,
            run_generation,
        ))?;

        if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
            return Err(MpvSttError::SttCancelled);
        }

        let segments = parse_response(&response, duration_ms, self.model)?;
        let output_path = PathBuf::from(output_prefix.as_ref()).with_extension("srt");
        let mut srt = SrtFile::new();
        for (index, segment) in segments.iter().enumerate() {
            let text = segment.text.trim();
            if text.is_empty() {
                continue;
            }
            let start_ms = (segment.start * 1000.0).round() as u32;
            let end_ms = ((segment.end * 1000.0).round() as u32).max(start_ms.saturating_add(1));
            srt.append_entry(SubtitleEntry {
                index: (index + 1) as u32,
                start_time: Timestamp::from_milliseconds(start_ms),
                end_time: Timestamp::from_milliseconds(end_ms),
                text: text.to_string(),
            });
        }

        if srt.entries.is_empty() {
            debug!(
                model = self.model.id(),
                duration_ms, "Cloudflare returned no speech for this chunk"
            );
        }
        srt.save(&output_path)?;
        debug!(
            model = self.model.id(),
            segments = segments.len(),
            entries = srt.entries.len(),
            duration_ms,
            path = %output_path.display(),
            "chunk transcribed"
        );
        Ok(())
    }

    async fn send_with_retry(
        &self,
        request_id: u64,
        audio: &[u8],
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<Vec<u8>> {
        let mut last_error = None;
        for attempt in 0..self.max_retry {
            if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
                return Err(MpvSttError::SttCancelled);
            }

            match self
                .send_request(request_id, audio, duration_ms, run_generation)
                .await
            {
                Ok(response) => return Ok(response),
                Err(error) => {
                    if attempt + 1 >= self.max_retry || !is_retryable(&error) {
                        return Err(error);
                    }
                    warn!(
                        attempt = attempt + 1,
                        of = self.max_retry,
                        error = %error,
                        cause = %crate::logging::err_chain(&error),
                        "Cloudflare transcription request failed; retrying"
                    );
                    last_error = Some(error);
                    tokio::select! {
                        () = tokio::time::sleep(Duration::from_millis(500)) => {}
                        () = Self::wait_for_cancellation(
                            &self.cancel_generation,
                            run_generation,
                        ) => return Err(MpvSttError::SttCancelled),
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MpvSttError::SttFailed("Cloudflare request ended without a response".to_string())
        }))
    }

    async fn send_request(
        &self,
        request_id: u64,
        audio: &[u8],
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<Vec<u8>> {
        let wall_start = Instant::now();
        let mut request = self
            .client
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .header("x-request-id", request_id.to_string())
            .header("x-duration-ms", duration_ms.to_string());

        match self.model {
            WhisperModel::Turbo => {
                let mut body = json!({
                    "audio": STANDARD.encode(audio),
                    "task": "transcribe"
                });
                if let Some(language) = self.language.as_deref() {
                    body["language"] = Value::String(language.to_string());
                }
                request = request.json(&body);
            }
            WhisperModel::Whisper => {
                // The regular Whisper schema accepts binary audio. Avoid
                // expanding the WAV into a large JSON array of byte values.
                request = request
                    .header(CONTENT_TYPE, "application/octet-stream")
                    .body(audio.to_vec());
            }
        }

        let endpoint = self.endpoint.to_string();
        let request_future = async {
            let response =
                request
                    .send()
                    .await
                    .map_err(|source| MpvSttError::TranslationRequest {
                        url: endpoint.clone(),
                        source,
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

            response
                .bytes()
                .await
                .map(|bytes| bytes.to_vec())
                .map_err(|e| MpvSttError::MalformedResponse {
                    server: endpoint.clone(),
                    context: format!("cannot read the response body: {e}"),
                })
        };
        let data = tokio::select! {
            result = request_future => result?,
            () = Self::wait_for_cancellation(
                &self.cancel_generation,
                run_generation,
            ) => return Err(MpvSttError::SttCancelled),
        };

        debug!(
            request_id,
            model = self.model.id(),
            account_id = %self.account_id,
            duration_ms,
            request_bytes = audio.len(),
            response_bytes = data.len(),
            wall_ms = wall_start.elapsed().as_millis() as u64,
            "Cloudflare transcription response received"
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
}

fn is_retryable(error: &MpvSttError) -> bool {
    match error {
        MpvSttError::HttpStatus { status, .. } => *status == 429 || *status >= 500,
        MpvSttError::TranslationRequest { source, .. } => {
            source.is_timeout() || source.is_connect()
        }
        _ => false,
    }
}

#[derive(Debug, Deserialize)]
struct CloudflareEnvelope {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    errors: Vec<Value>,
    result: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CloudflareResult {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Vec<RawSegment>,
    #[serde(default)]
    words: Vec<RawWord>,
    vtt: Option<String>,
    transcription_info: Option<TranscriptionInfo>,
}

#[derive(Debug, Deserialize)]
struct RawWord {
    start: Option<f64>,
    end: Option<f64>,
    #[serde(default, alias = "text")]
    word: String,
}

#[derive(Debug, Deserialize)]
struct TranscriptionInfo {
    duration: Option<f64>,
}

fn parse_response(json: &[u8], chunk_ms: u64, model: WhisperModel) -> Result<Vec<Segment>> {
    let envelope: CloudflareEnvelope =
        serde_json::from_slice(json).map_err(|e| MpvSttError::MalformedResponse {
            server: "Cloudflare Workers AI".to_string(),
            context: format!(
                "{e}; body starts with {:?}",
                crate::logging::one_line(&String::from_utf8_lossy(json), 200)
            ),
        })?;

    if !envelope.success || !envelope.errors.is_empty() {
        let details = serde_json::to_string(&envelope.errors)
            .unwrap_or_else(|_| "Cloudflare returned an error".to_string());
        return Err(MpvSttError::MalformedResponse {
            server: "Cloudflare Workers AI".to_string(),
            context: format!(
                "API envelope reported failure: {}",
                crate::logging::one_line(&details, 300)
            ),
        });
    }

    let result = envelope
        .result
        .ok_or_else(|| MpvSttError::MalformedResponse {
            server: "Cloudflare Workers AI".to_string(),
            context: "successful API envelope did not contain result".to_string(),
        })?;
    let result: CloudflareResult =
        serde_json::from_value(result).map_err(|e| MpvSttError::MalformedResponse {
            server: "Cloudflare Workers AI result".to_string(),
            context: e.to_string(),
        })?;
    let duration = result.transcription_info.and_then(|info| info.duration);

    let mut segments = result.segments;
    if segments.is_empty() {
        if let Some(vtt) = result.vtt.as_deref() {
            segments = parse_vtt(vtt);
        }
    }
    if segments.is_empty() && !result.words.is_empty() {
        segments = group_words(result.words);
    }

    normalize_segments(
        &result.text,
        segments,
        duration,
        chunk_ms,
        match model {
            WhisperModel::Whisper => "Cloudflare Whisper",
            WhisperModel::Turbo => "Cloudflare Whisper Large V3 Turbo",
        },
    )
}

fn group_words(words: Vec<RawWord>) -> Vec<RawSegment> {
    const MAX_CUE_CHARS: usize = 42;
    const MAX_CUE_SECONDS: f64 = 4.5;
    const PAUSE_SECONDS: f64 = 0.8;

    let mut words = words
        .into_iter()
        .filter_map(|word| {
            let (Some(start), Some(end)) = (word.start, word.end) else {
                return None;
            };
            if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
                return None;
            }
            let text = word.word.trim();
            if text.is_empty() {
                return None;
            }
            Some(RawSegment {
                start: Some(start),
                end: Some(end),
                text: text.to_string(),
            })
        })
        .collect::<Vec<_>>();
    words.sort_by(|a, b| {
        a.start
            .unwrap_or_default()
            .total_cmp(&b.start.unwrap_or_default())
    });

    let mut grouped = Vec::new();
    let mut current: Option<RawSegment> = None;
    for word in words {
        let start = word.start.unwrap_or_default();
        let end = word.end.unwrap_or(start);
        let should_break = current.as_ref().is_some_and(|segment| {
            let previous_end = segment.end.unwrap_or(segment.start.unwrap_or_default());
            let combined_chars = segment.text.chars().count() + word.text.chars().count();
            let duration = end - segment.start.unwrap_or(start);
            previous_end + PAUSE_SECONDS < start
                || segment_ends_sentence(&segment.text)
                || combined_chars > MAX_CUE_CHARS
                || duration > MAX_CUE_SECONDS
        });
        if should_break {
            if let Some(segment) = current.take() {
                grouped.push(segment);
            }
        }

        if let Some(segment) = current.as_mut() {
            let separator = if needs_space(&segment.text, &word.text) {
                " "
            } else {
                ""
            };
            segment.text.push_str(separator);
            segment.text.push_str(&word.text);
            segment.end = Some(end);
        } else {
            current = Some(word);
        }
    }
    if let Some(segment) = current {
        grouped.push(segment);
    }
    grouped
}

fn needs_space(previous: &str, next: &str) -> bool {
    let Some(previous_char) = previous.chars().next_back() else {
        return false;
    };
    let Some(next_char) = next.chars().next() else {
        return false;
    };
    if !next_char.is_alphanumeric() {
        return false;
    }
    if matches!(previous_char, '(' | '[' | '{' | '“' | '‘') {
        return false;
    }
    !(is_cjk(previous_char) && is_cjk(next_char))
}

fn is_cjk(character: char) -> bool {
    matches!(character as u32,
        0x2E80..=0x2EFF
            | 0x2F00..=0x2FDF
            | 0x3000..=0x303F
            | 0x3040..=0x30FF
            | 0x3100..=0x312F
            | 0x3130..=0x318F
            | 0x31A0..=0x31BF
            | 0x31F0..=0x31FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xAC00..=0xD7AF
            | 0xF900..=0xFAFF
            | 0xFF00..=0xFFEF
            | 0x20000..=0x2FA1F
    )
}

fn segment_ends_sentence(text: &str) -> bool {
    text.trim_end()
        .chars()
        .next_back()
        .is_some_and(|character| matches!(character, '.' | '!' | '?' | '。' | '！' | '？'))
}

fn parse_vtt(vtt: &str) -> Vec<RawSegment> {
    let lines = vtt.lines().collect::<Vec<_>>();
    let mut segments = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let Some((start, end)) = lines[index].split_once("-->") else {
            index += 1;
            continue;
        };
        let Some(start) = parse_vtt_timestamp(start.trim()) else {
            index += 1;
            continue;
        };
        let end = end.split_whitespace().next().unwrap_or_default();
        let Some(end) = parse_vtt_timestamp(end) else {
            index += 1;
            continue;
        };

        index += 1;
        let mut text_lines = Vec::new();
        while index < lines.len() && !lines[index].trim().is_empty() {
            let text = strip_vtt_tags(lines[index].trim());
            if !text.is_empty() {
                text_lines.push(text);
            }
            index += 1;
        }
        let text = text_lines.join(" ");
        if !text.is_empty() {
            segments.push(RawSegment {
                start: Some(start),
                end: Some(end),
                text,
            });
        }
    }
    segments
}

fn parse_vtt_timestamp(timestamp: &str) -> Option<f64> {
    let timestamp = timestamp.replace(',', ".");
    let parts = timestamp.split(':').collect::<Vec<_>>();
    if !(2..=3).contains(&parts.len()) {
        return None;
    }
    let last = parts.last()?.parse::<f64>().ok()?;
    let minutes = parts[parts.len() - 2].parse::<f64>().ok()?;
    let hours = if parts.len() == 3 {
        parts[0].parse::<f64>().ok()?
    } else {
        0.0
    };
    let seconds = hours * 3600.0 + minutes * 60.0 + last;
    seconds.is_finite().then_some(seconds)
}

fn strip_vtt_tags(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut in_tag = false;
    for character in text.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => output.push(character),
            _ => {}
        }
    }
    output
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .trim()
        .to_string()
}

impl SttBackend for CloudflareBackend {
    fn protocol(&self) -> SttProtocol {
        SttProtocol::Cloudflare
    }

    fn transcribe<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        self.transcribe_impl(audio_path, output_prefix, duration_ms)
    }

    fn transcribe_with_generation<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
        expected_generation: u64,
    ) -> Result<()> {
        self.transcribe_impl_with_generation(
            audio_path,
            output_prefix,
            duration_ms,
            expected_generation,
        )
    }

    fn cancel_inflight(&self) {
        self.cancel_generation.fetch_add(1, Ordering::AcqRel);
    }

    fn cancellation_generation(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.cancel_generation)
    }

    fn take_device_notice(&mut self) -> Option<SttDeviceNotice> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    #[derive(Debug)]
    struct CapturedRequest {
        headers: String,
        body: Vec<u8>,
    }

    fn mock_server(response: Value) -> (String, thread::JoinHandle<CapturedRequest>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let response = response.to_string();
        let thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(response.as_bytes()).unwrap();
            request
        });
        (format!("http://{address}/client/v4"), thread)
    }

    fn read_http_request(stream: &mut TcpStream) -> CapturedRequest {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        let mut header_end = None;
        let mut content_length = 0usize;
        loop {
            let count = stream.read(&mut chunk).unwrap();
            assert_ne!(count, 0, "connection closed before full request was read");
            bytes.extend_from_slice(&chunk[..count]);
            if header_end.is_none()
                && let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
            {
                let end = position + 4;
                let headers = String::from_utf8_lossy(&bytes[..end]);
                content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().unwrap())
                    })
                    .unwrap_or(0);
                header_end = Some(end);
            }
            if header_end.is_some_and(|end| bytes.len() >= end + content_length) {
                break;
            }
        }
        let header_end = header_end.unwrap();
        CapturedRequest {
            headers: String::from_utf8_lossy(&bytes[..header_end]).into_owned(),
            body: bytes[header_end..header_end + content_length].to_vec(),
        }
    }

    fn config(model: &str, language: Option<&str>) -> SttCloudflareConfig {
        SttCloudflareConfig {
            account_id: "account123".to_string(),
            model: model.to_string(),
            language: language.map(str::to_string),
            api_key: Some("test-token".to_string()),
            timeout_ms: 5_000,
            max_retry: 1,
        }
    }

    #[test]
    fn cancellation_during_audio_read_does_not_send_stale_chunk_and_next_chunk_works() {
        let (api_root, server) = mock_server(json!({
            "success": true,
            "errors": [],
            "result": {
                "text": "current chunk",
                "segments": [{"start": 0.0, "end": 0.8, "text": "current chunk"}]
            }
        }));
        let mut backend =
            CloudflareBackend::build(config(MODEL_WHISPER_TURBO, None), &api_root).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let audio_path = dir.path().join("chunk.wav");
        let prefix = dir.path().join("chunk");
        std::fs::write(&audio_path, b"RIFF fake wav").unwrap();

        let queued_generation = backend.cancel_generation.load(Ordering::Acquire);
        let cancel_generation = Arc::clone(&backend.cancel_generation);
        let cancelled = backend.transcribe_impl_with_reader(
            &audio_path,
            &prefix,
            2_000,
            queued_generation,
            move |_| {
                cancel_generation.fetch_add(1, Ordering::AcqRel);
                Ok(b"RIFF stale wav".to_vec())
            },
        );
        assert!(matches!(cancelled, Err(MpvSttError::SttCancelled)));
        assert!(
            !prefix.with_extension("srt").exists(),
            "a cancelled preflight must not leave a subtitle result"
        );

        backend.transcribe(&audio_path, &prefix, 2_000).unwrap();
        let request = server.join().unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["audio"], STANDARD.encode(b"RIFF fake wav"));
        let srt = std::fs::read_to_string(prefix.with_extension("srt")).unwrap();
        assert!(srt.contains("current chunk"));
    }

    #[test]
    fn turbo_request_uses_bearer_json_base64_and_language() {
        let audio = b"RIFF fake wav";
        let server_response = json!({
            "success": true,
            "errors": [],
            "result": {
                "text": "hello world",
                "segments": [
                    {"start": 0.0, "end": 0.8, "text": "hello"},
                    {"start": 0.8, "end": 1.6, "text": "world"}
                ],
                "transcription_info": {"duration": 1.6}
            }
        });
        let (api_root, server) = mock_server(server_response);
        let mut backend =
            CloudflareBackend::build(config(MODEL_WHISPER_TURBO, Some("ja")), &api_root).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let audio_path = dir.path().join("chunk.wav");
        let prefix = dir.path().join("chunk");
        std::fs::write(&audio_path, audio).unwrap();

        backend.transcribe(&audio_path, &prefix, 2_000).unwrap();
        let request = server.join().unwrap();
        assert!(request.headers.starts_with(
            "POST /client/v4/accounts/account123/ai/run/@cf/openai/whisper-large-v3-turbo HTTP/1.1"
        ));
        assert!(
            request
                .headers
                .to_ascii_lowercase()
                .contains("authorization: bearer test-token")
        );
        assert!(
            request
                .headers
                .to_ascii_lowercase()
                .contains("content-type: application/json")
        );
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["audio"], STANDARD.encode(audio));
        assert_eq!(body["language"], "ja");
        assert_eq!(body["task"], "transcribe");
        let srt = std::fs::read_to_string(prefix.with_extension("srt")).unwrap();
        assert!(srt.contains("hello"));
        assert!(srt.contains("world"));
    }

    #[test]
    fn regular_whisper_uses_binary_input_and_vtt_timestamps() {
        let audio = b"RIFF fake wav";
        let server_response = json!({
            "success": true,
            "errors": [],
            "result": {
                "text": "Hello, world.",
                "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.200\n<v Speaker>Hello, world.</v>\n"
            }
        });
        let (api_root, server) = mock_server(server_response);
        let mut backend = CloudflareBackend::build(config(MODEL_WHISPER, None), &api_root).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let audio_path = dir.path().join("chunk.wav");
        let prefix = dir.path().join("chunk");
        std::fs::write(&audio_path, audio).unwrap();

        backend.transcribe(&audio_path, &prefix, 2_000).unwrap();
        let request = server.join().unwrap();
        assert!(
            request.headers.starts_with(
                "POST /client/v4/accounts/account123/ai/run/@cf/openai/whisper HTTP/1.1"
            )
        );
        assert!(
            request
                .headers
                .to_ascii_lowercase()
                .contains("content-type: application/octet-stream")
        );
        assert_eq!(request.body, audio);
        let srt = std::fs::read_to_string(prefix.with_extension("srt")).unwrap();
        assert!(srt.contains("Hello, world."));
    }

    #[test]
    fn regular_whisper_word_times_are_grouped_into_readable_cues() {
        let words = vec![
            RawWord {
                start: Some(0.0),
                end: Some(0.25),
                word: "Hello,".to_string(),
            },
            RawWord {
                start: Some(0.3),
                end: Some(0.5),
                word: "world.".to_string(),
            },
            RawWord {
                start: Some(0.7),
                end: Some(1.0),
                word: "你好。".to_string(),
            },
        ];
        let segments = group_words(words);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "Hello, world.");
        assert_eq!(segments[1].text, "你好。");
        assert_eq!(segments[0].start, Some(0.0));
        assert_eq!(segments[0].end, Some(0.5));
    }

    #[test]
    fn configured_language_is_rejected_for_regular_whisper() {
        let error = CloudflareBackend::build(config(MODEL_WHISPER, Some("zh")), API_ROOT)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("does not accept a language hint"));
    }

    #[test]
    fn cloudflare_failure_envelope_is_not_treated_as_empty_speech() {
        let response =
            br#"{"success":false,"errors":[{"code":1000,"message":"bad token"}],"result":null}"#;
        let error = parse_response(response, 15_000, WhisperModel::Turbo).unwrap_err();
        assert!(error.to_string().contains("reported failure"));
        assert!(error.to_string().contains("bad token"));
    }

    #[test]
    fn text_without_timestamps_is_rejected() {
        let response = br#"{"success":true,"errors":[],"result":{"text":"hello"}}"#;
        let error = parse_response(response, 15_000, WhisperModel::Turbo).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without valid segment timestamps")
        );
    }

    #[test]
    fn empty_result_is_a_successful_no_speech_response() {
        let response = br#"{"success":true,"errors":[],"result":{"text":"","segments":[]}}"#;
        assert!(
            parse_response(response, 15_000, WhisperModel::Turbo)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn account_id_and_token_are_required_without_exposing_token() {
        let mut invalid_account_config = config(MODEL_WHISPER_TURBO, None);
        invalid_account_config.account_id = "../bad".to_string();
        let error = CloudflareBackend::build(invalid_account_config, API_ROOT)
            .err()
            .unwrap();
        assert!(error.to_string().contains("valid account_id"));

        let mut invalid_token_config = config(MODEL_WHISPER_TURBO, None);
        invalid_token_config.api_key = Some("  ".to_string());
        let error = CloudflareBackend::build(invalid_token_config, API_ROOT)
            .err()
            .unwrap();
        assert!(error.to_string().contains("requires api_key"));
        assert!(!error.to_string().contains("test-token"));
    }
}
