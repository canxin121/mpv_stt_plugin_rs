use super::{BackendKind, SttBackend, SttDeviceNotice};
use crate::common::{MpvSttError, Result};
use crate::srt::{SrtFile, SubtitleEntry, Timestamp};
use reqwest::Client;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, trace, warn};

pub type SttOpenAiConfig = crate::config::SttOpenAiConfig;

/// OpenAI-compatible backend: posts 16 kHz mono PCM WAV chunks to
/// `POST {server}/v1/audio/transcriptions` (multipart form) and turns the
/// returned `verbose_json` segments into an SRT file.
///
/// Only standard OpenAI fields are sent (`file` / `model` / `language` /
/// `response_format` / `timestamp_granularities[]`), so any compatible server
/// accepts the request: hosted APIs (e.g. Groq's
/// `https://api.groq.com/openai`) and self-hosted gateways alike.
///
/// Segments are requested with `response_format=verbose_json` plus
/// `timestamp_granularities[]=segment` (the default granularity, sent
/// explicitly). Both are standard OpenAI multipart fields; OpenAI itself only
/// returns the `segments` array for `verbose_json`. Servers that reject
/// `verbose_json` and answer with plain `{"text": ...}`, or that return no
/// segments, fall back to one subtitle per chunk (`parse_transcription`);
/// servers that emit segments anyway (Groq does) leave that fallback unused.
///
/// 16 kHz mono PCM is what the OpenAI API itself documents, and it is exactly
/// what `audio.rs`'s extractor emits (16000 Hz / 1 channel), so the payload is
/// already in the endpoint's native format.
pub struct OpenAiBackend {
    server_url: String,
    model: String,
    language: Option<String>,
    api_key: Option<String>,
    max_retry: usize,
    cancel_generation: Arc<AtomicU64>,
    client: Client,
}

impl OpenAiBackend {
    pub fn new(config: SttOpenAiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|e| MpvSttError::SttFailed(format!("HTTP client build failed: {}", e)))?;

        Ok(Self {
            server_url: normalize_server_url(&config.server_addr),
            model: config.model,
            language: config.language,
            api_key: config.api_key,
            max_retry: config.max_retry,
            cancel_generation: Arc::new(AtomicU64::new(0)),
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
            model = %self.model,
            audio = audio_str,
            duration_ms,
            "transcribing a chunk over the OpenAI protocol"
        );

        let run_generation = self.cancel_generation.load(Ordering::Relaxed);

        // The audio extractor always produces 16 kHz mono 16-bit PCM WAV; the
        // OpenAI endpoint accepts it as-is (the server resamples if needed).
        let audio_data =
            std::fs::read(&audio_path).map_err(|e| MpvSttError::MalformedResponse {
                server: "local WAV".to_string(),
                context: format!("cannot read {audio_str}: {e}"),
            })?;
        if audio_data.is_empty() {
            return Err(MpvSttError::MalformedResponse {
                server: "local WAV".to_string(),
                context: format!("{audio_str} is empty"),
            });
        }

        let request_id = self.generate_request_id();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| MpvSttError::SttFailed(format!("cannot build the async runtime: {e}")))?;
        let json = runtime.block_on(self.send_with_retry(
            request_id,
            &audio_data,
            duration_ms,
            run_generation,
        ))?;

        if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
            return Err(MpvSttError::SttCancelled);
        }

        let output_path = PathBuf::from(output_prefix.as_ref()).with_extension("srt");
        let segments = parse_transcription(&json, duration_ms)?;

        let mut srt = SrtFile::new();
        for (i, seg) in segments.iter().enumerate() {
            let text = seg.text.trim();
            if text.is_empty() {
                continue;
            }
            let start_ms = (seg.start * 1000.0).round() as u32;
            let end_ms = ((seg.end * 1000.0).round() as u32).max(start_ms.saturating_add(1));
            srt.append_entry(SubtitleEntry {
                index: (i + 1) as u32,
                start_time: Timestamp::from_milliseconds(start_ms),
                end_time: Timestamp::from_milliseconds(end_ms),
                text: text.to_string(),
            });
        }

        if srt.entries.is_empty() {
            // An empty SRT still replaces a real subtitle file upstream, so
            // report failure and leave the caller's file untouched.
            warn!(
                server = %self.server_url,
                model = %self.model,
                duration_ms,
                "the transcription server returned no text for this chunk"
            );
            return Err(MpvSttError::SttFailed(
                "the transcription server returned no text".to_string(),
            ));
        }

        srt.save(&output_path)?;
        debug!(
            segments = segments.len(),
            entries = srt.entries.len(),
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

    async fn send_with_retry(
        &self,
        request_id: u64,
        audio: &[u8],
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<Vec<u8>> {
        let mut last_error = None;
        let max_attempts = self.max_retry.max(1);

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
        // Build a standard OpenAI-style multipart form.
        let file_part = Part::bytes(audio.to_vec())
            .file_name("chunk.wav")
            .mime_str("audio/wav")
            .map_err(|e| MpvSttError::MalformedResponse {
                server: "local WAV".to_string(),
                context: format!("cannot label the upload as audio/wav: {e}"),
            })?;

        let mut form = Form::new()
            .part("file", file_part)
            .text("model", self.model.clone())
            .text("response_format", "verbose_json")
            .text("timestamp_granularities[]", "segment");
        if let Some(lang) = self.language.as_ref() {
            form = form.text("language", lang.clone());
        }

        let wall_start = Instant::now();
        let mut request = self
            .client
            .post(format!("{}/v1/audio/transcriptions", self.server_url))
            .multipart(form)
            .header("x-request-id", request_id.to_string())
            .header("x-duration-ms", duration_ms.to_string());
        if let Some(key) = self.api_key.as_ref() {
            request = request.bearer_auth(key);
        }
        let endpoint = format!("{}/v1/audio/transcriptions", self.server_url);
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
            server = %self.server_url,
            model = %self.model,
            duration_ms,
            bytes = data.len(),
            wall_ms = wall_start.elapsed().as_millis() as u64,
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
}

#[derive(Debug, Deserialize)]
struct Segment {
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct TranscriptionResponse {
    /// Present in every response shape: plain `json` / `text`, and also in
    /// `verbose_json` (which adds `segments` and timing metadata on top).
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Vec<Segment>,
    /// `verbose_json` only: the server's own measured audio duration.
    #[serde(default)]
    duration: Option<f64>,
}

/// Turn any OpenAI-compatible transcription response into subtitle segments.
///
/// `verbose_json` responses carry a `segments` array, which is used directly.
/// Anything else — a plain `{"text": ...}` from a server that ignores or
/// rejects `response_format`, or a server that returns no segments at all —
/// falls back to a single subtitle covering the whole chunk (`chunk_ms`). Text
/// without timings is still worth showing; a chunk with no text at all yields
/// no entries, which the caller reports as a failure rather than writing an
/// empty SRT over the user's existing subtitles.
fn parse_transcription(json: &[u8], chunk_ms: u64) -> Result<Vec<Segment>> {
    let resp: TranscriptionResponse =
        serde_json::from_slice(json).map_err(|e| MpvSttError::MalformedResponse {
            server: "transcription response".to_string(),
            context: format!(
                "{e}; body starts with {:?}",
                crate::logging::one_line(&String::from_utf8_lossy(json), 200)
            ),
        })?;

    if resp.segments.iter().any(|s| !s.text.trim().is_empty()) {
        return Ok(resp.segments);
    }

    let text = resp.text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }

    // Prefer the server's own duration (verbose_json); otherwise the chunk's
    // requested length is the best local estimate.
    let end_s = resp
        .duration
        .filter(|d| *d > 0.0)
        .unwrap_or(chunk_ms as f64 / 1000.0);

    Ok(vec![Segment {
        start: 0.0,
        end: end_s,
        text: text.to_string(),
    }])
}

fn normalize_server_url(raw: &str) -> String {
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{}", raw)
    }
}

impl SttBackend for OpenAiBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::OpenAi
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbose_json_segments_are_used_as_is() {
        let json = br#"{
            "text": "hello world",
            "duration": 2.5,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"}
            ]
        }"#;
        let segments = parse_transcription(json, 15_000).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[1].end, 2.0);
    }

    #[test]
    fn plain_text_response_falls_back_to_one_segment() {
        let json = r#"{"text": "  你好世界  "}"#.as_bytes();
        let segments = parse_transcription(json, 15_000).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "你好世界");
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 15.0);
    }

    #[test]
    fn empty_segments_fall_back_to_text_and_prefer_server_duration() {
        let json = r#"{"text": "auto 语言", "duration": 3.25, "segments": []}"#.as_bytes();
        let segments = parse_transcription(json, 15_000).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].end, 3.25);
    }

    #[test]
    fn blank_segments_do_not_shadow_the_text_field() {
        // Some servers return a segments array of empty entries; the text is
        // the only usable content.
        let json =
            r#"{"text": "はじめまして", "segments": [{"start": 0.0, "end": 1.0, "text": "  "}]}"#
                .as_bytes();
        let segments = parse_transcription(json, 8_000).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "はじめまして");
        assert_eq!(segments[0].end, 8.0);
    }

    #[test]
    fn response_without_text_or_segments_yields_nothing() {
        let segments = parse_transcription(br#"{"text": "   "}"#, 15_000).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn malformed_json_is_an_error() {
        assert!(parse_transcription(b"not json", 15_000).is_err());
    }

    /// End-to-end check against a live OpenAI-compatible server, exercising the
    /// real multipart request (including `timestamp_granularities[]`). Ignored
    /// by default: it needs actual speech, since a silent WAV transcribes to
    /// nothing. Run manually with the gateway on :8000:
    ///   MPV_STT_PLUGIN_RS_LIVE_AUDIO=/path/to/speech.wav \
    ///   cargo test -p mpv_stt_plugin_rs --lib -- --ignored openai_backend_against_live_server
    #[test]
    #[ignore]
    fn openai_backend_against_live_server() {
        let audio = std::env::var("MPV_STT_PLUGIN_RS_LIVE_AUDIO")
            .expect("set MPV_STT_PLUGIN_RS_LIVE_AUDIO to a speech WAV");
        let server = std::env::var("MPV_STT_PLUGIN_RS_LIVE_SERVER")
            .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
        let model = std::env::var("MPV_STT_PLUGIN_RS_LIVE_MODEL")
            .unwrap_or_else(|_| "sensevoice".to_string());

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("chunk");
        let mut backend = OpenAiBackend::new(SttOpenAiConfig {
            server_addr: server,
            model,
            // Match the shipped config: a language hint keeps the check
            // deterministic for the Japanese/Chinese clips it is run on.
            language: Some("ja".to_string()),
            ..Default::default()
        })
        .unwrap();

        backend
            .transcribe(audio.as_str(), prefix.to_str().unwrap(), 15_000)
            .unwrap();

        let srt = std::fs::read_to_string(prefix.with_extension("srt")).unwrap();
        assert!(srt.contains("-->"), "expected SRT cues, got: {srt}");
    }
}
