use mpv_client::{Event, Handle, mpv_handle};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};
use tempfile::TempDir;
use tracing::{Span, debug, error, info, trace, warn};
use url::Url;

use crate::audio::AudioExtractor;
use crate::common::MpvSttError;
use crate::config::{Config, SttRetryConfig, SttRetryStrategy};
use crate::logging::{self, LogSettings};
use crate::srt::SrtFile;
use crate::stt::{SttBackend, SttDeviceNotice, SttRunner};
use crate::subtitle_manager::SubtitleManager;
use crate::translate::{
    AsyncTranslationQueue, TranslationOutcome, TranslationTask, TranslatorConfig,
};

const SUBTITLE_TIMELINE_VERSION: u32 = 2;
const SUBTITLE_CACHE_SCHEMA_VERSION: u32 = 1;
const TRANSLATION_RETRY_BASE_SECS: u64 = 2;
const RETRY_MAX_DELAY_SECS: u64 = 60;
const TRANSLATION_SCAN_INTERVAL: Duration = Duration::from_secs(1);

struct TempPaths {
    _dir: TempDir,
    tmp_wav: PathBuf,
    tmp_sub: PathBuf,
    tmp_cache: PathBuf,
}

impl TempPaths {
    fn new() -> crate::common::Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix("mpv_stt_plugin_rs_")
            .tempdir()?;

        Ok(Self {
            tmp_wav: dir.path().join("audio.wav"),
            // `tmp_sub` is a prefix; intermediate files are derived via `format!("{}_append...", tmp_sub.display())`
            // and the main subtitle file is `tmp_sub.with_extension("srt")`.
            tmp_sub: dir.path().join("subs"),
            tmp_cache: dir.path().join("cache.mkv"),
            _dir: dir,
        })
    }

    fn cleanup_intermediate_subs(&self) {
        let _ = std::fs::remove_file(format!("{}_append.srt", self.tmp_sub.display()));
        let _ = std::fs::remove_file(format!("{}_append_offset.srt", self.tmp_sub.display()));
        let _ = std::fs::remove_file(format!("{}_append_offset_bi.srt", self.tmp_sub.display()));
    }

    fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.tmp_wav);
        let _ = std::fs::remove_file(self.tmp_sub.with_extension("srt"));
        self.cleanup_intermediate_subs();
        let _ = std::fs::remove_file(&self.tmp_cache);
    }
}

#[derive(Clone)]
struct CachePaths {
    subtitle_path: PathBuf,
    manifest_path: PathBuf,
    media_identity_hash: String,
    media_fingerprint: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranslationCacheEntry {
    start_ms: u32,
    original: String,
    translated: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct CacheManifest {
    #[serde(default)]
    schema_version: u32,
    /// Bump when subtitle timestamps can no longer be reused safely.
    #[serde(default)]
    timeline_version: u32,
    #[serde(default)]
    media_identity_hash: String,
    #[serde(default)]
    media_fingerprint: Option<String>,
    #[serde(default)]
    chunk_size_ms: u64,
    #[serde(default)]
    processed_chunks: Vec<u64>,
    #[serde(default)]
    translations: Vec<TranslationCacheEntry>,
}

enum ProcessingMode {
    Network,
    Local {
        media_path: String,
        file_length_ms: u64,
        subtitle_path: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlCommand {
    ToggleStt,
    ToggleTranslate,
    ClearCache,
}

impl ControlCommand {
    fn from_client_message(args: &[&str]) -> Option<Self> {
        // `script-message-to` normally puts the payload command in args[0].
        // Accept args[1] as well for compatibility with older mpv/IINA builds
        // and user input.conf entries that included an extra routing token.
        args.iter().take(2).find_map(|arg| match *arg {
            "toggle-stt" => Some(Self::ToggleStt),
            "toggle-translate" => Some(Self::ToggleTranslate),
            "clear-cache" => Some(Self::ClearCache),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeekTarget {
    position_ms: u64,
    forward: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CacheRange {
    start_sec: f64,
    end_sec: f64,
}

fn cache_ranges_from_node(node: &mpv_client::Node) -> Option<Vec<CacheRange>> {
    let mpv_client::Node::Map(state) = node else {
        return None;
    };
    let Some(mpv_client::Node::Array(ranges)) = state.get("seekable-ranges") else {
        return None;
    };

    let to_seconds = |value: &mpv_client::Node| match value {
        mpv_client::Node::Double(value) => Some(*value),
        mpv_client::Node::Int(value) => Some(*value as f64),
        _ => None,
    };

    Some(
        ranges
            .iter()
            .filter_map(|range| {
                let mpv_client::Node::Map(range) = range else {
                    return None;
                };
                let (start, end) = (range.get("start")?, range.get("end")?);
                let (start_sec, end_sec) = (to_seconds(start)?, to_seconds(end)?);
                (start_sec.is_finite() && end_sec.is_finite() && end_sec > start_sec)
                    .then_some(CacheRange { start_sec, end_sec })
            })
            .collect(),
    )
}

fn cache_ranges_have_single_cover(ranges: &[CacheRange], start_sec: f64, end_sec: f64) -> bool {
    if !start_sec.is_finite() || !end_sec.is_finite() || end_sec <= start_sec {
        return false;
    }

    let mut intersecting = 0usize;
    let mut request_is_covered = false;
    for range in ranges {
        if range.end_sec <= start_sec || range.start_sec >= end_sec {
            continue;
        }
        intersecting += 1;
        request_is_covered = range.start_sec <= start_sec && range.end_sec >= end_sec;
    }

    intersecting == 1 && request_is_covered
}

struct NetworkCacheRecovery<'a> {
    cursor_ms: u64,
    playback_ms: Option<u64>,
    chunk_ms: u64,
    cache_end_ms: Option<u64>,
    ranges: Option<&'a [CacheRange]>,
    processed_chunks: &'a HashSet<u64>,
    retry_pending: bool,
}

fn next_covered_chunk_after_playback(recovery: NetworkCacheRecovery<'_>) -> Option<u64> {
    if recovery.retry_pending {
        return None;
    }
    let (Some(playback_ms), Some(cache_end_ms), Some(ranges)) =
        (recovery.playback_ms, recovery.cache_end_ms, recovery.ranges)
    else {
        return None;
    };

    let chunk_ms = recovery.chunk_ms.max(1);
    let current_end_ms = recovery.cursor_ms.saturating_add(chunk_ms);
    if playback_ms < current_end_ms
        || cache_ranges_have_single_cover(
            ranges,
            recovery.cursor_ms as f64 / 1000.0,
            current_end_ms as f64 / 1000.0,
        )
    {
        return None;
    }

    let playback_chunk_ms = playback_ms - playback_ms % chunk_ms;
    let mut candidate_ms = playback_chunk_ms.max(current_end_ms);
    let remainder = candidate_ms % chunk_ms;
    if remainder != 0 {
        candidate_ms = candidate_ms.saturating_add(chunk_ms - remainder);
    }

    loop {
        let end_ms = candidate_ms.saturating_add(chunk_ms);
        if end_ms <= candidate_ms || end_ms > cache_end_ms {
            return None;
        }
        if !recovery.processed_chunks.contains(&candidate_ms)
            && cache_ranges_have_single_cover(
                ranges,
                candidate_ms as f64 / 1000.0,
                end_ms as f64 / 1000.0,
            )
        {
            return Some(candidate_ms);
        }
        candidate_ms = end_ms;
    }
}

fn detect_seek_target(
    playback_pos_ms: u64,
    last_pos_ms: Option<u64>,
    elapsed_ms: Option<u64>,
    current_pos_ms: u64,
    chunk_size_ms: u64,
    explicit_seek: bool,
) -> Option<SeekTarget> {
    let chunk_size_ms = chunk_size_ms.max(1);
    let comparison_pos_ms = last_pos_ms.unwrap_or(current_pos_ms);
    let delta_ms = playback_pos_ms.abs_diff(comparison_pos_ms);
    if delta_ms == 0 {
        return None;
    }

    let seek_threshold_ms = std::cmp::max(5_000, chunk_size_ms);
    if !explicit_seek {
        if delta_ms < seek_threshold_ms {
            return None;
        }
        if elapsed_ms.is_some_and(|elapsed| delta_ms <= elapsed.saturating_add(seek_threshold_ms)) {
            return None;
        }
    }

    Some(SeekTarget {
        position_ms: playback_pos_ms - playback_pos_ms % chunk_size_ms,
        forward: last_pos_ms
            .map(|last| playback_pos_ms > last)
            .unwrap_or(playback_pos_ms >= current_pos_ms),
    })
}

const KEY_BINDINGS: [(&str, &str); 3] = [
    ("Ctrl+Shift+S", "toggle-stt"),
    ("Ctrl+Shift+T", "toggle-translate"),
    ("Ctrl+Shift+C", "clear-cache"),
];

fn key_binding_section(target: &str) -> String {
    KEY_BINDINGS
        .iter()
        .map(|(key, command)| format!("{key} script-message-to {target} {command}"))
        .collect::<Vec<_>>()
        .join("\n")
}

struct TranscriptionJobInput {
    media_path: String,
    chunk_start_ms: u64,
    audio_start_ms: u64,
    /// Network `dump-cache` files may start before the requested chunk because
    /// mpv seeks to a preceding keyframe and rebases the dump timestamps.
    align_audio_to_chunk_end: bool,
    duration_ms: u64,
    wav_path: PathBuf,
    output_prefix: PathBuf,
    /// The chunk span the job was created in. Carried across the thread
    /// boundary and entered on the worker, so extraction and STT records attach
    /// to the chunk they belong to instead of floating free. `Span` is
    /// `Send + Sync`; note the span is *entered*, never *held* (a held `Entered`
    /// guard would not be `Send`).
    span: Span,
}

struct TranscriptionJob {
    generation: u64,
    stt_generation: u64,
    media_path: String,
    chunk_start_ms: u64,
    audio_start_ms: u64,
    align_audio_to_chunk_end: bool,
    duration_ms: u64,
    wav_path: PathBuf,
    output_prefix: PathBuf,
    span: Span,
}

struct TranscriptionWorkerResult {
    generation: u64,
    result: crate::common::Result<()>,
    device_notice: Option<SttDeviceNotice>,
}

struct TranscriptionWorker {
    job_sender: Sender<Option<TranscriptionJob>>,
    result_receiver: Receiver<TranscriptionWorkerResult>,
    worker_handle: Option<thread::JoinHandle<()>>,
    generation: Arc<AtomicU64>,
    audio_canceller: AudioExtractor,
    stt_cancel_generation: Arc<AtomicU64>,
}

impl TranscriptionWorker {
    fn new(audio_extractor: AudioExtractor, mut stt_runner: SttRunner) -> Self {
        let (job_sender, job_receiver) = channel::<Option<TranscriptionJob>>();
        let (result_sender, result_receiver) = channel::<TranscriptionWorkerResult>();
        let generation = Arc::new(AtomicU64::new(0));
        let worker_generation = Arc::clone(&generation);
        let worker_audio = audio_extractor.clone();
        let stt_cancel_generation = stt_runner.cancellation_generation();

        let worker_handle = thread::Builder::new()
            .name("mpv-stt-transcription".to_string())
            .spawn(move || {
                while let Ok(Some(job)) = job_receiver.recv() {
                    // Everything below runs inside the chunk span, so the
                    // extractor and the STT backend log against the chunk that
                    // caused them — even though this is a different thread from
                    // the one that created the span.
                    let _chunk = job.span.clone().entered();
                    if worker_generation.load(Ordering::Acquire) != job.generation {
                        trace!("Superseded chunk never started");
                        continue;
                    }
                    debug!(
                        media = %job.media_path,
                        chunk_start_ms = job.chunk_start_ms,
                        requested_audio_start_ms = job.audio_start_ms,
                        align_audio_to_chunk_end = job.align_audio_to_chunk_end,
                        "transcription job started"
                    );

                    let result = (|| {
                        let (extraction_start_ms, audio_end_ms) = if job.align_audio_to_chunk_end {
                            let audio_end_ms =
                                worker_audio.audio_end_relative_ms(job.media_path.as_str())?;
                            let extraction_start_ms =
                                audio_end_ms.checked_sub(job.duration_ms).ok_or_else(|| {
                                    MpvSttError::AudioExtractionFailed(format!(
                                        "dumped audio covers only {audio_end_ms}ms, shorter than the requested {}ms chunk",
                                        job.duration_ms
                                    ))
                                })?;
                            (extraction_start_ms, Some(audio_end_ms))
                        } else {
                            (job.audio_start_ms, None)
                        };
                        debug!(
                            extraction_start_ms,
                            audio_end_ms = ?audio_end_ms,
                            duration_ms = job.duration_ms,
                            "resolved the audio window inside the input"
                        );
                        worker_audio.extract_audio_segment(
                            job.media_path.as_str(),
                            job.wav_path.to_str().unwrap_or_default(),
                            extraction_start_ms,
                            job.duration_ms,
                        )?;
                        if worker_generation.load(Ordering::Acquire) != job.generation {
                            return Err(MpvSttError::SttCancelled);
                        }
                        stt_runner.transcribe_with_generation(
                            job.wav_path.to_str().unwrap_or_default(),
                            job.output_prefix.to_str().unwrap_or_default(),
                            job.duration_ms,
                            job.stt_generation,
                        )
                    })();
                    let device_notice = stt_runner.take_device_notice();

                    let _ = result_sender.send(TranscriptionWorkerResult {
                        generation: job.generation,
                        result,
                        device_notice,
                    });
                }
            })
            .expect("failed to spawn transcription worker");

        Self {
            job_sender,
            result_receiver,
            worker_handle: Some(worker_handle),
            generation,
            audio_canceller: audio_extractor,
            stt_cancel_generation,
        }
    }

    fn submit(&self, input: TranscriptionJobInput) -> Option<u64> {
        let generation = self.generation.load(Ordering::Acquire);
        let stt_generation = self.stt_cancel_generation.load(Ordering::Acquire);
        let job = TranscriptionJob {
            generation,
            stt_generation,
            media_path: input.media_path,
            chunk_start_ms: input.chunk_start_ms,
            audio_start_ms: input.audio_start_ms,
            align_audio_to_chunk_end: input.align_audio_to_chunk_end,
            duration_ms: input.duration_ms,
            wav_path: input.wav_path,
            output_prefix: input.output_prefix,
            span: input.span,
        };
        self.job_sender.send(Some(job)).ok().map(|()| generation)
    }

    fn try_recv(&self) -> Option<TranscriptionWorkerResult> {
        let current = self.generation.load(Ordering::Acquire);
        while let Ok(result) = self.result_receiver.try_recv() {
            if result.generation == current {
                return Some(result);
            }
            trace!(
                stale_gen = result.generation,
                current_gen = current,
                "dropping a stale transcription result"
            );
        }
        None
    }

    /// Current generation: the id every job and result of this attempt carries.
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn cancel_inflight(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.audio_canceller.cancel_inflight();
        self.stt_cancel_generation.fetch_add(1, Ordering::AcqRel);
    }

    fn shutdown(&mut self) {
        if self.worker_handle.is_none() {
            return;
        }
        self.cancel_inflight();
        let _ = self.job_sender.send(None);
        if let Some(handle) = self.worker_handle.take() {
            if handle.join().is_err() {
                warn!("Transcription worker panicked during shutdown");
            }
        }
    }
}

impl Drop for TranscriptionWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct PendingTranscription {
    generation: u64,
    start_ms: u64,
    duration_ms: u64,
    subtitle_path: Option<PathBuf>,
    /// The chunk span this job was submitted in; re-entered when the result
    /// lands so the merge, translation hand-off and SRT write are logged as part
    /// of the same chunk.
    span: Span,
}

struct ChunkRetryState {
    start_ms: u64,
    attempts: u32,
    retry_at: Instant,
}

struct TranslationRetryState {
    attempts: u32,
    retry_at: Instant,
}

struct PluginState {
    config: Config,
    stt_retry: SttRetryConfig,
    paths: TempPaths,
    transcription_worker: TranscriptionWorker,
    pending_transcription: Option<PendingTranscription>,
    transcription_retry: Option<ChunkRetryState>,
    async_translation_queue: Option<AsyncTranslationQueue>,
    subtitle_manager: SubtitleManager,
    translation_cache: HashMap<u32, (String, String)>,
    /// Cues transcribed during this process are known to contain original
    /// speech only, even if the recognizer placed the text on multiple lines.
    known_original_cues: HashSet<u32>,
    /// Cues currently waiting for their translation retry cooldown to expire.
    failed_translations: HashSet<u32>,
    /// Backoff state for cues whose last translation batch failed.
    translation_retries: HashMap<u32, TranslationRetryState>,
    /// Cues already submitted to the translation worker.
    pending_translations: HashSet<u32>,
    last_translation_scan: Option<Instant>,
    /// Whether the current failure burst has already produced a translation
    /// OSD, avoiding one message per failed cue.
    translation_failure_reported: bool,
    processed_chunks: HashSet<u64>,
    subtitle_cache: Option<CachePaths>,

    running: bool,
    shutting_down: bool,
    translate_enabled: bool, // Ctrl+Shift+t toggles whether new STT output gets translated
    subs_loaded: bool,
    current_pos_ms: u64,
    last_playback_pos_ms: Option<u64>,
    last_playback_instant: Option<Instant>,
    seek_pending: bool,
    chunk_dur: u64,
    mode: Option<ProcessingMode>,
    pending_auto_start: bool, // Delayed auto-start after file loads
    file_loaded: bool,        // Track if file is ready
    transcription_complete: bool,

    /// Monotonic id for the current media session, used only for log
    /// correlation. `0` means no session has started yet.
    session_id: u64,
    /// Chunks submitted in the current session, so a chunk can be referred to
    /// by number as well as by timestamp.
    chunk_seq: u64,
    /// Span covering the current session. Held (not only entered) so events
    /// recorded from this state stay attached to it.
    session_span: Option<Span>,
    /// When the current session's transcription started, for the completion
    /// summary.
    session_started: Option<Instant>,
    /// Total chunk failures in the current session, reported when it ends.
    session_failures: u64,
    cached_subtitle_path: Option<PathBuf>,
}

impl PluginState {
    fn new(config: Config) -> crate::common::Result<Self> {
        let chunk_dur = config.chunk.local_ms;
        let audio_extractor = AudioExtractor::default()
            .with_ffmpeg_timeout(config.timeout.ffmpeg_ms)
            .with_ffprobe_timeout(config.timeout.ffprobe_ms);

        let active_stt_source = config.stt.active_source_name().map_err(|e| {
            MpvSttError::SttFailed(format!("{e}; protocols: openai, ferrum, cloudflare"))
        })?;
        let stt_retry = config
            .stt
            .sources
            .get(active_stt_source)
            .expect("the resolved STT source is declared")
            .retry
            .clone();

        // Initialize the STT backend of the source named by `[stt] source`.
        // `from_config` resolves the name and selects the backend compiled for
        // the source's protocol.
        let stt_runner = SttRunner::from_config(&config.stt)?;
        let transcription_worker = TranscriptionWorker::new(audio_extractor, stt_runner);
        let paths = TempPaths::new()?;

        // Initialize async translation queue (always enabled)
        let async_translation_queue = Some(AsyncTranslationQueue::new(
            Self::build_translator_config(&config)?,
        ));

        Ok(Self {
            chunk_dur,
            config,
            stt_retry,
            paths,
            transcription_worker,
            pending_transcription: None,
            transcription_retry: None,
            async_translation_queue,
            subtitle_manager: SubtitleManager::new(),
            translation_cache: HashMap::new(),
            known_original_cues: HashSet::new(),
            failed_translations: HashSet::new(),
            translation_retries: HashMap::new(),
            pending_translations: HashSet::new(),
            last_translation_scan: None,
            translation_failure_reported: false,
            processed_chunks: HashSet::new(),
            subtitle_cache: None,
            running: false,
            shutting_down: false,
            translate_enabled: true,
            subs_loaded: false,
            current_pos_ms: 0,
            last_playback_pos_ms: None,
            last_playback_instant: None,
            seek_pending: false,
            mode: None,
            pending_auto_start: false,
            file_loaded: false,
            transcription_complete: false,
            session_id: 0,
            chunk_seq: 0,
            session_span: None,
            session_started: None,
            session_failures: 0,
            cached_subtitle_path: None,
        })
    }

    fn build_translator_config(config: &Config) -> crate::common::Result<TranslatorConfig> {
        // Every declared source is resolved up front, so a typo in one of them
        // fails at startup with a message naming the offending source rather
        // than on the first cue that happens to use it.
        let mut translator = TranslatorConfig::new(
            config.translate.from_lang.clone(),
            config.translate.to_lang.clone(),
        )
        .with_timeout_ms(config.timeout.translate_ms)
        .with_concurrency(config.translate.concurrency)
        .with_source(config.translate.source.clone());

        for (name, declared) in &config.translate.sources {
            translator = translator.with_source_config(name, declared)?;
        }

        // The active selector is resolved too: a name that matches nothing is
        // a configuration error, not something to discover mid-session.
        if translator.source != crate::translate::AUTO_SOURCE {
            translator.active_source()?;
        }

        Ok(translator)
    }

    fn local_chunk_size(&self) -> u64 {
        self.config.chunk.local_ms.max(1)
    }

    fn network_chunk_size(&self) -> u64 {
        self.config.chunk.network_ms.max(1)
    }

    fn active_chunk_size(&self) -> u64 {
        match self.mode {
            Some(ProcessingMode::Network) => self.network_chunk_size(),
            Some(ProcessingMode::Local { .. }) => self.local_chunk_size(),
            None => self.local_chunk_size(),
        }
    }

    fn cancel_translation_inflight(&mut self) {
        if let Some(queue) = self.async_translation_queue.as_ref() {
            queue.cancel_inflight();
        }
        self.pending_translations.clear();
        self.last_translation_scan = None;
    }

    /// Whether this cue is currently waiting for its retry cooldown.
    fn translation_failed(&self, start_ms: u32) -> bool {
        self.failed_translations.contains(&start_ms)
    }

    /// Allow every cue to be translated again: the user asked for translation
    /// explicitly (toggle), or the whole cache was dropped.
    fn reset_translation_failures(&mut self) {
        if !self.failed_translations.is_empty() {
            debug!(
                cues = self.failed_translations.len(),
                "clearing translation retry state; every cue may be retried"
            );
        }
        self.failed_translations.clear();
        self.translation_retries.clear();
        self.translation_failure_reported = false;
        self.last_translation_scan = None;
    }

    fn retry_delay_secs(attempts: u32, base_secs: u64) -> u64 {
        let shift = attempts.saturating_sub(1).min(6);
        base_secs
            .saturating_mul(1_u64 << shift)
            .min(RETRY_MAX_DELAY_SECS)
    }

    fn stt_retry_delay_secs(&self, attempts: u32, rate_limited: bool) -> u64 {
        match self.stt_retry.strategy {
            SttRetryStrategy::Fixed => self.stt_retry.interval_secs.max(1),
            SttRetryStrategy::Exponential => {
                let base_secs = if rate_limited {
                    self.stt_retry.rate_limit_interval_secs
                } else {
                    self.stt_retry.interval_secs
                };
                Self::retry_delay_secs(attempts, base_secs.max(1))
            }
        }
    }

    fn defer_transcription_retry(&mut self, start_ms: u64, reason: &str) -> Duration {
        self.defer_transcription_retry_with_rate_limit(start_ms, reason, false)
    }

    fn defer_transcription_retry_with_rate_limit(
        &mut self,
        start_ms: u64,
        reason: &str,
        rate_limited: bool,
    ) -> Duration {
        let previous_attempts = self
            .transcription_retry
            .as_ref()
            .filter(|retry| retry.start_ms == start_ms)
            .map(|retry| retry.attempts)
            .unwrap_or(0);
        let attempts = previous_attempts.saturating_add(1);
        let delay = Duration::from_secs(self.stt_retry_delay_secs(attempts, rate_limited));
        self.transcription_retry = Some(ChunkRetryState {
            start_ms,
            attempts,
            retry_at: Instant::now() + delay,
        });
        self.session_failures = self.session_failures.saturating_add(1);
        debug!(
            start_ms,
            attempts,
            retry_in_secs = delay.as_secs(),
            reason,
            "transcription chunk failed; scheduling an automatic retry"
        );
        delay
    }

    fn transcription_retry_waiting(&mut self) -> bool {
        let Some(retry) = self.transcription_retry.as_ref() else {
            return false;
        };
        if retry.start_ms != self.current_pos_ms {
            self.transcription_retry = None;
            return false;
        }

        let remaining = retry.retry_at.saturating_duration_since(Instant::now());
        if !remaining.is_zero() {
            trace!(
                start_ms = retry.start_ms,
                retry_in_ms = remaining.as_millis() as u64,
                "waiting for the transcription retry backoff"
            );
            return true;
        }
        false
    }

    fn clear_transcription_retry(&mut self, start_ms: u64) {
        if self
            .transcription_retry
            .as_ref()
            .is_some_and(|retry| retry.start_ms == start_ms)
        {
            self.transcription_retry = None;
        }
    }

    /// Periodically inspect every known cue, including subtitles restored from
    /// cache, and submit any untranslated cue whose retry delay has elapsed.
    fn enqueue_missing_translations(&mut self, force: bool) {
        if !self.translate_enabled {
            return;
        }
        let now = Instant::now();
        if !force
            && self
                .last_translation_scan
                .is_some_and(|last| now.duration_since(last) < TRANSLATION_SCAN_INTERVAL)
        {
            return;
        }
        self.last_translation_scan = Some(now);

        let entries = self.subtitle_manager.all_entries();

        if entries.is_empty() {
            return;
        }

        let mut pending_tasks = Vec::new();
        let mut already_translated = 0usize;
        let mut waiting_retry = 0usize;
        let retry_chunk_end_ms = self.transcription_retry.as_ref().map(|retry| {
            retry
                .start_ms
                .saturating_add(self.active_chunk_size())
                .min(u64::from(u32::MAX)) as u32
        });

        for (start_ms, entry) in entries {
            let original = entry.text.trim();
            if original.is_empty() {
                continue;
            }

            // Do not dispatch translations from a chunk whose transcription
            // result has not yet been successfully merged and saved.
            if self.transcription_retry.as_ref().is_some_and(|retry| {
                u64::from(start_ms) >= retry.start_ms
                    && retry_chunk_end_ms.is_some_and(|end_ms| start_ms < end_ms)
            }) {
                continue;
            }

            if self.pending_translations.contains(&start_ms) {
                continue;
            }

            if self.translation_failed(start_ms) {
                let retry_due = self
                    .translation_retries
                    .get(&start_ms)
                    .is_none_or(|retry| retry.retry_at <= now);
                if !retry_due {
                    waiting_retry += 1;
                    continue;
                }
                self.failed_translations.remove(&start_ms);
            }

            let cached_translation = self
                .translation_cache
                .get(&start_ms)
                .map(|(_, translated)| translated.clone());
            if let Some(translated) = cached_translation {
                if !translated.trim().is_empty() {
                    self.subtitle_manager
                        .update_translation(start_ms, &translated);
                    self.failed_translations.remove(&start_ms);
                    self.translation_retries.remove(&start_ms);
                    already_translated += 1;
                    continue;
                }
            }

            if SubtitleManager::text_has_translation(&entry.text) {
                if !self.known_original_cues.contains(&start_ms) {
                    self.failed_translations.remove(&start_ms);
                    self.translation_retries.remove(&start_ms);
                    already_translated += 1;
                    continue;
                }
            }

            pending_tasks.push(TranslationTask {
                start_ms,
                text: entry.text.clone(),
            });
        }

        if !pending_tasks.is_empty() {
            trace!(
                queued = pending_tasks.len(),
                already_translated, waiting_retry, "queueing missing translations"
            );
            if self.async_translation_queue.is_none() {
                return;
            }
            for task in pending_tasks {
                let start_ms = task.start_ms;
                let submitted = self
                    .async_translation_queue
                    .as_ref()
                    .is_some_and(|queue| queue.submit(task));
                if submitted {
                    self.pending_translations.insert(start_ms);
                } else {
                    self.defer_translation_retry(start_ms, "translation worker is unavailable");
                }
            }
        }
    }

    fn defer_translation_retry(&mut self, start_ms: u32, reason: &str) -> Duration {
        self.defer_translation_retry_with_base(start_ms, reason, TRANSLATION_RETRY_BASE_SECS)
    }

    fn defer_translation_retry_with_base(
        &mut self,
        start_ms: u32,
        reason: &str,
        base_secs: u64,
    ) -> Duration {
        self.pending_translations.remove(&start_ms);
        let attempts = self
            .translation_retries
            .get(&start_ms)
            .map(|retry| retry.attempts)
            .unwrap_or(0)
            .saturating_add(1);
        let delay = Duration::from_secs(Self::retry_delay_secs(attempts, base_secs));
        self.failed_translations.insert(start_ms);
        self.translation_retries.insert(
            start_ms,
            TranslationRetryState {
                attempts,
                retry_at: Instant::now() + delay,
            },
        );
        debug!(
            start_ms,
            attempts,
            retry_in_secs = delay.as_secs(),
            reason,
            "translation failed; scheduling an automatic retry"
        );
        delay
    }

    fn toggle_stt(&mut self, client: &mut Handle) {
        if self.running {
            info!(display = %logging::osd_line("STT: Off"), "disabling STT");
            self.stop_transcription();
        } else {
            info!(display = %logging::osd_line("STT: On"), "enabling STT");
            if self.mode.is_some() {
                self.stop_transcription();
            }
            self.running = true;
            self.start_transcription(client);
        }
    }

    fn toggle_translate(&mut self, _client: &mut Handle) {
        self.translate_enabled = !self.translate_enabled;
        let msg = if self.translate_enabled {
            "Translate: On"
        } else {
            "Translate: Off (new subtitles stay as original)"
        };
        info!(
            enabled = self.translate_enabled,
            display = %logging::osd_line(msg),
            "translation toggled"
        );
        if self.translate_enabled {
            // Turning translation back on is an explicit retry: forget earlier
            // backoff records and offer all untranslated cues to the backend,
            // which is how the user recovers after starting the gateway.
            self.reset_translation_failures();
            self.enqueue_missing_translations(true);
        }
    }

    fn remove_plugin_cache_files(paths: &[CachePaths]) -> usize {
        let mut removed = 0usize;
        for paths in paths {
            for path in [&paths.subtitle_path, &paths.manifest_path] {
                if !path.exists() {
                    continue;
                }
                match fs::remove_file(path) {
                    Ok(()) => removed += 1,
                    Err(error) => warn!(
                        error = %error,
                        cause = %logging::err_chain(&error),
                        path = %path.display(),
                        "cannot remove a plugin subtitle cache file"
                    ),
                }
            }
        }
        removed
    }

    /// Delete only the current media's plugin-owned subtitle cache, leaving
    /// ordinary `.srt` sidecars untouched.
    fn clear_cache(&mut self, client: &mut Handle) {
        let is_network = matches!(&self.mode, Some(ProcessingMode::Network))
            || self.detect_network_stream(client);
        let media_id = is_network
            .then(|| Self::media_id_for_cache(client))
            .flatten();
        let current_paths = self.subtitle_cache.clone().or_else(|| {
            if is_network {
                media_id
                    .as_deref()
                    .and_then(|id| self.cache_paths_for_network_media(id))
            } else {
                client
                    .get_property::<String>("path")
                    .ok()
                    .and_then(|path| Self::local_cache_paths_for_media_uri(&path))
            }
        });

        let mut paths_to_clear = Vec::new();
        if let Some(paths) = current_paths {
            paths_to_clear.push(paths);
        }
        if is_network
            && let (Some(media_id), Some(root)) = (media_id.as_deref(), Self::cache_root_dir())
        {
            paths_to_clear.push(Self::legacy_cache_paths_for_network_media_at(
                &root, media_id,
            ));
        }

        let removed = Self::remove_plugin_cache_files(&paths_to_clear);

        // Drop in-memory cues as well, so the next result cannot rewrite the
        // cache with subtitles the user just cleared.
        let chunk_entries = self.translation_cache.len();
        self.subtitle_manager.clear();
        self.translation_cache.clear();
        self.known_original_cues.clear();
        self.pending_translations.clear();
        self.cancel_translation_inflight();
        self.reset_translation_failures();
        self.processed_chunks.clear();
        self.transcription_retry = None;
        self.cached_subtitle_path = None;
        // Keep the loaded-track state: mpv may still hold the external subtitle
        // track after its file is removed, so the next save must reload it
        // rather than adding a duplicate track.

        info!(
            removed_files = removed,
            dropped_translations = chunk_entries,
            display = %logging::osd_line(&format!(
                "字幕缓存已清除: 删除 {removed} 个文件, 内存缓存 {chunk_entries} 条"
            )),
            "cleared the subtitle cache"
        );
    }

    fn schedule_transcription(
        &mut self,
        media_path: String,
        audio_start_ms: u64,
        align_audio_to_chunk_end: bool,
        duration_ms: u64,
        subtitle_path: Option<PathBuf>,
    ) -> bool {
        if self.pending_transcription.is_some() {
            trace!(
                start_ms = self.current_pos_ms,
                "chunk not submitted: the previous one is still running"
            );
            return false;
        }

        let output_prefix = PathBuf::from(format!("{}_append", self.paths.tmp_sub.display()));
        // An earlier attempt may have left a partial result behind. Never let
        // the next attempt merge an SRT file it did not produce itself.
        self.paths.cleanup_intermediate_subs();
        // The chunk span is created here, on the event thread, and travels with
        // the job so the worker's extractor/STT logs land inside it.
        let chunk_start_ms = self.current_pos_ms;
        let span = logging::chunk_span(
            self.session_id,
            self.chunk_seq,
            chunk_start_ms,
            duration_ms,
            self.transcription_worker.generation(),
        );
        let Some(generation) = self.transcription_worker.submit(TranscriptionJobInput {
            media_path,
            chunk_start_ms,
            audio_start_ms,
            align_audio_to_chunk_end,
            duration_ms,
            wav_path: self.paths.tmp_wav.clone(),
            output_prefix,
            span: span.clone(),
        }) else {
            let start_ms = self.current_pos_ms;
            let delay =
                self.defer_transcription_retry(start_ms, "transcription worker is unavailable");
            error!(
                start_ms,
                retry_in_secs = delay.as_secs(),
                display = %logging::osd_line(&format!(
                    "STT worker unavailable; retrying in {}s",
                    delay.as_secs()
                )),
                "transcription worker is unavailable; keeping the session active"
            );
            return false;
        };
        self.chunk_seq += 1;

        debug!(
            start_ms = chunk_start_ms,
            audio_start_ms,
            align_audio_to_chunk_end,
            dur_ms = duration_ms,
            gen = generation,
            "chunk submitted"
        );
        self.pending_transcription = Some(PendingTranscription {
            generation,
            start_ms: self.current_pos_ms,
            duration_ms,
            subtitle_path,
            span,
        });
        true
    }

    fn poll_transcription(&mut self, client: &mut Handle) {
        let Some(worker_result) = self.transcription_worker.try_recv() else {
            return;
        };
        let Some(pending_ref) = self.pending_transcription.as_ref() else {
            return;
        };
        if worker_result.generation != pending_ref.generation {
            return;
        }
        let Some(pending) = self.pending_transcription.take() else {
            return;
        };

        // Re-enter the chunk span for the second half of the round trip: the
        // merge, translation hand-off and SRT write belong to the same chunk as
        // the extraction and STT that produced them.
        let _chunk = pending.span.clone().entered();

        match worker_result.result {
            Ok(()) => {
                if self.check_seek(client) {
                    debug!("seek detected while the chunk was in flight; dropping its result");
                    self.paths.cleanup_intermediate_subs();
                    return;
                }
                self.current_pos_ms = pending.start_ms;
                if self.apply_transcription_result(
                    client,
                    pending.subtitle_path.as_deref(),
                    pending.start_ms,
                    worker_result.device_notice,
                ) {
                    self.clear_transcription_retry(pending.start_ms);
                    self.current_pos_ms = pending.start_ms.saturating_add(pending.duration_ms);
                    if !self.subs_loaded {
                        let main_srt = pending
                            .subtitle_path
                            .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));
                        let _ = client.command(&["sub-add", main_srt.to_str().unwrap_or_default()]);
                        self.subs_loaded = true;
                    }
                    if self.config.playback.show_progress {
                        let _ = client.command(&[
                            "show-text",
                            &format!("STT: {}", Self::format_progress(self.current_pos_ms)),
                        ]);
                    }
                } else {
                    let delay = self.defer_transcription_retry(
                        pending.start_ms,
                        "could not merge or save the subtitle chunk",
                    );
                    info!(
                        start_ms = pending.start_ms,
                        retry_in_secs = delay.as_secs(),
                        display = %logging::osd_line(&format!(
                            "字幕写入失败，{} 秒后重试",
                            delay.as_secs()
                        )),
                        "could not apply the transcription result; keeping the session active"
                    );
                }
            }
            Err(MpvSttError::SttCancelled | MpvSttError::AudioExtractionCancelled) => {
                debug!("chunk cancelled");
                self.paths.cleanup_intermediate_subs();
            }
            Err(err) => {
                self.current_pos_ms = pending.start_ms;
                if matches!(
                    &err,
                    MpvSttError::AudioExtractionFailed(_)
                        | MpvSttError::ProcessFailed(_)
                        | MpvSttError::ProcessTimeout(_)
                        | MpvSttError::Wav(_)
                ) && matches!(&self.mode, Some(ProcessingMode::Network))
                {
                    // A retained network dump that could not be decoded will
                    // not improve on another attempt; let mpv create a fresh
                    // dump from its cache if the range is still available.
                    let _ = std::fs::remove_file(&self.paths.tmp_cache);
                }
                let rate_limited = matches!(&err, MpvSttError::HttpStatus { status: 429, .. });
                let delay = self.defer_transcription_retry_with_rate_limit(
                    pending.start_ms,
                    &err.to_string(),
                    rate_limited,
                );
                error!(
                    start_ms = pending.start_ms,
                    error = %err,
                    cause = %logging::err_chain(&err),
                    retry_in_secs = delay.as_secs(),
                    display = %logging::osd_line(&format!(
                        "STT failed; retrying in {}s: {err}",
                        delay.as_secs()
                    )),
                    "chunk failed; keeping the session active"
                );
                self.paths.cleanup_intermediate_subs();
            }
        }
    }

    fn start_transcription(&mut self, client: &mut Handle) {
        // A new session: new log identity, fresh counters. Everything logged
        // from here until the session ends carries this span and id.
        self.session_id += 1;
        self.chunk_seq = 0;
        self.session_failures = 0;
        self.session_started = Some(Instant::now());
        self.transcription_complete = false;
        self.transcription_retry = None;
        self.last_translation_scan = None;
        self.subtitle_cache = None;
        self.cached_subtitle_path = None;
        self.subs_loaded = false;

        // Get current position
        let time_pos: f64 = client.get_property("time-pos").unwrap_or(0.0);
        self.current_pos_ms = (time_pos * 1000.0) as u64;
        self.last_playback_pos_ms = Some(self.current_pos_ms);
        self.last_playback_instant = Some(Instant::now());
        self.seek_pending = false;

        // Check if network stream - use multiple detection methods
        let is_network = self.detect_network_stream(client);

        let media = client
            .get_property::<String>("path")
            .unwrap_or_else(|_| "<unknown>".to_string());
        let duration_ms = client
            .get_property::<f64>("duration")
            .map(|d| (d * 1000.0) as u64)
            .unwrap_or(0);
        let mode = if is_network { "network" } else { "local" };
        let session_span =
            logging::session_span(self.session_id, &media, duration_ms, mode).entered();
        self.session_span = Some(session_span.clone());

        info!(
            start_ms = self.current_pos_ms,
            chunk_ms = self.chunk_dur,
            translate = self.translate_enabled,
            "transcription session started"
        );

        if is_network {
            // Network stream mode
            debug!(
                display = %logging::osd_line("STT: Starting network stream transcription..."),
                "network stream detected"
            );

            // Enable caching
            let _ = client.set_property("cache", true);

            // Set demuxer max bytes if configured (for better lookahead caching)
            if let Some(max_bytes) = self.config.network.demuxer_max_bytes {
                debug!(bytes = max_bytes, "setting demuxer-max-bytes");
                let _ = client.set_property("demuxer-max-bytes", max_bytes);
            }

            self.mode = Some(ProcessingMode::Network);
            self.subtitle_cache = None;
            let chunk_size = self.network_chunk_size();
            self.current_pos_ms -= self.current_pos_ms % chunk_size;

            if self.config.playback.save_srt {
                if let Some(media_id) = Self::media_id_for_cache(client) {
                    if let Some(cache_paths) = self.cache_paths_for_network_media(&media_id) {
                        Self::create_cache_parent(&cache_paths);
                        Self::migrate_legacy_network_cache(&media_id, &cache_paths);
                        if cache_paths.subtitle_path.exists()
                            && self.load_cached_subs(&cache_paths, self.network_chunk_size())
                        {
                            let _ = client
                                .command(&["sub-add", cache_paths.subtitle_path.to_str().unwrap()]);
                            self.subs_loaded = true;
                            self.cached_subtitle_path = Some(cache_paths.subtitle_path.clone());
                        }
                        self.subtitle_cache = Some(cache_paths);
                    }
                }
            }

            info!(
                start_ms = self.current_pos_ms,
                "network stream session ready"
            );
        } else {
            // Local file mode
            debug!("local file detected");
            let media_path: Result<String, _> = client.get_property("path");
            let duration: Result<f64, _> = client.get_property("duration");

            if let (Ok(path), Ok(dur)) = (media_path, duration) {
                let file_length_ms = (dur * 1000.0) as u64;

                // Save a namespaced SRT next to local media when the path is
                // filesystem-backed. Opaque URIs remain temporary-only.
                let cache_paths = if self.config.playback.save_srt {
                    Self::local_cache_paths_for_media_uri(&path)
                } else {
                    None
                };
                let subtitle_path = cache_paths
                    .as_ref()
                    .map(|cache| cache.subtitle_path.clone())
                    .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));

                info!(
                    display = %logging::osd_line("STT: Starting local file transcription..."),
                    path = %path,
                    "starting local file transcription"
                );

                // Start from beginning if configured
                let chunk_size = self.local_chunk_size();
                self.current_pos_ms -= self.current_pos_ms % chunk_size;

                self.mode = Some(ProcessingMode::Local {
                    media_path: path.clone(),
                    file_length_ms,
                    subtitle_path: subtitle_path.clone(),
                });
                self.subtitle_cache = None;

                if let Some(cache_paths) = cache_paths {
                    Self::create_cache_parent(&cache_paths);
                    if cache_paths.subtitle_path.exists()
                        && self.load_cached_subs(&cache_paths, self.local_chunk_size())
                    {
                        let _ = client
                            .command(&["sub-add", cache_paths.subtitle_path.to_str().unwrap()]);
                        self.subs_loaded = true;
                        self.cached_subtitle_path = Some(cache_paths.subtitle_path.clone());
                    }
                    self.subtitle_cache = Some(cache_paths);
                }

                // Create initial subtitles if this chunk hasn't been processed.
                if !self.is_chunk_processed(self.current_pos_ms) {
                    let remaining_ms = file_length_ms.saturating_sub(self.current_pos_ms);
                    self.chunk_dur = self.local_chunk_size().min(remaining_ms).max(1);
                    self.schedule_transcription(
                        path.clone(),
                        self.current_pos_ms,
                        false,
                        self.chunk_dur,
                        Some(subtitle_path.clone()),
                    );
                }

                info!(
                    file_length_ms,
                    start_ms = self.current_pos_ms,
                    subtitle_path = %subtitle_path.display(),
                    "local file session ready"
                );
            } else {
                self.running = false;
                self.mode = None;
                warn!(
                    display = %logging::osd_line("STT: Please open a playable media file first"),
                    "cannot start STT: mpv reports no playable media path/duration"
                );
            }
        }
    }

    /// Main processing loop - called on each event loop iteration
    fn tick(&mut self, client: &mut Handle) {
        if self.shutting_down {
            return;
        }

        if self.mode.is_some()
            && (self.running || self.transcription_complete)
            && self.check_seek(client)
        {
            return;
        }

        if !self.running {
            if self.transcription_complete {
                let subtitle_path = match &self.mode {
                    Some(ProcessingMode::Network) => self
                        .subtitle_cache
                        .as_ref()
                        .map(|cache| cache.subtitle_path.clone()),
                    Some(ProcessingMode::Local { subtitle_path, .. }) => {
                        Some(subtitle_path.clone())
                    }
                    None => None,
                };
                self.process_translation_results(client, subtitle_path.as_deref());
                self.enqueue_missing_translations(false);
            }
            return;
        }

        self.poll_transcription(client);
        if !self.running || self.shutting_down {
            return;
        }

        match &self.mode {
            Some(ProcessingMode::Network) => self.tick_network(client),
            Some(ProcessingMode::Local {
                media_path,
                file_length_ms,
                subtitle_path,
            }) => {
                let media_path = media_path.clone();
                let file_length_ms = *file_length_ms;
                let subtitle_path = subtitle_path.clone();
                self.tick_local(client, &media_path, file_length_ms, &subtitle_path);
            }
            None => {}
        }
    }

    fn tick_network(&mut self, client: &mut Handle) {
        let subtitle_path = self
            .subtitle_cache
            .as_ref()
            .map(|cache| cache.subtitle_path.clone());

        // Check for completed translations from async queue
        self.process_translation_results(client, subtitle_path.as_deref());
        self.enqueue_missing_translations(false);

        if self.transcription_retry_waiting() {
            return;
        }
        let retry_dump_available = self
            .transcription_retry
            .as_ref()
            .is_some_and(|retry| retry.start_ms == self.current_pos_ms)
            && std::fs::metadata(&self.paths.tmp_cache).is_ok_and(|metadata| metadata.len() > 0);

        // Get cache end time
        let cache_end_sec: Option<f64> = client.get_property("demuxer-cache-time").ok();
        if cache_end_sec.is_none() && !retry_dump_available {
            trace!("demuxer cache has not reported a time yet");
            return; // Cache not ready yet
        }
        let cache_end_ms = cache_end_sec
            .map(|cache_end| (cache_end * 1000.0) as u64)
            .unwrap_or_else(|| {
                self.current_pos_ms
                    .saturating_add(self.network_chunk_size())
            });
        let chunk_ms = self.network_chunk_size();

        // Catch-up mode: check if we're too far behind playback (always enabled)
        // Look-ahead processing for network streams (always enabled)
        // Check how far ahead playback is from processing
        if let Some(playback_pos_ms) = self.last_playback_pos_ms {
            let _ahead = if self.current_pos_ms > playback_pos_ms {
                self.current_pos_ms - playback_pos_ms
            } else {
                0
            };

            // No lookahead limit; we rely on cache availability below
        }

        if self.pending_transcription.is_some() {
            return;
        }

        while self.is_chunk_processed(self.current_pos_ms) {
            self.current_pos_ms = self.current_pos_ms.saturating_add(chunk_ms);
        }

        let mut chunk_end_ms = self.current_pos_ms.saturating_add(chunk_ms);
        if !retry_dump_available {
            let cache_ranges = Self::demuxer_cache_ranges(client);
            let current_chunk_covered = cache_ranges.as_deref().is_some_and(|ranges| {
                cache_ranges_have_single_cover(
                    ranges,
                    self.current_pos_ms as f64 / 1000.0,
                    chunk_end_ms as f64 / 1000.0,
                )
            });
            if !current_chunk_covered {
                let retry_pending = self
                    .transcription_retry
                    .as_ref()
                    .is_some_and(|retry| retry.start_ms == self.current_pos_ms);
                let resume_pos_ms = next_covered_chunk_after_playback(NetworkCacheRecovery {
                    cursor_ms: self.current_pos_ms,
                    playback_ms: self.last_playback_pos_ms,
                    chunk_ms,
                    cache_end_ms: Some(cache_end_ms),
                    ranges: cache_ranges.as_deref(),
                    processed_chunks: &self.processed_chunks,
                    retry_pending,
                });
                if let Some(resume_pos_ms) = resume_pos_ms {
                    let skipped_start_ms = self.current_pos_ms;
                    self.current_pos_ms = resume_pos_ms;
                    chunk_end_ms = self.current_pos_ms.saturating_add(chunk_ms);
                    warn!(
                        skipped_start_ms,
                        resume_start_ms = self.current_pos_ms,
                        playback_pos_ms = self.last_playback_pos_ms,
                        "skipping a network chunk without a single covering cache range"
                    );
                } else {
                    trace!(
                        start_sec = self.current_pos_ms as f64 / 1000.0,
                        end_sec = chunk_end_ms as f64 / 1000.0,
                        "waiting for one continuous demuxer cache range to cover the chunk"
                    );
                    return;
                }
            }
        }

        let available_ms = cache_end_ms.saturating_sub(self.current_pos_ms);
        if available_ms < chunk_ms && !retry_dump_available {
            trace!(
                needed_ms = self.current_pos_ms.saturating_add(chunk_ms),
                cached_ms = cache_end_ms,
                "waiting for the demuxer cache to grow"
            );
            return;
        }
        let lookahead_limit_ms =
            chunk_ms.saturating_mul(self.config.prefetch.lookahead_chunks.max(1) as u64);
        if let Some(playback_pos_ms) = self.last_playback_pos_ms {
            let ahead_end_ms = chunk_end_ms.saturating_sub(playback_pos_ms);
            if ahead_end_ms > lookahead_limit_ms {
                trace!(
                    ahead_ms = ahead_end_ms,
                    limit_ms = lookahead_limit_ms,
                    "look-ahead limit reached; waiting for playback to catch up"
                );
                return;
            }
        }

        if chunk_end_ms > cache_end_ms && !retry_dump_available {
            trace!(
                needed_ms = chunk_end_ms,
                cached_ms = cache_end_ms,
                "waiting for the demuxer cache to grow"
            );
            return;
        }

        debug!(start_ms = self.current_pos_ms, "scheduling a network chunk");
        self.process_chunk(client, chunk_ms, subtitle_path.as_deref());
    }

    fn tick_local(
        &mut self,
        client: &mut Handle,
        media_path: &str,
        file_length_ms: u64,
        subtitle_path: &Path,
    ) {
        // Check for completed translations from async queue
        self.process_translation_results(client, Some(subtitle_path));
        self.enqueue_missing_translations(false);

        // Calculate remaining time
        let time_left = if file_length_ms > self.current_pos_ms {
            file_length_ms - self.current_pos_ms
        } else {
            0
        };

        // Adjust chunk size for last chunk
        let local_chunk_size = self.local_chunk_size();
        if time_left > 0 && time_left < local_chunk_size {
            self.chunk_dur = time_left;
        } else {
            self.chunk_dur = local_chunk_size;
        }

        if time_left > 0 {
            if self.pending_transcription.is_some() {
                return;
            }

            while self.is_chunk_processed(self.current_pos_ms) {
                self.current_pos_ms = self.current_pos_ms.saturating_add(local_chunk_size);
                if self.current_pos_ms >= file_length_ms {
                    return;
                }
            }

            if self.transcription_retry_waiting() {
                return;
            }

            let lookahead_limit_ms = local_chunk_size
                .saturating_mul(self.config.prefetch.lookahead_chunks.max(1) as u64);
            let chunk_end_ms = self.current_pos_ms.saturating_add(self.chunk_dur);
            if let Some(playback_pos_ms) = self.last_playback_pos_ms {
                let ahead_end_ms = chunk_end_ms.saturating_sub(playback_pos_ms);
                if ahead_end_ms > lookahead_limit_ms {
                    trace!(
                        ahead_ms = ahead_end_ms,
                        limit_ms = lookahead_limit_ms,
                        "look-ahead limit reached; waiting for playback to catch up"
                    );
                    return;
                }
            }

            debug!(
                start_ms = self.current_pos_ms,
                remaining_ms = time_left,
                "scheduling a local chunk"
            );
            self.process_chunk_local(media_path, subtitle_path);
        } else {
            // Finished processing
            if !self.transcription_complete {
                let elapsed_ms = self
                    .session_started
                    .map(|started| started.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                let on_screen = if self.config.playback.save_srt {
                    format!("STT: Saved subtitles to {}", subtitle_path.display())
                } else {
                    "STT: Transcription complete".to_string()
                };
                info!(
                    chunks = self.chunk_seq,
                    subtitles = self.subtitle_manager.len(),
                    translations = self.translation_cache.len(),
                    failures = self.session_failures,
                    elapsed_ms,
                    path = %subtitle_path.display(),
                    display = %logging::osd_line(&on_screen),
                    "finished transcribing the local file"
                );
                self.running = false;
                self.transcription_complete = true;
            }
        }
    }

    fn check_seek(&mut self, client: &mut Handle) -> bool {
        let Ok(playback_pos) = client.get_property::<f64>("time-pos") else {
            return false;
        };
        let playback_pos_ms = (playback_pos * 1000.0) as u64;
        let now = Instant::now();
        let last_pos_ms = self.last_playback_pos_ms;
        let elapsed_ms = self.last_playback_instant.map(|last| {
            now.duration_since(last)
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX)
        });
        let explicit_seek = self.seek_pending;

        self.last_playback_pos_ms = Some(playback_pos_ms);
        self.last_playback_instant = Some(now);
        self.seek_pending = false;

        // `current_pos_ms` is the processing cursor and may run ahead of playback
        // when cache is available. Use the playback sample baseline instead.
        let Some(target) = detect_seek_target(
            playback_pos_ms,
            last_pos_ms,
            elapsed_ms,
            self.current_pos_ms,
            self.active_chunk_size(),
            explicit_seek,
        ) else {
            return false;
        };

        if target.position_ms == self.current_pos_ms {
            debug!(
                new_pos = target.position_ms,
                "seek landed inside the chunk in flight; keeping its tasks"
            );
            return false;
        }

        debug!(
            direction = if target.forward {
                "forward"
            } else {
                "backward"
            },
            from_ms = last_pos_ms.unwrap_or(self.current_pos_ms),
            to_ms = target.position_ms,
            delta_ms = playback_pos_ms.abs_diff(last_pos_ms.unwrap_or(self.current_pos_ms)),
            explicit = explicit_seek,
            "seeked"
        );

        if self.apply_seek_target(target) {
            // Drawn directly: the seek overlay is transient feedback for an
            // action the user just took, not a diagnostic worth queueing.
            let _ = client.command(&[
                "show-text",
                &format!(
                    "STT: Seeked to {}",
                    Self::format_progress(target.position_ms)
                ),
                "3000",
            ]);
            true
        } else {
            false
        }
    }

    fn apply_seek_target(&mut self, target: SeekTarget) -> bool {
        if target.position_ms == self.current_pos_ms {
            return false;
        }

        let completed_session = !self.running && self.transcription_complete && self.mode.is_some();
        if completed_session && self.is_chunk_processed(target.position_ms) {
            debug!(
                target_ms = target.position_ms,
                "completed session already covers the seek target; keeping its results"
            );
            return false;
        }

        if completed_session {
            self.running = true;
            self.transcription_complete = false;
            debug!(
                target_ms = target.position_ms,
                "resuming transcription for an uncovered seek target"
            );
        }

        self.current_pos_ms = target.position_ms;
        self.handle_seek_to(target.position_ms, target.forward);
        true
    }

    /// Realign the session after a seek to `new_pos` (already chunk-aligned).
    ///
    /// Seeking forward keeps the subtitles already generated: they still cover
    /// the part of the file the user is skipping over, and dropping them would
    /// leave a hole if the user seeks back. Seeking backward is different — the
    /// chunks past the seek target were generated for a timeline the user is
    /// rewinding into, so they are dropped and will be re-derived in order.
    fn handle_seek_to(&mut self, new_pos: u64, forward: bool) {
        self.cancel_translation_inflight();
        self.transcription_worker.cancel_inflight();
        self.pending_transcription = None;

        if !forward {
            self.subtitle_manager
                .remove_after(new_pos.min(u64::from(u32::MAX)) as u32);
            self.known_original_cues
                .retain(|start_ms| u64::from(*start_ms) <= new_pos);
            // The seek target is chunk-aligned. Its cues were removed above
            // unless they start exactly at the boundary, so process that chunk
            // again along with every later chunk.
            self.processed_chunks.retain(|start_ms| *start_ms < new_pos);
        }

        self.transcription_retry = None;
        if !forward {
            self.failed_translations
                .retain(|start_ms| u64::from(*start_ms) <= new_pos);
            self.translation_retries
                .retain(|start_ms, _| u64::from(*start_ms) <= new_pos);
        }
        self.enqueue_missing_translations(true);
    }

    /// Process one chunk from network cache
    fn process_chunk(
        &mut self,
        client: &mut Handle,
        chunk_ms: u64,
        subtitle_path: Option<&Path>,
    ) -> bool {
        // Keep the last network dump while retrying the same chunk. A live
        // stream may evict that media range from mpv's cache before a long
        // backoff expires; the retained dump lets us retry the exact audio.
        if self
            .transcription_retry
            .as_ref()
            .is_some_and(|retry| retry.start_ms == self.current_pos_ms)
            && std::fs::metadata(&self.paths.tmp_cache).is_ok_and(|metadata| metadata.len() > 0)
        {
            return self.schedule_transcription(
                self.paths.tmp_cache.to_string_lossy().into_owned(),
                0,
                true,
                chunk_ms,
                subtitle_path.map(Path::to_path_buf),
            );
        }

        // Dump cache
        let start_sec = self.current_pos_ms as f64 / 1000.0;
        let end_sec = self.current_pos_ms.saturating_add(chunk_ms) as f64 / 1000.0;
        if !Self::single_cache_range_covers(client, start_sec, end_sec) {
            trace!(
                start_sec,
                end_sec, "waiting for one continuous demuxer cache range to cover the chunk"
            );
            return false;
        }
        trace!(
            start_sec,
            end_sec, "dumping the demuxer cache to a temp file"
        );

        // Never mistake an older chunk's retained dump for this request if
        // mpv fails to create a new cache file.
        let _ = std::fs::remove_file(&self.paths.tmp_cache);

        let dump_result = client.command(&[
            "dump-cache",
            &start_sec.to_string(),
            &end_sec.to_string(),
            self.paths.tmp_cache.to_str().unwrap(),
        ]);

        if dump_result.is_err() {
            let _ = std::fs::remove_file(&self.paths.tmp_cache);
            let delay = self.defer_transcription_retry(
                self.current_pos_ms,
                "mpv refused to dump the network cache",
            );
            error!(
                start_sec,
                end_sec,
                retry_in_secs = delay.as_secs(),
                display = %logging::osd_line(&format!(
                    "无法读取网络音频，{} 秒后重试",
                    delay.as_secs()
                )),
                "mpv refused to dump the demuxer cache; keeping the session active"
            );
            return false;
        }

        self.schedule_transcription(
            self.paths.tmp_cache.to_string_lossy().into_owned(),
            0,
            true,
            chunk_ms,
            subtitle_path.map(Path::to_path_buf),
        )
    }

    /// `dump-cache` concatenates every cached range that intersects its
    /// request. If a request crosses a gap or overlapping ranges, mpv may
    /// rebase discontinuous packets into one output timeline. Such a dump has
    /// no single reliable offset back to the source media.
    fn demuxer_cache_ranges(client: &mut Handle) -> Option<Vec<CacheRange>> {
        let Ok(state) = client.get_property::<mpv_client::Node>("demuxer-cache-state") else {
            return None;
        };
        cache_ranges_from_node(&state)
    }

    fn single_cache_range_covers(client: &mut Handle, start_sec: f64, end_sec: f64) -> bool {
        Self::demuxer_cache_ranges(client)
            .is_some_and(|ranges| cache_ranges_have_single_cover(&ranges, start_sec, end_sec))
    }

    /// Process one chunk from local file
    fn process_chunk_local(&mut self, media_path: &str, subtitle_path: &Path) -> bool {
        self.schedule_transcription(
            media_path.to_string(),
            self.current_pos_ms,
            false,
            self.chunk_dur,
            Some(subtitle_path.to_path_buf()),
        )
    }

    /// Apply a completed worker result on the mpv event thread.
    fn apply_transcription_result(
        &mut self,
        client: &mut Handle,
        subtitle_path: Option<&Path>,
        chunk_start_ms: u64,
        device_notice: Option<SttDeviceNotice>,
    ) -> bool {
        let tmp_sub_prefix = self.paths.tmp_sub.to_string_lossy().to_string();
        let append_path = format!("{}_append", &tmp_sub_prefix);
        let main_srt = subtitle_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));
        self.show_device_notice(client, device_notice);

        // Offset timestamps
        let append_srt = format!("{}.srt", append_path);
        let offset_srt = format!("{}_append_offset.srt", &tmp_sub_prefix);

        if let Ok(meta) = std::fs::metadata(&append_srt) {
            if meta.len() == 0 {
                info!("chunk produced no subtitles; marking it done without merging");
                self.mark_chunk_processed(chunk_start_ms);
                self.paths.cleanup_intermediate_subs();
                return true;
            }
        }

        if let Err(e) = crate::srt::offset_srt_file(&append_srt, &offset_srt, chunk_start_ms as i64)
        {
            error!(
                error = %e,
                cause = %logging::err_chain(&e),
                offset_ms = chunk_start_ms,
                "cannot shift the chunk's subtitles onto the media timeline"
            );
            return false;
        }

        // Add original subtitles first so recognition updates immediately.
        // Everything past this point is additive: the original text is already
        // in the manager and on disk, so a later failure cannot lose it.
        let srt_file = match SrtFile::parse(&offset_srt) {
            Ok(srt) => srt,
            Err(e) => {
                error!(
                    error = %e,
                    cause = %logging::err_chain(&e),
                    path = %offset_srt,
                    "cannot parse the chunk's subtitles; original text stays on screen"
                );
                return false;
            }
        };
        self.subtitle_manager.add_from_srt(&srt_file);
        for entry in &srt_file.entries {
            let start_ms = Self::timestamp_to_millis(entry.start_time);
            if !entry.text.trim().is_empty() {
                self.known_original_cues.insert(start_ms);
            }
        }
        debug!(
            entries = srt_file.entries.len(),
            total = self.subtitle_manager.len(),
            "subtitles merged"
        );

        let already_processed = self.is_chunk_processed(chunk_start_ms);
        self.mark_chunk_processed(chunk_start_ms);
        if !self.save_subs(client, &main_srt) {
            if !already_processed {
                self.processed_chunks.remove(&chunk_start_ms);
            }
            return false;
        }

        // Only a successfully persisted chunk counts as complete. Scanning all
        // entries also catches untranslated cues restored from an SRT cache.
        if self.translate_enabled {
            self.last_translation_scan = None;
            self.enqueue_missing_translations(true);
        }

        // Keep only the main subtitle file on disk during playback to reduce clutter.
        // The `_append*` files are per-chunk intermediates and will be regenerated each chunk.
        self.paths.cleanup_intermediate_subs();

        debug!(next_ms = self.current_pos_ms, "chunk merged");
        true
    }

    fn show_device_notice(&mut self, _client: &mut Handle, device_notice: Option<SttDeviceNotice>) {
        let Some(notice) = device_notice else {
            return;
        };

        let mut msg = format!("STT device: {}", notice.effective);
        if notice.effective.is_gpu() {
            msg.push_str(&format!(" (gpu_device: {})", notice.gpu_device));
        }
        if notice.effective != notice.requested {
            msg.push_str(&format!(
                " (fallback from {}: {})",
                notice.requested, notice.reason
            ));
        }

        info!(
            requested = %notice.requested,
            effective = %notice.effective,
            gpu_device = notice.gpu_device,
            reason = %notice.reason,
            display = %logging::osd_line(&msg),
            "STT device notice"
        );
    }

    /// Process completed translation outcomes from the async queue. A failed
    /// batch enters a per-cue cooldown; the periodic missing-translation scan
    /// will submit it again after the backoff expires.
    fn process_translation_results(&mut self, client: &mut Handle, subtitle_path: Option<&Path>) {
        let Some(ref queue) = self.async_translation_queue else {
            return;
        };
        let outcomes = queue.try_recv_results();
        if outcomes.is_empty() {
            return;
        }

        let main_srt = subtitle_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));

        let mut translated = 0usize;
        let mut failed = 0usize;
        let mut last_reason = String::new();
        let mut last_retry_secs = 0u64;

        for outcome in outcomes {
            match outcome {
                TranslationOutcome::Translated(result) => {
                    self.pending_translations.remove(&result.start_ms);
                    self.failed_translations.remove(&result.start_ms);
                    self.translation_retries.remove(&result.start_ms);
                    self.known_original_cues.remove(&result.start_ms);
                    self.translation_cache.insert(
                        result.start_ms,
                        (result.original.clone(), result.translated.clone()),
                    );
                    self.subtitle_manager
                        .update_translation(result.start_ms, &result.translated);
                    let _ = self.save_subs(client, &main_srt);
                    translated += 1;
                }
                TranslationOutcome::Failed(failure) => {
                    self.pending_translations.remove(&failure.start_ms);
                    let delay = self.defer_translation_retry_with_base(
                        failure.start_ms,
                        &failure.cause,
                        failure.retry_base_secs,
                    );
                    debug!(
                        start_ms = failure.start_ms,
                        cause = %failure.cause,
                        reason = %failure.reason,
                        retry_in_secs = delay.as_secs(),
                        "translation batch failed; the original stays on screen while retry is scheduled"
                    );
                    failed += 1;
                    last_reason = failure.reason;
                    last_retry_secs = delay.as_secs();
                }
            }
        }
        trace!(translated, failed, "translation outcomes applied");
        if translated > 0 {
            self.translation_failure_reported = false;
        }
        self.last_translation_scan = None;

        if failed > 0 {
            // The reason is worth seeing, but a backend that is down fails
            // every cue, so keep the OSD to one message per failure burst.
            if !self.translation_failure_reported {
                self.translation_failure_reported = true;
                info!(
                    failed,
                    reason = %last_reason,
                    retry_in_secs = last_retry_secs,
                    display = %logging::osd_line(&format!(
                        "翻译暂时失败 ({failed} 条)，{last_retry_secs} 秒后自动重试: {last_reason}"
                    )),
                    "translation backend failed; retries remain scheduled"
                );
            }
        }
    }

    fn timestamp_to_millis(ts: crate::srt::Timestamp) -> u32 {
        let (h, m, s, ms) = ts.get();
        crate::srt::Timestamp::convert_to_milliseconds(h, m, s, ms)
    }

    fn save_subs(&mut self, client: &mut Handle, main_srt: &Path) -> bool {
        let entries = self.subtitle_manager.len();
        if let Err(e) = self.subtitle_manager.save_to_file(main_srt) {
            error!(
                error = %e,
                cause = %logging::err_chain(&e),
                path = %main_srt.display(),
                entries,
                "cannot write the subtitle file"
            );
            return false;
        }
        if self.subs_loaded {
            let _ = client.command(&["sub-reload"]);
        } else {
            trace!(path = %main_srt.display(), "subtitle file written (not yet loaded into mpv)");
        }
        self.save_cache_manifest_if_needed();
        true
    }

    /// Stop the current media/transcription session while keeping the plugin
    /// alive. This path is used by the toggle, EndFile, and media changes, so
    /// it must remain restartable.
    fn stop_transcription(&mut self) {
        // Report inside the session span before dropping it, so the summary
        // carries the session id and media it belongs to.
        if let Some(span) = self.session_span.take() {
            let _session = span.entered();
            let elapsed_ms = self
                .session_started
                .take()
                .map(|started| started.elapsed().as_millis() as u64)
                .unwrap_or(0);
            info!(
                chunks = self.chunk_seq,
                subtitles = self.subtitle_manager.len(),
                translations = self.translation_cache.len(),
                failures = self.session_failures,
                elapsed_ms,
                "transcription session ended"
            );
        }
        self.session_started = None;
        self.chunk_seq = 0;
        self.session_failures = 0;
        self.cached_subtitle_path = None;
        self.transcription_retry = None;

        self.running = false;
        self.transcription_worker.cancel_inflight();
        self.pending_transcription = None;

        // Cancel tasks belonging to this media, but keep the worker alive so
        // Ctrl+Shift+S and the next file can start a fresh session.
        self.cancel_translation_inflight();

        self.paths.cleanup();
        self.subtitle_manager.clear();
        self.translation_cache.clear();
        self.known_original_cues.clear();
        self.reset_translation_failures();
        self.processed_chunks.clear();
        self.subtitle_cache = None;
        self.subs_loaded = false;
        self.current_pos_ms = 0;
        self.last_playback_pos_ms = None;
        self.last_playback_instant = None;
        self.seek_pending = false;
        self.mode = None;
        self.transcription_complete = false;
    }

    /// Permanently shut down resources immediately before the mpv client is
    /// destroyed. Unlike `stop_transcription`, this is terminal.
    fn shutdown(&mut self) {
        if self.shutting_down {
            return;
        }

        self.shutting_down = true;
        info!("shutting down");
        self.stop_transcription();
        self.transcription_worker.shutdown();
        if let Some(mut queue) = self.async_translation_queue.take() {
            queue.shutdown();
        }
        info!("shutdown complete");
    }

    fn format_progress(ms: u64) -> String {
        let seconds = ms / 1000;
        let minutes = seconds / 60;
        let hours = minutes / 60;

        let seconds = seconds % 60;
        let minutes = minutes % 60;
        let millis = ms % 1000;

        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
    }

    fn is_chunk_processed(&self, start_ms: u64) -> bool {
        self.processed_chunks.contains(&start_ms)
    }

    fn mark_chunk_processed(&mut self, start_ms: u64) {
        self.processed_chunks.insert(start_ms);
    }

    fn media_id_for_cache(client: &mut Handle) -> Option<String> {
        if let Ok(id) = client.get_property::<String>("stream-open-filename") {
            if !id.trim().is_empty() {
                return Some(id);
            }
        }
        if let Ok(id) = client.get_property::<String>("path") {
            if !id.trim().is_empty() {
                return Some(id);
            }
        }
        None
    }

    fn cache_root_dir() -> Option<PathBuf> {
        let base = directories::BaseDirs::new()?;
        Some(
            base.config_dir()
                .join("mpv")
                .join("mpv_stt_plugin_rs_cache"),
        )
    }

    fn cache_paths_for_network_media(&self, media_id: &str) -> Option<CachePaths> {
        let root = Self::cache_root_dir()?;
        Some(Self::cache_paths_for_network_media_at(&root, media_id))
    }

    fn cache_paths_for_network_media_at(root: &Path, media_id: &str) -> CachePaths {
        let normalized_id = Self::normalize_network_media_id(media_id);
        let media_identity_hash = Self::sha256_hex(&normalized_id);
        let basename = Self::safe_media_basename(media_id);
        let stem = format!("{basename}.mpv_stt_plugin_rs-{media_identity_hash}");
        CachePaths {
            subtitle_path: root.join(format!("{stem}.srt")),
            manifest_path: root.join(format!("{stem}.json")),
            media_identity_hash,
            media_fingerprint: None,
        }
    }

    fn legacy_cache_paths_for_network_media_at(root: &Path, media_id: &str) -> CachePaths {
        let stem = format!("{:016x}", Self::fnv1a_hash64(media_id));
        CachePaths {
            subtitle_path: root.join(format!("{stem}.srt")),
            manifest_path: root.join(format!("{stem}.json")),
            media_identity_hash: String::new(),
            media_fingerprint: None,
        }
    }

    fn local_cache_paths_for_media_uri(media_uri: &str) -> Option<CachePaths> {
        let media_path = Self::canonical_local_media_path(media_uri)?;
        let filename = media_path.file_name()?.to_string_lossy();
        let parent = media_path.parent()?;
        let identity = media_path.to_string_lossy();
        let media_identity_hash = Self::sha256_hex(&identity);
        let media_fingerprint = Self::local_media_fingerprint(&media_path);
        Some(CachePaths {
            subtitle_path: parent.join(format!("{filename}.mpv_stt_plugin_rs.srt")),
            manifest_path: parent.join(format!("{filename}.mpv_stt_plugin_rs.json")),
            media_identity_hash,
            media_fingerprint,
        })
    }

    fn canonical_local_media_path(media_uri: &str) -> Option<PathBuf> {
        let media_uri = media_uri.trim();
        let path = match Url::parse(media_uri) {
            Ok(url) if url.scheme().eq_ignore_ascii_case("file") => url.to_file_path().ok()?,
            _ if media_uri.contains("://") => return None,
            _ => PathBuf::from(media_uri),
        };
        let absolute = if path.is_absolute() {
            path
        } else {
            std::env::current_dir().ok()?.join(path)
        };
        Some(
            fs::canonicalize(&absolute)
                .unwrap_or_else(|_| Self::normalize_absolute_path(&absolute)),
        )
    }

    fn normalize_absolute_path(path: &Path) -> PathBuf {
        use std::path::Component;

        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::CurDir => {}
                Component::ParentDir => {
                    normalized.pop();
                }
                component => normalized.push(component.as_os_str()),
            }
        }
        normalized
    }

    fn local_media_fingerprint(path: &Path) -> Option<String> {
        let metadata = fs::metadata(path).ok()?;
        let modified_ns = metadata
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos();
        Some(format!("{}:{modified_ns}", metadata.len()))
    }

    fn normalize_network_media_id(media_id: &str) -> String {
        let Ok(mut url) = Url::parse(media_id.trim()) else {
            return media_id.trim().to_string();
        };
        url.set_fragment(None);
        // Authentication in URL userinfo must not split cache identity or be
        // retained in any value used to derive a persistent cache name.
        let _ = url.set_username("");
        let _ = url.set_password(None);

        let mut query: Vec<(String, String)> = url
            .query_pairs()
            .filter(|(key, _)| !Self::is_volatile_auth_query_parameter(key))
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();
        query.sort_unstable();
        url.set_query(None);
        if !query.is_empty() {
            url.query_pairs_mut().extend_pairs(query);
        }
        url.to_string()
    }

    fn is_volatile_auth_query_parameter(key: &str) -> bool {
        matches!(
            key.to_ascii_lowercase().as_str(),
            "access_token"
                | "auth"
                | "expires"
                | "expiry"
                | "hdnea"
                | "hdnts"
                | "jwt"
                | "key-pair-id"
                | "policy"
                | "sig"
                | "signature"
                | "token"
                | "x-amz-credential"
                | "x-amz-date"
                | "x-amz-expires"
                | "x-amz-security-token"
                | "x-amz-signature"
        )
    }

    fn safe_media_basename(media_id: &str) -> String {
        let candidate = Url::parse(media_id)
            .ok()
            .and_then(|url| {
                url.path_segments()
                    .and_then(Iterator::last)
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned)
            })
            .or_else(|| {
                Path::new(media_id)
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "stream".to_string());
        let safe: String = candidate
            .chars()
            .take(80)
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                    character
                } else {
                    '_'
                }
            })
            .collect();
        let safe = safe.trim_matches('.');
        if safe.is_empty() {
            "stream".to_string()
        } else {
            safe.to_string()
        }
    }

    fn sha256_hex(input: &str) -> String {
        format!("{:x}", Sha256::digest(input.as_bytes()))
    }

    fn fnv1a_hash64(input: &str) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut hash = FNV_OFFSET;
        for byte in input.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn create_cache_parent(paths: &CachePaths) {
        if let Some(parent) = paths.subtitle_path.parent()
            && let Err(error) = fs::create_dir_all(parent)
        {
            warn!(
                error = %error,
                cause = %logging::err_chain(&error),
                dir = %parent.display(),
                "cannot create the subtitle cache directory"
            );
        }
    }

    fn migrate_legacy_network_cache(media_id: &str, new_paths: &CachePaths) {
        if new_paths.subtitle_path.exists() || new_paths.manifest_path.exists() {
            return;
        }
        let Some(root) = new_paths.manifest_path.parent() else {
            return;
        };
        let legacy = Self::legacy_cache_paths_for_network_media_at(root, media_id);
        if !legacy.subtitle_path.exists() || !legacy.manifest_path.exists() {
            return;
        }

        let Some(mut manifest) = Self::read_cache_manifest(&legacy.manifest_path) else {
            return;
        };
        if manifest.schema_version != 0
            || manifest.timeline_version != SUBTITLE_TIMELINE_VERSION
            || !manifest.media_identity_hash.is_empty()
            || SrtFile::parse(&legacy.subtitle_path).is_err()
        {
            return;
        }

        manifest.schema_version = SUBTITLE_CACHE_SCHEMA_VERSION;
        manifest.media_identity_hash = new_paths.media_identity_hash.clone();
        manifest.media_fingerprint = new_paths.media_fingerprint.clone();
        let Ok(content) = serde_json::to_string(&manifest) else {
            return;
        };
        if let Err(error) = fs::copy(&legacy.subtitle_path, &new_paths.subtitle_path) {
            warn!(
                error = %error,
                path = %legacy.subtitle_path.display(),
                "cannot copy a legacy network subtitle cache"
            );
            return;
        }
        if let Err(error) = fs::write(&new_paths.manifest_path, content) {
            let _ = fs::remove_file(&new_paths.subtitle_path);
            warn!(
                error = %error,
                path = %new_paths.manifest_path.display(),
                "cannot migrate the legacy network cache manifest"
            );
            return;
        }

        for legacy_path in [&legacy.subtitle_path, &legacy.manifest_path] {
            if let Err(error) = fs::remove_file(legacy_path) {
                warn!(
                    error = %error,
                    path = %legacy_path.display(),
                    "cannot remove a migrated legacy cache file"
                );
            }
        }
        info!(
            path = %new_paths.subtitle_path.display(),
            "migrated the current network subtitle cache"
        );
    }

    fn cache_manifest_matches(paths: &CachePaths, manifest: &CacheManifest) -> bool {
        manifest.schema_version == SUBTITLE_CACHE_SCHEMA_VERSION
            && manifest.timeline_version == SUBTITLE_TIMELINE_VERSION
            && manifest.media_identity_hash == paths.media_identity_hash
            && manifest.media_fingerprint == paths.media_fingerprint
    }

    fn load_cached_subs(&mut self, paths: &CachePaths, chunk_size_ms: u64) -> bool {
        if !paths.subtitle_path.exists() {
            return false;
        }

        let Some(manifest) = self.load_cache_manifest(&paths.manifest_path) else {
            debug!(
                path = %paths.subtitle_path.display(),
                "ignoring subtitles without a readable plugin manifest"
            );
            return false;
        };
        if !Self::cache_manifest_matches(paths, &manifest) {
            debug!(
                path = %paths.subtitle_path.display(),
                "ignoring subtitles with a stale or mismatched plugin manifest"
            );
            return false;
        }

        let srt_path = &paths.subtitle_path;
        let srt_file = match SrtFile::parse(srt_path) {
            Ok(srt) => srt,
            Err(err) => {
                warn!(
                    error = %err,
                    cause = %logging::err_chain(&err),
                    path = %srt_path.display(),
                    "cannot parse the cached subtitles; transcribing from scratch"
                );
                return false;
            }
        };

        self.subtitle_manager.clear();
        self.translation_cache.clear();
        self.known_original_cues.clear();
        // A fresh media/session: earlier retry cooldowns say nothing about it.
        self.reset_translation_failures();
        self.processed_chunks.clear();
        self.subtitle_manager.add_from_srt(&srt_file);

        let chunk_size = chunk_size_ms.max(1);
        if manifest.chunk_size_ms == chunk_size {
            self.processed_chunks.extend(manifest.processed_chunks);
        }
        for entry in manifest.translations {
            if !entry.translated.trim().is_empty() {
                self.translation_cache
                    .insert(entry.start_ms, (entry.original, entry.translated));
            }
        }

        debug!(
            path = %srt_path.display(),
            entries = srt_file.entries.len(),
            translations = self.translation_cache.len(),
            "reused cached subtitles"
        );
        true
    }

    fn load_cache_manifest(&self, path: &Path) -> Option<CacheManifest> {
        Self::read_cache_manifest(path)
    }

    fn read_cache_manifest(path: &Path) -> Option<CacheManifest> {
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(err) => {
                debug!(
                    error = %err,
                    path = %path.display(),
                    "no usable cache manifest"
                );
                return None;
            }
        };
        match serde_json::from_str(&content) {
            Ok(manifest) => Some(manifest),
            Err(err) => {
                warn!(
                    error = %err,
                    path = %path.display(),
                    "cache manifest is unreadable"
                );
                None
            }
        }
    }

    fn save_cache_manifest_if_needed(&self) {
        let Some(cache) = &self.subtitle_cache else {
            return;
        };

        let mut processed_chunks: Vec<u64> = self.processed_chunks.iter().copied().collect();
        processed_chunks.sort_unstable();

        let translations = self
            .translation_cache
            .iter()
            .map(|(start_ms, (original, translated))| TranslationCacheEntry {
                start_ms: *start_ms,
                original: original.clone(),
                translated: translated.clone(),
            })
            .collect();

        let chunk_size_ms = match self.mode.as_ref() {
            Some(ProcessingMode::Network) => self.network_chunk_size(),
            Some(ProcessingMode::Local { .. }) => self.local_chunk_size(),
            None => self.chunk_dur,
        };
        let manifest = CacheManifest {
            schema_version: SUBTITLE_CACHE_SCHEMA_VERSION,
            timeline_version: SUBTITLE_TIMELINE_VERSION,
            media_identity_hash: cache.media_identity_hash.clone(),
            media_fingerprint: cache.media_fingerprint.clone(),
            chunk_size_ms,
            processed_chunks,
            translations,
        };

        let content = match serde_json::to_string(&manifest) {
            Ok(data) => data,
            Err(err) => {
                warn!(error = %err, "cannot serialize the cache manifest");
                return;
            }
        };

        if let Err(err) = fs::write(&cache.manifest_path, content) {
            warn!(
                error = %err,
                path = %cache.manifest_path.display(),
                "cannot write the cache manifest; the next run will re-transcribe"
            );
        } else {
            debug!(
                chunks = manifest.processed_chunks.len(),
                translations = manifest.translations.len(),
                path = %cache.manifest_path.display(),
                "saved the cache manifest"
            );
        }
    }

    /// Detect if current media is a network stream
    fn detect_network_stream(&self, client: &mut Handle) -> bool {
        // Method 1: Check path/filename for http/https URLs
        if let Ok(path) = client.get_property::<String>("path") {
            trace!(path, "checking whether the media is a network stream");
            if path.starts_with("http://") || path.starts_with("https://") {
                trace!(signal = "url-scheme", "network stream detected");
                return true;
            }
        }

        // Method 2: Check stream-open-filename
        if let Ok(filename) = client.get_property::<String>("stream-open-filename") {
            trace!(filename, "checking stream-open-filename");
            if filename.starts_with("http://") || filename.starts_with("https://") {
                trace!(signal = "stream-open-filename", "network stream detected");
                return true;
            }
        }

        // Method 3: Check demuxer-via-network property
        if let Ok(via_network) = client.get_property::<String>("demuxer-via-network") {
            trace!(via_network, "checking demuxer-via-network");
            if via_network == "yes" {
                trace!(signal = "demuxer-via-network", "network stream detected");
                return true;
            }
        }

        trace!("no network stream signal; treating the media as a local file");
        false
    }
}

impl Drop for PluginState {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// MPV C plugin entry point
#[unsafe(no_mangle)]
pub extern "C" fn mpv_open_cplugin(handle: *mut mpv_handle) -> std::os::raw::c_int {
    let result = std::panic::catch_unwind(|| {
        // Config first: the logging setup itself is configured by `[log]`, so it
        // cannot come before the config is read.
        let env_cfg_path = Config::config_path_from_env();
        let default_cfg_path = Config::default_config_path();
        let config = Config::load();

        let log_settings = LogSettings::from_config(
            &config.log,
            config
                .log_dir(env_cfg_path.as_ref().or(default_cfg_path.as_ref()))
                .as_deref(),
        );
        let _log_guard = logging::init(&log_settings);
        logging::install_panic_hook();

        let client = Handle::from_ptr(handle);

        // The effective configuration, field by field. Never the whole
        // `Config`: it carries API keys, encryption keys and auth secrets.
        info!(
            stt_source = %config.stt.source,
            stt_sources = %config.stt.sources.keys().cloned().collect::<Vec<_>>().join(","),
            translate_source = %config.translate.source,
            translate_sources = %config.translate.sources.keys().cloned().collect::<Vec<_>>().join(","),
            translate = %format!("{}->{}", config.translate.from_lang, config.translate.to_lang),
            local_chunk_ms = config.chunk.local_ms,
            network_chunk_ms = config.chunk.network_ms,
            auto_start = config.playback.auto_start,
            save_srt = config.playback.save_srt,
            log_filter = %log_settings.filter,
            log_format = %log_settings.format,
            log_file = %log_settings
                .file
                .as_ref()
                .map(|f| f.dir.join(&f.stem).display().to_string())
                .unwrap_or_else(|| "off".to_string()),
            log_osd = log_settings.osd,
            "effective configuration"
        );

        // Print welcome message
        info!(
            client = client.name(),
            display = %logging::osd_line("mpv_stt_plugin_rs Rust plugin loaded!"),
            "plugin loaded"
        );
        let auto_start = config.playback.auto_start;
        let mut state = match PluginState::new(config) {
            Ok(state) => state,
            Err(err) => {
                error!(
                    error = %err,
                    cause = %logging::err_chain(&err),
                    "cannot initialize the plugin"
                );
                // Drawn directly rather than through the OSD queue: a failed
                // init returns before the event loop that drains the queue.
                let _ = client.command(&[
                    "show-text",
                    &format!(
                        "STT plugin initialization failed: {}",
                        logging::osd_line(&err.to_string())
                    ),
                    "8000",
                ]);
                return -1;
            }
        };

        // Target the numeric client ID rather than a filename-derived name.
        // IINA can create multiple mpv cores and mpv may suffix duplicate
        // client names; IDs are unambiguous for the lifetime of this client.
        let client_name = client.name().to_string();
        let client_target = format!("@{}", client.id());

        // Use a forced section so IINA/default input bindings cannot silently
        // shadow the plugin controls. Users explicitly chose these shortcuts.
        let key_bindings = key_binding_section(&client_target);
        let section_name = format!("{}-input", client_name);

        if let Err(err) = client.command(&["define-section", &section_name, &key_bindings, "force"])
        {
            error!(
                error = %err,
                section = %section_name,
                display = %logging::osd_line("STT plugin: failed to register shortcuts (see log)"),
                "cannot define the input section"
            );
        } else if let Err(err) = client.command(&["enable-section", &section_name]) {
            error!(
                error = %err,
                section = %section_name,
                display = %logging::osd_line("STT plugin: failed to enable shortcuts (see log)"),
                "cannot enable the input section"
            );
        } else {
            info!(
                client = %client_name,
                target = %client_target,
                keys = "Ctrl+Shift+S/T/C",
                "registered the forced shortcut section"
            );
        }

        // Set auto-start flag (will start after file loads)
        if auto_start {
            info!("auto-start enabled; waiting for a file to load");
            state.pending_auto_start = true;
        }

        // If a file is already loaded when the plugin is attached (e.g., script reload),
        // try to start immediately instead of waiting for the next FileLoaded event.
        if auto_start && !state.running {
            if client.get_property::<f64>("duration").is_ok() {
                debug!("auto-start: media already loaded, starting now");
                state.file_loaded = true;
                state.pending_auto_start = false;
                state.running = true;
                state.start_transcription(client);
            }
        }

        // Main event loop with short timeout for continuous processing
        loop {
            // Use 0.1 second timeout to allow continuous processing
            let event = client.wait_event(0.1);
            // Anything `warn` and above raised since the last iteration goes to
            // the OSD here: non-blocking, and it cannot delay an event.
            drain_osd(client);
            match event {
                Event::Shutdown => {
                    state.shutdown();
                    return 0;
                }
                Event::ClientMessage(msg) => {
                    if state.shutting_down {
                        continue;
                    }
                    let args = msg.args();
                    if let Some(command) = ControlCommand::from_client_message(&args) {
                        match command {
                            ControlCommand::ToggleStt => {
                                debug!("toggling STT");
                                state.toggle_stt(client);
                            }
                            ControlCommand::ToggleTranslate => {
                                debug!("toggling translation");
                                state.toggle_translate(client);
                            }
                            ControlCommand::ClearCache => {
                                debug!("clearing the subtitle cache");
                                state.clear_cache(client);
                            }
                        }
                    }
                }
                Event::StartFile(_) => {
                    if state.shutting_down {
                        continue;
                    }
                    debug!("mpv start-file");
                    // Be defensive if a frontend switches files without an
                    // EndFile event reaching this client.
                    if state.running || state.mode.is_some() {
                        state.stop_transcription();
                    }
                    state.file_loaded = false;
                    state.seek_pending = false;
                    state.pending_auto_start = state.config.playback.auto_start;
                }
                Event::FileLoaded => {
                    if state.shutting_down {
                        continue;
                    }
                    debug!("mpv file-loaded");
                    state.file_loaded = true;

                    // Trigger auto-start if pending
                    if state.pending_auto_start && !state.running {
                        info!("auto-starting STT after file load");
                        state.pending_auto_start = false;
                        state.running = true;
                        state.start_transcription(client);
                    }
                }
                Event::Seek => {
                    if state.shutting_down {
                        continue;
                    }
                    // mpv emits Seek before the new playback position is fully
                    // settled. Keep the marker until the following restart or
                    // tick, where `time-pos` is reconciled without chunk-size
                    // heuristics.
                    state.seek_pending = true;
                }
                Event::PlaybackRestart => {
                    if state.shutting_down {
                        continue;
                    }
                    debug!("mpv playback-restart");

                    // Also trigger auto-start on playback restart (backup mechanism)
                    if state.pending_auto_start && !state.running && state.file_loaded {
                        info!("auto-starting STT after playback restart");
                        state.pending_auto_start = false;
                        state.running = true;
                        state.start_transcription(client);
                    }

                    state.tick(client);
                }
                Event::EndFile(_) => {
                    if !state.shutting_down && (state.running || state.mode.is_some()) {
                        state.stop_transcription();
                    }
                    state.file_loaded = false; // Reset for next file
                    state.seek_pending = false;
                }
                Event::None => {
                    if state.shutting_down {
                        continue;
                    }
                    // Timeout - use this to tick the processing
                    state.tick(client);
                }
                _ => {
                    if state.shutting_down || state.seek_pending {
                        continue;
                    }
                    // Other events - still tick unless a seek is awaiting its
                    // settled position from PlaybackRestart or the next timeout.
                    state.tick(client);
                }
            }
        }
    });

    if let Err(payload) = result {
        report_escaped_panic(payload.as_ref());
        return -1;
    }

    0
}

/// Draw the log records queued for the OSD since the last call.
///
/// Drained from the event loop rather than from the logging path so a burst of
/// failures can never block the thread that has to keep answering mpv.
fn drain_osd(client: &mut Handle) {
    let notices = logging::take_osd_notices();
    for line in logging::format_osd_notices(&notices) {
        let _ = client.command(&["show-text", &line, "4000"]);
    }
}

/// Record a panic that escaped the event loop. The hook installed by
/// `logging::install_panic_hook` normally catches these first and already sent
/// them to every sink; this is the last resort for a panic raised outside it.
fn report_escaped_panic(payload: &(dyn std::any::Any + Send)) {
    let detail = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string());
    error!(target: logging::TARGET_INTERNAL, "event loop aborted by panic: {detail}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    use std::io::Read;
    #[cfg(feature = "stt_cloudflare")]
    use std::io::Write;
    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    use std::net::TcpListener;
    #[cfg(feature = "stt_cloudflare")]
    use std::net::TcpStream;
    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    use std::time::Duration;
    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    use std::time::Instant;

    #[test]
    fn control_messages_support_all_shortcuts_and_legacy_shape() {
        assert_eq!(
            ControlCommand::from_client_message(&["toggle-stt"]),
            Some(ControlCommand::ToggleStt)
        );
        assert_eq!(
            ControlCommand::from_client_message(&["toggle-translate"]),
            Some(ControlCommand::ToggleTranslate)
        );
        assert_eq!(
            ControlCommand::from_client_message(&["clear-cache"]),
            Some(ControlCommand::ClearCache)
        );
        assert_eq!(
            ControlCommand::from_client_message(&["legacy-route", "toggle-translate"]),
            Some(ControlCommand::ToggleTranslate)
        );
        assert_eq!(ControlCommand::from_client_message(&["unknown"]), None);
    }

    #[test]
    fn key_section_targets_the_exact_mpv_client() {
        let section = key_binding_section("@42");
        assert_eq!(section.lines().count(), KEY_BINDINGS.len());
        for (key, command) in KEY_BINDINGS {
            assert!(
                section.contains(&format!("{key} script-message-to @42 {command}")),
                "missing {key}/{command} binding in {section:?}"
            );
        }
    }

    #[test]
    fn local_cache_paths_are_stable_and_distinguish_container_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let mkv = dir.path().join("movie.mkv");
        let mp4 = dir.path().join("movie.mp4");
        fs::write(&mkv, b"mkv media").unwrap();
        fs::write(&mp4, b"mp4 media").unwrap();

        let paths = PluginState::local_cache_paths_for_media_uri(mkv.to_str().unwrap()).unwrap();
        let alias = dir.path().join(".").join("movie.mkv");
        let alias_paths =
            PluginState::local_cache_paths_for_media_uri(alias.to_str().unwrap()).unwrap();
        let file_uri = Url::from_file_path(&mkv).unwrap().to_string();
        let uri_paths = PluginState::local_cache_paths_for_media_uri(&file_uri).unwrap();
        let mp4_paths =
            PluginState::local_cache_paths_for_media_uri(mp4.to_str().unwrap()).unwrap();
        let canonical_dir = fs::canonicalize(dir.path()).unwrap();

        assert_eq!(
            paths.subtitle_path,
            canonical_dir.join("movie.mkv.mpv_stt_plugin_rs.srt")
        );
        assert_eq!(
            paths.manifest_path,
            canonical_dir.join("movie.mkv.mpv_stt_plugin_rs.json")
        );
        assert_eq!(paths.subtitle_path, alias_paths.subtitle_path);
        assert_eq!(paths.media_identity_hash, alias_paths.media_identity_hash);
        assert_eq!(paths.subtitle_path, uri_paths.subtitle_path);
        assert_eq!(paths.media_identity_hash, uri_paths.media_identity_hash);
        assert_ne!(paths.subtitle_path, mp4_paths.subtitle_path);
        assert_ne!(paths.media_identity_hash, mp4_paths.media_identity_hash);
    }

    #[test]
    fn network_cache_paths_ignore_only_volatile_url_identity_parts() {
        let dir = tempfile::tempdir().unwrap();
        let first = "HTTPS://user:old-secret@EXAMPLE.com:443/video.mkv?z=2&token=old&a=1#chapter";
        let refreshed = "https://EXAMPLE.com/video.mkv?a=1&token=new&z=2#other";
        let different_content = "https://example.com/video.mkv?a=1&token=new&v=2&z=2";

        let first_paths = PluginState::cache_paths_for_network_media_at(dir.path(), first);
        let refreshed_paths = PluginState::cache_paths_for_network_media_at(dir.path(), refreshed);
        let different_paths =
            PluginState::cache_paths_for_network_media_at(dir.path(), different_content);

        assert_eq!(first_paths.subtitle_path, refreshed_paths.subtitle_path);
        assert_eq!(
            first_paths.media_identity_hash,
            refreshed_paths.media_identity_hash
        );
        assert_ne!(first_paths.subtitle_path, different_paths.subtitle_path);
        let filename = first_paths
            .subtitle_path
            .file_name()
            .unwrap()
            .to_string_lossy();
        assert!(filename.contains("video.mkv.mpv_stt_plugin_rs-"));
        assert!(!filename.contains("old-secret"));
        assert!(!filename.contains("token"));
    }

    #[test]
    fn cache_manifest_must_match_schema_timeline_identity_and_fingerprint() {
        let paths = CachePaths {
            subtitle_path: PathBuf::from("movie.srt"),
            manifest_path: PathBuf::from("movie.json"),
            media_identity_hash: "media-hash".to_string(),
            media_fingerprint: Some("12:345".to_string()),
        };
        let manifest =
            |schema_version, timeline_version, identity: &str, fingerprint: Option<&str>| {
                CacheManifest {
                    schema_version,
                    timeline_version,
                    media_identity_hash: identity.to_string(),
                    media_fingerprint: fingerprint.map(str::to_string),
                    chunk_size_ms: 15_000,
                    processed_chunks: Vec::new(),
                    translations: Vec::new(),
                }
            };

        assert!(PluginState::cache_manifest_matches(
            &paths,
            &manifest(
                SUBTITLE_CACHE_SCHEMA_VERSION,
                SUBTITLE_TIMELINE_VERSION,
                "media-hash",
                Some("12:345")
            )
        ));
        assert!(!PluginState::cache_manifest_matches(
            &paths,
            &manifest(0, SUBTITLE_TIMELINE_VERSION, "media-hash", Some("12:345"))
        ));
        assert!(!PluginState::cache_manifest_matches(
            &paths,
            &manifest(
                SUBTITLE_CACHE_SCHEMA_VERSION,
                SUBTITLE_TIMELINE_VERSION + 1,
                "media-hash",
                Some("12:345")
            )
        ));
        assert!(!PluginState::cache_manifest_matches(
            &paths,
            &manifest(
                SUBTITLE_CACHE_SCHEMA_VERSION,
                SUBTITLE_TIMELINE_VERSION,
                "other-media",
                Some("12:345")
            )
        ));
        assert!(!PluginState::cache_manifest_matches(
            &paths,
            &manifest(
                SUBTITLE_CACHE_SCHEMA_VERSION,
                SUBTITLE_TIMELINE_VERSION,
                "media-hash",
                Some("12:346")
            )
        ));
    }

    #[test]
    fn clearing_plugin_cache_files_preserves_regular_srt_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let media = dir.path().join("movie.mkv");
        let ordinary_srt = dir.path().join("movie.srt");
        fs::write(&media, b"media").unwrap();
        fs::write(&ordinary_srt, b"user subtitle").unwrap();
        let paths = PluginState::local_cache_paths_for_media_uri(media.to_str().unwrap()).unwrap();
        fs::write(&paths.subtitle_path, b"plugin subtitle").unwrap();
        fs::write(&paths.manifest_path, b"plugin manifest").unwrap();

        assert_eq!(
            PluginState::remove_plugin_cache_files(std::slice::from_ref(&paths)),
            2
        );
        assert!(media.exists());
        assert!(ordinary_srt.exists());
        assert!(!paths.subtitle_path.exists());
        assert!(!paths.manifest_path.exists());
    }

    #[test]
    fn valid_legacy_network_cache_is_migrated_to_the_namespaced_path() {
        let dir = tempfile::tempdir().unwrap();
        let media_id = "https://example.test/video.mkv?token=old";
        let paths = PluginState::cache_paths_for_network_media_at(dir.path(), media_id);
        let legacy = PluginState::legacy_cache_paths_for_network_media_at(dir.path(), media_id);
        fs::write(
            &legacy.subtitle_path,
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        )
        .unwrap();
        let manifest = CacheManifest {
            schema_version: 0,
            timeline_version: SUBTITLE_TIMELINE_VERSION,
            media_identity_hash: String::new(),
            media_fingerprint: None,
            chunk_size_ms: 15_000,
            processed_chunks: vec![0],
            translations: Vec::new(),
        };
        fs::write(
            &legacy.manifest_path,
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        PluginState::migrate_legacy_network_cache(media_id, &paths);

        assert!(paths.subtitle_path.exists());
        assert!(paths.manifest_path.exists());
        assert!(!legacy.subtitle_path.exists());
        assert!(!legacy.manifest_path.exists());
        let migrated = PluginState::read_cache_manifest(&paths.manifest_path).unwrap();
        assert_eq!(migrated.schema_version, SUBTITLE_CACHE_SCHEMA_VERSION);
        assert_eq!(migrated.media_identity_hash, paths.media_identity_hash);
        assert!(PluginState::cache_manifest_matches(&paths, &migrated));
    }

    #[test]
    fn explicit_short_seeks_align_forward_and_backward_without_chunk_threshold() {
        let forward = detect_seek_target(17_500, Some(16_000), Some(100), 60_000, 15_000, true)
            .expect("explicit forward seek should be detected");
        assert_eq!(
            forward,
            SeekTarget {
                position_ms: 15_000,
                forward: true,
            }
        );

        let backward = detect_seek_target(14_000, Some(16_000), Some(100), 60_000, 15_000, true)
            .expect("explicit backward seek should be detected");
        assert_eq!(
            backward,
            SeekTarget {
                position_ms: 0,
                forward: false,
            }
        );
    }

    #[test]
    fn repeated_small_explicit_seeks_are_not_lost_to_polling_thresholds() {
        let first = detect_seek_target(15_500, Some(14_500), Some(100), 60_000, 15_000, true);
        let second = detect_seek_target(17_000, Some(15_500), Some(100), 60_000, 15_000, true);
        assert_eq!(first.map(|target| target.position_ms), Some(15_000));
        assert_eq!(second.map(|target| target.position_ms), Some(15_000));
    }

    #[test]
    fn ordinary_playback_progress_is_not_mistaken_for_a_seek() {
        assert_eq!(
            detect_seek_target(10_100, Some(10_000), Some(100), 0, 15_000, false),
            None
        );
        assert_eq!(
            detect_seek_target(30_000, Some(10_000), Some(20_000), 0, 15_000, false),
            None
        );
    }

    fn cache_range_ms(start_ms: u64, end_ms: u64) -> CacheRange {
        CacheRange {
            start_sec: start_ms as f64 / 1000.0,
            end_sec: end_ms as f64 / 1000.0,
        }
    }

    fn recovery_target(
        playback_ms: u64,
        ranges: &[CacheRange],
        processed_chunks: &HashSet<u64>,
        retry_pending: bool,
    ) -> Option<u64> {
        next_covered_chunk_after_playback(NetworkCacheRecovery {
            cursor_ms: 15_000,
            playback_ms: Some(playback_ms),
            chunk_ms: 15_000,
            cache_end_ms: Some(60_000),
            ranges: Some(ranges),
            processed_chunks,
            retry_pending,
        })
    }

    #[test]
    fn cache_recovery_waits_until_playback_passes_the_chunk() {
        let ranges = [cache_range_ms(30_000, 60_000)];
        assert_eq!(
            recovery_target(29_999, &ranges, &HashSet::new(), false),
            None
        );
    }

    #[test]
    fn cache_recovery_selects_the_first_covered_chunk_at_the_playhead() {
        let ranges = [cache_range_ms(30_000, 60_000)];
        assert_eq!(
            recovery_target(30_000, &ranges, &HashSet::new(), false),
            Some(30_000)
        );
    }

    #[test]
    fn cache_recovery_skips_processed_chunks_and_requires_one_raw_range() {
        let ranges = [cache_range_ms(30_000, 60_000)];
        assert_eq!(
            recovery_target(30_000, &ranges, &HashSet::from([30_000]), false),
            Some(45_000)
        );

        let overlapping_ranges = [
            cache_range_ms(30_000, 45_000),
            cache_range_ms(40_000, 60_000),
        ];
        assert!(!cache_ranges_have_single_cover(
            &overlapping_ranges,
            30.0,
            45.0
        ));
        assert_eq!(
            recovery_target(30_000, &overlapping_ranges, &HashSet::new(), false),
            Some(45_000)
        );

        let adjacent_ranges = [
            cache_range_ms(30_000, 45_000),
            cache_range_ms(45_000, 60_000),
        ];
        assert!(!cache_ranges_have_single_cover(
            &adjacent_ranges,
            30.0,
            60.0
        ));

        let disjoint_ranges = [
            cache_range_ms(30_000, 40_000),
            cache_range_ms(50_000, 60_000),
        ];
        assert_eq!(
            recovery_target(30_000, &disjoint_ranges, &HashSet::new(), false),
            None
        );
    }

    #[test]
    fn cache_recovery_waits_when_no_later_chunk_is_covered_or_a_retry_is_pending() {
        let partial_range = [cache_range_ms(30_000, 44_000)];
        assert_eq!(
            recovery_target(30_000, &partial_range, &HashSet::new(), false),
            None
        );

        let later_range = [cache_range_ms(30_000, 60_000)];
        assert_eq!(
            recovery_target(30_000, &later_range, &HashSet::new(), true),
            None
        );
    }

    #[test]
    fn cache_recovery_keeps_a_chunk_that_still_has_single_range_coverage() {
        let ranges = [cache_range_ms(15_000, 30_000)];
        assert_eq!(
            recovery_target(30_000, &ranges, &HashSet::new(), false),
            None
        );
    }

    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    #[test]
    fn a_seek_inside_the_current_chunk_keeps_its_inflight_work() {
        let mut state = PluginState::new(test_config()).unwrap();
        state.running = true;
        state.current_pos_ms = 30_000;
        let generation = state.transcription_worker.generation();

        let changed = state.apply_seek_target(SeekTarget {
            position_ms: 30_000,
            forward: true,
        });

        assert!(!changed);
        assert_eq!(state.transcription_worker.generation(), generation);
    }

    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    #[test]
    fn completed_local_session_resumes_for_an_unprocessed_seek_target() {
        let mut state = PluginState::new(test_config()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        state.mode = Some(ProcessingMode::Local {
            media_path: "movie.mkv".to_string(),
            file_length_ms: 60_000,
            subtitle_path: dir.path().join("movie.srt"),
        });
        state.current_pos_ms = 60_000;
        state.running = false;
        state.transcription_complete = true;
        state.processed_chunks.extend([15_000, 30_000, 45_000]);

        assert!(state.apply_seek_target(SeekTarget {
            position_ms: 0,
            forward: false,
        }));

        assert!(state.running);
        assert!(!state.transcription_complete);
        assert_eq!(state.current_pos_ms, 0);
        assert!(state.processed_chunks.is_empty());
    }

    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    #[test]
    fn completed_local_session_does_not_restart_for_a_processed_chunk() {
        let mut state = PluginState::new(test_config()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        state.mode = Some(ProcessingMode::Local {
            media_path: "movie.mkv".to_string(),
            file_length_ms: 60_000,
            subtitle_path: dir.path().join("movie.srt"),
        });
        state.current_pos_ms = 60_000;
        state.running = false;
        state.transcription_complete = true;
        state.processed_chunks.insert(30_000);
        let generation = state.transcription_worker.generation();

        assert!(!state.apply_seek_target(SeekTarget {
            position_ms: 30_000,
            forward: false,
        }));

        assert!(!state.running);
        assert!(state.transcription_complete);
        assert_eq!(state.current_pos_ms, 60_000);
        assert_eq!(state.transcription_worker.generation(), generation);
        assert!(state.is_chunk_processed(30_000));
    }

    /// A config with one declared STT source. The plugin refuses to start
    /// without one, so every test that builds a `PluginState` needs this.
    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    fn test_config() -> Config {
        let mut config = Config::default();
        let protocol = if cfg!(feature = "stt_openai") {
            crate::config::SttProtocol::OpenAi
        } else {
            crate::config::SttProtocol::Cloudflare
        };
        config.stt.sources.insert(
            "local".to_string(),
            crate::config::SttSourceConfig {
                protocol: Some(protocol),
                server_addr: Some("http://127.0.0.1:8000".to_string()),
                account_id: Some("test-account".to_string()),
                api_key: Some("test-token".to_string()),
                ..Default::default()
            },
        );
        config
    }

    #[cfg(any(feature = "stt_openai", feature = "stt_cloudflare"))]
    #[test]
    fn cached_srt_requires_a_matching_manifest_and_chunks_come_from_manifest_only() {
        let dir = tempfile::tempdir().unwrap();
        let srt_path = dir.path().join("movie.mkv.mpv_stt_plugin_rs.srt");
        let manifest_path = dir.path().join("movie.mkv.mpv_stt_plugin_rs.json");
        fs::write(&srt_path, "1\n00:00:00,000 --> 00:00:01,000\nhello\n").unwrap();
        let paths = CachePaths {
            subtitle_path: srt_path.clone(),
            manifest_path: manifest_path.clone(),
            media_identity_hash: "current-media".to_string(),
            media_fingerprint: Some("5:100".to_string()),
        };

        let mut state = PluginState::new(test_config()).unwrap();
        assert!(!state.load_cached_subs(&paths, 15_000));
        fs::write(
            &manifest_path,
            serde_json::to_string(&CacheManifest {
                schema_version: SUBTITLE_CACHE_SCHEMA_VERSION,
                timeline_version: SUBTITLE_TIMELINE_VERSION,
                media_identity_hash: paths.media_identity_hash.clone(),
                media_fingerprint: paths.media_fingerprint.clone(),
                chunk_size_ms: 15_000,
                processed_chunks: Vec::new(),
                translations: Vec::new(),
            })
            .unwrap(),
        )
        .unwrap();

        assert!(state.load_cached_subs(&paths, 15_000));
        assert_eq!(state.subtitle_manager.len(), 1);
        assert!(!state.is_chunk_processed(0));

        let mismatched_dir = tempfile::tempdir().unwrap();
        let mismatched_paths = CachePaths {
            subtitle_path: mismatched_dir.path().join("movie.srt"),
            manifest_path: mismatched_dir.path().join("movie.json"),
            media_identity_hash: "current-media".to_string(),
            media_fingerprint: Some("5:100".to_string()),
        };
        fs::write(
            &mismatched_paths.subtitle_path,
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        )
        .unwrap();
        fs::write(
            &mismatched_paths.manifest_path,
            serde_json::to_string(&CacheManifest {
                schema_version: SUBTITLE_CACHE_SCHEMA_VERSION,
                timeline_version: SUBTITLE_TIMELINE_VERSION,
                media_identity_hash: "different-media".to_string(),
                media_fingerprint: mismatched_paths.media_fingerprint.clone(),
                chunk_size_ms: 15_000,
                processed_chunks: vec![0],
                translations: Vec::new(),
            })
            .unwrap(),
        )
        .unwrap();
        let mut mismatched_state = PluginState::new(test_config()).unwrap();
        assert!(!mismatched_state.load_cached_subs(&mismatched_paths, 15_000));
        assert_eq!(mismatched_state.subtitle_manager.len(), 0);
        assert!(!mismatched_state.is_chunk_processed(0));
    }

    /// Failed cues are held during their backoff, and an explicit reset lets
    /// the user immediately retry after correcting the translation backend.
    #[cfg(feature = "stt_openai")]
    #[test]
    fn translation_retry_state_can_be_reset_explicitly() {
        let mut state = PluginState::new(test_config()).unwrap();

        assert!(!state.translation_failed(1_000));
        state.failed_translations.insert(1_000);
        state.translation_retries.insert(
            1_000,
            TranslationRetryState {
                attempts: 1,
                retry_at: Instant::now() + Duration::from_secs(2),
            },
        );
        assert!(state.translation_failed(1_000));

        // A cooldown is per-cue: its neighbours remain eligible for translation.
        assert!(!state.translation_failed(2_000));

        state.reset_translation_failures();
        assert!(!state.translation_failed(1_000));
        assert!(
            !state.translation_failure_reported,
            "the next failure should be allowed to speak again"
        );
    }

    /// Ending a session (stop / new file / plugin shutdown) starts a clean
    /// slate, so a stale cooldown cannot mute the next media's subtitles.
    #[cfg(feature = "stt_openai")]
    #[test]
    fn stopping_a_session_clears_translation_failures() {
        let mut state = PluginState::new(test_config()).unwrap();
        state.failed_translations.insert(1_000);
        state.translation_retries.insert(
            1_000,
            TranslationRetryState {
                attempts: 1,
                retry_at: Instant::now() + Duration::from_secs(2),
            },
        );
        state.translation_failure_reported = true;

        state.stop_transcription();

        assert!(state.failed_translations.is_empty());
        assert!(state.translation_retries.is_empty());
        assert!(!state.translation_failure_reported);
    }

    #[cfg(feature = "stt_openai")]
    #[test]
    fn stopping_a_session_does_not_permanently_shutdown_the_plugin() {
        let mut state = PluginState::new(test_config()).unwrap();
        state.running = true;
        state.mode = Some(ProcessingMode::Network);

        state.stop_transcription();

        assert!(!state.running);
        assert!(!state.shutting_down);
        assert!(state.mode.is_none());
        assert!(
            state.async_translation_queue.is_some(),
            "the translation worker must remain available for the next start"
        );

        // A terminal shutdown is deliberately separate and idempotent.
        state.shutdown();
        state.shutdown();
        assert!(state.shutting_down);
        assert!(state.async_translation_queue.is_none());
    }

    #[cfg(feature = "stt_openai")]
    #[test]
    fn transcription_worker_cancels_a_blocked_stt_request_promptly() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut byte = [0u8; 1];
            let _ = stream.read(&mut byte);
            let _ = request_seen_tx.send(());
            thread::sleep(Duration::from_secs(2));
        });

        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.wav");
        let output_wav = temp.path().join("chunk.wav");
        let output_prefix = temp.path().join("subs_append");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&input, spec).unwrap();
        for _ in 0..1_600 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();

        let mut config = Config::default();
        config.stt.sources.insert(
            "stub".to_string(),
            crate::config::SttSourceConfig {
                protocol: Some(crate::config::SttProtocol::OpenAi),
                server_addr: Some(format!("http://{addr}")),
                timeout_ms: Some(30_000),
                max_retry: Some(1),
                ..Default::default()
            },
        );
        let audio = AudioExtractor::default().with_ffmpeg_timeout(5_000);
        let stt = SttRunner::from_config(&config.stt).unwrap();
        let mut worker = TranscriptionWorker::new(audio, stt);
        worker
            .submit(TranscriptionJobInput {
                media_path: input.to_string_lossy().into_owned(),
                chunk_start_ms: 0,
                audio_start_ms: 0,
                align_audio_to_chunk_end: false,
                duration_ms: 100,
                wav_path: output_wav,
                output_prefix,
                span: logging::chunk_span(1, 0, 0, 100, 0),
            })
            .expect("failed to submit transcription job");
        request_seen_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("STT worker never started the request");

        let started = Instant::now();
        worker.shutdown();
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "worker shutdown waited for the blocked STT request: {:?}",
            started.elapsed()
        );
        assert!(worker.worker_handle.is_none());
    }

    #[cfg(feature = "stt_cloudflare")]
    fn accept_with_timeout(listener: &TcpListener) -> TcpStream {
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < deadline,
                        "timed out waiting for mock request"
                    );
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("mock server accept failed: {error}"),
            }
        }
    }

    #[cfg(feature = "stt_cloudflare")]
    fn read_complete_http_request(stream: &mut TcpStream) {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        let mut header_end = None;
        let mut content_length = 0usize;
        loop {
            let count = stream.read(&mut chunk).unwrap();
            assert_ne!(count, 0, "mock request ended before its body was read");
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
                return;
            }
        }
    }

    #[cfg(feature = "stt_cloudflare")]
    #[test]
    fn transcription_worker_processes_the_next_cloudflare_job_after_seek_cancel() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let (first_request_tx, first_request_rx) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let mut first = accept_with_timeout(&listener);
            let mut byte = [0u8; 1];
            first.read_exact(&mut byte).unwrap();
            first_request_tx.send(()).unwrap();
            // Keep the first request pending until the test cancels it.
            thread::sleep(Duration::from_millis(150));
            drop(first);

            let mut second = accept_with_timeout(&listener);
            read_complete_http_request(&mut second);
            let body = r#"{"success":true,"errors":[],"result":{"text":"replacement chunk","segments":[{"start":0.0,"end":0.8,"text":"replacement chunk"}]}}"#;
            write!(
                second,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.wav");
        let first_wav = temp.path().join("first.wav");
        let second_wav = temp.path().join("second.wav");
        let output_prefix = temp.path().join("chunk_append");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&input, spec).unwrap();
        for _ in 0..1_600 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();

        let stt = SttRunner::cloudflare_for_test(
            crate::stt::SttCloudflareConfig {
                account_id: "account123".to_string(),
                model: "@cf/openai/whisper-large-v3-turbo".to_string(),
                language: None,
                api_key: Some("test-token".to_string()),
                timeout_ms: 5_000,
                max_retry: 1,
            },
            &format!("http://{addr}/client/v4"),
        )
        .unwrap();
        let audio = AudioExtractor::default().with_ffmpeg_timeout(5_000);
        let mut worker = TranscriptionWorker::new(audio, stt);
        let media_path = input.to_string_lossy().into_owned();
        worker
            .submit(TranscriptionJobInput {
                media_path: media_path.clone(),
                chunk_start_ms: 0,
                audio_start_ms: 0,
                align_audio_to_chunk_end: false,
                duration_ms: 100,
                wav_path: first_wav,
                output_prefix: output_prefix.clone(),
                span: logging::chunk_span(1, 0, 0, 100, 0),
            })
            .expect("failed to submit the original chunk");
        first_request_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("Cloudflare worker never started the original request");

        worker.cancel_inflight();
        worker
            .submit(TranscriptionJobInput {
                media_path,
                chunk_start_ms: 15_000,
                audio_start_ms: 0,
                align_audio_to_chunk_end: false,
                duration_ms: 100,
                wav_path: second_wav,
                output_prefix: output_prefix.clone(),
                span: logging::chunk_span(1, 1, 15_000, 100, 1),
            })
            .expect("failed to submit the seek replacement chunk");

        let deadline = Instant::now() + Duration::from_secs(8);
        let result = loop {
            if let Some(result) = worker.try_recv() {
                break result;
            }
            assert!(
                Instant::now() < deadline,
                "replacement Cloudflare job did not finish"
            );
            thread::sleep(Duration::from_millis(10));
        };
        assert!(
            result.result.is_ok(),
            "replacement job failed: {:?}",
            result.result
        );
        assert!(
            std::fs::read_to_string(output_prefix.with_extension("srt"))
                .unwrap()
                .contains("replacement chunk")
        );
        worker.shutdown();
        server.join().unwrap();
    }
}
