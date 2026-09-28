use mpv_client::{Event, Handle, mpv_handle};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::Instant;
use tempfile::TempDir;
use tracing::{Span, debug, error, info, trace, warn};

use crate::audio::AudioExtractor;
use crate::common::MpvSttError;
use crate::config::Config;
use crate::logging::{self, LogSettings};
use crate::srt::SrtFile;
use crate::stt::{SttBackend, SttDeviceNotice, SttRunner};
use crate::subtitle_manager::SubtitleManager;
use crate::translate::{
    AsyncTranslationQueue, TranslationOutcome, TranslationTask, TranslatorConfig,
};

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
}

#[derive(Debug, Serialize, Deserialize)]
struct TranslationCacheEntry {
    start_ms: u32,
    original: String,
    translated: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct CacheManifest {
    chunk_size_ms: u64,
    processed_chunks: Vec<u64>,
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

struct TranscriptionJob {
    generation: u64,
    media_path: String,
    audio_start_ms: u64,
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
                        start_ms = job.audio_start_ms,
                        "transcription job started"
                    );

                    let result = worker_audio
                        .extract_audio_segment(
                            job.media_path.as_str(),
                            job.wav_path.to_str().unwrap_or_default(),
                            job.audio_start_ms,
                            job.duration_ms,
                        )
                        .and_then(|()| {
                            if worker_generation.load(Ordering::Acquire) != job.generation {
                                return Err(MpvSttError::SttCancelled);
                            }
                            stt_runner.transcribe(
                                job.wav_path.to_str().unwrap_or_default(),
                                job.output_prefix.to_str().unwrap_or_default(),
                                job.duration_ms,
                            )
                        });
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

    fn submit(
        &self,
        media_path: String,
        audio_start_ms: u64,
        duration_ms: u64,
        wav_path: PathBuf,
        output_prefix: PathBuf,
        span: Span,
    ) -> Option<u64> {
        let generation = self.generation.load(Ordering::Acquire);
        let job = TranscriptionJob {
            generation,
            media_path,
            audio_start_ms,
            duration_ms,
            wav_path,
            output_prefix,
            span,
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

struct PluginState {
    config: Config,
    paths: TempPaths,
    transcription_worker: TranscriptionWorker,
    pending_transcription: Option<PendingTranscription>,
    async_translation_queue: Option<AsyncTranslationQueue>,
    subtitle_manager: SubtitleManager,
    translation_cache: HashMap<u32, (String, String)>,
    /// Cues the translation backend definitively failed on, so they are not
    /// re-queued on every seek/chunk boundary. Cleared when the session is
    /// restarted, when the cache is cleared, or when the user toggles
    /// translation back on (an explicit "try again").
    failed_translations: HashSet<u32>,
    /// Whether the current session has already told the user that translation
    /// stopped working. Keeps a broken backend to one OSD message per session
    /// instead of one per chunk.
    translation_failure_reported: bool,
    processed_chunks: HashSet<u64>,
    network_cache: Option<CachePaths>,

    running: bool,
    shutting_down: bool,
    translate_enabled: bool, // Ctrl+Shift+t toggles whether new STT output gets translated
    subs_loaded: bool,
    current_pos_ms: u64,
    last_playback_pos_ms: Option<u64>,
    last_playback_instant: Option<Instant>,
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

        // Initialize the STT backend chosen at runtime by [stt] backend key.
        // Both remote backends are compiled in; `from_config` matches the
        // `config.stt.backend` enum to the active one.
        let stt_runner = SttRunner::from_config(&config.stt)?;
        let transcription_worker = TranscriptionWorker::new(audio_extractor, stt_runner);
        let paths = TempPaths::new()?;

        // Initialize async translation queue (always enabled)
        let async_translation_queue = Some(AsyncTranslationQueue::new(
            Self::build_translator_config(&config),
        ));

        Ok(Self {
            chunk_dur,
            config,
            paths,
            transcription_worker,
            pending_transcription: None,
            async_translation_queue,
            subtitle_manager: SubtitleManager::new(),
            translation_cache: HashMap::new(),
            failed_translations: HashSet::new(),
            translation_failure_reported: false,
            processed_chunks: HashSet::new(),
            network_cache: None,
            running: false,
            shutting_down: false,
            translate_enabled: true,
            subs_loaded: false,
            current_pos_ms: 0,
            last_playback_pos_ms: None,
            last_playback_instant: None,
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

    fn build_translator_config(config: &Config) -> TranslatorConfig {
        let default_libretranslate = crate::config::TranslateLibreTranslateConfig::default();
        let libretranslate = config
            .translate
            .libretranslate
            .as_ref()
            .unwrap_or(&default_libretranslate);
        let default_google = crate::config::TranslateGoogleConfig::default();
        let google = config.translate.google.as_ref().unwrap_or(&default_google);
        let default_edge = crate::config::TranslateEdgeConfig::default();
        let edge = config.translate.edge.as_ref().unwrap_or(&default_edge);
        let default_alibaba = crate::config::TranslateAlibabaConfig::default();
        let alibaba = config
            .translate
            .alibaba
            .as_ref()
            .unwrap_or(&default_alibaba);

        TranslatorConfig::new(
            config.translate.from_lang.clone(),
            config.translate.to_lang.clone(),
        )
        .with_backend(config.translate.backend)
        .with_timeout_ms(config.timeout.translate_ms)
        .with_concurrency(config.translate.concurrency)
        .with_server_addr(config.translate.server_addr.clone())
        .with_api_key(config.translate.api_key.clone())
        .with_libretranslate_server_addr(libretranslate.server_addr.clone())
        .with_libretranslate_api_key(libretranslate.api_key.clone())
        .with_google_server_addr(google.server_addr.clone())
        .with_edge_server_addr(edge.server_addr.clone())
        .with_edge_api_key(edge.api_key.clone())
        .with_alibaba_server_addr(alibaba.server_addr.clone())
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
    }

    /// Whether this cue's translation has already been given up on.
    fn translation_failed(&self, start_ms: u32) -> bool {
        self.failed_translations.contains(&start_ms)
    }

    /// Allow every cue to be translated again: the user asked for translation
    /// explicitly (toggle), or the whole cache was dropped.
    fn reset_translation_failures(&mut self) {
        if !self.failed_translations.is_empty() {
            debug!(
                cues = self.failed_translations.len(),
                "clearing the failed-translation records; every cue may be retried"
            );
        }
        self.failed_translations.clear();
        self.translation_failure_reported = false;
    }

    fn enqueue_missing_translations_for_chunk(&mut self, chunk_start_ms: u64) {
        let Some(queue) = self.async_translation_queue.as_ref() else {
            return;
        };

        let chunk_end = chunk_start_ms.saturating_add(self.active_chunk_size());
        let entries = self.subtitle_manager.entries_in_range(
            chunk_start_ms as u32,
            chunk_end.min(u64::from(u32::MAX)) as u32,
        );

        if entries.is_empty() {
            return;
        }

        let mut pending_tasks = Vec::new();
        let mut already_translated = 0usize;
        let mut skipped_failed = 0usize;

        for (start_ms, entry) in entries {
            let original = entry.text.trim();
            if original.is_empty() {
                continue;
            }

            if self.translation_failed(start_ms) {
                skipped_failed += 1;
                continue;
            }

            if let Some((_, translated)) = self.translation_cache.get(&start_ms) {
                if !translated.trim().is_empty() {
                    self.subtitle_manager
                        .update_translation(start_ms, translated);
                    already_translated += 1;
                    continue;
                }
            }

            if SubtitleManager::text_has_translation(&entry.text) {
                already_translated += 1;
                continue;
            }

            pending_tasks.push(TranslationTask {
                start_ms,
                text: entry.text.clone(),
            });
        }

        if skipped_failed > 0 {
            trace!(
                skipped_failed,
                "not re-queueing cues whose translation already failed"
            );
        }

        if !pending_tasks.is_empty() {
            trace!(
                queued = pending_tasks.len(),
                already_translated, "re-queueing missing translations"
            );
            for task in pending_tasks {
                queue.submit(task);
            }
        }
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
            // give-ups and offer the untranslated cues to the backend again,
            // which is how the user recovers after starting the gateway.
            self.reset_translation_failures();
            self.enqueue_missing_translations_for_chunk(self.current_pos_ms);
        }
    }

    /// Delete the current media's on-disk subtitle/translation cache and drop
    /// the in-memory translation + processed-chunk state, so replaying the same
    /// file re-transcribes instead of reusing stale cached subtitles.
    fn clear_cache(&mut self, client: &mut Handle) {
        let mut removed = 0usize;
        if let Some(media_id) = Self::media_id_for_cache(client) {
            if let Some(paths) = self.cache_paths_for_media(&media_id) {
                for p in [&paths.subtitle_path, &paths.manifest_path] {
                    if p.exists() {
                        match std::fs::remove_file(p) {
                            Ok(()) => removed += 1,
                            // A cache file the user asked to delete but could
                            // not be: worth recording, not worth a session
                            // teardown or a line on screen.
                            Err(e) => warn!(
                                error = %e,
                                cause = %logging::err_chain(&e),
                                path = %p.display(),
                                "cannot remove a cache file"
                            ),
                        }
                    }
                }
            }
        }
        // Drop in-memory state so a fresh playback re-transcribes from scratch.
        let chunk_entries = self.translation_cache.len();
        self.translation_cache.clear();
        self.reset_translation_failures();
        self.processed_chunks.clear();

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
        duration_ms: u64,
        subtitle_path: Option<PathBuf>,
    ) -> bool {
        if self.pending_transcription.is_some() {
            trace!(
                start_ms = audio_start_ms,
                "chunk not submitted: the previous one is still running"
            );
            return false;
        }

        let output_prefix = PathBuf::from(format!("{}_append", self.paths.tmp_sub.display()));
        // The chunk span is created here, on the event thread, and travels with
        // the job so the worker's extractor/STT logs land inside it.
        let span = logging::chunk_span(
            self.session_id,
            self.chunk_seq,
            audio_start_ms,
            duration_ms,
            self.transcription_worker.generation(),
        );
        let Some(generation) = self.transcription_worker.submit(
            media_path,
            audio_start_ms,
            duration_ms,
            self.paths.tmp_wav.clone(),
            output_prefix,
            span.clone(),
        ) else {
            error!(
                display = %logging::osd_line("the transcription worker is unavailable; restart playback"),
                "transcription worker is unavailable; the plugin cannot process audio"
            );
            return false;
        };
        self.chunk_seq += 1;

        debug!(
            start_ms = audio_start_ms,
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
        let Some(pending) = self.pending_transcription.take() else {
            return;
        };
        if worker_result.generation != pending.generation {
            return;
        }

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
                }
            }
            Err(MpvSttError::SttCancelled | MpvSttError::AudioExtractionCancelled) => {
                debug!("chunk cancelled");
                self.paths.cleanup_intermediate_subs();
            }
            Err(err) => {
                self.session_failures += 1;
                // The OSD gets the short form, the log keeps the cause chain.
                error!(
                    error = %err,
                    cause = %logging::err_chain(&err),
                    display = %logging::osd_line(&format!("STT failed: {err}")),
                    "chunk failed; ending the session"
                );
                self.stop_transcription();
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

        // Get current position
        let time_pos: f64 = client.get_property("time-pos").unwrap_or(0.0);
        self.current_pos_ms = (time_pos * 1000.0) as u64;
        self.last_playback_pos_ms = Some(self.current_pos_ms);

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
            self.network_cache = None;
            let chunk_size = self.network_chunk_size();
            self.current_pos_ms -= self.current_pos_ms % chunk_size;

            if self.config.playback.save_srt {
                if let Some(media_id) = Self::media_id_for_cache(client) {
                    if let Some(cache_paths) = self.cache_paths_for_media(&media_id) {
                        if let Some(parent) = cache_paths.subtitle_path.parent() {
                            if let Err(err) = fs::create_dir_all(parent) {
                                warn!(
                                    error = %err,
                                    cause = %logging::err_chain(&err),
                                    dir = %parent.display(),
                                    "cannot create the subtitle cache directory"
                                );
                            }
                        }
                        if cache_paths.subtitle_path.exists() {
                            if self.load_cached_subs(
                                &cache_paths.subtitle_path,
                                Some(&cache_paths.manifest_path),
                                self.network_chunk_size(),
                            ) {
                                let _ = client.command(&[
                                    "sub-add",
                                    cache_paths.subtitle_path.to_str().unwrap(),
                                ]);
                                self.subs_loaded = true;
                                self.cached_subtitle_path = Some(cache_paths.subtitle_path.clone());
                            }
                        }
                        self.network_cache = Some(cache_paths);
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

                // Calculate subtitle path next to the video file when possible.
                // SAF content:// URIs are not writable as filesystem paths.
                let subtitle_path = if self.config.playback.save_srt {
                    Self::get_subtitle_path_for_media_uri(&path)
                        .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"))
                } else {
                    self.paths.tmp_sub.with_extension("srt")
                };

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
                self.network_cache = None;

                if self.config.playback.save_srt && subtitle_path.exists() {
                    if self.load_cached_subs(&subtitle_path, None, self.local_chunk_size()) {
                        let _ = client.command(&["sub-add", subtitle_path.to_str().unwrap()]);
                        self.subs_loaded = true;
                        self.cached_subtitle_path = Some(subtitle_path.clone());
                    }
                }

                // Create initial subtitles if this chunk hasn't been processed.
                if !self.is_chunk_processed(self.current_pos_ms) {
                    let remaining_ms = file_length_ms.saturating_sub(self.current_pos_ms);
                    self.chunk_dur = self.local_chunk_size().min(remaining_ms).max(1);
                    self.schedule_transcription(
                        path.clone(),
                        self.current_pos_ms,
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

        if !self.running {
            if self.transcription_complete {
                let subtitle_path = match &self.mode {
                    Some(ProcessingMode::Network) => self
                        .network_cache
                        .as_ref()
                        .map(|cache| cache.subtitle_path.clone()),
                    Some(ProcessingMode::Local { subtitle_path, .. }) => {
                        Some(subtitle_path.clone())
                    }
                    None => None,
                };
                self.process_translation_results(client, subtitle_path.as_deref());
            }
            return;
        }

        if self.pending_transcription.is_some() && self.check_seek(client) {
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
            .network_cache
            .as_ref()
            .map(|cache| cache.subtitle_path.clone());

        // Check for seek first. If cache isn't ready yet after a seek, we still want to update
        // `current_pos_ms` so we don't keep generating subtitles for the old position.
        if self.check_seek(client) {
            return;
        }

        // Check for completed translations from async queue
        self.process_translation_results(client, subtitle_path.as_deref());

        // Get cache end time
        let cache_end_sec: Option<f64> = client.get_property("demuxer-cache-time").ok();
        if cache_end_sec.is_none() {
            trace!("demuxer cache has not reported a time yet");
            return; // Cache not ready yet
        }
        let cache_end_ms = (cache_end_sec.unwrap() * 1000.0) as u64;
        let available_ms = cache_end_ms.saturating_sub(self.current_pos_ms);
        let chunk_ms = self.network_chunk_size();

        if available_ms < chunk_ms {
            trace!(
                needed_ms = self.current_pos_ms + chunk_ms,
                cached_ms = cache_end_ms,
                "waiting for the demuxer cache to grow"
            );
            return;
        }

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

        let chunk_end_ms = self.current_pos_ms.saturating_add(chunk_ms);
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

        if chunk_end_ms > cache_end_ms {
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
        // Check for seek
        if self.check_seek(client) {
            return;
        }

        // Check for completed translations from async queue
        self.process_translation_results(client, Some(subtitle_path));

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
        let playback_pos: Option<f64> = client.get_property("time-pos").ok();
        if let Some(pos) = playback_pos {
            let playback_pos_ms = (pos * 1000.0) as u64;
            let now = Instant::now();

            // Detect user seek by comparing against the last observed playback position.
            // IMPORTANT: `current_pos_ms` is the *processing cursor* (next chunk start), which can
            // legitimately run ahead of playback when the cache is full. Comparing playback to
            // `current_pos_ms` causes false "seek backward" detections and makes subtitles vanish.
            let Some(last_ms) = self.last_playback_pos_ms.replace(playback_pos_ms) else {
                self.last_playback_instant = Some(now);
                return false;
            };
            let last_instant = self.last_playback_instant.replace(now);

            // Avoid treating normal playback progression (or time spent inside STT/translate)
            // as a seek. Since we process in chunk units, only treat jumps of >= 1 chunk as seek.
            let chunk_size = self.active_chunk_size();
            let seek_threshold_ms = std::cmp::max(5_000, chunk_size);
            let delta_ms = playback_pos_ms.abs_diff(last_ms);
            if delta_ms < seek_threshold_ms {
                return false;
            }
            if let Some(last_instant) = last_instant {
                let elapsed_ms = now
                    .duration_since(last_instant)
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX);
                if delta_ms <= elapsed_ms.saturating_add(seek_threshold_ms) {
                    return false;
                }
            }

            let new_pos = playback_pos_ms - (playback_pos_ms % chunk_size);
            if new_pos == self.current_pos_ms {
                debug!(
                    new_pos,
                    "seek landed inside the chunk in flight; keeping its tasks"
                );
                return false;
            }

            let forward = playback_pos_ms > last_ms;
            debug!(
                direction = if forward { "forward" } else { "backward" },
                from_ms = last_ms,
                to_ms = new_pos,
                delta_ms,
                "seeked"
            );
            // Drawn directly: the seek overlay is transient feedback for an
            // action the user just took, not a diagnostic worth queueing.
            let _ = client.command(&[
                "show-text",
                &format!("STT: Seeked to {}", Self::format_progress(new_pos)),
                "3000",
            ]);

            self.current_pos_ms = new_pos;
            self.handle_seek_to(new_pos, forward);
            return true;
        }
        false
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
        }

        if self.is_chunk_processed(new_pos) {
            self.enqueue_missing_translations_for_chunk(new_pos);
        }
    }

    /// Process one chunk from network cache
    fn process_chunk(
        &mut self,
        client: &mut Handle,
        chunk_ms: u64,
        subtitle_path: Option<&Path>,
    ) -> bool {
        // Dump cache
        let start_sec = self.current_pos_ms as f64 / 1000.0;
        let end_sec = (self.current_pos_ms + chunk_ms) as f64 / 1000.0;
        trace!(
            start_sec,
            end_sec, "dumping the demuxer cache to a temp file"
        );

        let dump_result = client.command(&[
            "dump-cache",
            &start_sec.to_string(),
            &end_sec.to_string(),
            self.paths.tmp_cache.to_str().unwrap(),
        ]);

        if dump_result.is_err() {
            error!(
                start_sec,
                end_sec, "mpv refused to dump the demuxer cache; this chunk cannot be transcribed"
            );
            return false;
        }

        self.schedule_transcription(
            self.paths.tmp_cache.to_string_lossy().into_owned(),
            0,
            chunk_ms,
            subtitle_path.map(Path::to_path_buf),
        )
    }

    /// Process one chunk from local file
    fn process_chunk_local(&mut self, media_path: &str, subtitle_path: &Path) -> bool {
        self.schedule_transcription(
            media_path.to_string(),
            self.current_pos_ms,
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
        self.mark_chunk_processed(chunk_start_ms);
        debug!(
            entries = srt_file.entries.len(),
            total = self.subtitle_manager.len(),
            "subtitles merged"
        );

        let mut pending_tasks = Vec::new();
        let mut already_translated = 0usize;
        let mut skipped_failed = 0usize;

        for entry in &srt_file.entries {
            let original = entry.text.trim();
            if original.is_empty() {
                continue;
            }
            let start_ms = Self::timestamp_to_millis(entry.start_time);
            if SubtitleManager::text_has_translation(&entry.text) {
                already_translated += 1;
                continue;
            }
            // Already attempted and given up on (e.g. this session restarted
            // mid-file, or the seek path re-queued this chunk): the original is
            // on screen and stays there rather than being retried forever.
            if self.translation_failed(start_ms) {
                skipped_failed += 1;
                continue;
            }

            pending_tasks.push(TranslationTask {
                start_ms,
                text: entry.text.clone(),
            });
        }

        if !self.save_subs(client, &main_srt) {
            return false;
        }

        // Translate using async translation queue (Ctrl+Shift+t toggles translate_enabled).
        // Purely additive: the originals are already saved above, so nothing
        // here can take them away.
        if self.translate_enabled && !pending_tasks.is_empty() {
            if let Some(ref queue) = self.async_translation_queue {
                let queued = pending_tasks.len();
                for task in pending_tasks {
                    queue.submit(task);
                }
                debug!(
                    queued,
                    already_translated, skipped_failed, "submitted cues for translation"
                );
            }
        } else if !self.translate_enabled && !pending_tasks.is_empty() {
            debug!(
                cues = pending_tasks.len(),
                "translation is off; keeping the original text"
            );
        } else if already_translated > 0 {
            debug!(already_translated, "every cue already had a translation");
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

    /// Process completed translation outcomes from the async queue.
    ///
    /// Translations are merged into the subtitle text. Give-ups are recorded
    /// instead, so the cue is neither retried forever nor silently lost: the
    /// original line stays on screen, and the user is told once that
    /// translation is not coming. Nothing here can affect the STT session —
    /// a dead translation backend leaves subtitles working and untranslated.
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

        for outcome in outcomes {
            match outcome {
                TranslationOutcome::Translated(result) => {
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
                    warn!(
                        start_ms = failure.start_ms,
                        cause = %failure.cause,
                        reason = %failure.reason,
                        "giving up on this cue's translation; the original stays on screen"
                    );
                    // Remembering the failure is what keeps it from being
                    // re-queued on every seek or chunk boundary.
                    self.failed_translations.insert(failure.start_ms);
                    failed += 1;
                    last_reason = failure.reason;
                }
            }
        }
        trace!(translated, failed, "translation outcomes applied");

        if failed > 0 {
            // The reason is worth seeing, but a backend that is down fails
            // every cue, so say it once per session and keep the count in the
            // log instead of on screen.
            if !self.translation_failure_reported {
                self.translation_failure_reported = true;
                info!(
                    failed,
                    reason = %last_reason,
                    display = %logging::osd_line(&format!(
                        "翻译失败 ({failed} 条), 仅显示原文: {last_reason}"
                    )),
                    "translation backend is not answering"
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
    /// alive. This path is used by the toggle, EndFile, completed media and
    /// recoverable STT errors, so it must remain restartable.
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

        self.running = false;
        self.transcription_worker.cancel_inflight();
        self.pending_transcription = None;

        // Cancel tasks belonging to this media, but keep the worker alive so
        // Ctrl+Shift+S and the next file can start a fresh session.
        if let Some(queue) = self.async_translation_queue.as_ref() {
            queue.cancel_inflight();
        }

        self.paths.cleanup();
        self.subtitle_manager.clear();
        self.translation_cache.clear();
        self.reset_translation_failures();
        self.processed_chunks.clear();
        self.network_cache = None;
        self.subs_loaded = false;
        self.current_pos_ms = 0;
        self.last_playback_pos_ms = None;
        self.last_playback_instant = None;
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

    fn cache_paths_for_media(&self, media_id: &str) -> Option<CachePaths> {
        let dir = Self::cache_root_dir()?;
        let hash = Self::fnv1a_hash64(media_id);
        let stem = format!("{:016x}", hash);
        Some(CachePaths {
            subtitle_path: dir.join(format!("{stem}.srt")),
            manifest_path: dir.join(format!("{stem}.json")),
        })
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

    fn load_cached_subs(
        &mut self,
        srt_path: &Path,
        manifest_path: Option<&Path>,
        chunk_size_ms: u64,
    ) -> bool {
        if !srt_path.exists() {
            return false;
        }

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
        // A fresh media/session: earlier give-ups say nothing about this one.
        self.reset_translation_failures();
        self.processed_chunks.clear();
        self.subtitle_manager.add_from_srt(&srt_file);

        let chunk_size = chunk_size_ms.max(1);
        for entry in &srt_file.entries {
            let start_ms = Self::timestamp_to_millis(entry.start_time) as u64;
            let chunk_start = start_ms - (start_ms % chunk_size);
            self.processed_chunks.insert(chunk_start);
        }

        if let Some(path) = manifest_path {
            if let Some(manifest) = self.load_cache_manifest(path) {
                if manifest.chunk_size_ms == chunk_size {
                    for chunk in manifest.processed_chunks {
                        self.processed_chunks.insert(chunk);
                    }
                }
                for entry in manifest.translations {
                    if !entry.translated.trim().is_empty() {
                        self.translation_cache
                            .insert(entry.start_ms, (entry.original, entry.translated));
                    }
                }
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
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(err) => {
                debug!(
                    error = %err,
                    path = %path.display(),
                    "no usable cache manifest; deriving progress from the subtitles"
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
                    "cache manifest is unreadable; deriving progress from the subtitles"
                );
                None
            }
        }
    }

    fn save_cache_manifest_if_needed(&self) {
        let Some(cache) = &self.network_cache else {
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

        let manifest = CacheManifest {
            chunk_size_ms: self.network_chunk_size(),
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

    /// Get subtitle path for a media file (same directory, same name, .srt extension)
    fn get_subtitle_path_for_media(media_path: &str) -> PathBuf {
        let path = Path::new(media_path);
        if let Some(stem) = path.file_stem() {
            if let Some(parent) = path.parent() {
                return parent.join(format!("{}.srt", stem.to_string_lossy()));
            }
        }
        // Fallback: just append .srt
        PathBuf::from(format!("{}.srt", media_path))
    }

    /// Try to map a media path/URI to a writable filesystem subtitle path.
    /// Returns None for non-filesystem URIs like content://.
    fn get_subtitle_path_for_media_uri(media_path: &str) -> Option<PathBuf> {
        if let Some(rest) = media_path.strip_prefix("file://") {
            return Some(Self::get_subtitle_path_for_media(rest));
        }
        if media_path.contains("://") {
            return None;
        }
        Some(Self::get_subtitle_path_for_media(media_path))
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
            backend = %config.stt.backend,
            translate_backend = %config.translate.backend,
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
                }
                Event::None => {
                    if state.shutting_down {
                        continue;
                    }
                    // Timeout - use this to tick the processing
                    state.tick(client);
                }
                _ => {
                    if state.shutting_down {
                        continue;
                    }
                    // Other events - still tick
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
    use std::io::Read;
    use std::net::TcpListener;
    use std::time::{Duration, Instant};

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

    /// A cue whose translation was given up on must not be offered to the
    /// backend again, and the user must be able to retry explicitly (toggle /
    /// clear cache) once the backend is back.
    #[test]
    fn failed_translations_are_not_requeued_until_explicitly_reset() {
        let mut state = PluginState::new(Config::default()).unwrap();

        assert!(!state.translation_failed(1_000));
        state.failed_translations.insert(1_000);
        assert!(state.translation_failed(1_000));

        // A give-up is per-cue: its neighbours are still offered.
        assert!(!state.translation_failed(2_000));

        state.reset_translation_failures();
        assert!(!state.translation_failed(1_000));
        assert!(
            !state.translation_failure_reported,
            "the next failure should be allowed to speak again"
        );
    }

    /// Ending a session (stop / new file / plugin shutdown) starts a clean
    /// slate, so a stale give-up cannot mute the next media's subtitles.
    #[test]
    fn stopping_a_session_clears_translation_failures() {
        let mut state = PluginState::new(Config::default()).unwrap();
        state.failed_translations.insert(1_000);
        state.translation_failure_reported = true;

        state.stop_transcription();

        assert!(state.failed_translations.is_empty());
        assert!(!state.translation_failure_reported);
    }

    #[test]
    fn stopping_a_session_does_not_permanently_shutdown_the_plugin() {
        let mut state = PluginState::new(Config::default()).unwrap();
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
        let openai = config.stt.openai.as_mut().unwrap();
        openai.server_addr = format!("http://{addr}");
        openai.timeout_ms = 30_000;
        openai.max_retry = 1;
        let audio = AudioExtractor::default().with_ffmpeg_timeout(5_000);
        let stt = SttRunner::from_config(&config.stt).unwrap();
        let mut worker = TranscriptionWorker::new(audio, stt);
        worker
            .submit(
                input.to_string_lossy().into_owned(),
                0,
                100,
                output_wav,
                output_prefix,
                logging::chunk_span(1, 0, 0, 100, 0),
            )
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
}
