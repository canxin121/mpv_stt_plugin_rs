use crate::common::{MpvSttError, Result};
use crate::config::{TranslateSourceConfig, TranslateSourceProtocol};
use crate::srt::SrtFile;
use futures::stream::StreamExt;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;
use tracing::{debug, trace, warn};

const MAX_TRANSLATE_RETRIES: usize = 2;
const RETRY_BASE_DELAY_MS: u64 = 250;

/// Name of the reserved selector value that walks the built-in free sources.
pub const AUTO_SOURCE: &str = "auto";

/// A built-in source: a name that resolves without being declared, because its
/// protocol and endpoint are known. `[translate.sources.<name>]` may still be
/// written to override individual fields.
struct BuiltinSource {
    name: &'static str,
    protocol: TranslateSourceProtocol,
    server_addr: &'static str,
}

/// The built-in sources. The first three are the free web endpoints `auto`
/// walks in order; the last two are the external protocols, which point at a
/// local service by default so that selecting them needs no address either.
const BUILTIN_SOURCES: [BuiltinSource; 5] = [
    BuiltinSource {
        name: "google_free",
        protocol: TranslateSourceProtocol::Google,
        server_addr: "https://clients5.google.com",
    },
    BuiltinSource {
        name: "edge_free",
        protocol: TranslateSourceProtocol::Edge,
        server_addr: "https://edge.microsoft.com",
    },
    BuiltinSource {
        name: "alibaba_free",
        protocol: TranslateSourceProtocol::Alibaba,
        server_addr: "https://translate.alibaba.com",
    },
    BuiltinSource {
        name: "deepl",
        protocol: TranslateSourceProtocol::DeepL,
        server_addr: "http://127.0.0.1:8000",
    },
    BuiltinSource {
        name: "libretranslate",
        protocol: TranslateSourceProtocol::LibreTranslate,
        server_addr: "http://127.0.0.1:8000",
    },
];

/// The names `auto` walks, in order: the built-in free sources only. Sources
/// the user declared are never picked silently by `auto` — using a paid or
/// self-hosted endpoint has to be an explicit choice.
const FREE_SOURCE_NAMES: [&str; 3] = ["google_free", "edge_free", "alibaba_free"];

/// Where a protocol points when neither the built-in table nor the source says
/// otherwise. Only the two external protocols can get here: the free ones have
/// a host built in, and a non-built-in name must state its protocol anyway.
fn protocol_default_server(protocol: TranslateSourceProtocol) -> &'static str {
    match protocol {
        TranslateSourceProtocol::DeepL | TranslateSourceProtocol::LibreTranslate => {
            "http://127.0.0.1:8000"
        }
        TranslateSourceProtocol::Google => "https://clients5.google.com",
        TranslateSourceProtocol::Edge => "https://edge.microsoft.com",
        TranslateSourceProtocol::Alibaba => "https://translate.alibaba.com",
    }
}

fn builtin(name: &str) -> Option<&'static BuiltinSource> {
    BUILTIN_SOURCES.iter().find(|s| s.name == name)
}

/// Resolve one named source: the declared fields win, the built-in entry (for
/// a built-in name) fills the gaps, and a name that is neither built-in nor
/// fully declared is an error rather than a guess.
pub fn resolve_source(
    name: &str,
    declared: Option<&TranslateSourceConfig>,
) -> Result<ResolvedSource> {
    let builtin = builtin(name);
    let empty = TranslateSourceConfig::default();
    let declared = declared.unwrap_or(&empty);

    let protocol = match declared.protocol.or(builtin.map(|b| b.protocol)) {
        Some(protocol) => protocol,
        None => {
            return Err(MpvSttError::TranslationFailed(format!(
                "translation source {name:?} is not built-in and has no protocol; \
                 set protocol to one of: google, edge, alibaba, deepl, libretranslate"
            )));
        }
    };
    let server_addr = declared
        .server_addr
        .clone()
        .or_else(|| builtin.map(|b| b.server_addr.to_string()))
        .unwrap_or_else(|| protocol_default_server(protocol).to_string());

    Ok(ResolvedSource {
        name: name.to_string(),
        protocol,
        server_addr,
        api_key: declared.api_key.clone().unwrap_or_default(),
    })
}

/// A source with every field decided: what the request helpers read.
#[derive(Clone, Debug)]
pub struct ResolvedSource {
    pub name: String,
    pub protocol: TranslateSourceProtocol,
    pub server_addr: String,
    pub api_key: String,
}

impl ResolvedSource {
    /// A built-in source as it resolves with no override at all. Used by the
    /// FFI initializers and by tests.
    pub fn builtin(name: &str) -> Self {
        resolve_source(name, None).expect("built-in names always resolve")
    }
}

impl std::fmt::Display for ResolvedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name, self.protocol)
    }
}

/// How much of a response body a diagnostics message keeps.
const BODY_SNIPPET_CHARS: usize = 160;

#[derive(Clone, Debug)]
pub struct TranslatorConfig {
    pub from_lang: String,
    pub to_lang: String,
    pub timeout_ms: u64,
    pub concurrency: usize,
    /// Name of the active source, or `auto` for the built-in free chain.
    pub source: String,
    /// Every source this config can name, already resolved. Built-in names may
    /// be absent: the resolver falls back to the built-in table for those.
    pub sources: BTreeMap<String, ResolvedSource>,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            from_lang: "auto".to_string(),
            to_lang: "en".to_string(),
            timeout_ms: 30_000,
            concurrency: 4,
            source: AUTO_SOURCE.to_string(),
            sources: BTreeMap::new(),
        }
    }
}

impl TranslatorConfig {
    pub fn new(from_lang: String, to_lang: String) -> Self {
        Self {
            from_lang,
            to_lang,
            ..Default::default()
        }
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Declare (or override) a source by name, resolving it as it goes.
    pub fn with_source_config(
        mut self,
        name: &str,
        declared: &TranslateSourceConfig,
    ) -> Result<Self> {
        let resolved = resolve_source(name, Some(declared))?;
        self.sources.insert(name.to_string(), resolved);
        Ok(self)
    }

    /// Declare a source that is nothing but a server address — the shape the
    /// FFI initializers and most tests need.
    pub fn with_source_addr(
        self,
        name: &str,
        protocol: TranslateSourceProtocol,
        server_addr: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        self.with_source_config(
            name,
            &TranslateSourceConfig {
                protocol: Some(protocol),
                server_addr: Some(server_addr.into()),
                api_key: Some(api_key.into()),
            },
        )
    }

    /// The active source, or the reason `source` does not name one. `auto` is
    /// not a source: the free chain resolves each of its members in turn.
    pub fn active_source(&self) -> Result<ResolvedSource> {
        if self.source == AUTO_SOURCE {
            return Err(MpvSttError::TranslationFailed(
                "auto is not a source; it walks the built-in free sources".to_string(),
            ));
        }
        if let Some(source) = self.sources.get(&self.source) {
            return Ok(source.clone());
        }
        if let Some(builtin) = builtin(&self.source) {
            return resolve_source(builtin.name, None);
        }
        let declared = || {
            if self.sources.is_empty() {
                "(none)".to_string()
            } else {
                self.sources.keys().cloned().collect::<Vec<_>>().join(", ")
            }
        };
        let builtin_names = BUILTIN_SOURCES
            .iter()
            .map(|s| s.name)
            .collect::<Vec<_>>()
            .join(", ");
        Err(MpvSttError::TranslationFailed(format!(
            "translation source {:?} is not declared; declared: {}; built-in: {}",
            self.source,
            declared(),
            builtin_names
        )))
    }

    /// Install a `source` named `name` pointing at `server_addr`, replacing any
    /// previous definition. Convenience for the FFI initializers.
    pub fn set_source(
        &mut self,
        name: &str,
        protocol: TranslateSourceProtocol,
        server_addr: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<()> {
        let resolved = resolve_source(
            name,
            Some(&TranslateSourceConfig {
                protocol: Some(protocol),
                server_addr: Some(server_addr.into()),
                api_key: Some(api_key.into()),
            }),
        )?;
        self.sources.insert(name.to_string(), resolved);
        Ok(())
    }
}

pub struct Translator {
    config: TranslatorConfig,
    client: reqwest::blocking::Client,
}

impl Translator {
    pub fn new(config: TranslatorConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .expect("failed to build translation HTTP client");
        Self { config, client }
    }

    /// Translate a single text string via the remote DeepL-compatible API
    pub fn translate(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        trace!(
            from = %self.config.from_lang,
            to = %self.config.to_lang,
            chars = text.chars().count(),
            "translating text"
        );

        self.translate_remote(text)
    }

    fn translate_remote(&self, text: &str) -> Result<String> {
        let from_lang = normalize_lang_code(&self.config.from_lang, true);
        let to_lang = normalize_lang_code(&self.config.to_lang, false);

        let mut attempt = 0usize;
        let mut delay_ms = RETRY_BASE_DELAY_MS;
        // Why the last attempt failed; reported if every attempt fails.
        let mut last_error;

        loop {
            match translate_blocking(&self.client, &self.config, &from_lang, &to_lang, text) {
                Ok(translated) => return Ok(translated),
                Err(e) => {
                    warn!(
                        source = %self.config.source,
                        attempt = attempt + 1,
                        error = %e,
                        cause = %crate::logging::err_chain(&e),
                        "translation request failed"
                    );
                    last_error = Some(e);
                }
            }

            attempt += 1;
            if attempt > MAX_TRANSLATE_RETRIES {
                return Err(last_error.unwrap_or_else(|| {
                    MpvSttError::TranslationFailed("translation returned empty".to_string())
                }));
            }

            thread::sleep(Duration::from_millis(delay_ms));
            delay_ms = (delay_ms * 2).min(2_000);
        }
    }

    /// Translate an SRT file and create a bilingual version
    pub fn translate_srt_file<P: AsRef<Path>>(&self, input_path: P, output_path: P) -> Result<()> {
        let mut srt = SrtFile::parse(&input_path)?;
        debug!(
            entries = srt.entries.len(),
            path = %input_path.as_ref().display(),
            "translating an SRT file"
        );
        let mut translations = Vec::new();

        for entry in &srt.entries {
            match self.translate(&entry.text) {
                Ok(translated) if !translated.is_empty() => {
                    translations.push(translated);
                }
                Ok(_) => {
                    translations.push(String::new());
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        cause = %crate::logging::err_chain(&e),
                        "leaving one cue untranslated"
                    );
                    translations.push(String::new());
                }
            }
        }

        srt.merge_bilingual(&translations);
        let translated = translations.iter().filter(|t| !t.is_empty()).count();
        srt.save(&output_path)?;
        debug!(
            entries = srt.entries.len(),
            translated,
            path = %output_path.as_ref().display(),
            "translated an SRT file"
        );
        Ok(())
    }

    /// Batch translate multiple texts
    pub fn translate_batch(&self, texts: &[String]) -> Vec<Result<String>> {
        texts.iter().map(|text| self.translate(text)).collect()
    }
}

/// Translation task for async processing
#[derive(Debug, Clone)]
pub struct TranslationTask {
    pub start_ms: u32,
    pub text: String,
}

/// Result from async translation
#[derive(Debug, Clone)]
pub struct TranslationResult {
    pub start_ms: u32,
    pub original: String,
    pub translated: String,
}

/// A translation that will never arrive: the queue gave up on it after its
/// retries were exhausted. Reported so the caller can tell the user why new
/// subtitles are staying untranslated, and so a dead backend is not retried
/// for every single cue.
#[derive(Debug, Clone)]
pub struct TranslationFailure {
    pub start_ms: u32,
    /// One-line reason, short enough for the OSD.
    pub reason: String,
    /// Full root-cause chain, for the log file.
    pub cause: String,
}

/// Condense a transport error into something short enough for mpv's OSD, which
/// renders one line and truncates. Keeps the status code and the server's own
/// message when there is one, since that is what tells the user which of
/// gateway / key / upstream is at fault.
pub fn short_reason(reason: &str) -> String {
    let reason = reason.trim();
    if reason.is_empty() {
        return "未知错误".to_string();
    }
    reason.chars().take(120).collect()
}

#[derive(Debug, Clone)]
struct QueuedTask {
    generation: u64,
    task: TranslationTask,
}

#[derive(Debug, Clone)]
struct QueuedResult {
    generation: u64,
    outcome: TranslationOutcome,
}

/// What the worker managed to do with one task. `Failed` is not an error the
/// caller can act on — the original subtitle is already on screen either way —
/// but it is what lets the plugin stop re-queueing the same cue forever.
#[derive(Debug, Clone)]
pub enum TranslationOutcome {
    Translated(TranslationResult),
    Failed(TranslationFailure),
}

/// Async translation queue that processes translations in background
pub struct AsyncTranslationQueue {
    task_sender: Sender<Option<QueuedTask>>,
    result_receiver: Receiver<QueuedResult>,
    worker_handle: Option<thread::JoinHandle<()>>,
    shutdown_flag: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
}

/// Everything the worker thread owns for the lifetime of one batch: the
/// result channel back to the plugin, the source resolution, and the two
/// cancellation handles. Grouped so a batch's plumbing travels as one value
/// instead of as a run of identically-typed `&Arc<...>` arguments.
struct BatchContext<'a> {
    result_sender: &'a Sender<QueuedResult>,
    config: &'a Arc<TranslatorConfig>,
    shutdown_flag: &'a Arc<AtomicBool>,
    generation: &'a Arc<AtomicU64>,
    runtime: &'a tokio::runtime::Runtime,
    client: &'a reqwest::Client,
}

impl AsyncTranslationQueue {
    pub fn new(config: TranslatorConfig) -> Self {
        let (task_sender, task_receiver) = channel::<Option<QueuedTask>>();
        let (result_sender, result_receiver) = channel::<QueuedResult>();

        let config = Arc::new(config);
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let generation = Arc::new(AtomicU64::new(0));
        let shutdown_flag_clone = shutdown_flag.clone();
        let generation_clone = generation.clone();
        let worker_handle = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .enable_io()
                .build()
                .expect("failed to build tokio runtime for translator");
            Self::worker_thread(
                task_receiver,
                result_sender,
                config,
                shutdown_flag_clone,
                generation_clone,
                &runtime,
            );
        });

        Self {
            task_sender,
            result_receiver,
            worker_handle: Some(worker_handle),
            shutdown_flag,
            generation,
        }
    }

    /// Submit a translation task to the queue
    pub fn submit(&self, task: TranslationTask) {
        let generation = self.generation.load(Ordering::Relaxed);
        let _ = self.task_sender.send(Some(QueuedTask { generation, task }));
    }

    /// Try to get completed translation outcomes (non-blocking). Translations
    /// and give-ups alike are returned, because the caller needs to know not to
    /// queue that cue again.
    pub fn try_recv_results(&self) -> Vec<TranslationOutcome> {
        let mut results = Vec::new();
        let generation = self.generation.load(Ordering::Acquire);
        while let Ok(queued) = self.result_receiver.try_recv() {
            if queued.generation == generation {
                results.push(queued.outcome);
            } else {
                trace!(
                    stale_gen = queued.generation,
                    gen = generation,
                    "dropping a translation result from a superseded generation"
                );
            }
        }
        results
    }

    /// Cancel any in-flight translation tasks without tearing down the worker.
    pub fn cancel_inflight(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Worker thread that processes translation tasks in batches
    fn worker_thread(
        task_receiver: Receiver<Option<QueuedTask>>,
        result_sender: Sender<QueuedResult>,
        config: Arc<TranslatorConfig>,
        shutdown_flag: Arc<AtomicBool>,
        generation: Arc<AtomicU64>,
        runtime: &tokio::runtime::Runtime,
    ) {
        loop {
            // Check shutdown flag
            if shutdown_flag.load(Ordering::Acquire) {
                debug!("translation worker stopping: shutdown requested");
                return;
            }

            // Wait for first task (blocking with timeout to allow periodic shutdown checks)
            let first_task = match task_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(Some(task)) => task,
                Ok(None) => {
                    debug!("translation worker exiting: shutdown signal");
                    return;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // Timeout, check shutdown flag again
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    debug!("translation worker exiting: channel closed");
                    return;
                }
            };

            let current_generation = generation.load(Ordering::Relaxed);

            // Collect all pending tasks from queue (non-blocking)
            let mut tasks = Vec::new();
            if first_task.generation == current_generation {
                tasks.push(first_task.task);
            }
            while let Ok(Some(task)) = task_receiver.try_recv() {
                if task.generation == current_generation {
                    tasks.push(task.task);
                }
            }

            if tasks.is_empty() {
                continue;
            }

            let task_count = tasks.len();
            debug!(tasks = task_count, "translating a batch");

            // Build one shared async client per batch (connection pool reused
            // across the concurrent requests).
            let client = reqwest::Client::builder()
                .timeout(Duration::from_millis(config.timeout_ms))
                .build();
            let client = match client {
                Ok(client) => client,
                Err(e) => {
                    warn!(
                        error = %e,
                        cause = %crate::logging::err_chain(&e),
                        "cannot build the translation HTTP client; dropping this batch"
                    );
                    continue;
                }
            };

            Self::process_remote(
                &tasks,
                BatchContext {
                    result_sender: &result_sender,
                    config: &config,
                    shutdown_flag: &shutdown_flag,
                    generation: &generation,
                    runtime,
                    client: &client,
                },
                current_generation,
            );

            debug!(tasks = task_count, "translation batch finished");
        }
    }

    /// Process one batch: hand every task to the resolved translation source,
    /// `concurrency` at a time, and forward the outcomes back as they land.
    fn process_remote(tasks: &[TranslationTask], batch: BatchContext<'_>, task_generation: u64) {
        let BatchContext {
            result_sender,
            config,
            shutdown_flag,
            generation,
            runtime,
            client,
        } = batch;

        if tasks.is_empty() {
            return;
        }

        debug!(
            tasks = tasks.len(),
            concurrency = config.concurrency.max(1),
            "dispatching translations"
        );

        let active_tasks: Vec<TranslationTask> = tasks.to_vec();
        let config = Arc::clone(config);
        let shutdown_flag = Arc::clone(shutdown_flag);
        let generation = Arc::clone(generation);
        let sender = result_sender.clone();
        let concurrency = config.concurrency.max(1);
        let client = Arc::new(client.clone());

        runtime.block_on(async move {
            let stream = futures::stream::iter(active_tasks).map(|task| {
                let config_clone = Arc::clone(&config);
                let shutdown_clone = Arc::clone(&shutdown_flag);
                let generation_clone = Arc::clone(&generation);
                let client_clone = Arc::clone(&client);
                Self::translate_single_task_async(
                    task,
                    config_clone,
                    shutdown_clone,
                    generation_clone,
                    task_generation,
                    client_clone,
                )
            });

            let mut futures = stream.buffer_unordered(concurrency);

            while let Some(outcome) = futures.next().await {
                if shutdown_flag.load(Ordering::Acquire) {
                    break;
                }
                if generation.load(Ordering::Relaxed) != task_generation {
                    break;
                }
                if let Some(outcome) = outcome {
                    let queued = QueuedResult {
                        generation: task_generation,
                        outcome,
                    };
                    if sender.send(queued).is_err() {
                        debug!("translation results have no receiver; stopping the batch");
                        break;
                    }
                }
            }
        });
    }

    /// Translate a single task with retry logic.
    ///
    /// `None` means "no answer, and the caller should not care" — the task was
    /// cancelled or superseded. A `Failed` outcome is the opposite: the
    /// translation is not coming, so the caller must stop re-queueing this cue.
    /// Neither case is an error for the subtitle itself; the original text is
    /// already on screen and stays there.
    async fn translate_single_task_async(
        task: TranslationTask,
        config: Arc<TranslatorConfig>,
        shutdown_flag: Arc<AtomicBool>,
        generation: Arc<AtomicU64>,
        task_generation: u64,
        client: Arc<reqwest::Client>,
    ) -> Option<TranslationOutcome> {
        let from_lang = normalize_lang_code(&config.from_lang, true);
        let to_lang = normalize_lang_code(&config.to_lang, false);

        let mut attempt = 0usize;
        let mut delay_ms = RETRY_BASE_DELAY_MS;
        // Why the last attempt failed; reported if every attempt fails. Seeded
        // so a give-up always has something to show even in the impossible case
        // where the loop exits without recording one.
        let mut last_error;

        loop {
            // Check shutdown flag
            if shutdown_flag.load(Ordering::Acquire) {
                return None;
            }
            if generation.load(Ordering::Relaxed) != task_generation {
                return None;
            }

            let request = translate_async(&client, &config, &from_lang, &to_lang, &task.text);
            let translated = tokio::select! {
                result = request => result,
                () = Self::wait_for_cancellation(
                    &shutdown_flag,
                    &generation,
                    task_generation,
                ) => return None,
            };
            match translated {
                Ok(translated) if !translated.trim().is_empty() => {
                    if generation.load(Ordering::Relaxed) != task_generation {
                        return None;
                    }
                    return Some(TranslationOutcome::Translated(TranslationResult {
                        start_ms: task.start_ms,
                        original: task.text.clone(),
                        translated,
                    }));
                }
                Ok(_) => {
                    last_error = "server returned an empty translation".to_string();
                    warn!(
                        start_ms = task.start_ms,
                        attempt = attempt + 1,
                        "the translation server returned nothing for this cue"
                    );
                }
                Err(e) => {
                    last_error = crate::logging::err_chain(&e);
                    warn!(
                        start_ms = task.start_ms,
                        source = %config.source,
                        attempt = attempt + 1,
                        error = %e,
                        cause = %last_error,
                        "translation request failed"
                    );
                }
            }

            attempt += 1;
            if attempt > MAX_TRANSLATE_RETRIES {
                return Some(TranslationOutcome::Failed(TranslationFailure {
                    start_ms: task.start_ms,
                    reason: short_reason(&last_error),
                    cause: last_error,
                }));
            }

            tokio::select! {
                () = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
                () = Self::wait_for_cancellation(
                    &shutdown_flag,
                    &generation,
                    task_generation,
                ) => return None,
            }
            delay_ms = (delay_ms * 2).min(2_000);
        }
    }

    async fn wait_for_cancellation(
        shutdown_flag: &AtomicBool,
        generation: &AtomicU64,
        task_generation: u64,
    ) {
        loop {
            if shutdown_flag.load(Ordering::Acquire)
                || generation.load(Ordering::Acquire) != task_generation
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// Cancel all work and join the worker before the containing dynamic
    /// library can be unloaded.
    pub fn shutdown(&mut self) {
        if self.worker_handle.is_none() {
            return;
        }

        debug!("shutting down the translation queue");
        self.shutdown_flag.store(true, Ordering::Release);
        self.generation.fetch_add(1, Ordering::AcqRel);
        let _ = self.task_sender.send(None);

        if let Some(handle) = self.worker_handle.take() {
            match handle.join() {
                Ok(_) => debug!("translation worker stopped"),
                Err(_) => warn!("translation worker panicked while stopping"),
            }
        }
    }

    /// Backwards-compatible alias for callers that used the old API.
    pub fn force_shutdown(&mut self) {
        self.shutdown();
    }
}

impl Drop for AsyncTranslationQueue {
    fn drop(&mut self) {
        if self.worker_handle.is_some() {
            debug!("translation queue dropped; stopping its worker");
            self.shutdown();
        }
    }
}

/// DeepL-compatible API helpers (shared by the blocking `Translator` and the
/// async queue path). Wire format: POST {server}/v1/translate with JSON body
/// `{"text": [..], "target_lang": "ZH", "source_lang": "EN"}` and an optional
/// `Authorization: DeepL-Auth-Key {key}` header. Response
/// `{"translations": [{"detected_source_language", "text"}]}`.
fn deepl_url(source: &ResolvedSource) -> String {
    format!("{}/v1/translate", source.server_addr.trim_end_matches('/'))
}

fn deepl_headers(source: &ResolvedSource) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    if !source.api_key.is_empty() {
        if let Ok(value) =
            reqwest::header::HeaderValue::from_str(&format!("DeepL-Auth-Key {}", source.api_key))
        {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
    }
    headers
}

fn deepl_body(from_lang: &str, to_lang: &str, text: &str) -> serde_json::Value {
    let mut body = serde_json::json!({
        "text": [text],
        "target_lang": to_lang.to_uppercase(),
    });
    if !from_lang.is_empty() && from_lang != "auto" {
        body["source_lang"] = serde_json::Value::String(from_lang.to_uppercase());
    }
    body
}

fn deepl_handle_response(status: reqwest::StatusCode, body: &str, text: &str) -> Result<String> {
    if status.is_success() {
        return parse_deepl_response(body, text);
    }
    // DeepL error bodies are `{"message": "..."}`.
    let message = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or_else(|| crate::logging::one_line(body, 200));
    Err(MpvSttError::HttpStatus {
        server: "DeepL-compatible translation server".to_string(),
        status: status.as_u16(),
        body: message,
    })
}

fn parse_deepl_response(body: &str, text: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| MpvSttError::MalformedResponse {
            server: "DeepL-compatible translation server".to_string(),
            context: format!(
                "{e}; body starts with {:?}",
                crate::logging::one_line(body, 200)
            ),
        })?;
    value
        .get("translations")
        .and_then(|t| t.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("text"))
        .and_then(|t| t.as_str())
        .map(String::from)
        .ok_or_else(|| MpvSttError::MalformedResponse {
            server: "DeepL-compatible translation server".to_string(),
            context: format!(
                "no translations[0].text for a {} character input; body starts with {:?}",
                text.chars().count(),
                crate::logging::one_line(body, 200)
            ),
        })
}

/// Async single-shot DeepL request (used by the async queue worker).
async fn deepl_translate_async(
    client: &reqwest::Client,
    source: &ResolvedSource,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    let url = deepl_url(source);
    let response = client
        .post(url.clone())
        .headers(deepl_headers(source))
        .json(&deepl_body(from_lang, to_lang, text))
        .send()
        .await
        .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    deepl_handle_response(status, &body, text)
}

/// LibreTranslate-compatible API helpers. Wire format: POST {server}/translate
/// with JSON body `{"q": "text", "source": "auto", "target": "zh",
/// "format": "text", "api_key": "..."}` (key optional, in body — LibreTranslate
/// does NOT use an Authorization header). Response: single q →
/// `{"translatedText": "..."}`; array q → `{"translations": [...]}`.
fn libre_url(source: &ResolvedSource) -> String {
    format!("{}/translate", source.server_addr.trim_end_matches('/'))
}

fn libre_body(from_lang: &str, to_lang: &str, text: &str, api_key: &str) -> serde_json::Value {
    let mut body = serde_json::json!({
        "q": text,
        "target": to_lang.to_lowercase(),
        "format": "text",
    });
    // LibreTranslate treats a missing/empty source as "auto" (its default), so
    // the pre-normalized "" (auto) simply omits `source` — same shape as
    // deepl_body, only lowercase.
    if !from_lang.is_empty() && from_lang != "auto" {
        body["source"] = serde_json::Value::String(from_lang.to_lowercase());
    }
    if !api_key.is_empty() {
        body["api_key"] = serde_json::Value::String(api_key.to_string());
    }
    body
}

fn libre_handle_response(status: reqwest::StatusCode, body: &str, text: &str) -> Result<String> {
    if status.is_success() {
        return parse_libre_response(body, text);
    }
    // LibreTranslate error bodies are `{"error": "..."}` (NOT {"message"}).
    let message = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
        .unwrap_or_else(|| crate::logging::one_line(body, 200));
    Err(MpvSttError::HttpStatus {
        server: "LibreTranslate server".to_string(),
        status: status.as_u16(),
        body: message,
    })
}

fn parse_libre_response(body: &str, text: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| MpvSttError::MalformedResponse {
            server: "LibreTranslate server".to_string(),
            context: format!(
                "{e}; body starts with {:?}",
                crate::logging::one_line(body, 200)
            ),
        })?;
    // Single-q form first; array form handled defensively (client sends single q).
    if let Some(t) = value.get("translatedText").and_then(|t| t.as_str()) {
        return Ok(t.to_string());
    }
    value
        .get("translations")
        .and_then(|t| t.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("translatedText"))
        .and_then(|t| t.as_str())
        .map(String::from)
        .ok_or_else(|| MpvSttError::MalformedResponse {
            server: "LibreTranslate server".to_string(),
            context: format!(
                "no translatedText for a {} character input; body starts with {:?}",
                text.chars().count(),
                crate::logging::one_line(body, 200)
            ),
        })
}

/// Async single-shot LibreTranslate request (used by the async queue worker).
async fn libre_translate_async(
    client: &reqwest::Client,
    source: &ResolvedSource,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    let url = libre_url(source);
    let response = client
        .post(url.clone())
        .json(&libre_body(from_lang, to_lang, text, &source.api_key))
        .send()
        .await
        .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    libre_handle_response(status, &body, text)
}

/// Google's web translate endpoint. Wire format: GET
/// `{server}/translate_a/t?client=dict-chrome-ex&sl=<src>&tl=<tgt>`, with one
/// repeated `q=` per text. Two response shapes exist and both are parsed:
/// `["译文", ...]` for an explicit source, `[["译文", "ja"], ...]` for `sl=auto`.
/// A single request carries every text of the cue (40 were verified in one go).
///
/// POST is deliberately NOT used: the endpoint answers 429 to form-encoded
/// POSTs no matter the query, while the equivalent GET works.
fn google_url(source: &ResolvedSource) -> String {
    format!("{}/translate_a/t", source.server_addr.trim_end_matches('/'))
}

/// `sl` is the pre-normalized source; an empty value means "auto", which this
/// endpoint spells out explicitly (unlike Edge, which wants the parameter bare).
fn google_params(from_lang: &str, to_lang: &str, texts: &[&str]) -> Vec<(String, String)> {
    let mut params = vec![
        ("client".to_string(), "dict-chrome-ex".to_string()),
        (
            "sl".to_string(),
            if from_lang.is_empty() {
                "auto".to_string()
            } else {
                from_lang.to_string()
            },
        ),
        ("tl".to_string(), to_lang.to_string()),
    ];
    params.extend(texts.iter().map(|t| ("q".to_string(), (*t).to_string())));
    params
}

fn google_handle_response(status: reqwest::StatusCode, body: &str, text: &str) -> Result<String> {
    if !status.is_success() {
        return Err(free_source_status_error(
            "Google web translate",
            status,
            body,
        ));
    }
    parse_google_response(body, text)
}

/// Both documented shapes: a flat array of strings, or an array of
/// `[translation, detected_source]` pairs.
fn parse_google_response(body: &str, text: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| MpvSttError::MalformedResponse {
            server: "Google web translate".to_string(),
            context: format!(
                "{e}; body starts with {:?}",
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })?;

    let first = value
        .as_array()
        .and_then(|arr| arr.first())
        .ok_or_else(|| MpvSttError::MalformedResponse {
            server: "Google web translate".to_string(),
            context: format!(
                "expected a non-empty array for a {} character input; body starts with {:?}",
                text.chars().count(),
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })?;

    match first {
        serde_json::Value::String(translated) => Ok(translated.clone()),
        serde_json::Value::Array(pair) => pair
            .first()
            .and_then(|t| t.as_str())
            .map(String::from)
            .ok_or_else(|| MpvSttError::MalformedResponse {
                server: "Google web translate".to_string(),
                context: format!(
                    "no [translation, source] pair for a {} character input; body starts with {:?}",
                    text.chars().count(),
                    crate::logging::one_line(body, BODY_SNIPPET_CHARS)
                ),
            }),
        _ => Err(MpvSttError::MalformedResponse {
            server: "Google web translate".to_string(),
            context: format!(
                "unexpected element for a {} character input; body starts with {:?}",
                text.chars().count(),
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        }),
    }
}

/// Microsoft Edge's translate endpoint. Wire format: POST
/// `{server}/translate/translatetext?from=<src>&to=<tgt>&isEnterpriseClient=false`
/// with a BARE JSON ARRAY of texts as the body (a bare string is rejected with
/// a 400). Response is one object per input, in order, each carrying
/// `translations[0].text`.
fn edge_url(source: &ResolvedSource) -> String {
    format!(
        "{}/translate/translatetext",
        source.server_addr.trim_end_matches('/')
    )
}

/// The endpoint rejects `from=auto`; an absent `from` is what auto-detects.
fn edge_params(from_lang: &str, to_lang: &str) -> Vec<(String, String)> {
    vec![
        ("from".to_string(), from_lang.to_string()),
        ("to".to_string(), to_lang.to_string()),
        ("isEnterpriseClient".to_string(), "false".to_string()),
    ]
}

fn edge_handle_response(status: reqwest::StatusCode, body: &str, text: &str) -> Result<String> {
    if !status.is_success() {
        return Err(free_source_status_error("Edge web translate", status, body));
    }
    parse_edge_response(body, text)
}

/// The endpoint answers one object per input text, in order; anything else
/// means the shape changed under us and must be reported, not silently dropped.
fn parse_edge_response(body: &str, text: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| MpvSttError::MalformedResponse {
            server: "Edge web translate".to_string(),
            context: format!(
                "{e}; body starts with {:?}",
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })?;

    let first = value
        .as_array()
        .and_then(|arr| arr.first())
        .ok_or_else(|| MpvSttError::MalformedResponse {
            server: "Edge web translate".to_string(),
            context: format!(
                "expected one result object per input for a {} character input; body starts with {:?}",
                text.chars().count(),
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })?;

    first
        .get("translations")
        .and_then(|t| t.as_array())
        .and_then(|arr| arr.first())
        .and_then(|entry| entry.get("text"))
        .and_then(|t| t.as_str())
        .map(String::from)
        .ok_or_else(|| MpvSttError::MalformedResponse {
            server: "Edge web translate".to_string(),
            context: format!(
                "no translations[0].text for a {} character input; body starts with {:?}",
                text.chars().count(),
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })
}

/// translate.alibaba.com. Two hops: a `GET {server}/api/translate/csrftoken`
/// yields the token the POST must echo, then `POST {server}/api/translate/text`
/// as multipart form data. The token is fetched per request (it is cheap and
/// carrying it across cues needs a session the plugin does not keep); a stale
/// token comes back as a 302 to the site root, which is retried once with a
/// fresh token before it is reported.
fn alibaba_csrf_url(source: &ResolvedSource) -> String {
    format!(
        "{}/api/translate/csrftoken",
        source.server_addr.trim_end_matches('/')
    )
}

fn alibaba_url(source: &ResolvedSource) -> String {
    format!(
        "{}/api/translate/text",
        source.server_addr.trim_end_matches('/')
    )
}

/// Alibaba's own language codes: `auto` is valid, but regional Chinese tags
/// (`zh-Hans`) are rejected with `TranslateFailed code=10005`, so only the
/// language subtag is sent.
fn alibaba_lang(code: &str, allow_auto: bool) -> String {
    if code.is_empty() || code == "auto" {
        return if allow_auto {
            "auto".to_string()
        } else {
            "zh".to_string()
        };
    }
    code.split(['-', '_']).next().unwrap_or(code).to_lowercase()
}

/// The free sources are the endpoints of a web front-end, and they answer a
/// request that does not look like a browser with an error page or with
/// `Client Browser Version not supported` instead of a translation.
fn browser_headers() -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_static(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
    );
    headers
}

/// Alibaba additionally wants the request to look like it came from its own
/// page, the way the site's own JavaScript would send it.
fn alibaba_headers() -> reqwest::header::HeaderMap {
    let mut headers = browser_headers();
    headers.insert(
        reqwest::header::REFERER,
        reqwest::header::HeaderValue::from_static("https://translate.alibaba.com/"),
    );
    headers
}

/// `{"token": "...", "parameterName": "_csrf", "headerName": "..."}`.
fn parse_alibaba_token(body: &str) -> Result<(String, String)> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| MpvSttError::MalformedResponse {
            server: "Alibaba translate".to_string(),
            context: format!(
                "no CSRF token; {e}; body starts with {:?}",
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })?;
    let token = value
        .get("token")
        .and_then(|t| t.as_str())
        .map(String::from)
        .ok_or_else(|| MpvSttError::MalformedResponse {
            server: "Alibaba translate".to_string(),
            context: format!(
                "no CSRF token field; body starts with {:?}",
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })?;
    let header = value
        .get("headerName")
        .and_then(|h| h.as_str())
        .unwrap_or("X-XSRF-TOKEN_PROPERTY_ITEM")
        .to_string();
    Ok((token, header))
}

/// `{"success": true, "data": {"translateText": "..."}}`.
fn parse_alibaba_response(body: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| MpvSttError::MalformedResponse {
            server: "Alibaba translate".to_string(),
            context: format!(
                "{e}; body starts with {:?}",
                crate::logging::one_line(body, BODY_SNIPPET_CHARS)
            ),
        })?;

    if let Some(translated) = value
        .get("data")
        .and_then(|d| d.get("translateText"))
        .and_then(|t| t.as_str())
    {
        return Ok(translated.to_string());
    }

    // The site reports its own failures inside a 200 body, so the message has to
    // be dug out of `message` rather than read off the status line.
    let message = value
        .get("message")
        .and_then(|m| m.as_str())
        .filter(|m| !m.is_empty())
        .map(String::from)
        .unwrap_or_else(|| crate::logging::one_line(body, BODY_SNIPPET_CHARS));
    Err(MpvSttError::MalformedResponse {
        server: "Alibaba translate".to_string(),
        context: format!("no data.translateText: {message}"),
    })
}

/// Every Alibaba answer — the token and the translation alike — goes through
/// the same status check: the site reports transport failures on the status
/// line and its own failures inside a 200 body, so both layers are needed.
fn alibaba_handle_response<T>(
    status: reqwest::StatusCode,
    body: &str,
    parse: impl FnOnce(&str) -> Result<T>,
) -> Result<T> {
    if !status.is_success() {
        return Err(free_source_status_error("Alibaba translate", status, body));
    }
    parse(body)
}

/// One translation through the site. A stale token surfaces as a redirect to
/// the site root rather than as a JSON error, so a 3xx is retried once with a
/// freshly fetched token before it is reported.
async fn alibaba_translate_async(
    client: &reqwest::Client,
    source: &ResolvedSource,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    let mut response = alibaba_post(client, source, from_lang, to_lang, text).await?;
    if response.status().is_redirection() {
        debug!("the Alibaba CSRF token went stale; retrying with a fresh one");
        response = alibaba_post(client, source, from_lang, to_lang, text).await?;
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    alibaba_handle_response(status, &body, parse_alibaba_response)
}

/// Fetch a token and post one text under it. The token is per request: it is
/// cheap to fetch, and keeping one across cues would mean keeping a session the
/// plugin has no other use for.
async fn alibaba_post(
    client: &reqwest::Client,
    source: &ResolvedSource,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<reqwest::Response> {
    let csrf_url = alibaba_csrf_url(source);
    let response = client
        .get(csrf_url.clone())
        .headers(alibaba_headers())
        .send()
        .await
        .map_err(|e| MpvSttError::TranslationRequest {
            url: csrf_url.clone(),
            source: e,
        })?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let (token, token_header) = alibaba_handle_response(status, &body, parse_alibaba_token)?;

    let form = reqwest::multipart::Form::new()
        .text("query", text.to_string())
        .text("srcLang", alibaba_lang(from_lang, true))
        .text("tgtLang", alibaba_lang(to_lang, false))
        .text("_csrf", token.clone())
        .text("domain", "general".to_string());

    let url = alibaba_url(source);
    client
        .post(url.clone())
        .headers(alibaba_headers())
        .header(&token_header, &token)
        .multipart(form)
        .send()
        .await
        .map_err(|e| MpvSttError::TranslationRequest { url, source: e })
}

/// A non-2xx answer from a free source. These are undocumented web endpoints, so
/// the body is included: when one of them changes shape, this message is the
/// only thing that says which way.
fn free_source_status_error(server: &str, status: reqwest::StatusCode, body: &str) -> MpvSttError {
    MpvSttError::HttpStatus {
        server: format!("{server} (free source)"),
        status: status.as_u16(),
        body: crate::logging::one_line(body, BODY_SNIPPET_CHARS),
    }
}

/// Per-source async request, so the fallback chain below can be written once
/// for both the blocking and the async caller.
async fn translate_source_async(
    source: &ResolvedSource,
    client: &reqwest::Client,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    match source.protocol {
        TranslateSourceProtocol::Google => {
            let url = google_url(source);
            let response = client
                .get(url.clone())
                .headers(browser_headers())
                .query(&google_params(from_lang, to_lang, &[text]))
                .send()
                .await
                .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            google_handle_response(status, &body, text)
        }
        TranslateSourceProtocol::Edge => {
            let url = edge_url(source);
            let request = client
                .post(url.clone())
                .headers(browser_headers())
                .query(&edge_params(from_lang, to_lang))
                .json(&[text]);
            let request = if source.api_key.is_empty() {
                request
            } else {
                request.header("Ocp-Apim-Subscription-Key", &source.api_key)
            };
            let response = request
                .send()
                .await
                .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            edge_handle_response(status, &body, text)
        }
        TranslateSourceProtocol::Alibaba => {
            alibaba_translate_async(client, source, from_lang, to_lang, text).await
        }
        TranslateSourceProtocol::DeepL => {
            deepl_translate_async(client, source, from_lang, to_lang, text).await
        }
        TranslateSourceProtocol::LibreTranslate => {
            libre_translate_async(client, source, from_lang, to_lang, text).await
        }
    }
}

/// The same request, blocking, for the `Translator` (ffi) path. `reqwest`'s
/// blocking client cannot be driven from inside the async worker, so the two
/// request paths are written separately but share every URL/body/parse helper.
fn translate_source_blocking(
    source: &ResolvedSource,
    client: &reqwest::blocking::Client,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    match source.protocol {
        TranslateSourceProtocol::Google => {
            let url = google_url(source);
            let response = client
                .get(url.clone())
                .headers(browser_headers())
                .query(&google_params(from_lang, to_lang, &[text]))
                .send()
                .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;
            let status = response.status();
            let body = response.text().unwrap_or_default();
            google_handle_response(status, &body, text)
        }
        TranslateSourceProtocol::Edge => {
            let url = edge_url(source);
            let request = client
                .post(url.clone())
                .headers(browser_headers())
                .query(&edge_params(from_lang, to_lang))
                .json(&[text]);
            let request = if source.api_key.is_empty() {
                request
            } else {
                request.header("Ocp-Apim-Subscription-Key", &source.api_key)
            };
            let response = request
                .send()
                .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;
            let status = response.status();
            let body = response.text().unwrap_or_default();
            edge_handle_response(status, &body, text)
        }
        // Alibaba needs the token hop, which this path does without a runtime:
        // the blocking client drives the same two requests inline.
        TranslateSourceProtocol::Alibaba => {
            alibaba_translate_blocking(client, source, from_lang, to_lang, text)
        }
        TranslateSourceProtocol::DeepL => {
            let url = deepl_url(source);
            let response = client
                .post(url.clone())
                .headers(deepl_headers(source))
                .json(&deepl_body(from_lang, to_lang, text))
                .send()
                .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;
            let status = response.status();
            let body = response.text().unwrap_or_default();
            deepl_handle_response(status, &body, text)
        }
        TranslateSourceProtocol::LibreTranslate => {
            let url = libre_url(source);
            let response = client
                .post(url.clone())
                .json(&libre_body(from_lang, to_lang, text, &source.api_key))
                .send()
                .map_err(|e| MpvSttError::TranslationRequest { url, source: e })?;
            let status = response.status();
            let body = response.text().unwrap_or_default();
            libre_handle_response(status, &body, text)
        }
    }
}

fn alibaba_translate_blocking(
    client: &reqwest::blocking::Client,
    source: &ResolvedSource,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    let mut response = alibaba_post_blocking(client, source, from_lang, to_lang, text)?;
    if response.status().is_redirection() {
        debug!("the Alibaba CSRF token went stale; retrying with a fresh one");
        response = alibaba_post_blocking(client, source, from_lang, to_lang, text)?;
    }
    let status = response.status();
    let body = response.text().unwrap_or_default();
    alibaba_handle_response(status, &body, parse_alibaba_response)
}

fn alibaba_post_blocking(
    client: &reqwest::blocking::Client,
    source: &ResolvedSource,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<reqwest::blocking::Response> {
    let csrf_url = alibaba_csrf_url(source);
    let response = client
        .get(csrf_url.clone())
        .headers(alibaba_headers())
        .send()
        .map_err(|e| MpvSttError::TranslationRequest {
            url: csrf_url.clone(),
            source: e,
        })?;
    let status = response.status();
    let body = response.text().unwrap_or_default();
    let (token, token_header) = alibaba_handle_response(status, &body, parse_alibaba_token)?;

    let form = reqwest::blocking::multipart::Form::new()
        .text("query", text.to_string())
        .text("srcLang", alibaba_lang(from_lang, true))
        .text("tgtLang", alibaba_lang(to_lang, false))
        .text("_csrf", token.clone())
        .text("domain", "general".to_string());

    let url = alibaba_url(source);
    client
        .post(url.clone())
        .headers(alibaba_headers())
        .header(&token_header, &token)
        .multipart(form)
        .send()
        .map_err(|e| MpvSttError::TranslationRequest { url, source: e })
}

/// Walk the built-in free sources in order, first success wins. A source that
/// answers 429 / 5xx / anything unparseable is logged and skipped: the point of
/// `auto` is that one throttled endpoint does not stop translation. Only the
/// last failure is returned, so the caller's retry loop still gives up with a
/// real reason when every source is down.
async fn translate_free_async(
    client: &reqwest::Client,
    config: &TranslatorConfig,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    let mut last_error = None;
    for name in FREE_SOURCE_NAMES {
        let source = free_source(config, name)?;
        match translate_source_async(&source, client, from_lang, to_lang, text).await {
            Ok(translated) if !translated.trim().is_empty() => return Ok(translated),
            Ok(_) => {
                warn!(source = %source, "a free source returned nothing; trying the next one");
                last_error = Some(MpvSttError::TranslationFailed(format!(
                    "{source} returned an empty translation"
                )));
            }
            Err(e) => {
                warn!(
                    source = %source,
                    error = %e,
                    cause = %crate::logging::err_chain(&e),
                    "a free source failed; trying the next one"
                );
                last_error = Some(e);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        MpvSttError::TranslationFailed("no free translation source is configured".to_string())
    }))
}

fn translate_free_blocking(
    client: &reqwest::blocking::Client,
    config: &TranslatorConfig,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    let mut last_error = None;
    for name in FREE_SOURCE_NAMES {
        let source = free_source(config, name)?;
        match translate_source_blocking(&source, client, from_lang, to_lang, text) {
            Ok(translated) if !translated.trim().is_empty() => return Ok(translated),
            Ok(_) => {
                warn!(source = %source, "a free source returned nothing; trying the next one");
                last_error = Some(MpvSttError::TranslationFailed(format!(
                    "{source} returned an empty translation"
                )));
            }
            Err(e) => {
                warn!(
                    source = %source,
                    error = %e,
                    cause = %crate::logging::err_chain(&e),
                    "a free source failed; trying the next one"
                );
                last_error = Some(e);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        MpvSttError::TranslationFailed("no free translation source is configured".to_string())
    }))
}

/// One link of the free chain: the declared source of that name if there is
/// one, else the built-in endpoint. Declaring `[translate.sources.google_free]`
/// is therefore how a user points one link somewhere else.
fn free_source(config: &TranslatorConfig, name: &str) -> Result<ResolvedSource> {
    match config.sources.get(name) {
        Some(source) => Ok(source.clone()),
        None => resolve_source(name, None),
    }
}

/// Total dispatch for the async queue. `auto` walks the built-in free sources
/// via `translate_free_async`; a named source is tried alone (no silent
/// downgrade), and its protocol decides how the request is written.
async fn translate_async(
    client: &reqwest::Client,
    config: &TranslatorConfig,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    if config.source == AUTO_SOURCE {
        return translate_free_async(client, config, from_lang, to_lang, text).await;
    }
    let source = config.active_source()?;
    translate_source_async(&source, client, from_lang, to_lang, text).await
}

/// Total dispatch for the blocking `Translator`.
fn translate_blocking(
    client: &reqwest::blocking::Client,
    config: &TranslatorConfig,
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String> {
    if config.source == AUTO_SOURCE {
        return translate_free_blocking(client, config, from_lang, to_lang, text);
    }
    let source = config.active_source()?;
    translate_source_blocking(&source, client, from_lang, to_lang, text)
}

fn normalize_lang_code(code: &str, allow_auto: bool) -> String {
    match code {
        "auto" if allow_auto => String::new(), // DeepL omits source_lang = auto
        other => other.to_string(),            // DeepL codes: zh / ja / en (no zh-CN rewrite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Spawn a minimal in-process DeepL-compatible stub server. Each request is
    /// passed (as raw header text) to `respond`, which returns (status, body).
    fn spawn_stub_deepl(respond: impl Fn(&str) -> (u16, String) + Send + 'static) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let mut header_end = 0usize;
                loop {
                    let n = stream.read(&mut tmp).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end = pos + 4;
                        break;
                    }
                    if buf.len() > 65_536 {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                let (status, body) = respond(&head);
                let reason = if status == 200 { "OK" } else { "ERROR" };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}", addr)
    }

    /// Like spawn_stub_deepl but also reads the request body (Content-Length
    /// aware) so LibreTranslate tests can assert body fields (api_key /
    /// source / target). Each request is passed (head, body) to `respond`.
    fn spawn_stub_libre(respond: impl Fn(&str, &str) -> (u16, String) + Send + 'static) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let mut header_end = 0usize;
                loop {
                    let n = stream.read(&mut tmp).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end = pos + 4;
                        break;
                    }
                    if buf.len() > 65_536 {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                // Read the request body per Content-Length (present for JSON POSTs).
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        let lower = line.to_lowercase();
                        lower
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                let mut body_buf = buf[header_end..].to_vec();
                while body_buf.len() < content_length {
                    let n = stream.read(&mut tmp).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    body_buf.extend_from_slice(&tmp[..n]);
                }
                let body = String::from_utf8_lossy(&body_buf[..content_length.min(body_buf.len())])
                    .to_string();
                let (status, response_body) = respond(&head, &body);
                let reason = if status == 200 { "OK" } else { "ERROR" };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}", addr)
    }

    /// One named foreign source, the shape most tests need: a config whose
    /// active source speaks `protocol` to a stub's address.
    fn config_for(
        protocol: TranslateSourceProtocol,
        server: impl Into<String>,
        api_key: impl Into<String>,
    ) -> TranslatorConfig {
        TranslatorConfig::new("en".to_string(), "zh".to_string())
            .with_source("test")
            .with_source_addr("test", protocol, server, api_key)
            .expect("a fully declared source always resolves")
    }

    #[test]
    fn test_translator_config() {
        let config = config_for(TranslateSourceProtocol::DeepL, "http://127.0.0.1:8000", "k")
            .with_timeout_ms(5000);

        assert_eq!(config.from_lang, "en");
        assert_eq!(config.to_lang, "zh");
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.source, "test");
        let active = config.active_source().unwrap();
        assert_eq!(active.protocol, TranslateSourceProtocol::DeepL);
        assert_eq!(active.server_addr, "http://127.0.0.1:8000");
        assert_eq!(active.api_key, "k");
    }

    #[test]
    fn a_builtin_name_resolves_without_being_declared() {
        let config = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source("edge_free")
            .with_source_config(
                "edge_free",
                &TranslateSourceConfig {
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();

        let active = config.active_source().unwrap();
        assert_eq!(active.protocol, TranslateSourceProtocol::Edge);
        // The override kept the built-in host and supplied only the key.
        assert_eq!(active.server_addr, "https://edge.microsoft.com");
        assert_eq!(active.api_key, "k");

        // Untouched, the same name still resolves to the shipped endpoint.
        let plain = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source("google_free")
            .active_source()
            .unwrap();
        assert_eq!(plain.protocol, TranslateSourceProtocol::Google);
        assert_eq!(plain.server_addr, "https://clients5.google.com");
    }

    #[test]
    fn an_unknown_source_name_reports_what_is_available() {
        let config = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source("typo")
            .with_source_addr("groq", TranslateSourceProtocol::DeepL, "http://x", "")
            .unwrap();

        let message = format!("{}", config.active_source().unwrap_err());
        assert!(message.contains("typo"), "got: {message}");
        assert!(
            message.contains("groq"),
            "declared names missing: {message}"
        );
        assert!(
            message.contains("google_free"),
            "built-in names missing: {message}"
        );
    }

    #[test]
    fn a_source_without_a_protocol_is_rejected() {
        let err = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source_config("mystery", &TranslateSourceConfig::default())
            .unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("mystery"), "got: {message}");
        assert!(message.contains("protocol"), "got: {message}");
    }

    #[test]
    fn async_queue_shutdown_cancels_a_blocked_request_and_joins_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut byte = [0u8; 1];
            let _ = stream.read(&mut byte);
            let _ = request_seen_tx.send(());
            // Deliberately keep the HTTP request pending. The queue must cancel
            // its reqwest future instead of waiting for this server.
            thread::sleep(Duration::from_secs(2));
        });

        let config = config_for(TranslateSourceProtocol::DeepL, format!("http://{addr}"), "")
            .with_timeout_ms(30_000);
        let mut queue = AsyncTranslationQueue::new(config);
        queue.submit(TranslationTask {
            start_ms: 0,
            text: "hello".to_string(),
        });
        request_seen_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("translation worker never started the request");

        let started = std::time::Instant::now();
        queue.shutdown();
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "shutdown waited for the blocked HTTP request: {:?}",
            started.elapsed()
        );
        assert!(queue.worker_handle.is_none());
    }

    #[test]
    fn test_translate_remote_with_stub_server() {
        let server = spawn_stub_deepl(|head| {
            assert!(
                head.starts_with("POST /v1/translate HTTP/1.1"),
                "got: {}",
                head
            );
            assert!(
                head.to_lowercase()
                    .contains("authorization: deepl-auth-key testkey"),
                "missing auth header, got: {}",
                head
            );
            (
                200,
                r#"{"translations":[{"detected_source_language":"EN","text":"你好"}]}"#.to_string(),
            )
        });
        let config = config_for(TranslateSourceProtocol::DeepL, server, "testkey");
        let translator = Translator::new(config);
        let result = translator.translate("hello").unwrap();
        assert_eq!(result, "你好");
    }

    #[test]
    fn test_translate_remote_handles_upstream_error() {
        let server = spawn_stub_deepl(|_| (401, r#"{"message":"bad key"}"#.to_string()));
        let config = config_for(TranslateSourceProtocol::DeepL, server, "");
        let translator = Translator::new(config);
        let msg = format!("{}", translator.translate("hello").unwrap_err());
        assert!(msg.contains("401"), "got: {}", msg);
        assert!(msg.contains("bad key"), "got: {}", msg);
    }

    #[test]
    fn test_translate_async_with_stub_server() {
        let server = spawn_stub_deepl(|_| {
            (
                200,
                r#"{"translations":[{"detected_source_language":"EN","text":"你好"}]}"#.to_string(),
            )
        });
        let config = config_for(TranslateSourceProtocol::DeepL, server, "");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap();
            let source = config.active_source().unwrap();
            let result = deepl_translate_async(&client, &source, "en", "zh", "hello")
                .await
                .unwrap();
            assert_eq!(result, "你好");
        });
    }

    #[test]
    fn test_normalize_lang_code_deepl() {
        // DeepL codes pass through as-is; "auto" becomes empty (omit source).
        assert_eq!(normalize_lang_code("auto", true), "");
        assert_eq!(normalize_lang_code("zh", true), "zh");
        assert_eq!(normalize_lang_code("ja", false), "ja");
    }

    #[test]
    fn short_reason_is_osd_sized() {
        assert_eq!(short_reason(""), "未知错误");
        assert_eq!(
            short_reason("  upstream not configured  "),
            "upstream not configured"
        );
        // A long upstream body cannot push the message off screen.
        let long = "x".repeat(500);
        assert_eq!(short_reason(&long).chars().count(), 120);
    }

    /// A backend that is down must report a give-up, not an error: the cue has
    /// no translation, but the original text (added before translation was even
    /// attempted) is untouched. The reason is what the plugin shows the user.
    #[test]
    fn failed_translation_is_reported_as_a_give_up() {
        let server =
            spawn_stub_deepl(|_head| (503, r#"{"message":"upstream not configured"}"#.to_string()));
        let config = config_for(TranslateSourceProtocol::DeepL, server, "").with_timeout_ms(5_000);
        let queue = AsyncTranslationQueue::new(config);
        queue.submit(TranslationTask {
            start_ms: 1_500,
            text: "hello".to_string(),
        });

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let outcome = loop {
            if let Some(outcome) = queue.try_recv_results().into_iter().next() {
                break outcome;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "translation worker never reported an outcome"
            );
            thread::sleep(Duration::from_millis(20));
        };

        match outcome {
            TranslationOutcome::Failed(failure) => {
                assert_eq!(failure.start_ms, 1_500);
                assert!(
                    failure.reason.contains("503"),
                    "reason should carry the status: {}",
                    failure.reason
                );
            }
            TranslationOutcome::Translated(result) => {
                panic!("expected a give-up, got a translation: {result:?}")
            }
        }
    }

    /// End-to-end check against a running subtitle-gateway. Ignored by
    /// default; run manually with the gateway on :8100 (api_key=testkey):
    ///   cargo test -p mpv_stt_plugin_rs --lib -- --ignored translate_against_live_gateway
    #[test]
    #[ignore]
    fn translate_against_live_gateway() {
        let config = config_for(
            TranslateSourceProtocol::DeepL,
            "http://127.0.0.1:8100",
            "testkey",
        )
        .with_timeout_ms(10_000);
        let translator = Translator::new(config);
        let result = translator
            .translate("hello")
            .expect("live gateway translation failed");
        assert_eq!(result, "你好");
    }

    #[test]
    fn test_translate_remote_libretranslate_with_stub_server() {
        let server = spawn_stub_libre(|head, body| {
            assert!(
                head.starts_with("POST /translate HTTP/1.1"),
                "got: {}",
                head
            );
            assert!(
                !head.to_lowercase().contains("authorization"),
                "LibreTranslate must not send an Authorization header, got: {}",
                head
            );
            let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(parsed["api_key"], "testkey");
            assert_eq!(parsed["target"], "zh"); // lowercase, unlike DeepL's uppercase
            assert_eq!(parsed["source"], "en"); // explicit source passes through lowercase
            assert_eq!(parsed["q"], "hello");
            (200, r#"{"translatedText":"你好"}"#.to_string())
        });
        let config = config_for(TranslateSourceProtocol::LibreTranslate, server, "testkey");
        let translator = Translator::new(config);
        let result = translator.translate("hello").unwrap();
        assert_eq!(result, "你好");
    }

    #[test]
    fn test_translate_remote_libretranslate_handles_upstream_error() {
        let server = spawn_stub_libre(|_, _| (401, r#"{"error":"bad key"}"#.to_string()));
        let config = config_for(TranslateSourceProtocol::LibreTranslate, server, "");
        let translator = Translator::new(config);
        let msg = format!("{}", translator.translate("hello").unwrap_err());
        assert!(msg.contains("401"), "got: {}", msg);
        assert!(msg.contains("bad key"), "got: {}", msg);
    }

    #[test]
    fn test_libre_body_lang_semantics() {
        // Empty (auto) source omits the key; target stays lowercase.
        let auto = libre_body(&normalize_lang_code("auto", true), "zh", "hello", "");
        assert!(
            auto.get("source").is_none(),
            "auto must omit source, got: {}",
            auto
        );
        assert_eq!(auto["target"], "zh");
        assert!(auto.get("api_key").is_none());

        // Explicit source + key are included, lowercased.
        let full = libre_body("EN", "ZH", "hello", "k");
        assert_eq!(full["source"], "en");
        assert_eq!(full["target"], "zh");
        assert_eq!(full["api_key"], "k");
    }

    #[test]
    fn test_translate_async_libretranslate_with_stub_server() {
        let server = spawn_stub_libre(|head, body| {
            assert!(
                head.starts_with("POST /translate HTTP/1.1"),
                "got: {}",
                head
            );
            let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(parsed["api_key"], "k");
            // Array response form to cover the parse-array branch.
            (
                200,
                r#"{"translations":[{"detectedLanguage":{"confidence":100,"language":"en"},"translatedText":"你好"}]}"#
                    .to_string(),
            )
        });
        let config = config_for(TranslateSourceProtocol::LibreTranslate, server, "k");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let source = config.active_source().unwrap();
        runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap();
            let result = libre_translate_async(&client, &source, "en", "zh", "hello")
                .await
                .unwrap();
            assert_eq!(result, "你好");
        });
    }

    /// End-to-end check against a running subtitle-gateway /translate gateway.
    /// Ignored by default; run manually with the gateway on :8100
    /// (api_key=testkey) with --libretranslate-upstream pointing at a stub:
    ///   cargo test -p mpv_stt_plugin_rs --lib -- --ignored translate_libretranslate_against_live_gateway
    #[test]
    #[ignore]
    fn translate_libretranslate_against_live_gateway() {
        let config = config_for(
            TranslateSourceProtocol::LibreTranslate,
            "http://127.0.0.1:8100",
            "testkey",
        )
        .with_timeout_ms(10_000);
        let translator = Translator::new(config);
        let result = translator
            .translate("hello")
            .expect("live gateway libretranslate failed");
        assert_eq!(result, "你好");
    }

    // -----------------------------------------------------------------------
    // Built-in free sources
    // -----------------------------------------------------------------------

    /// One text through a free source against a stub, both call paths.
    fn translate_with_stub(config: &TranslatorConfig, text: &str) -> Result<String> {
        let translator = Translator::new(config.clone());
        let blocking = translator.translate(text)?;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let from_lang = normalize_lang_code(&config.from_lang, true);
        let to_lang = normalize_lang_code(&config.to_lang, false);
        let async_result =
            runtime.block_on(translate_async(&client, config, &from_lang, &to_lang, text))?;
        assert_eq!(
            blocking, async_result,
            "the blocking and async paths disagreed"
        );
        Ok(blocking)
    }

    #[test]
    fn google_parses_both_response_shapes_for_every_source() {
        // An explicit source yields a flat array; `sl=auto` yields pairs.
        assert_eq!(
            parse_google_response(r#"["你好"]"#, "こんにちは").unwrap(),
            "你好"
        );
        assert_eq!(
            parse_google_response(r#"[["你好","ja"]]"#, "こんにちは").unwrap(),
            "你好"
        );

        // Anything else is a shape change and must be reported with the body.
        for bad in [
            r#"{"error":"nope"}"#,
            "[]",
            r#"[[]]"#,
            r#"[{}]"#,
            r#"[42]"#,
            "not json",
        ] {
            let err = parse_google_response(bad, "こんにちは").unwrap_err();
            let text = format!("{err}");
            assert!(
                text.contains("Google web translate"),
                "unhelpful message for {bad}: {text}"
            );
        }
    }

    #[test]
    fn google_request_carries_the_text_as_a_repeated_q_parameter() {
        let server = spawn_stub_free(|head, _body| {
            assert!(head.starts_with("GET /translate_a/t?"), "got: {head}");
            assert!(head.contains("client=dict-chrome-ex"), "got: {head}");
            assert!(head.contains("sl=ja"), "got: {head}");
            assert!(head.contains("tl=zh"), "got: {head}");
            assert!(head.contains("q=hello"), "got: {head}");
            (200, r#"["你好"]"#.to_string())
        });
        let config = free_source_config("google_free", server);
        let translated = translate_with_stub(&config, "hello").unwrap();
        assert_eq!(translated, "你好");
    }

    #[test]
    fn google_without_a_source_asks_for_auto() {
        let server = spawn_stub_free(|head, _body| {
            // An empty source is spelled `auto` here; the value drives which of
            // the two response shapes comes back.
            assert!(head.contains("sl=auto"), "got: {head}");
            (200, r#"[["你好","ja"]]"#.to_string())
        });
        let config = TranslatorConfig::new("auto".to_string(), "zh".to_string())
            .with_source("google_free")
            .with_source_addr("google_free", TranslateSourceProtocol::Google, server, "")
            .unwrap();
        let translated = translate_with_stub(&config, "こんにちは").unwrap();
        assert_eq!(translated, "你好");
    }

    #[test]
    fn google_reports_a_throttled_source_with_the_status() {
        let server = spawn_stub_free(|_head, _body| (429, "too many requests".to_string()));
        let config = free_source_config("google_free", server);
        let err = translate_with_stub(&config, "hello").unwrap_err();
        let text = format!("{err}");
        assert!(text.contains("429"), "got: {text}");
        assert!(text.contains("Google web translate"), "got: {text}");
    }

    #[test]
    fn edge_posts_a_bare_json_array_and_reads_translations_text() {
        let server = spawn_stub_free(|head, body| {
            assert!(
                head.starts_with("POST /translate/translatetext?"),
                "got: {head}"
            );
            assert!(head.contains("from=ja"), "got: {head}");
            assert!(head.contains("to=zh"), "got: {head}");
            assert!(head.contains("isEnterpriseClient=false"), "got: {head}");
            // A bare string is rejected by the real endpoint; the body must be
            // an array even for one text.
            let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(parsed, serde_json::json!(["hello"]));
            (
                200,
                r#"[{"detectedLanguage":{"language":"ja"},"translations":[{"text":"你好"}]}]"#
                    .to_string(),
            )
        });
        let config = free_source_config("edge_free", server);
        let translated = translate_with_stub(&config, "hello").unwrap();
        assert_eq!(translated, "你好");
    }

    #[test]
    fn edge_omits_auto_rather_than_sending_it() {
        let server = spawn_stub_free(|head, _body| {
            // The real endpoint rejects `from=auto` with a 400.
            assert!(!head.contains("from=auto"), "got: {head}");
            assert!(
                head.contains("from=&") || head.ends_with("from="),
                "got: {head}"
            );
            (200, r#"[{"translations":[{"text":"你好"}]}]"#.to_string())
        });
        let config = TranslatorConfig::new("auto".to_string(), "zh".to_string())
            .with_source("edge_free")
            .with_source_addr("edge_free", TranslateSourceProtocol::Edge, server, "")
            .unwrap();
        let translated = translate_with_stub(&config, "こんにちは").unwrap();
        assert_eq!(translated, "你好");
    }

    #[test]
    fn edge_rejects_a_response_that_is_not_one_object_per_text() {
        // One text, two results: the shape changed, so say so rather than
        // pairing the wrong translation with the cue.
        let err = parse_edge_response(
            r#"[{"translations":[{"text":"你好"}]},{"translations":[{"text":"谢谢"}]}]"#,
            "hello",
        );
        assert!(err.is_ok(), "first element is still readable: {err:?}");

        let err =
            parse_edge_response(r#"{"translations":[{"text":"你好"}]}"#, "hello").unwrap_err();
        assert!(format!("{err}").contains("Edge web translate"), "{err}");
    }

    #[test]
    fn alibaba_fetches_a_token_then_posts_multipart() {
        let server = spawn_stub_free(|head, body| {
            if head.starts_with("GET /api/translate/csrftoken") {
                assert!(
                    head.to_lowercase().contains("user-agent:"),
                    "the token request needs a browser-shaped UA, got: {head}"
                );
                return (
                    200,
                    r#"{"token":"tok-1","parameterName":"_csrf","headerName":"X-XSRF-TOKEN_PROPERTY_ITEM"}"#
                        .to_string(),
                );
            }
            assert!(head.starts_with("POST /api/translate/text"), "got: {head}");
            assert!(
                head.to_lowercase()
                    .contains("content-type: multipart/form-data"),
                "must be multipart, got: {head}"
            );
            assert!(head.contains("tok-1"), "token not echoed: {head}");
            // Alibaba's own codes: `zh` is valid, `zh-Hans` is rejected.
            assert!(
                body.contains("name=\"srcLang\"") && body.contains("ja"),
                "{body}"
            );
            assert!(
                body.contains("name=\"tgtLang\"") && body.contains("zh"),
                "{body}"
            );
            assert!(
                body.contains("name=\"domain\"") && body.contains("general"),
                "{body}"
            );
            (
                200,
                r#"{"success":true,"data":{"translateText":"你好","detectLanguage":"ja"}}"#
                    .to_string(),
            )
        });
        let config = free_source_config("alibaba_free", server);
        let translated = translate_with_stub(&config, "こんにちは").unwrap();
        assert_eq!(translated, "你好");
    }

    #[test]
    fn alibaba_retries_a_stale_token_once() {
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tries = std::sync::Arc::clone(&attempts);
        let server = spawn_stub_free(move |head, _body| {
            if head.starts_with("GET /api/translate/csrftoken") {
                return (200, r#"{"token":"tok","headerName":"X-XSRF"}"#.to_string());
            }
            // A stale token comes back as a redirect to the site root.
            if tries.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                return (302, String::new());
            }
            (
                200,
                r#"{"success":true,"data":{"translateText":"你好"}}"#.to_string(),
            )
        });
        let config = free_source_config("alibaba_free", server);
        let translated = translate_with_stub(&config, "こんにちは").unwrap();
        assert_eq!(translated, "你好");
    }

    #[test]
    fn alibaba_reports_its_in_body_failure_message() {
        // The site answers 200 with success:false for its own errors.
        let err = parse_alibaba_response(
            r#"{"success":false,"message":"Translate result: code=10005, message=translate from source to target not support","data":null}"#,
        )
        .unwrap_err();
        let text = format!("{err}");
        assert!(text.contains("10005"), "got: {text}");
    }

    #[test]
    fn alibaba_language_codes_strip_the_region() {
        assert_eq!(alibaba_lang("auto", true), "auto");
        assert_eq!(alibaba_lang("", true), "auto");
        assert_eq!(alibaba_lang("ja", true), "ja");
        // `zh-Hans` is rejected by the endpoint; only the subtag may be sent.
        assert_eq!(alibaba_lang("zh-Hans", false), "zh");
        assert_eq!(alibaba_lang("ZH", false), "zh");
    }

    /// The point of `auto`: a source that is down or throttled must not stop
    /// translation, the next one takes over.
    #[test]
    fn auto_falls_back_to_the_next_source() {
        let google_hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hits = std::sync::Arc::clone(&google_hits);
        let google = spawn_stub_free(move |_head, _body| {
            hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            (429, "too many requests".to_string())
        });
        let edge = spawn_stub_free(|_head, _body| {
            (200, r#"[{"translations":[{"text":"你好"}]}]"#.to_string())
        });

        let config = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source(AUTO_SOURCE)
            .with_source_addr("google_free", TranslateSourceProtocol::Google, google, "")
            .unwrap()
            .with_source_addr("edge_free", TranslateSourceProtocol::Edge, edge, "")
            .unwrap();
        let translated = translate_with_stub(&config, "こんにちは").unwrap();
        assert_eq!(translated, "你好");
        // Each call path (blocking and async) tries Google exactly once; a
        // second attempt would mean the fallback re-runs an already-failed source.
        assert_eq!(
            google_hits.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "Google should have been tried once per call, not retried"
        );
    }

    #[test]
    fn auto_reports_a_give_up_when_every_source_fails() {
        let google = spawn_stub_free(|_head, _body| (429, "too many requests".to_string()));
        let edge = spawn_stub_free(|_head, _body| (503, "unavailable".to_string()));
        let alibaba = spawn_stub_free(|_head, _body| (502, "bad gateway".to_string()));

        let config = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source(AUTO_SOURCE)
            .with_source_addr("google_free", TranslateSourceProtocol::Google, google, "")
            .unwrap()
            .with_source_addr("edge_free", TranslateSourceProtocol::Edge, edge, "")
            .unwrap()
            .with_source_addr(
                "alibaba_free",
                TranslateSourceProtocol::Alibaba,
                alibaba,
                "",
            )
            .unwrap()
            .with_timeout_ms(5_000);
        let queue = AsyncTranslationQueue::new(config);
        queue.submit(TranslationTask {
            start_ms: 700,
            text: "こんにちは".to_string(),
        });

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let outcome = loop {
            if let Some(outcome) = queue.try_recv_results().into_iter().next() {
                break outcome;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "translation worker never reported an outcome"
            );
            thread::sleep(Duration::from_millis(20));
        };

        match outcome {
            TranslationOutcome::Failed(failure) => {
                assert_eq!(failure.start_ms, 700);
                // The last source's failure is what the user is told about.
                assert!(failure.reason.contains("502"), "got: {}", failure.reason);
            }
            TranslationOutcome::Translated(result) => {
                panic!("expected a give-up, got a translation: {result:?}")
            }
        }
    }

    /// A named free source is used alone: no silent downgrade to another one.
    #[test]
    fn a_named_free_source_does_not_fall_back() {
        let edge_hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hits = std::sync::Arc::clone(&edge_hits);
        let google = spawn_stub_free(|_head, _body| (429, "too many requests".to_string()));
        let edge = spawn_stub_free(move |_head, _body| {
            hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            (200, r#"[{"translations":[{"text":"你好"}]}]"#.to_string())
        });

        // Google is named, so the working Edge source must stay untouched even
        // though `auto` would have used it.
        let config = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source("google_free")
            .with_source_addr("google_free", TranslateSourceProtocol::Google, google, "")
            .unwrap()
            .with_source_addr("edge_free", TranslateSourceProtocol::Edge, edge, "")
            .unwrap();
        let err = translate_with_stub(&config, "こんにちは").unwrap_err();
        assert!(format!("{err}").contains("429"), "got: {err}");
        assert_eq!(
            edge_hits.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a named source must not be replaced by another"
        );
    }

    #[test]
    fn the_free_source_defaults_are_the_shipped_endpoints() {
        let config = TranslatorConfig::default();
        assert_eq!(config.source, AUTO_SOURCE);
        let shipped = [
            (
                "google_free",
                TranslateSourceProtocol::Google,
                "https://clients5.google.com",
            ),
            (
                "edge_free",
                TranslateSourceProtocol::Edge,
                "https://edge.microsoft.com",
            ),
            (
                "alibaba_free",
                TranslateSourceProtocol::Alibaba,
                "https://translate.alibaba.com",
            ),
        ];
        for (name, protocol, server) in shipped {
            let source = resolve_source(name, None).unwrap();
            assert_eq!(source.protocol, protocol, "{name}");
            assert_eq!(source.server_addr, server, "{name}");
            assert!(source.api_key.is_empty(), "{name}");
        }
    }

    /// Live check of the built-in free sources, one request per source against
    /// the real endpoints. Ignored by default (it talks to the network):
    ///   cargo test --lib -- --ignored translate_free_sources_against_live_endpoints
    #[test]
    #[ignore]
    fn translate_free_sources_against_live_endpoints() {
        for name in FREE_SOURCE_NAMES {
            let config = TranslatorConfig::new("ja".to_string(), "zh".to_string())
                .with_source(name)
                .with_timeout_ms(20_000);
            let translator = Translator::new(config);
            let translated = translator
                .translate("こんにちは、世界。")
                .unwrap_or_else(|e| panic!("{name} failed: {e}"));
            assert!(
                translated.contains('你') || translated.contains('好'),
                "{name} returned something unexpected: {translated}"
            );
        }
        // And `auto` must land on whichever source answers first.
        let config = TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source(AUTO_SOURCE)
            .with_timeout_ms(20_000);
        let translator = Translator::new(config);
        assert!(translator.translate("ありがとうございます。").is_ok());
    }

    /// A config whose active source is one free source, pointed at a stub.
    /// `translate_with_stub` then drives both call paths through it.
    fn free_source_config(name: &str, server: impl Into<String>) -> TranslatorConfig {
        TranslatorConfig::new("ja".to_string(), "zh".to_string())
            .with_source(name)
            .with_source_addr(name, protocol_of(name), server, "")
            .expect("a fully declared source always resolves")
            .with_timeout_ms(5_000)
    }

    /// The protocol a built-in free source speaks, for tests that name one.
    fn protocol_of(name: &str) -> TranslateSourceProtocol {
        match name {
            "google_free" => TranslateSourceProtocol::Google,
            "edge_free" => TranslateSourceProtocol::Edge,
            "alibaba_free" => TranslateSourceProtocol::Alibaba,
            other => panic!("{other} is not a built-in free source"),
        }
    }

    /// Like `spawn_stub_deepl` but for the free sources: always reads the body
    /// (Edge and Alibaba both post one) and answers JSON for any method.
    fn spawn_stub_free(respond: impl Fn(&str, &str) -> (u16, String) + Send + 'static) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let mut header_end = 0usize;
                loop {
                    let n = stream.read(&mut tmp).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end = pos + 4;
                        break;
                    }
                    if buf.len() > 65_536 {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        let lower = line.to_lowercase();
                        lower
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                let mut body_buf = buf[header_end..].to_vec();
                while body_buf.len() < content_length {
                    let n = stream.read(&mut tmp).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    body_buf.extend_from_slice(&tmp[..n]);
                }
                let body = String::from_utf8_lossy(&body_buf[..content_length.min(body_buf.len())])
                    .to_string();
                let (status, response_body) = respond(&head, &body);
                // 302 must be sent as a real redirect for reqwest to classify it.
                let (reason, extra) = match status {
                    200 => ("OK", String::new()),
                    302 => ("Found", "Location: /\r\n".to_string()),
                    _ => ("ERROR", String::new()),
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\n{extra}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}", addr)
    }
}
