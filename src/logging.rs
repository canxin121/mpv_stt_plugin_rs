//! The plugin's only logging assembly point.
//!
//! Everything the plugin logs goes through `tracing`; this module decides where
//! those records end up. Three sinks are installed:
//!
//! * **stderr** — what a terminal-launched mpv shows. Compact, one line per
//!   event, ANSI only when stderr is a terminal.
//! * **a rotating file** — what a GUI-launched IINA leaves behind. mpv is a
//!   GUI app when started from Finder, so its stderr is discarded; without the
//!   file sink there is nothing to read after a failure. Fuller format than
//!   stderr: span open/close lines carry the per-chunk and per-request timings.
//! * **mpv's OSD** — the records a user should see are queued and drawn by the
//!   plugin's event loop, so a broken backend says so on screen instead of only
//!   in a log the user cannot see. Reaching the OSD is opt-in: a record carries
//!   a `display` field when it is meant for the screen, and one without it goes
//!   to the terminal and the file only. Most `warn`s are plumbing (a retry
//!   about to be attempted again, a cache file that could not be written) and
//!   would otherwise drown out the one message that matters.
//!
//! On Android the stderr sink is replaced by a logcat layer; the file sink and
//! the OSD queue are identical.
//!
//! The module is also where the log *vocabulary* lives: the session/chunk/request
//! spans every subsystem records into, and the helpers that turn an error into a
//! loggable cause chain.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Mutex, OnceLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Span, Subscriber};
use tracing_error::ErrorLayer;
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::fmt as tsfmt;

/// Where the file sink writes when the config says `file = "auto"`.
pub const DEFAULT_LOG_FILE_NAME: &str = "mpv_stt_plugin_rs.log";

/// Environment variable that overrides `[log] level`.
pub const LOG_ENV: &str = "MPV_STT_PLUGIN_RS_LOG";

/// Tag the logcat layer uses and the OSD prefixes messages with.
pub const LOG_TAG: &str = "mpv_stt_plugin_rs";

/// Target for records that are not tied to a module's normal flow: panics and
/// anything the logging setup itself has to report. Filter it with
/// `mpv_stt_plugin_rs::logging=debug`.
pub const TARGET_INTERNAL: &str = "mpv_stt_plugin_rs::logging";

/// Line format for a sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// One dense line per event, with the span context inlined as `name{fields}`.
    #[default]
    Compact,
    /// One line per event, plus one line each for every span open and close,
    /// annotated with `time.busy` / `time.idle`.
    Full,
    /// One JSON object per event.
    Json,
}

impl fmt::Display for LogFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            LogFormat::Compact => "compact",
            LogFormat::Full => "full",
            LogFormat::Json => "json",
        };
        write!(f, "{label}")
    }
}

impl LogFormat {
    /// Parse a `[log] format` value. Unknown values fall back to `Compact`
    /// rather than failing the whole config.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "full" => LogFormat::Full,
            "json" => LogFormat::Json,
            _ => LogFormat::Compact,
        }
    }
}

/// A rotating file the log is mirrored into.
#[derive(Debug, Clone)]
pub struct FileSink {
    /// Directory holding the rotated files.
    pub dir: PathBuf,
    /// File name stem; rotated files become `{stem}.{date}`.
    pub stem: String,
    /// How many rotated files to keep.
    pub max_files: usize,
}

/// Everything the subscriber needs, resolved from config plus the environment.
///
/// It is `Debug` on purpose — but note the fields are log *configuration*, and
/// the plugin's rule about never logging secrets still applies to callers that
/// print it.
#[derive(Debug, Clone)]
pub struct LogSettings {
    /// `EnvFilter` directive string, e.g. `info` or `mpv_stt_plugin_rs::stt=trace,warn`.
    pub filter: String,
    /// Directive string for the file sink when it should differ from `filter`.
    /// `None` means "reuse `filter`" (what the `file_level` key produces).
    pub file_filter: Option<String>,
    pub format: LogFormat,
    /// `None` = colorize only when the sink is a terminal.
    pub ansi: Option<bool>,
    pub file: Option<FileSink>,
    /// Queue `warn` and above for the plugin to draw on mpv's OSD.
    pub osd: bool,
}

impl Default for LogSettings {
    fn default() -> Self {
        Self {
            filter: "info".to_string(),
            file_filter: None,
            format: LogFormat::Compact,
            ansi: None,
            file: None,
            osd: true,
        }
    }
}

impl LogSettings {
    /// Resolve settings from the `[log]` config section, letting the
    /// `MPV_STT_PLUGIN_RS_LOG` environment variable win over `log.level`.
    ///
    /// `default_file_dir` is where `file = "auto"` points (the directory the
    /// config itself lives in); `None` disables the automatic file sink, which
    /// is what platforms with no writable config directory get.
    pub fn from_config(config: &crate::config::LogConfig, default_file_dir: Option<&Path>) -> Self {
        let env_filter = std::env::var(LOG_ENV).ok().filter(|s| !s.trim().is_empty());

        let level = config.level.trim();
        let filter = env_filter
            .clone()
            .unwrap_or_else(|| if level.is_empty() { "info".into() } else { level.into() });

        // The file keeps more than the terminal on purpose: the terminal is for
        // watching, the file is for diagnosing after the fact.
        let mut file_filter = if config.file_level.trim().is_empty() {
            None
        } else {
            Some(config.file_level.trim().to_string())
        };
        // An explicit MPV_STT_PLUGIN_RS_LOG is a deliberate global override, so
        // it applies to every sink instead of being silently overridden by the
        // file's own (lower) default.
        if env_filter.is_some() {
            file_filter = None;
        }

        let file = match config.file.trim() {
            "" | "off" | "none" => None,
            "auto" => default_file_dir.map(|dir| FileSink {
                dir: dir.to_path_buf(),
                stem: DEFAULT_LOG_FILE_NAME.to_string(),
                max_files: config.file_max_files.max(1),
            }),
            path => {
                let path = PathBuf::from(path);
                let dir = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| default_file_dir.unwrap_or(Path::new(".")).to_path_buf());
                let stem = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| DEFAULT_LOG_FILE_NAME.to_string());
                Some(FileSink {
                    dir,
                    stem,
                    max_files: config.file_max_files.max(1),
                })
            }
        };

        Self {
            filter,
            file_filter,
            format: LogFormat::parse(&config.format),
            ansi: match config.ansi.trim().to_ascii_lowercase().as_str() {
                "true" | "yes" | "on" | "1" => Some(true),
                "false" | "no" | "off" | "0" => Some(false),
                _ => None,
            },
            file,
            osd: config.osd,
        }
    }
}

/// Keeps the file sink's writer alive for as long as the subscriber needs it.
///
/// `RollingFileAppender` is moved into the layer, which lives in the global
/// subscriber for the process lifetime, so there is nothing to flush by hand —
/// the guard exists to make the lifetime explicit at the call site.
pub struct LogGuard {
    _private: (),
}

/// Install the global subscriber. Returns `None` if one is already installed
/// (a second plugin instance, or a test that ran first) instead of panicking.
///
/// Must be called on a thread that already has no subscriber active.
pub fn init(settings: &LogSettings) -> Option<LogGuard> {
    // `osd` needs the queue to exist before any record can be emitted.
    if settings.osd {
        let _ = osd_sender();
    }

    if build_subscriber(settings).try_init().is_err() {
        return None;
    }

    Some(LogGuard { _private: () })
}

/// The whole subscriber: `tracing-error`'s layer (so a log line can carry a span
/// trace) plus every configured sink.
///
/// The sinks are added as one `Vec` of boxed layers rather than one `with` call
/// each, so the subscriber's concrete type does not depend on how many of them
/// the config enabled — and the result can be boxed before the OSD layer is
/// added, which is the only part that needs `LookupSpan`.
fn build_subscriber(settings: &LogSettings) -> Box<dyn Subscriber + Send + Sync> {
    let registry = tracing_subscriber::registry().with(ErrorLayer::default());
    let mut layers: Vec<Box<dyn Layer<_> + Send + Sync>> = Vec::new();
    if let Some(file) = file_sink(settings) {
        layers.push(file);
    }
    layers.push(event_sink(settings));
    let subscriber = registry.with(layers);
    if settings.osd {
        Box::new(subscriber.with(OsdLayer::new(build_filter(&settings.filter))))
    } else {
        Box::new(subscriber)
    }
}

/// The rotating file layer, or `None` when the config disables it or the
/// directory cannot be written to. A read-only config directory must not cost
/// the user their terminal logs, so the failure is reported and skipped.
#[cfg(not(target_os = "android"))]
fn file_sink<S>(settings: &LogSettings) -> Option<Box<dyn Layer<S> + Send + Sync>>
where
    S: Subscriber + for<'a> LookupSpan<'a> + Send + Sync + 'static,
{
    let sink = settings.file.as_ref()?;

    let appender = match tracing_appender::rolling::Builder::new()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix(sink.stem.clone())
        .max_log_files(sink.max_files)
        .build(&sink.dir)
    {
        Ok(appender) => appender,
        Err(err) => {
            eprintln!(
                "{LOG_TAG}: cannot write logs to {}: {err} (continuing without a log file)",
                sink.dir.join(&sink.stem).display()
            );
            return None;
        }
    };

    let filter = build_filter(settings.file_filter.as_deref().unwrap_or(&settings.filter));
    // The file is read in a text editor, so ANSI escapes would be noise.
    let layer: Box<dyn Layer<S> + Send + Sync> =
        Box::new(event_layer(settings.format, appender, false));
    Some(Box::new(layer.with_filter(filter)))
}

/// Android has no per-app directory the plugin can rely on being writable, and
/// logcat is already where this process's output belongs.
#[cfg(target_os = "android")]
fn file_sink<S>(settings: &LogSettings) -> Option<Box<dyn Layer<S> + Send + Sync>>
where
    S: Subscriber + for<'a> LookupSpan<'a> + Send + Sync + 'static,
{
    let _ = settings;
    None
}

/// The layer that reaches whoever is watching the process run: logcat on
/// Android, stderr everywhere else.
#[cfg(not(target_os = "android"))]
fn event_sink<S>(settings: &LogSettings) -> Box<dyn Layer<S> + Send + Sync>
where
    S: Subscriber + for<'a> LookupSpan<'a> + Send + Sync + 'static,
{
    let ansi = settings.ansi.unwrap_or_else(|| {
        use std::io::IsTerminal;
        std::io::stderr().is_terminal()
    });
    let filter = build_filter(&settings.filter);
    let layer: Box<dyn Layer<S> + Send + Sync> =
        Box::new(event_layer(settings.format, std::io::stderr, ansi));
    Box::new(layer.with_filter(filter))
}

#[cfg(target_os = "android")]
fn event_sink<S>(settings: &LogSettings) -> Box<dyn Layer<S> + Send + Sync>
where
    S: Subscriber + for<'a> LookupSpan<'a> + Send + Sync + 'static,
{
    let filter = build_filter(&settings.filter);
    let layer: Box<dyn Layer<S> + Send + Sync> = Box::new(
        tracing_android::layer(LOG_TAG)
            .expect("the plugin never installs a global logger, so this cannot fail"),
    );
    Box::new(layer.with_filter(filter))
}

/// Every module of this crate lives under this prefix, so it is the target that
/// a bare level is scoped to and the one the EnvFilter directives match.
const CRATE_TARGET: &str = "mpv_stt_plugin_rs";

/// Directives for one sink. A malformed directive is a typo, not a reason to
/// lose every log line, so the parse is lossy and keeps whatever it understood.
fn build_filter(directives: &str) -> EnvFilter {
    let directives = if directives.trim().is_empty() {
        "info"
    } else {
        directives
    };
    EnvFilter::builder()
        // Anything the user did not name stays off. `warn` here rather than a
        // level would let every linked crate (hyper, reqwest, tokio) log at its
        // own default volume; the plugin's own records are the point.
        .with_default_directive(LevelFilter::OFF.into())
        .parse_lossy(scope(directives))
}

/// Rewrite a directive string so a bare level means *this plugin* at that level.
///
/// `debug` says "the plugin, verbosely" — not "everything linked into this
/// process, verbosely", which would bury the plugin's own lines under a
/// connection-by-connection account of hyper's HTTP client. A directive that
/// already names a target is passed through, so `hyper_util=trace` is how
/// someone asks for that traffic deliberately.
fn scope(directives: &str) -> String {
    directives
        .split(',')
        .map(str::trim)
        .filter(|directive| !directive.is_empty())
        .map(|directive| {
            // A target directive (`stt=debug`), or a span directive
            // (`[request]=trace`), is already scoped by the user.
            if directive.contains('=') || directive.contains('[') {
                directive.to_string()
            } else {
                format!("{CRATE_TARGET}={directive}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// One event per line, optionally with span lifecycle lines.
///
/// `W: for<'w> MakeWriter<'w>` rather than a concrete writer so the same
/// function serves stderr and the rotating appender.
fn event_layer<S, W>(
    format: LogFormat,
    writer: W,
    ansi: bool,
) -> Box<dyn Layer<S> + Send + Sync>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    W: for<'w> tsfmt::MakeWriter<'w> + Send + Sync + 'static,
{
    // The three formats have three different types, so each arm boxes its own.
    let layer = tsfmt::layer()
        .with_writer(writer)
        .with_target(true)
        .with_ansi(ansi);
    match format {
        LogFormat::Compact => Box::new(layer.compact()),
        LogFormat::Full => Box::new(
            layer.with_span_events(tsfmt::format::FmtSpan::NEW | tsfmt::format::FmtSpan::CLOSE),
        ),
        LogFormat::Json => Box::new(layer.json()),
    }
}

// ---------------------------------------------------------------------------
// OSD sink
// ---------------------------------------------------------------------------

/// A `warn`/`error` record waiting to be shown on mpv's OSD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OsdNotice {
    pub level: Level,
    pub message: String,
}

/// mpv's OSD renders a single line and truncates; keep messages well inside it.
const OSD_MAX_CHARS: usize = 110;

/// How many notices one drain may draw. A failure storm must not fight the
/// progress text for the screen.
const OSD_MAX_PER_DRAIN: usize = 3;

static OSD_SENDER: OnceLock<Mutex<Sender<OsdNotice>>> = OnceLock::new();
static OSD_RECEIVER: OnceLock<Mutex<std::sync::mpsc::Receiver<OsdNotice>>> = OnceLock::new();

fn osd_sender() -> &'static Mutex<Sender<OsdNotice>> {
    OSD_SENDER.get_or_init(|| {
        let (sender, receiver) = channel();
        let _ = OSD_RECEIVER.set(Mutex::new(receiver));
        Mutex::new(sender)
    })
}

/// Hand a notice to the plugin. Dropped when nobody is listening, which is the
/// case for every process that uses this crate as a library (the FFI surface).
fn queue_osd(notice: OsdNotice) {
    if let Ok(sender) = osd_sender().lock() {
        // A full queue is impossible (unbounded) and a disconnected one just
        // means the plugin is gone; neither is worth a log record about logging.
        let _ = sender.send(notice);
    }
}

/// Take the notices accumulated since the last call. Non-blocking: the caller
/// is mpv's event thread and must never wait on the log path.
pub fn take_osd_notices() -> Vec<OsdNotice> {
    let Some(receiver) = OSD_RECEIVER.get() else {
        return Vec::new();
    };
    let Ok(receiver) = receiver.lock() else {
        return Vec::new();
    };
    let mut notices = Vec::new();
    while let Ok(notice) = receiver.try_recv() {
        notices.push(notice);
    }
    notices
}

/// Collapse a drain into the lines to draw: consecutive duplicates merged with a
/// count (a dead backend repeats the same failure per chunk), capped so a storm
/// cannot monopolize the OSD.
///
/// The lines are the display text as the call site wrote it, with no level
/// prefix added: that text is already written for the viewer (`STT: …`,
/// `翻译失败 …`), so decorating it here would read as `STT WARN: STT: …`.
pub fn format_osd_notices(notices: &[OsdNotice]) -> Vec<String> {
    let mut out: Vec<(Level, String, usize)> = Vec::new();
    for notice in notices {
        match out.last_mut() {
            Some((level, message, count)) if *level == notice.level && *message == notice.message => {
                *count += 1;
            }
            _ => out.push((notice.level, notice.message.clone(), 1)),
        }
    }
    out.into_iter()
        .take(OSD_MAX_PER_DRAIN)
        .map(|(_, message, count)| {
            if count > 1 {
                format!("{message} (x{count})")
            } else {
                message
            }
        })
        .collect()
}

/// Field a record sets to put itself on screen.
///
/// A record reaches the OSD when it carries this field. Its value is the text
/// to draw, which is *not* the log message: a log line says what happened to
/// the developer (`chunk failed; ending the session`), the on-screen line says
/// what happened to the viewer (`STT failed: cannot reach the server`).
pub const FIELD_DISPLAY: &str = "display";

/// Bound a line to what mpv's OSD can show, for use as the `display` field:
/// `warn!(display = %logging::osd_line(&text), "…")`.
pub fn osd_line(text: &str) -> String {
    one_line(text, OSD_MAX_CHARS)
}

/// Level below which a record is never drawn, whatever its fields say. `info`
/// is the lowest milestone a user should see; a `debug!` or `trace!` carrying a
/// `display` field is a mistake, not a request.
const OSD_MIN_LEVEL: Level = Level::INFO;

/// Layer that copies the records marked for display into the OSD queue.
///
/// It carries the same filter as the visible sinks, so `log.level = "warn"`
/// quiets the screen along with the terminal — the level setting is the user's
/// one noise dial, and having it silence the log but not the OSD would be
/// surprising.
struct OsdLayer {
    min_level: Level,
    filter: EnvFilter,
}

impl OsdLayer {
    fn new(filter: EnvFilter) -> Self {
        Self {
            min_level: OSD_MIN_LEVEL,
            filter,
        }
    }
}

/// Pulls the `display` field out of an event. Its presence is what puts the
/// record on screen, so nothing outside this field is looked at.
#[derive(Default)]
struct DisplayVisitor {
    display: Option<String>,
}

impl Visit for DisplayVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == FIELD_DISPLAY {
            self.display = Some(format!("{value:?}"));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == FIELD_DISPLAY {
            self.display = Some(value.to_string());
        }
    }
}

impl<S> Layer<S> for OsdLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let metadata = event.metadata();
        if *metadata.level() > self.min_level || !self.filter.enabled(metadata, ctx) {
            return;
        }
        let mut visitor = DisplayVisitor::default();
        event.record(&mut visitor);
        let Some(display) = visitor.display else {
            return;
        };
        queue_osd(OsdNotice {
            level: *metadata.level(),
            message: one_line(&display, OSD_MAX_CHARS),
        });
    }
}

/// Flatten whitespace and cut to `max` characters, so a multi-line or
/// server-supplied message still fits mpv's single OSD line.
pub fn one_line(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    let mut out: String = flat.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

// ---------------------------------------------------------------------------
// Panics
// ---------------------------------------------------------------------------

/// Route panics through the logger so they reach every sink — logcat included,
/// where a panic that only wrote to stderr would be lost entirely.
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let payload = payload_of(info);
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!(
            target: TARGET_INTERNAL,
            location = %location,
            "plugin panicked: {payload}"
        );
        tracing::debug!(target: TARGET_INTERNAL, backtrace = %backtrace, "panic backtrace");
    }));
}

fn payload_of(info: &std::panic::PanicHookInfo<'_>) -> String {
    if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

// ---------------------------------------------------------------------------
// Spans
// ---------------------------------------------------------------------------

/// Start a media session span: one loaded medium, from the first chunk to the
/// last. Held in `PluginState` for the whole session, so every event recorded
/// while it is entered carries the session id and the file it belongs to.
pub fn session_span(session: u64, media: &str, duration_ms: u64, mode: &str) -> Span {
    tracing::info_span!(
        "session",
        session,
        media = %one_line(media, 120),
        duration_ms,
        mode
    )
}

/// Start a chunk span: one audio segment's round trip through ffmpeg, the
/// remote recognizer, the subtitle merge and the translation hand-off.
///
/// `gen` is the backend's cancellation generation, which is what tells a
/// superseded chunk apart from a live one when both appear in the log.
pub fn chunk_span(session: u64, seq: u64, start_ms: u64, duration_ms: u64, generation: u64) -> Span {
    tracing::debug_span!(
        "chunk",
        session,
        seq,
        start_ms,
        dur_ms = duration_ms,
        gen = generation
    )
}

// ---------------------------------------------------------------------------
// Error causes
// ---------------------------------------------------------------------------

/// Render an error together with its `source()` chain, root cause first.
///
/// The chain is what makes an `error!` line actionable: the outer message says
/// which step failed, the root cause says why (connection refused, invalid
/// JSON, TLS handshake). Log it as its own field so the message stays readable.
pub fn err_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut causes = vec![error.to_string()];
    let mut source = error.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        // Nested wrappers commonly repeat themselves; keep the chain readable.
        if !causes.contains(&text) {
            causes.push(text);
        }
        source = cause.source();
    }
    causes.reverse();
    causes.join(" <- ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex as StdMutex};

    /// A `MakeWriter` the tests can read back.
    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<StdMutex<Vec<u8>>>);

    impl SharedBuffer {
        fn contents(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    struct BufferWriter(SharedBuffer);

    impl Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'w> tsfmt::MakeWriter<'w> for SharedBuffer {
        type Writer = BufferWriter;

        fn make_writer(&'w self) -> Self::Writer {
            BufferWriter(self.clone())
        }
    }

    /// The OSD queue is process-wide, so tests that write to it must not run
    /// alongside each other or they would read one another's notices.
    static OSD_TEST_LOCK: StdMutex<()> = StdMutex::new(());

    /// A subscriber with just the OSD layer, so the test can inspect the queue
    /// without touching the global one.
    fn with_osd_layer<F: FnOnce()>(f: F) -> Vec<OsdNotice> {
        let _guard = OSD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Drain anything a previous test left behind.
        let _ = take_osd_notices();
        let subscriber =
            tracing_subscriber::registry().with(OsdLayer::new(build_filter("info")));
        tracing::subscriber::with_default(subscriber, f);
        take_osd_notices()
    }

    #[test]
    fn env_filter_accepts_default_and_target_directives() {
        // The plain level form the config documents.
        assert!(!build_filter("info").to_string().is_empty());
        // The per-module form: this is how a user turns on one subsystem only.
        let filter = build_filter("mpv_stt_plugin_rs::stt=trace,warn");
        let rendered = filter.to_string();
        assert!(
            rendered.contains("mpv_stt_plugin_rs::stt") || rendered.contains("warn"),
            "target directive should survive parsing: {rendered}"
        );
        // A typo must not cost the user all their logs.
        assert!(!build_filter("stt=not-a-level").to_string().is_empty());
        // An empty directive set falls back to the default level.
        assert!(!build_filter("").to_string().is_empty());
    }

    #[test]
    fn a_bare_level_scopes_to_this_crate() {
        assert_eq!(scope("debug"), "mpv_stt_plugin_rs=debug");
        assert_eq!(scope("info"), "mpv_stt_plugin_rs=info");
        // Spaces and empty entries are tolerated.
        assert_eq!(scope(" info , "), "mpv_stt_plugin_rs=info");
        // A user naming a target keeps it, including one outside the crate.
        assert_eq!(scope("hyper_util=trace"), "hyper_util=trace");
        assert_eq!(
            scope("mpv_stt_plugin_rs::stt=trace,warn"),
            "mpv_stt_plugin_rs::stt=trace,mpv_stt_plugin_rs=warn"
        );
        // A span directive is already scoped; leave it alone.
        assert_eq!(scope("[request]=trace"), "[request]=trace");
    }

    #[test]
    fn only_this_crates_records_survive_a_bare_level() {
        // A bare `debug` must not turn on the HTTP client under the plugin —
        // that is the whole reason the targets are scoped. Records are emitted
        // with an explicit `target:` so the test can play both roles.
        let buffer = SharedBuffer::default();
        let subscriber = tracing_subscriber::registry().with(
            event_layer(LogFormat::Compact, buffer.clone(), false)
                .with_filter(build_filter("debug")),
        );
        tracing::subscriber::with_default(subscriber, || {
            tracing::debug!(
                target: "hyper_util::client::legacy::connect::http",
                "connecting to 127.0.0.1:18001"
            );
            tracing::debug!(target: "mpv_stt_plugin_rs::stt", "chunk transcribed");
        });

        let written = buffer.contents();
        assert!(
            written.contains("chunk transcribed"),
            "the plugin's own records pass: {written}"
        );
        assert!(
            !written.contains("connecting to"),
            "third-party records stay off: {written}"
        );
    }

    #[test]
    fn format_parsing_falls_back_to_compact() {
        assert_eq!(LogFormat::parse("full"), LogFormat::Full);
        assert_eq!(LogFormat::parse("JSON"), LogFormat::Json);
        assert_eq!(LogFormat::parse("compat"), LogFormat::Compact);
        assert_eq!(LogFormat::parse(""), LogFormat::Compact);
    }

    #[test]
    fn osd_layer_forwards_only_records_marked_for_display() {
        let notices = with_osd_layer(|| {
            tracing::trace!(display = %osd_line("trace noise"), "trace noise");
            tracing::debug!(display = %osd_line("debug noise"), "debug noise");
            // A record with no display field is plumbing: it belongs in the log,
            // not on the viewer's screen.
            tracing::warn!("retrying the request");
            tracing::warn!(display = %osd_line("gateway is down"), "translation backend is not answering");
            tracing::error!(display = %osd_line("session ended"), "chunk failed; ending the session");
        });

        assert_eq!(
            notices.len(),
            2,
            "only records marked for display belong on the OSD: {notices:?}"
        );
        assert_eq!(notices[0].level, Level::WARN);
        assert_eq!(notices[0].message, "gateway is down");
        assert_eq!(notices[1].level, Level::ERROR);
        assert_eq!(notices[1].message, "session ended");
    }

    #[test]
    fn osd_layer_ignores_a_display_field_below_info() {
        let notices = with_osd_layer(|| {
            tracing::debug!(display = %osd_line("a debug line"), "debug with display");
        });
        assert!(notices.is_empty(), "debug never reaches the screen: {notices:?}");
    }

    #[test]
    fn osd_notice_is_one_bounded_line() {
        let long = format!("上游返回了很长的错误 {}\n第二行", "x".repeat(400));
        let notices = with_osd_layer(|| tracing::warn!(display = %osd_line(&long), "a long failure"));
        assert_eq!(notices.len(), 1);
        let message = &notices[0].message;
        assert!(
            message.chars().count() <= OSD_MAX_CHARS,
            "OSD line must fit mpv's single line: {} chars",
            message.chars().count()
        );
        assert!(!message.contains('\n'), "OSD line must not wrap: {message}");
        assert!(message.ends_with('…'));
    }

    #[test]
    fn osd_drain_merges_repeats_and_caps_the_batch() {
        let notice = |message: &str| OsdNotice {
            level: Level::WARN,
            message: message.to_string(),
        };
        let drained = vec![
            notice("translation failed"),
            notice("translation failed"),
            notice("translation failed"),
            notice("stt 503"),
        ];
        let lines = format_osd_notices(&drained);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("(x3)"), "repeats collapse: {lines:?}");
        assert!(lines[1].contains("stt 503"));

        // A storm is capped rather than drawn in full.
        let storm: Vec<_> = (0..10).map(|i| notice(&format!("failure {i}"))).collect();
        assert_eq!(format_osd_notices(&storm).len(), OSD_MAX_PER_DRAIN);
    }

    #[test]
    fn err_chain_keeps_the_root_cause() {
        use std::error::Error;
        use std::fmt;

        #[derive(Debug)]
        struct Inner;
        impl fmt::Display for Inner {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "connection refused")
            }
        }
        impl Error for Inner {}

        #[derive(Debug)]
        struct Outer(Inner);
        impl fmt::Display for Outer {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "request to http://127.0.0.1:8000 failed")
            }
        }
        impl Error for Outer {
            fn source(&self) -> Option<&(dyn Error + 'static)> {
                Some(&self.0)
            }
        }

        let chain = err_chain(&Outer(Inner));
        assert!(chain.starts_with("connection refused"), "{chain}");
        assert!(chain.contains("<-"), "{chain}");
        assert!(chain.contains("http://127.0.0.1:8000"), "{chain}");
    }

    #[test]
    fn compact_events_land_in_the_writer() {
        let buffer = SharedBuffer::default();
        let subscriber = tracing_subscriber::registry().with(
            event_layer(LogFormat::Compact, buffer.clone(), false)
                .with_filter(build_filter("info")),
        );
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(chunk = 3, "merged chunk");
        });
        let written = buffer.contents();
        assert!(written.contains("merged chunk"), "{written}");
        assert!(written.contains("chunk=3"), "fields must be rendered: {written}");
        assert!(!written.contains('\u{1b}'), "no ANSI codes in a plain writer");
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn file_layer_writes_a_rotating_log() {
        let dir = tempfile::tempdir().unwrap();
        let sink = FileSink {
            dir: dir.path().to_path_buf(),
            stem: DEFAULT_LOG_FILE_NAME.to_string(),
            max_files: 2,
        };
        let settings = LogSettings {
            filter: "info".to_string(),
            file_filter: Some("debug".to_string()),
            format: LogFormat::Full,
            ansi: Some(false),
            file: Some(sink.clone()),
            osd: false,
        };

        let subscriber = tracing_subscriber::registry()
            .with(ErrorLayer::default())
            .with({
                let appender = tracing_appender::rolling::Builder::new()
                    .rotation(tracing_appender::rolling::Rotation::DAILY)
                    .filename_prefix(sink.stem.clone())
                    .max_log_files(sink.max_files)
                    .build(&sink.dir)
                    .unwrap();
                event_layer(LogFormat::Full, appender, false)
                    .with_filter(build_filter(&settings.filter))
            });
        tracing::subscriber::with_default(subscriber, || {
            tracing::info_span!("chunk", start_ms = 15_000u64).in_scope(|| {
                tracing::info!("chunk done");
            });
        });

        let written = std::fs::read_dir(&sink.dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| std::fs::read_to_string(entry.path()).unwrap())
            .collect::<String>();
        assert!(written.contains("chunk done"), "{written}");
        assert!(written.contains("start_ms=15000"), "span fields: {written}");
        assert!(!written.contains('\u{1b}'), "the file must stay plain text");
    }

    #[test]
    fn settings_from_config_let_the_environment_win() {
        use crate::config::LogConfig;

        let dir = Path::new("/tmp/example");
        let config = LogConfig {
            level: "info".to_string(),
            file_level: "debug".to_string(),
            file: "auto".to_string(),
            file_max_files: 3,
            ..LogConfig::default()
        };

        let settings = LogSettings::from_config(&config, Some(dir));
        assert_eq!(settings.filter, "info");
        assert_eq!(settings.file_filter.as_deref(), Some("debug"));
        assert_eq!(settings.file.as_ref().unwrap().dir, dir);
        assert_eq!(settings.file.as_ref().unwrap().max_files, 3);

        // `file = ""` turns the sink off without touching anything else.
        let off = LogConfig {
            file: String::new(),
            ..config.clone()
        };
        assert!(LogSettings::from_config(&off, Some(dir)).file.is_none());

        // No config directory to write to means no automatic sink.
        assert!(LogSettings::from_config(&config, None).file.is_none());
    }

    #[test]
    fn explicit_file_paths_split_into_directory_and_stem() {
        use crate::config::LogConfig;

        let config = LogConfig {
            file: "/var/log/stt.log".to_string(),
            ..LogConfig::default()
        };
        let settings = LogSettings::from_config(&config, None);
        let sink = settings.file.expect("an explicit path always yields a sink");
        assert_eq!(sink.dir, PathBuf::from("/var/log"));
        assert_eq!(sink.stem, "stt.log");

        // A bare filename lands in the default directory rather than nowhere.
        let config = LogConfig {
            file: "stt.log".to_string(),
            ..LogConfig::default()
        };
        let sink = LogSettings::from_config(&config, Some(Path::new("/tmp/example")))
            .file
            .unwrap();
        assert_eq!(sink.dir, PathBuf::from("/tmp/example"));
        assert_eq!(sink.stem, "stt.log");
    }
}
