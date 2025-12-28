use log::{debug, error, info, trace, warn};
use mpv_client::{Event, Handle, mpv_handle};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

use crate::audio::AudioExtractor;
use crate::config::Config;
use crate::srt::{self, SrtFile};
use crate::subtitle_manager::SubtitleManager;
use crate::translate::{AsyncTranslationQueue, TranslationTask, TranslatorConfig};
use crate::whisper::{WhisperConfig, WhisperRunner};

struct TempPaths {
    _dir: TempDir,
    tmp_wav: PathBuf,
    tmp_sub: PathBuf,
    tmp_cache: PathBuf,
}

impl TempPaths {
    fn new() -> Self {
        let dir = tempfile::Builder::new()
            .prefix("mpv_whispersubs_rs_")
            .tempdir()
            .expect("failed to create temp dir");

        Self {
            tmp_wav: dir.path().join("audio.wav"),
            // `tmp_sub` is a prefix; intermediate files are derived via `format!("{}_append...", tmp_sub.display())`
            // and the main subtitle file is `tmp_sub.with_extension("srt")`.
            tmp_sub: dir.path().join("subs"),
            tmp_cache: dir.path().join("cache.mkv"),
            _dir: dir,
        }
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

enum ProcessingMode {
    Network,
    Local {
        media_path: String,
        file_length_ms: u64,
        subtitle_path: PathBuf,
    },
}

struct PluginState {
    config: Config,
    paths: TempPaths,
    audio_extractor: AudioExtractor,
    whisper_runner: WhisperRunner,
    async_translation_queue: Option<AsyncTranslationQueue>,
    subtitle_manager: SubtitleManager,

    running: bool,
    shutting_down: bool,
    subs_loaded: bool,
    current_pos_ms: u64,
    last_playback_pos_ms: Option<u64>,
    chunk_dur: u64,
    mode: Option<ProcessingMode>,
    pending_auto_start: bool, // Delayed auto-start after file loads
    file_loaded: bool,        // Track if file is ready
}

impl PluginState {
    fn new(config: Config) -> Self {
        let chunk_dur = config.chunk_size_ms;
        let audio_extractor = AudioExtractor::default()
            .with_ffmpeg_timeout(config.ffmpeg_timeout_ms)
            .with_ffprobe_timeout(config.ffprobe_timeout_ms);

        // Initialize Whisper
        let whisper_config = WhisperConfig::new(config.model_path.clone())
            .with_threads(config.threads)
            .with_language(config.language.clone())
            .with_inference_device(config.inference_device)
            .with_gpu_device(config.gpu_device)
            .with_cuda_flash_attn(config.cuda_flash_attn)
            .with_timeout_ms(config.whisper_timeout_ms);

        let whisper_runner = WhisperRunner::new(whisper_config);

        // Initialize Translator (builtin Google Translate)
        let translator_config =
            TranslatorConfig::new(config.from_lang.clone(), config.to_lang.clone())
                .with_timeout_ms(config.translate_timeout_ms);

        // Initialize async translation queue (always enabled)
        let async_translation_queue = Some(AsyncTranslationQueue::new(translator_config));

        Self {
            chunk_dur,
            config,
            paths: TempPaths::new(),
            audio_extractor,
            whisper_runner,
            async_translation_queue,
            subtitle_manager: SubtitleManager::new(),
            running: false,
            shutting_down: false,
            subs_loaded: false,
            current_pos_ms: 0,
            last_playback_pos_ms: None,
            mode: None,
            pending_auto_start: false,
            file_loaded: false,
        }
    }

    fn toggle_whisper(&mut self, client: &mut Handle) {
        if self.running {
            info!("Disabling Whisper transcription");
            self.running = false;
            let _ = client.command(&["show-text", "Whisper: Off"]);
            self.cleanup(client);
        } else {
            info!("Enabling Whisper transcription");
            self.running = true;
            let _ = client.command(&["show-text", "Whisper: On"]);
            self.start_transcription(client);
        }
    }

    fn start_transcription(&mut self, client: &mut Handle) {
        debug!("Starting transcription");
        // Get current position
        let time_pos: f64 = client.get_property("time-pos").unwrap_or(0.0);
        self.current_pos_ms = (time_pos * 1000.0) as u64;
        self.last_playback_pos_ms = Some(self.current_pos_ms);
        trace!("Current playback position: {}ms", self.current_pos_ms);

        // Check if network stream - use multiple detection methods
        let is_network = self.detect_network_stream(client);

        if is_network {
            // Network stream mode
            debug!("Detected network stream, entering network mode");
            let _ = client.command(&[
                "show-text",
                "Whisper: Starting network stream transcription...",
            ]);

            // Enable caching
            let _ = client.set_property("cache", true);

            // Set demuxer max bytes if configured (for better lookahead caching)
            if let Some(max_bytes) = self.config.demuxer_max_bytes {
                debug!("Setting demuxer-max-bytes to {} bytes", max_bytes);
                let _ = client.set_property("demuxer-max-bytes", max_bytes);
            }

            self.mode = Some(ProcessingMode::Network);

            // Create initial subtitles
            if self.process_chunk(client) {
                let main_srt = self.paths.tmp_sub.with_extension("srt");
                let _ = client.command(&["sub-add", main_srt.to_str().unwrap()]);
                self.subs_loaded = true;
            }

            info!(
                "Network stream mode active, current_pos: {}ms",
                self.current_pos_ms
            );
        } else {
            // Local file mode
            debug!("Detected local file, entering local mode");
            let media_path: Result<String, _> = client.get_property("path");
            let duration: Result<f64, _> = client.get_property("duration");

            if let (Ok(path), Ok(dur)) = (media_path, duration) {
                let file_length_ms = (dur * 1000.0) as u64;
                trace!("Media file: {}, duration: {}ms", path, file_length_ms);

                // Calculate subtitle path next to the video file when possible.
                // SAF content:// URIs are not writable as filesystem paths.
                let subtitle_path = Self::get_subtitle_path_for_media_uri(&path)
                    .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));
                info!("Subtitle will be saved to: {}", subtitle_path.display());

                let _ =
                    client.command(&["show-text", "Whisper: Starting local file transcription..."]);

                // Start from beginning if configured
                if self.config.start_at_zero {
                    self.current_pos_ms = 0;
                }

                self.mode = Some(ProcessingMode::Local {
                    media_path: path.clone(),
                    file_length_ms,
                    subtitle_path: subtitle_path.clone(),
                });

                // Create initial subtitles
                if self.process_chunk_local(client, &path, &subtitle_path) {
                    let _ = client.command(&["sub-add", subtitle_path.to_str().unwrap()]);
                    self.subs_loaded = true;
                }

                info!(
                    "Local file mode: {}, length: {}ms, start: {}ms",
                    path, file_length_ms, self.current_pos_ms
                );
            } else {
                let _ = client.command(&["show-text", "Whisper: Failed to get file info"]);
            }
        }
    }

    /// Main processing loop - called on each event loop iteration
    fn tick(&mut self, client: &mut Handle) {
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
        // Update playback position in translation queue
        if let Some(ref queue) = self.async_translation_queue {
            if let Ok(time_pos) = client.get_property::<f64>("time-pos") {
                queue.update_playback_position((time_pos * 1000.0) as u64);
            }
        }

        // Check for seek first. If cache isn't ready yet after a seek, we still want to update
        // `current_pos_ms` so we don't keep generating subtitles for the old position.
        self.check_seek(client, None);

        // Check for completed translations from async queue
        self.process_translation_results(client, None);

        // Get cache end time
        let cache_end_sec: Option<f64> = client.get_property("demuxer-cache-time").ok();
        if cache_end_sec.is_none() {
            trace!("Cache not ready yet");
            return; // Cache not ready yet
        }
        let cache_end_ms = (cache_end_sec.unwrap() * 1000.0) as u64;

        // Catch-up mode: check if we're too far behind playback (always enabled)
        if let Some(playback_pos_ms) = self.last_playback_pos_ms {
            let lag = if playback_pos_ms > self.current_pos_ms {
                playback_pos_ms - self.current_pos_ms
            } else {
                0
            };

            if lag > self.config.catchup_threshold_ms {
                // We're too far behind, skip to near current playback position
                let new_pos = playback_pos_ms - (playback_pos_ms % self.chunk_dur);
                info!(
                    "Catch-up: skipping from {}ms to {}ms (lag: {}ms)",
                    self.current_pos_ms, new_pos, lag
                );
                self.current_pos_ms = new_pos;
                let _ = client.command(&[
                    "show-text",
                    &format!("Whisper: Catching up to {}", Self::format_progress(new_pos)),
                    "3000",
                ]);
            }
        }

        // Look-ahead processing for network streams (always enabled)
        // Check how far ahead playback is from processing
        if let Some(playback_pos_ms) = self.last_playback_pos_ms {
            let ahead = if self.current_pos_ms > playback_pos_ms {
                self.current_pos_ms - playback_pos_ms
            } else {
                0
            };

            // Don't process too far ahead of playback
            if ahead > self.config.lookahead_limit_ms {
                trace!(
                    "Look-ahead limit reached: {}ms ahead, waiting for playback to catch up",
                    ahead
                );
                return;
            }
        }

        // Try to process multiple chunks ahead if they're cached
        let chunks_to_process = self.config.lookahead_chunks.max(1);
        for i in 0..chunks_to_process {
            let chunk_start_ms = self.current_pos_ms + (i as u64 * self.chunk_dur);
            let chunk_end_ms = chunk_start_ms + self.chunk_dur;

            // Check if this chunk is fully cached
            if chunk_end_ms > cache_end_ms {
                if i == 0 {
                    // Current chunk not cached, wait
                    trace!(
                        "Waiting for more cache: need {}ms, have {}ms",
                        chunk_end_ms, cache_end_ms
                    );
                } else {
                    // Future chunks not cached yet, that's fine
                    trace!(
                        "Look-ahead: chunk {} not cached yet (need {}ms, have {}ms)",
                        i + 1,
                        chunk_end_ms,
                        cache_end_ms
                    );
                }
                break; // Stop processing future chunks
            }

            // This chunk is cached, process it
            if i == 0 {
                debug!("Processing network chunk at {}ms", self.current_pos_ms);
            } else {
                debug!(
                    "Look-ahead: processing chunk {} at {}ms",
                    i + 1,
                    chunk_start_ms
                );
            }

            if self.process_chunk(client) {
                self.current_pos_ms += self.chunk_dur;

                if self.config.show_progress && i == 0 {
                    let _ = client.command(&[
                        "show-text",
                        &format!("Whisper: {}", Self::format_progress(self.current_pos_ms)),
                    ]);
                }
            } else {
                break; // Stop if processing failed
            }
        }
    }

    fn tick_local(
        &mut self,
        client: &mut Handle,
        media_path: &str,
        file_length_ms: u64,
        subtitle_path: &Path,
    ) {
        // Update playback position in translation queue
        if let Some(ref queue) = self.async_translation_queue {
            if let Ok(time_pos) = client.get_property::<f64>("time-pos") {
                queue.update_playback_position((time_pos * 1000.0) as u64);
            }
        }

        // Check for seek
        self.check_seek(client, Some(subtitle_path));

        // Check for completed translations from async queue
        self.process_translation_results(client, Some(subtitle_path));

        // Check if we're too far ahead of playback (look-ahead limit, always enabled)
        if let Some(playback_pos_ms) = self.last_playback_pos_ms {
            let ahead = if self.current_pos_ms > playback_pos_ms {
                self.current_pos_ms - playback_pos_ms
            } else {
                0
            };

            if ahead > self.config.lookahead_limit_ms {
                trace!(
                    "Look-ahead limit reached: {}ms ahead, waiting for playback to catch up",
                    ahead
                );
                return; // Wait for playback to catch up
            }
        }

        // Calculate remaining time
        let time_left = if file_length_ms > self.current_pos_ms {
            file_length_ms - self.current_pos_ms
        } else {
            0
        };

        // Adjust chunk size for last chunk
        if time_left > 0 && time_left < self.config.chunk_size_ms {
            self.chunk_dur = time_left;
        } else {
            self.chunk_dur = self.config.chunk_size_ms;
        }

        if time_left > 0 {
            // Look-ahead processing: process multiple chunks ahead (always enabled)
            let chunks_to_process = self.config.lookahead_chunks.max(1);

            for i in 0..chunks_to_process {
                let chunk_pos = self.current_pos_ms + (i as u64 * self.chunk_dur);
                if chunk_pos >= file_length_ms {
                    break; // Don't process beyond file end
                }

                // Catch-up mode: check if we're too far behind playback (always enabled)
                if let Some(playback_pos_ms) = self.last_playback_pos_ms {
                    let lag = if playback_pos_ms > self.current_pos_ms {
                        playback_pos_ms - self.current_pos_ms
                    } else {
                        0
                    };

                    if lag > self.config.catchup_threshold_ms {
                        // Skip to near current playback position
                        let new_pos = playback_pos_ms - (playback_pos_ms % self.chunk_dur);
                        info!(
                            "Catch-up: skipping from {}ms to {}ms (lag: {}ms)",
                            self.current_pos_ms, new_pos, lag
                        );
                        self.current_pos_ms = new_pos.min(file_length_ms);
                        let _ = client.command(&[
                            "show-text",
                            &format!("Whisper: Catching up to {}", Self::format_progress(new_pos)),
                            "3000",
                        ]);
                        break; // Don't process old chunks, restart loop with new position
                    }
                }

                // Process current chunk
                if i == 0 {
                    debug!(
                        "Processing local chunk at {}ms, remaining: {}ms",
                        self.current_pos_ms, time_left
                    );
                } else {
                    debug!("Look-ahead: processing chunk {} at {}ms", i + 1, chunk_pos);
                }

                if self.process_chunk_local(client, media_path, subtitle_path) {
                    self.current_pos_ms += self.chunk_dur;

                    if self.config.show_progress && i == 0 {
                        let _ = client.command(&[
                            "show-text",
                            &format!("Whisper: {}", Self::format_progress(self.current_pos_ms)),
                        ]);
                    }
                } else {
                    break; // Stop if processing failed
                }

                // Update remaining time for next iteration
                let new_time_left = if file_length_ms > self.current_pos_ms {
                    file_length_ms - self.current_pos_ms
                } else {
                    0
                };
                if new_time_left == 0 {
                    break;
                }
            }
        } else {
            // Finished processing
            info!("Finished processing local file");
            let msg = format!("Whisper: Saved subtitles to {}", subtitle_path.display());
            let _ = client.command(&["show-text", &msg, "5000"]);

            self.running = false;
            self.cleanup(client);
        }
    }

    fn check_seek(&mut self, client: &mut Handle, subtitle_path: Option<&Path>) {
        let playback_pos: Option<f64> = client.get_property("time-pos").ok();
        if let Some(pos) = playback_pos {
            let playback_pos_ms = (pos * 1000.0) as u64;

            // Update translation queue with current playback position
            if let Some(ref queue) = self.async_translation_queue {
                queue.update_playback_position(playback_pos_ms);
            }

            // Detect user seek by comparing against the last observed playback position.
            // IMPORTANT: `current_pos_ms` is the *processing cursor* (next chunk start), which can
            // legitimately run ahead of playback when the cache is full. Comparing playback to
            // `current_pos_ms` causes false "seek backward" detections and makes subtitles vanish.
            let Some(last_ms) = self.last_playback_pos_ms.replace(playback_pos_ms) else {
                return;
            };

            // Avoid treating normal playback progression (or time spent inside Whisper/translate)
            // as a seek. Since we process in chunk units, only treat jumps of >= 1 chunk as seek.
            let seek_threshold_ms = std::cmp::max(5_000, self.chunk_dur);
            let delta_ms = playback_pos_ms.abs_diff(last_ms);
            if delta_ms < seek_threshold_ms {
                return;
            }

            // Detect seek forward (user skipped ahead)
            if playback_pos_ms > last_ms {
                let new_pos = playback_pos_ms - (playback_pos_ms % self.chunk_dur);
                debug!(
                    "User seeked forward from {}ms to {}ms (delta: {}ms)",
                    last_ms, new_pos, delta_ms
                );
                let _ = client.command(&[
                    "show-text",
                    &format!("Whisper: Jumped to {}", Self::format_progress(new_pos)),
                    "3000",
                ]);

                // Keep existing subtitles, just update processing cursor.
                self.current_pos_ms = new_pos;
            }
            // Detect seek backward
            else {
                let new_pos = playback_pos_ms - (playback_pos_ms % self.chunk_dur);
                debug!(
                    "User seeked backward from {}ms to {}ms (delta: {}ms)",
                    last_ms, new_pos, delta_ms
                );
                let _ = client.command(&[
                    "show-text",
                    &format!("Whisper: Seeked back to {}", Self::format_progress(new_pos)),
                    "3000",
                ]);

                // Remove subtitles after the new position
                let new_pos_u32 = match u32::try_from(new_pos) {
                    Ok(v) => v,
                    Err(_) => {
                        warn!(
                            "Seek position {}ms exceeds subtitle timestamp range",
                            new_pos
                        );
                        u32::MAX
                    }
                };

                self.subtitle_manager.remove_after(new_pos_u32);
                self.current_pos_ms = new_pos;

                // Save updated subtitles - use subtitle_path for local files, tmp for network
                let srt_path = subtitle_path
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));
                let _ = self.subtitle_manager.save_to_file(&srt_path);
                if self.subs_loaded {
                    let _ = client.command(&["sub-reload"]);
                }
            }
        }
    }

    /// Process one chunk from network cache
    fn process_chunk(&mut self, client: &mut Handle) -> bool {
        // Dump cache
        let start_sec = self.current_pos_ms as f64 / 1000.0;
        let end_sec = (self.current_pos_ms + self.chunk_dur) as f64 / 1000.0;
        trace!("Dumping cache from {}s to {}s", start_sec, end_sec);

        let dump_result = client.command(&[
            "dump-cache",
            &start_sec.to_string(),
            &end_sec.to_string(),
            self.paths.tmp_cache.to_str().unwrap(),
        ]);

        if dump_result.is_err() {
            error!("dump-cache failed");
            return false;
        }

        // Extract audio from cache
        if !self.create_wav(self.paths.tmp_cache.to_str().unwrap(), 0) {
            return false;
        }

        self.transcribe_and_update(client, None)
    }

    /// Process one chunk from local file
    fn process_chunk_local(
        &mut self,
        client: &mut Handle,
        media_path: &str,
        subtitle_path: &Path,
    ) -> bool {
        // Extract audio directly from local file
        if !self.create_wav(media_path, self.current_pos_ms) {
            return false;
        }

        self.transcribe_and_update(client, Some(subtitle_path))
    }

    /// Common transcription and subtitle update logic
    fn transcribe_and_update(&mut self, client: &mut Handle, subtitle_path: Option<&Path>) -> bool {
        let tmp_sub_prefix = self.paths.tmp_sub.to_string_lossy().to_string();
        let append_path = format!("{}_append", &tmp_sub_prefix);
        let main_srt = subtitle_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));

        trace!("Starting Whisper transcription for current chunk");
        // Run Whisper transcription
        if let Err(e) = self.whisper_runner.transcribe(
            self.paths.tmp_wav.to_str().unwrap(),
            append_path.as_str(),
            self.chunk_dur,
        ) {
            error!("Whisper transcription failed: {}", e);
            return false;
        }

        // Offset timestamps
        let append_srt = format!("{}.srt", append_path);
        let offset_srt = format!("{}_append_offset.srt", &tmp_sub_prefix);

        if let Err(e) = srt::offset_srt_file(&append_srt, &offset_srt, self.current_pos_ms as i64) {
            error!("SRT offset failed: {}", e);
            return false;
        }

        // Add original subtitles first so recognition updates immediately
        let srt_file = match SrtFile::parse(&offset_srt) {
            Ok(srt) => srt,
            Err(_) => return false,
        };
        self.subtitle_manager.add_from_srt(&srt_file);
        if !self.save_subs(client, &main_srt) {
            return false;
        }

        // Translate using async translation queue (always enabled)
        if let Some(ref queue) = self.async_translation_queue {
            trace!("Submitting subtitles to async translation queue");
            for entry in &srt_file.entries {
                if entry.text.trim().is_empty() {
                    continue;
                }
                let start_ms = Self::timestamp_to_millis(entry.start_time);
                queue.submit(TranslationTask {
                    start_ms,
                    text: entry.text.clone(),
                });
            }
            debug!(
                "Submitted {} entries to async translation",
                srt_file.entries.len()
            );
        }

        // Keep only the main subtitle file on disk during playback to reduce clutter.
        // The `_append*` files are per-chunk intermediates and will be regenerated each chunk.
        self.paths.cleanup_intermediate_subs();

        debug!(
            "Processed chunk at {}ms, total subs: {}",
            self.current_pos_ms,
            self.subtitle_manager.len()
        );
        true
    }

    /// Process completed translation results from async queue
    fn process_translation_results(&mut self, client: &mut Handle, subtitle_path: Option<&Path>) {
        if let Some(ref queue) = self.async_translation_queue {
            let results = queue.try_recv_results();
            if !results.is_empty() {
                debug!("Received {} translation results", results.len());

                // Update subtitles with translations
                for result in results {
                    self.subtitle_manager
                        .update_translation(result.start_ms, &result.translated);
                }

                // Save updated subtitles
                let main_srt = subtitle_path
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| self.paths.tmp_sub.with_extension("srt"));
                let _ = self.save_subs(client, &main_srt);
            }
        }
    }

    fn timestamp_to_millis(ts: srtlib::Timestamp) -> u32 {
        let (h, m, s, ms) = ts.get();
        srtlib::Timestamp::convert_to_milliseconds(h, m, s, ms)
    }

    fn save_subs(&mut self, client: &mut Handle, main_srt: &Path) -> bool {
        if let Err(e) = self.subtitle_manager.save_to_file(main_srt) {
            error!("Failed to save subtitles: {}", e);
            return false;
        }
        if self.subs_loaded {
            let _ = client.command(&["sub-reload"]);
        }
        true
    }

    fn create_wav(&self, media_path: &str, start_ms: u64) -> bool {
        let result = self.audio_extractor.extract_audio_segment(
            media_path,
            self.paths.tmp_wav.to_str().unwrap(),
            start_ms,
            self.config.wav_chunk_size_ms,
        );

        match result {
            Ok(_) => true,
            Err(e) => {
                error!("Audio extraction failed: {}", e);
                false
            }
        }
    }

    fn cleanup(&mut self, _client: &mut Handle) {
        debug!("Cleaning up temporary files and state");

        // Set shutting down flag to stop any ongoing processing
        self.shutting_down = true;

        // Shutdown async translation queue if it exists
        if let Some(ref mut queue) = self.async_translation_queue {
            queue.force_shutdown();
        }

        self.paths.cleanup();
        self.subtitle_manager.clear();
        self.subs_loaded = false;
        self.current_pos_ms = 0;
        self.last_playback_pos_ms = None;
        self.mode = None;
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

    /// Detect if current media is a network stream
    fn detect_network_stream(&self, client: &mut Handle) -> bool {
        // Method 1: Check path/filename for http/https URLs
        if let Ok(path) = client.get_property::<String>("path") {
            debug!("Checking path for network stream: {}", path);
            if path.starts_with("http://") || path.starts_with("https://") {
                debug!("Detected network stream by URL prefix");
                return true;
            }
        }

        // Method 2: Check stream-open-filename
        if let Ok(filename) = client.get_property::<String>("stream-open-filename") {
            debug!("Checking stream-open-filename: {}", filename);
            if filename.starts_with("http://") || filename.starts_with("https://") {
                debug!("Detected network stream by stream-open-filename");
                return true;
            }
        }

        // Method 3: Check demuxer-via-network property
        if let Ok(via_network) = client.get_property::<String>("demuxer-via-network") {
            debug!("demuxer-via-network: {}", via_network);
            if via_network == "yes" {
                debug!("Detected network stream by demuxer-via-network");
                return true;
            }
        }

        debug!("Not detected as network stream, treating as local file");
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

/// MPV C plugin entry point
#[unsafe(no_mangle)]
pub extern "C" fn mpv_open_cplugin(handle: *mut mpv_handle) -> std::os::raw::c_int {
    init_logger();

    let client = Handle::from_ptr(handle);

    info!("WhisperSubs Rust plugin initializing...");

    // Print welcome message
    let _ = client.command(&["show-text", "WhisperSubs Rust plugin loaded!", "3000"]);
    info!("Plugin loaded, client name: {}", client.name());

    // Initialize plugin state with configuration
    let config = Config::load();
    let auto_start = config.auto_start;
    let mut state = PluginState::new(config);

    // Get client name first
    let client_name = client.name().to_string();

    // Register key binding
    let key_binding = format!("Ctrl+. script-message-to {} toggle-whisper", client_name);
    let section_name = format!("{}-input", client_name);

    let _ = client.command(&["define-section", &section_name, &key_binding, "default"]);
    let _ = client.command(&["enable-section", &section_name]);

    // Set auto-start flag (will start after file loads)
    if auto_start {
        info!("Auto-start enabled, waiting for file to load...");
        state.pending_auto_start = true;
    }

    // Main event loop with short timeout for continuous processing
    loop {
        // Use 0.1 second timeout to allow continuous processing
        match client.wait_event(0.1) {
            Event::Shutdown => {
                info!("Shutting down...");
                state.shutting_down = true;
                state.running = false;
                state.cleanup(client);
                info!("Shutdown complete");
                return 0;
            }
            Event::ClientMessage(msg) => {
                if state.shutting_down {
                    continue;
                }
                let args = msg.args();
                if !args.is_empty() {
                    let command = if args[0] == "toggle-whisper" {
                        Some("toggle-whisper")
                    } else if args.len() > 1 && args[1] == "toggle-whisper" {
                        Some("toggle-whisper")
                    } else {
                        None
                    };

                    if command.is_some() {
                        debug!("Toggling whisper...");
                        state.toggle_whisper(client);
                    }
                }
            }
            Event::FileLoaded => {
                if state.shutting_down {
                    continue;
                }
                debug!("File loaded event received");
                state.file_loaded = true;

                // Trigger auto-start if pending
                if state.pending_auto_start && !state.running {
                    info!("Auto-starting Whisper transcription after file load");
                    state.pending_auto_start = false;
                    state.running = true;
                    state.start_transcription(client);
                }
            }
            Event::PlaybackRestart => {
                if state.shutting_down {
                    continue;
                }
                debug!("Playback restart event received");

                // Also trigger auto-start on playback restart (backup mechanism)
                if state.pending_auto_start && !state.running && state.file_loaded {
                    info!("Auto-starting Whisper transcription after playback restart");
                    state.pending_auto_start = false;
                    state.running = true;
                    state.start_transcription(client);
                }

                state.tick(client);
            }
            Event::EndFile(_) => {
                if state.running && !state.shutting_down {
                    state.running = false;
                    state.cleanup(client);
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
}

fn init_logger() {
    // Set WHISPERSUBS_LOG environment variable to control log level (e.g., WHISPERSUBS_LOG=debug)
    #[cfg(target_os = "android")]
    {
        use log::LevelFilter;

        let level = std::env::var("WHISPERSUBS_LOG")
            .ok()
            .and_then(|s| {
                s.parse::<LevelFilter>()
                    .or_else(|_| s.to_lowercase().parse())
                    .ok()
            })
            .unwrap_or(LevelFilter::Info);

        let config = android_logger::Config::default()
            .with_tag("whispersubs_rs")
            .with_max_level(level);
        let _ = android_logger::init_once(config);
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = env_logger::Builder::from_env(
            env_logger::Env::new().filter_or("WHISPERSUBS_LOG", "info"),
        )
        .format_timestamp_millis()
        .try_init();
    }
}
