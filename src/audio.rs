use crate::common::{MpvSttError, Result};
use ffmpeg::format::Sample;
use ffmpeg::format::sample::Type as SampleType;
use ffmpeg::util::mathematics::Rounding;
use ffmpeg::util::mathematics::rescale;
use ffmpeg::util::mathematics::rescale::Rescale;
use ffmpeg_next as ffmpeg;
use std::path::Path;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use tracing::{debug, trace};

static FFMPEG_INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

fn ensure_ffmpeg() -> Result<()> {
    match FFMPEG_INIT.get_or_init(|| ffmpeg::init().map_err(|e| e.to_string())) {
        Ok(()) => Ok(()),
        Err(err) => Err(MpvSttError::AudioExtractionFailed(format!(
            "ffmpeg init failed: {err}"
        ))),
    }
}

fn ffmpeg_err(context: &str, err: impl std::fmt::Display) -> MpvSttError {
    MpvSttError::AudioExtractionFailed(format!("{context}: {err}"))
}

fn check_timeout(start: Instant, timeout: Duration, label: &str) -> Result<()> {
    if timeout.as_millis() == 0 {
        return Ok(());
    }
    if start.elapsed() > timeout {
        return Err(MpvSttError::ProcessTimeout(format!(
            "{label} timed out after {}ms",
            timeout.as_millis()
        )));
    }
    Ok(())
}

fn output_channel_layout(channels: u8) -> ffmpeg::channel_layout::ChannelLayout {
    match channels {
        1 => ffmpeg::channel_layout::ChannelLayout::MONO,
        2 => ffmpeg::channel_layout::ChannelLayout::STEREO,
        ch => ffmpeg::channel_layout::ChannelLayout::default(ch as i32),
    }
}

/// Decoded frames may carry an UNSPEC channel order (empty mask), e.g. WAV
/// files without an explicit channel mask. FFmpeg 9's `swr_convert_frame`
/// strict-compares the frame layout against the configured source layout and
/// returns `AVERROR_INPUT_CHANGED` on any mismatch. Normalize the frame layout
/// to a concrete one derived from the channel count so it matches the
/// resampler configuration.
fn normalize_frame_layout(frame: &mut ffmpeg::frame::Audio) {
    if frame.channel_layout().is_empty() {
        frame.set_channel_layout(output_channel_layout(frame.channels() as u8));
    }
}

/// Align output samples with `target_time_us`, given the PTS of the first
/// decoded frame. A backward seek can still land slightly after the target;
/// report that as leading silence instead of shifting the remaining speech to
/// time zero.
fn sample_alignment(
    target_time_us: i64,
    frame_pts: i64,
    stream_time_base: ffmpeg::Rational,
    output_sample_rate: u32,
) -> (u64, u64) {
    let output_time_base = (1, output_sample_rate as i32);
    let target_sample =
        target_time_us.rescale_with(rescale::TIME_BASE, output_time_base, Rounding::Up);
    let first_sample = frame_pts.rescale_with(stream_time_base, output_time_base, Rounding::Down);

    if first_sample >= target_sample {
        (0, first_sample.saturating_sub(target_sample) as u64)
    } else {
        (target_sample.saturating_sub(first_sample) as u64, 0)
    }
}

#[derive(Clone)]
pub struct AudioExtractor {
    output_sample_rate: u32,
    output_channels: u8,
    ffmpeg_timeout: Duration,
    ffprobe_timeout: Duration,
    cancel_generation: Arc<AtomicU64>,
}

impl Default for AudioExtractor {
    fn default() -> Self {
        Self {
            output_sample_rate: 16000,
            output_channels: 1,
            ffmpeg_timeout: Duration::from_secs(30),
            ffprobe_timeout: Duration::from_secs(10),
            cancel_generation: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl AudioExtractor {
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            output_sample_rate: sample_rate,
            output_channels: channels,
            ..Default::default()
        }
    }

    pub fn with_ffmpeg_timeout(mut self, timeout_ms: u64) -> Self {
        self.ffmpeg_timeout = Duration::from_millis(timeout_ms);
        self
    }

    pub fn with_ffprobe_timeout(mut self, timeout_ms: u64) -> Self {
        self.ffprobe_timeout = Duration::from_millis(timeout_ms);
        self
    }

    pub fn cancel_inflight(&self) {
        self.cancel_generation.fetch_add(1, Ordering::Relaxed);
    }

    fn check_cancel(&self, generation: u64) -> Result<()> {
        if self.cancel_generation.load(Ordering::Relaxed) != generation {
            return Err(MpvSttError::AudioExtractionCancelled);
        }
        Ok(())
    }

    /// Estimate the relative end of the selected audio stream in the input.
    ///
    /// mpv's `dump-cache` starts at a preceding seek point and rebases packet
    /// timestamps to zero. The requested network chunk therefore does not
    /// necessarily begin at zero in the dumped file. The final audio packet is
    /// the one immediately before the requested end; its midpoint estimates
    /// that end to within half an encoded audio packet. We use this only to
    /// locate the chunk within the dumped file, avoiding the potentially much
    /// larger video-keyframe pre-roll.
    pub fn audio_end_relative_ms<P: AsRef<Path>>(&self, input_path: P) -> Result<u64> {
        ensure_ffmpeg()?;
        let start_time = Instant::now();
        let run_generation = self.cancel_generation.load(Ordering::Relaxed);
        let input_path = input_path.as_ref();
        let cancel_generation = Arc::clone(&self.cancel_generation);
        let ffmpeg_timeout = self.ffmpeg_timeout;

        let mut ictx = ffmpeg::format::input_with_interrupt(input_path, move || {
            (ffmpeg_timeout.as_millis() != 0 && start_time.elapsed() > ffmpeg_timeout)
                || cancel_generation.load(Ordering::Relaxed) != run_generation
        })
        .map_err(|e| ffmpeg_err("open input failed while locating dumped audio", e))?;

        check_timeout(start_time, self.ffmpeg_timeout, "ffmpeg")?;
        self.check_cancel(run_generation)?;

        let format_start_time = unsafe { (*ictx.as_ptr()).start_time };
        let (stream_index, stream_time_base, stream_start_time, stream_duration) = {
            let stream = ictx
                .streams()
                .best(ffmpeg::media::Type::Audio)
                .ok_or_else(|| {
                    MpvSttError::AudioExtractionFailed(
                        "No audio stream found in the dumped cache".to_string(),
                    )
                })?;
            (
                stream.index(),
                stream.time_base(),
                stream.start_time(),
                stream.duration(),
            )
        };

        let mut last_packet_midpoint = None;
        for (stream, packet) in ictx.packets() {
            if stream.index() != stream_index {
                continue;
            }
            check_timeout(start_time, self.ffmpeg_timeout, "ffmpeg")?;
            self.check_cancel(run_generation)?;

            let Some(packet_pts) = packet.pts().or_else(|| packet.dts()) else {
                continue;
            };
            let packet_midpoint = packet_pts.saturating_add(packet.duration().max(0) / 2);
            last_packet_midpoint = Some(
                last_packet_midpoint.map_or(packet_midpoint, |last: i64| last.max(packet_midpoint)),
            );
        }

        let end_pts = last_packet_midpoint.or_else(|| {
            (stream_start_time != ffmpeg::ffi::AV_NOPTS_VALUE && stream_duration > 0)
                .then(|| stream_start_time.saturating_add(stream_duration))
        });
        let Some(end_pts) = end_pts else {
            return Err(MpvSttError::AudioExtractionFailed(
                "cannot determine the audio timeline in the dumped cache".to_string(),
            ));
        };

        let earliest_stream_start_us = ictx
            .streams()
            .filter_map(|stream| {
                let stream_start = stream.start_time();
                let time_base = stream.time_base();
                (stream_start != ffmpeg::ffi::AV_NOPTS_VALUE && time_base.denominator() != 0)
                    .then(|| stream_start.rescale(time_base, rescale::TIME_BASE))
            })
            .min();
        let format_start_time_us = if format_start_time == ffmpeg::ffi::AV_NOPTS_VALUE {
            earliest_stream_start_us.unwrap_or(0)
        } else {
            format_start_time
        };
        let end_time_us = end_pts.rescale(stream_time_base, rescale::TIME_BASE);
        let relative_end_us = end_time_us.saturating_sub(format_start_time_us);
        if relative_end_us <= 0 {
            return Err(MpvSttError::AudioExtractionFailed(
                "the dumped audio has no positive timeline duration".to_string(),
            ));
        }

        Ok((relative_end_us / 1000) as u64)
    }

    /// Extract audio segment from media file using ffmpeg libraries
    pub fn extract_audio_segment<P: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: P,
        start_ms: u64,
        duration_ms: u64,
    ) -> Result<()> {
        ensure_ffmpeg()?;
        let start_time = Instant::now();
        let run_generation = self.cancel_generation.load(Ordering::Relaxed);

        let input_str = input_path
            .as_ref()
            .to_str()
            .ok_or_else(|| MpvSttError::InvalidPath("Invalid input path".to_string()))?;
        let output_str = output_path
            .as_ref()
            .to_str()
            .ok_or_else(|| MpvSttError::InvalidPath("Invalid output path".to_string()))?;

        trace!(
            start_ms,
            end_ms = start_ms.saturating_add(duration_ms),
            input = input_str,
            output = output_str,
            "extracting audio with ffmpeg"
        );

        let cancel_generation = Arc::clone(&self.cancel_generation);
        let ffmpeg_timeout = self.ffmpeg_timeout;
        let mut ictx = ffmpeg::format::input_with_interrupt(&input_path, move || {
            if ffmpeg_timeout.as_millis() != 0 && start_time.elapsed() > ffmpeg_timeout {
                return true;
            }
            cancel_generation.load(Ordering::Relaxed) != run_generation
        })
        .map_err(|e| ffmpeg_err("open input failed", e))?;

        check_timeout(start_time, self.ffmpeg_timeout, "ffmpeg")?;
        self.check_cancel(run_generation)?;

        // `time-pos` is relative to the media start, while FFmpeg frame PTS
        // values are on the demuxer's absolute timeline. Keep both in the
        // same clock so non-zero container start times are handled correctly.
        // AVFormatContext.start_time is in AV_TIME_BASE units.
        let format_start_time = unsafe { (*ictx.as_ptr()).start_time };
        let earliest_stream_start_us = ictx
            .streams()
            .filter_map(|stream| {
                let start_time = stream.start_time();
                let time_base = stream.time_base();
                (start_time != ffmpeg::ffi::AV_NOPTS_VALUE && time_base.denominator() != 0)
                    .then(|| start_time.rescale(time_base, rescale::TIME_BASE))
            })
            .min();
        let format_start_time_us = if format_start_time == ffmpeg::ffi::AV_NOPTS_VALUE {
            earliest_stream_start_us.unwrap_or(0)
        } else {
            format_start_time
        };
        let target_time_us = format_start_time_us
            .saturating_add((start_ms as i64).rescale((1, 1000), rescale::TIME_BASE));

        let (stream_index, stream_time_base, stream_start_time) = {
            let input_stream =
                ictx.streams()
                    .best(ffmpeg::media::Type::Audio)
                    .ok_or_else(|| {
                        MpvSttError::AudioExtractionFailed("No audio stream found".to_string())
                    })?;
            (
                input_stream.index(),
                input_stream.time_base(),
                input_stream.start_time(),
            )
        };

        let mut seeked = false;
        if start_ms > 0 {
            if let Err(err) = ictx.seek(target_time_us, ..target_time_us) {
                trace!(error = %err, "ffmpeg seek failed; decoding from the start and skipping");
            } else {
                seeked = true;
            }
        }

        let input_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| {
                MpvSttError::AudioExtractionFailed("No audio stream found".to_string())
            })?;
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
                .map_err(|e| ffmpeg_err("decoder context failed", e))?;
        let mut decoder = context_decoder
            .decoder()
            .audio()
            .map_err(|e| ffmpeg_err("audio decoder failed", e))?;

        // Containers with a coarse index (MPEG-PS, MPEG-TS) seek to a keyframe
        // before the requested position, so the demuxer hands the decoder
        // packets that belong to the old timeline. Without a flush the codec
        // keeps its pre-seek state and rejects the first packets after the
        // seek with "Invalid data found when processing input", which aborts
        // the whole extraction.
        if seeked {
            decoder.flush();
        }

        let output_layout = output_channel_layout(self.output_channels);
        let output_format = Sample::I16(SampleType::Packed);

        // The decoder is not opened yet, so its `channel_layout()` may carry an
        // empty mask (channels > 0 but layout unknown). FFmpeg 9's swr compares
        // this against the decoded frame's layout and fails with
        // AVERROR_INPUT_CHANGED. Derive a concrete source layout from the
        // channel count instead, so it matches the frames once decoded.
        let src_layout = if decoder.channel_layout().is_empty() {
            output_channel_layout(decoder.channels() as u8)
        } else {
            decoder.channel_layout()
        };
        let mut resampler = ffmpeg::software::resampling::Context::get(
            decoder.format(),
            src_layout,
            decoder.rate() as u32,
            output_format,
            output_layout,
            self.output_sample_rate,
        )
        .map_err(|e| ffmpeg_err("resampler init failed", e))?;

        let spec = hound::WavSpec {
            channels: self.output_channels as u16,
            sample_rate: self.output_sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(output_path, spec)?;

        let fallback_first_frame_us = if stream_start_time == ffmpeg::ffi::AV_NOPTS_VALUE {
            format_start_time_us
        } else {
            stream_start_time.rescale(stream_time_base, rescale::TIME_BASE)
        };
        let fallback_alignment = sample_alignment(
            target_time_us,
            fallback_first_frame_us,
            rescale::TIME_BASE,
            self.output_sample_rate,
        );
        let target_frames = duration_ms
            .saturating_mul(self.output_sample_rate as u64)
            .saturating_div(1000);

        let mut written_frames = 0u64;
        let mut first_frame_seen = false;
        let mut first_frame_pts = None;
        let mut frames_to_skip = None;
        let mut leading_silence_frames = 0u64;

        let mut decoded = ffmpeg::frame::Audio::empty();

        for (stream, packet) in ictx.packets() {
            if stream.index() != stream_index {
                continue;
            }

            check_timeout(start_time, self.ffmpeg_timeout, "ffmpeg")?;
            self.check_cancel(run_generation)?;
            if let Err(err) = decoder.send_packet(&packet) {
                // Containers with a coarse index (MPEG-PS above all) seek to a
                // point that is not a packet boundary, so the demuxer hands the
                // decoder a truncated first packet. One malformed packet is not
                // worth aborting a chunk the user is waiting on: drop it and
                // keep reading, which is what ffmpeg's own CLI does. Every other
                // error still fails the extraction.
                if err == ffmpeg::Error::InvalidData {
                    trace!(
                        pos = packet.pts(),
                        "skipping a malformed packet after the seek"
                    );
                    continue;
                }
                return Err(ffmpeg_err("send packet failed", err));
            }

            while decoder.receive_frame(&mut decoded).is_ok() {
                check_timeout(start_time, self.ffmpeg_timeout, "ffmpeg")?;
                self.check_cancel(run_generation)?;

                if !first_frame_seen {
                    first_frame_seen = true;
                    first_frame_pts = decoded.timestamp().or_else(|| decoded.pts());
                    if first_frame_pts.is_none() && seeked {
                        return Err(MpvSttError::AudioExtractionFailed(
                            "cannot align audio after seek: the first decoded frame has no timestamp"
                                .to_string(),
                        ));
                    }
                }

                normalize_frame_layout(&mut decoded);

                let mut resampled = ffmpeg::frame::Audio::empty();
                let _ = resampler
                    .run(&decoded, &mut resampled)
                    .map_err(|e| ffmpeg_err("resample failed", e))?;

                let frames = resampled.samples();
                if frames == 0 {
                    continue;
                }
                let channels = self.output_channels as usize;
                let data = resampled.data(0);
                let sample_count = data.len() / std::mem::size_of::<i16>();
                if sample_count < frames * channels {
                    return Err(MpvSttError::AudioExtractionFailed(
                        "resampled frame shorter than expected".to_string(),
                    ));
                }

                if frames_to_skip.is_none() {
                    let (skip, pad) = first_frame_pts.map_or(fallback_alignment, |pts| {
                        sample_alignment(
                            target_time_us,
                            pts,
                            stream_time_base,
                            self.output_sample_rate,
                        )
                    });
                    frames_to_skip = Some(skip);
                    leading_silence_frames = pad;
                    trace!(
                        target_time_us,
                        first_frame_pts = ?first_frame_pts,
                        skip_frames = skip,
                        leading_silence_frames,
                        "aligning extracted audio to the requested start"
                    );
                }

                let samples = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i16, sample_count)
                };

                while leading_silence_frames > 0
                    && (target_frames == 0 || written_frames < target_frames)
                {
                    for _ in 0..channels {
                        writer.write_sample(0)?;
                    }
                    written_frames += 1;
                    leading_silence_frames -= 1;
                }

                for frame_idx in 0..frames {
                    if frames_to_skip.is_some_and(|remaining| remaining > 0) {
                        frames_to_skip = frames_to_skip.map(|remaining| remaining - 1);
                        continue;
                    }
                    if target_frames > 0 && written_frames >= target_frames {
                        break;
                    }

                    let base = frame_idx * channels;
                    for ch in 0..channels {
                        writer.write_sample(samples[base + ch])?;
                    }
                    written_frames += 1;
                }

                if target_frames > 0 && written_frames >= target_frames {
                    break;
                }
            }

            if target_frames > 0 && written_frames >= target_frames {
                break;
            }
        }

        if target_frames == 0 || written_frames < target_frames {
            decoder
                .send_eof()
                .map_err(|e| ffmpeg_err("send eof failed", e))?;
            while decoder.receive_frame(&mut decoded).is_ok() {
                check_timeout(start_time, self.ffmpeg_timeout, "ffmpeg")?;
                self.check_cancel(run_generation)?;

                if !first_frame_seen {
                    first_frame_seen = true;
                    first_frame_pts = decoded.timestamp().or_else(|| decoded.pts());
                    if first_frame_pts.is_none() && seeked {
                        return Err(MpvSttError::AudioExtractionFailed(
                            "cannot align audio after seek: the first decoded frame has no timestamp"
                                .to_string(),
                        ));
                    }
                }

                normalize_frame_layout(&mut decoded);

                let mut resampled = ffmpeg::frame::Audio::empty();
                let _ = resampler
                    .run(&decoded, &mut resampled)
                    .map_err(|e| ffmpeg_err("resample failed", e))?;

                let frames = resampled.samples();
                if frames == 0 {
                    continue;
                }
                let channels = self.output_channels as usize;
                let data = resampled.data(0);
                let sample_count = data.len() / std::mem::size_of::<i16>();
                if sample_count < frames * channels {
                    return Err(MpvSttError::AudioExtractionFailed(
                        "resampled frame shorter than expected".to_string(),
                    ));
                }

                if frames_to_skip.is_none() {
                    let (skip, pad) = first_frame_pts.map_or(fallback_alignment, |pts| {
                        sample_alignment(
                            target_time_us,
                            pts,
                            stream_time_base,
                            self.output_sample_rate,
                        )
                    });
                    frames_to_skip = Some(skip);
                    leading_silence_frames = pad;
                    trace!(
                        target_time_us,
                        first_frame_pts = ?first_frame_pts,
                        skip_frames = skip,
                        leading_silence_frames,
                        "aligning extracted audio to the requested start"
                    );
                }

                let samples = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i16, sample_count)
                };

                while leading_silence_frames > 0
                    && (target_frames == 0 || written_frames < target_frames)
                {
                    for _ in 0..channels {
                        writer.write_sample(0)?;
                    }
                    written_frames += 1;
                    leading_silence_frames -= 1;
                }

                for frame_idx in 0..frames {
                    if frames_to_skip.is_some_and(|remaining| remaining > 0) {
                        frames_to_skip = frames_to_skip.map(|remaining| remaining - 1);
                        continue;
                    }
                    if target_frames > 0 && written_frames >= target_frames {
                        break;
                    }

                    let base = frame_idx * channels;
                    for ch in 0..channels {
                        writer.write_sample(samples[base + ch])?;
                    }
                    written_frames += 1;
                }

                if target_frames > 0 && written_frames >= target_frames {
                    break;
                }
            }
        }

        self.check_cancel(run_generation)?;
        writer.finalize()?;
        if target_frames > 0 && written_frames == 0 {
            return Err(MpvSttError::AudioExtractionFailed(
                "no audio samples decoded".to_string(),
            ));
        }

        debug!(frames = written_frames, "audio extraction finished");
        Ok(())
    }

    /// Check if audio file exists and is valid
    pub fn validate_audio<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        ensure_ffmpeg()?;
        if !path.as_ref().exists() {
            return Ok(false);
        }

        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| MpvSttError::InvalidPath("Invalid path".to_string()))?;

        let start_time = Instant::now();
        let ictx = ffmpeg::format::input(&path).map_err(|e| ffmpeg_err("open input failed", e))?;
        check_timeout(start_time, self.ffprobe_timeout, "ffprobe")?;

        let has_audio = ictx.streams().best(ffmpeg::media::Type::Audio).is_some();
        trace!(
            path = path_str,
            has_audio, "checked whether the file carries audio"
        );
        Ok(has_audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Every container `scripts/gen-test-media.sh` derives from the committed
    /// clip. The list lives here as well as in the script because the test is
    /// what has to notice a container regressing; the script only has to build
    /// them.
    const CONTAINERS: [&str; 11] = [
        "mp4", "mkv", "mov", "avi", "webm", "ts", "flv", "wmv", "mpg", "mp3", "m4a",
    ];

    /// The clip's speech peaks around -9 dBFS, an all-silent track is digital
    /// silence. Anything above this is unambiguously the real audio rather than
    /// a demuxer that handed back an empty track.
    const SILENCE_PEAK: i16 = 500;

    #[test]
    fn test_audio_extractor_new() {
        let extractor = AudioExtractor::new(48000, 2);
        assert_eq!(extractor.output_sample_rate, 48000);
        assert_eq!(extractor.output_channels, 2);
    }

    #[test]
    fn test_audio_extractor_default() {
        let extractor = AudioExtractor::default();
        assert_eq!(extractor.output_sample_rate, 16000);
        assert_eq!(extractor.output_channels, 1);
    }

    /// Peak absolute sample of a WAV written by the extractor.
    fn peak_of(path: &Path) -> i16 {
        let mut reader = hound::WavReader::open(path).expect("open extracted WAV");
        reader
            .samples::<i16>()
            .map(|s| s.expect("read sample").unsigned_abs() as i16)
            .max()
            .unwrap_or(0)
    }

    /// Length of an extracted WAV in milliseconds.
    fn duration_ms_of(path: &Path) -> u64 {
        let reader = hound::WavReader::open(path).expect("open extracted WAV");
        let spec = reader.spec();
        u64::from(reader.duration()) * 1000 / u64::from(spec.sample_rate)
    }

    /// The same Japanese speech, muxed into every container the plugin is
    /// likely to be handed, must extract to the same thing: five seconds of
    /// 16 kHz mono audio that is not silence. A container that demuxes to an
    /// empty track produces no subtitles at all, and an `Ok` return hides it.
    ///
    /// The clip is `testdata/ja_all.mp4`; the matrix comes from
    /// `scripts/gen-test-media.sh` (build artifact, so it is not in the repo):
    ///
    ///   ./scripts/gen-test-media.sh
    ///   ./scripts/cargo-with-deps.sh test --lib -- --ignored audio_extraction_covers_every_container
    ///
    /// Without the generated matrix the test reports and passes, so running
    /// every `#[ignore]`d test on a fresh checkout does not fail.
    #[test]
    #[ignore = "needs the container matrix from scripts/gen-test-media.sh"]
    fn audio_extraction_covers_every_container() {
        let media_dir = std::env::var_os("MPV_STT_PLUGIN_RS_TEST_MEDIA")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/testmedia"));

        if !media_dir.is_dir() {
            eprintln!(
                "skipping: {} is missing; run scripts/gen-test-media.sh first",
                media_dir.display()
            );
            return;
        }

        // 55 s in lands inside the longest clip, clear of the 0.8 s silence
        // gaps that separate the source clips. Seeking there exercises the
        // `start_ms > 0` branch, including the fall-back to decoding from the
        // start when a container cannot seek.
        let mid_start_ms = 55_000;
        let window_ms = 5_000;

        let dir = tempfile::tempdir().expect("create temp dir");
        let extractor = AudioExtractor::default();

        for container in CONTAINERS {
            let media = media_dir.join(format!("ja_all.{container}"));
            assert!(media.is_file(), "{container}: missing {}", media.display());

            assert!(
                extractor.validate_audio(&media).unwrap_or(false),
                "{container}: validate_audio did not find an audio stream"
            );

            for (label, start_ms) in [("head", 0), ("mid", mid_start_ms)] {
                let wav = dir.path().join(format!("{container}-{label}.wav"));
                extractor
                    .extract_audio_segment(&media, &wav, start_ms, window_ms)
                    .unwrap_or_else(|e| {
                        panic!("{container}/{label}: extract at {start_ms} ms failed: {e}")
                    });

                let spec = hound::WavReader::open(&wav)
                    .expect("open extracted WAV")
                    .spec();
                assert_eq!(spec.sample_rate, 16_000, "{container}/{label}: sample rate");
                assert_eq!(spec.channels, 1, "{container}/{label}: channels");

                let peak = peak_of(&wav);
                assert!(
                    peak > SILENCE_PEAK,
                    "{container}/{label}: extracted {window_ms} ms starting at {start_ms} ms \
                     came back silent (peak {peak}); the demuxer found no audio"
                );

                let duration = duration_ms_of(&wav);
                assert!(
                    duration.abs_diff(window_ms) < 400,
                    "{container}/{label}: expected ~{window_ms} ms, got {duration} ms"
                );
            }
        }
    }
}
