use crate::common::{MpvSttError, Result};
use serde::Deserialize;
use tracing::warn;

#[derive(Debug, Deserialize)]
pub(super) struct RawSegment {
    pub(super) start: Option<f64>,
    pub(super) end: Option<f64>,
    #[serde(default)]
    pub(super) text: String,
}

#[derive(Debug)]
pub(super) struct Segment {
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) text: String,
}

/// Validate and clamp timestamped response segments to the current audio chunk.
/// Text without usable timestamps is rejected instead of being assigned a
/// guessed whole-chunk interval.
pub(super) fn normalize_segments(
    response_text: &str,
    response_segments: Vec<RawSegment>,
    response_duration: Option<f64>,
    chunk_ms: u64,
    server: &str,
) -> Result<Vec<Segment>> {
    // SRT timestamps are represented as milliseconds in u32 throughout this
    // plugin; do not let a huge response overflow while converting seconds.
    let chunk_duration_s = chunk_ms.min(u32::MAX as u64) as f64 / 1000.0;
    let duration_s = response_duration
        .filter(|duration| duration.is_finite() && *duration > 0.0)
        .map(|duration| duration.min(chunk_duration_s))
        .unwrap_or(chunk_duration_s);
    let response_has_text = !response_text.trim().is_empty()
        || response_segments
            .iter()
            .any(|segment| !segment.text.trim().is_empty());

    let mut segments = Vec::with_capacity(response_segments.len());
    let mut rejected_segments = 0usize;
    for segment in response_segments {
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }

        let (Some(start), Some(end)) = (segment.start, segment.end) else {
            rejected_segments += 1;
            continue;
        };
        if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
            rejected_segments += 1;
            continue;
        }

        // A model can round its final segment a little past the WAV duration.
        // Keep the part inside this audio chunk, but never move a segment that
        // begins after the chunk back onto its final millisecond.
        if start >= duration_s || end <= 0.0 {
            rejected_segments += 1;
            continue;
        }
        let end = end.min(duration_s);
        if end <= start {
            rejected_segments += 1;
            continue;
        }

        segments.push(Segment {
            start,
            end,
            text: text.to_string(),
        });
    }

    if rejected_segments > 0 {
        warn!(
            server,
            rejected_segments,
            accepted_segments = segments.len(),
            chunk_ms,
            "discarded transcription segments with invalid or out-of-range timestamps"
        );
    }

    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    if !segments.is_empty() {
        return Ok(segments);
    }

    if !response_has_text {
        return Ok(Vec::new());
    }

    Err(MpvSttError::SttFailed(
        "the transcription server returned text without valid segment timestamps; use a model/server that returns timed segments or words".to_string(),
    ))
}
