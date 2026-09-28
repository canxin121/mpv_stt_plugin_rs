#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/testdata/ja_all.mp4"
OUT_DIR="${MPV_STT_PLUGIN_RS_TEST_MEDIA:-${ROOT_DIR}/target/testmedia}"
FORCE=""
VERBOSE="${MPV_STT_PLUGIN_RS_VERBOSE:-}"

log() {
  if [[ -n "${VERBOSE}" ]]; then
    echo "$@"
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/gen-test-media.sh [--force]

Transcodes testdata/ja_all.mp4 into the container matrix the audio-extraction
tests and scripts/e2e-media-matrix.sh run against. The derived files are build
artifacts: they live under target/ and are never committed.

Options:
  --force   Rebuild files that already exist.
  -h, --help

Env:
  MPV_STT_PLUGIN_RS_TEST_MEDIA   Output directory (default: target/testmedia)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE="--force"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${SRC}" ]]; then
  echo "missing ${SRC}" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not on PATH" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# Video containers. The codec pairs are the ones this ffmpeg can actually mux
# (checked one by one): h264/aac for the ISO-BMFF family, mp3 in AVI (a format
# users still hand the plugin), VP9/opus for WebM, MPEG-2 picture in the
# transport and program streams, and the Flash/ASF pair for the two legacy
# containers. Audio is forced to what the source already is - 16 kHz mono -
# which is also what the extractor resamples to, so the containers differ only
# in the demuxer and not in the audio itself.
#
# The two MPEG pairings are the ones MPEG-PS accepts (mp1/mp2/mp3/ac3/dts) and
# the transport stream handles; AAC in MPEG-PS is rejected by the muxer.
video_formats=(
  "mp4:-c:v libx264 -c:a aac"
  "mkv:-c:v libx264 -c:a aac"
  "mov:-c:v libx264 -c:a aac"
  "avi:-c:v mpeg4 -c:a libmp3lame"
  "webm:-c:v libvpx-vp9 -c:a libopus"
  "ts:-c:v mpeg2video -c:a aac"
  "flv:-c:v flv -c:a aac"
  "wmv:-c:v wmv2 -c:a wmav2 -f asf"
  "mpg:-c:v mpeg2video -c:a libmp3lame -f mpeg"
)

# Audio-only containers: the plugin is handed plain audio just as often, and
# mp3 in particular has no timestamps to seek with.
audio_formats=(
  "mp3:-c:a libmp3lame"
  "m4a:-c:a aac"
)

encode() {
  local ext="$1"; shift
  local out="${OUT_DIR}/ja_all.${ext}"

  if [[ -z "${FORCE}" && -f "${out}" ]]; then
    log "keeping ${out}"
    return 0
  fi

  log "building ${out}"
  ffmpeg -hide_banner -loglevel error -y -i "${SRC}" "$@" -ac 1 -ar 16000 "${out}"
}

for spec in "${video_formats[@]}"; do
  encode "${spec%%:*}" ${spec#*:}
done

for spec in "${audio_formats[@]}"; do
  encode "${spec%%:*}" ${spec#*:} -vn
done

# Everything below is a check on the artifacts, not on the encoder: a container
# that lost its audio stream would make the extraction tests pass for the wrong
# reason, so each file must carry one track and it must not be silence.
failures=0
for spec in "${video_formats[@]}" "${audio_formats[@]}"; do
  ext="${spec%%:*}"
  out="${OUT_DIR}/ja_all.${ext}"
  file_failed=0

  # `grep -c` rather than `grep -q`: under `pipefail` a `-q` that exits early
  # closes the pipe on ffprobe, whose SIGPIPE then reads as a failed check. The
  # count is not compared exactly because ffprobe lists a transport stream's
  # program and its streams both.
  if [[ ! -s "${out}" ]]; then
    echo "${ext}: missing ${out}" >&2
    file_failed=1
  elif [[ "$(ffprobe -hide_banner -v error -show_entries stream=codec_type -of csv=p=0 "${out}" | grep -c '^audio$')" -lt 1 ]]; then
    echo "${ext}: no audio stream" >&2
    file_failed=1
  else
    # volumedetect reports the peak of an all-silent track as -inf, and a real
    # one somewhere above -30 dBFS; the threshold sits well below speech.
    peak="$(ffmpeg -hide_banner -v info -nostats -i "${out}" -af volumedetect -f null - 2>&1 | awk '/max_volume/ { print $(NF - 1) }')"
    if [[ -z "${peak}" || "${peak}" == "-inf" || "${peak%%.*}" -lt -30 ]]; then
      echo "${ext}: audio track is silent (max_volume: ${peak:-none} dB)" >&2
      file_failed=1
    fi
  fi

  if [[ "${file_failed}" == "0" ]]; then
    echo "${ext}: ok"
  else
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" != "0" ]]; then
  echo "${failures} container(s) failed to build" >&2
  exit 1
fi

echo "container matrix ready in ${OUT_DIR}"