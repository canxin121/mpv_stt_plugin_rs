#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDIA_DIR="${MPV_STT_PLUGIN_RS_TEST_MEDIA:-${ROOT_DIR}/target/testmedia}"
PLUGIN_SRC="${ROOT_DIR}/target/release/libmpv_stt_plugin_rs.dylib"
VERBOSE="${MPV_STT_PLUGIN_RS_VERBOSE:-}"
LENGTH="${MPV_STT_PLUGIN_RS_E2E_LENGTH:-25}"

CONTAINERS=(mp4 mkv mov avi webm ts flv wmv mpg mp3 m4a)

log() {
  if [[ -n "${VERBOSE}" ]]; then
    echo "$@"
  fi
}

usage() {
  cat <<'EOF'
Usage: MPV_STT_PLUGIN_RS_CONFIG=… scripts/e2e-media-matrix.sh

Plays every container of the matrix through mpv with the plugin attached, and
checks that each run produced subtitles with no failed chunks. Sound is never
opened: every run passes --ao=null (and --vo=null).

Requires a running OpenAI-compatible STT server (subtitle-gateway on :8000, or
Groq) and a config that points at it. The script passes the config path
through untouched - it never reads the file, which holds an API key.

Options:
  -h, --help

Env:
  MPV_STT_PLUGIN_RS_CONFIG       Config file for the run (required).
  MPV_STT_PLUGIN_RS_TEST_MEDIA   Container matrix directory (default: target/testmedia)
  MPV_STT_PLUGIN_RS_TEST_PLUGIN  Plugin binary (default: target/release/libmpv_stt_plugin_rs.dylib)
  MPV_STT_PLUGIN_RS_E2E_LENGTH   Seconds to play per container (default: 25)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${MPV_STT_PLUGIN_RS_CONFIG:-}" ]]; then
  echo "set MPV_STT_PLUGIN_RS_CONFIG to a config pointing at a live STT server" >&2
  exit 2
fi

if [[ ! -f "${MPV_STT_PLUGIN_RS_CONFIG}" ]]; then
  echo "config not found: ${MPV_STT_PLUGIN_RS_CONFIG}" >&2
  exit 2
fi

if [[ -n "${MPV_STT_PLUGIN_RS_TEST_PLUGIN:-}" ]]; then
  PLUGIN_SRC="${MPV_STT_PLUGIN_RS_TEST_PLUGIN}"
fi

if [[ ! -f "${PLUGIN_SRC}" ]]; then
  echo "${PLUGIN_SRC} is missing; run ./scripts/cargo-with-deps.sh build --release" >&2
  exit 2
fi

if ! command -v mpv >/dev/null 2>&1; then
  echo "mpv is not on PATH" >&2
  exit 2
fi

if [[ ! -d "${MEDIA_DIR}" ]]; then
  log "generating the container matrix"
  bash "${ROOT_DIR}/scripts/gen-test-media.sh"
fi

# mpv picks a C plugin's backend by file extension and names the client after
# the file stem, so the binary is staged under the name the shortcuts and the
# startup message use.
PLUGIN_DIR="${MEDIA_DIR}/plugin"
PLUGIN_SO="${PLUGIN_DIR}/mpv_stt_plugin_rs.so"
mkdir -p "${PLUGIN_DIR}"
cp "${PLUGIN_SRC}" "${PLUGIN_SO}"

# Every container shares the source clip's stem. The plugin must preserve the
# full media filename in its namespaced sidecar so these outputs never collide.
failures=0
for container in "${CONTAINERS[@]}"; do
  media="${MEDIA_DIR}/ja_all.${container}"
  subtitle="${media}.mpv_stt_plugin_rs.srt"
  manifest="${media}.mpv_stt_plugin_rs.json"
  log_file="${MEDIA_DIR}/ja_all.${container}.log"

  if [[ ! -f "${media}" ]]; then
    echo "${container}: FAIL missing ${media}" >&2
    failures=$((failures + 1))
    continue
  fi

  rm -f "${subtitle}" "${manifest}" "${log_file}"

  # --ao=null keeps the run silent and --vo=null skips video output. Nothing
  # starts STT on its own (the shipped default is auto_start = false), so the
  # run drives the same shortcut a user would press; the message has to be
  # addressed because a plugin's own client name is not the addressed default.
  MPV_STT_PLUGIN_RS_CONFIG="${MPV_STT_PLUGIN_RS_CONFIG}" \
  MPV_STT_PLUGIN_RS_LOG_FILE=off \
    mpv --no-config --vo=null --ao=null \
        --script="${PLUGIN_SO}" \
        --input-commands="script-message-to mpv_stt_plugin_rs toggle-stt" \
        --length="${LENGTH}" \
        "${media}" > "${log_file}" 2>&1 || true

  verdict=""
  if [[ ! -s "${subtitle}" ]]; then
    verdict="no subtitles written to ${subtitle}"
  elif ! grep -q "failures=0" "${log_file}"; then
    failed_chunks="$(grep -oE "failures=[0-9]+" "${log_file}" | grep -v "failures=0" | head -1)"
    verdict="chunks failed (${failed_chunks:-no session summary})"
  fi

  if [[ -z "${verdict}" ]]; then
    cues="$(grep -cE '^[0-9]+$' "${subtitle}" || true)"
    echo "${container}: ok (${cues} cues)"
    log "  log: ${log_file}"
  else
    echo "${container}: FAIL ${verdict}" >&2
    log "  log: ${log_file}"
    failures=$((failures + 1))
  fi
done

# The namespaced subtitle caches and staged plugin are this script's own
# outputs, produced beside the build artifacts; drop them after the matrix.
for container in "${CONTAINERS[@]}"; do
  media="${MEDIA_DIR}/ja_all.${container}"
  rm -f "${media}.mpv_stt_plugin_rs.srt" "${media}.mpv_stt_plugin_rs.json"
done
rm -rf "${PLUGIN_DIR}"

if [[ "${failures}" != "0" ]]; then
  echo "${failures} container(s) failed" >&2
  exit 1
fi

echo "every container produced subtitles"