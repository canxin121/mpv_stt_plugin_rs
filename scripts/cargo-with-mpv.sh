#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${ROOT_DIR}/target/mpv-headers"
MPV_REPO="${MPV_REPO:-https://github.com/mpv-player/mpv.git}"
VERBOSE="${MPV_STT_PLUGIN_RS_VERBOSE:-}"

log() {
  if [[ -n "${VERBOSE}" ]]; then
    echo "$@"
  fi
}

if [[ ! -d "${CACHE_DIR}" ]]; then
  log "Cloning mpv headers (depth=1) into ${CACHE_DIR}..."
  git clone --depth 1 "${MPV_REPO}" "${CACHE_DIR}"
fi

INC_CANDIDATES=(
  "${CACHE_DIR}/include"
  "${CACHE_DIR}/libmpv"
  "${CACHE_DIR}"
)
MPV_INCLUDE_DIR=""
for p in "${INC_CANDIDATES[@]}"; do
  if [[ -f "${p}/mpv/client.h" ]]; then
    MPV_INCLUDE_DIR="${p}"
    break
  fi
done

if [[ -z "${MPV_INCLUDE_DIR}" ]]; then
  echo "mpv/client.h not found after clone; please set MPV_INCLUDE_DIR manually" >&2
  exit 1
fi

export MPV_INCLUDE_DIR
export BINDGEN_EXTRA_CLANG_ARGS="-I${MPV_INCLUDE_DIR}"
# Silence deprecated warnings from upstream mpv-client-sys bindgen usage
export RUSTFLAGS="${RUSTFLAGS:-} -A deprecated"

args_joined=" $* "
if [[ "${args_joined}" == *"stt_remote_udp"* ]] || [[ "${args_joined}" == *"--all-features"* ]]; then
  log "stt_remote_udp uses libopusenc-static-sys (static libopusenc + opus); no system Opus required."
  export CMAKE_INSTALL_LIBDIR="${CMAKE_INSTALL_LIBDIR:-lib}"
  log "Using CMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR} for Opus static install layout."
fi

log "Using MPV_INCLUDE_DIR=${MPV_INCLUDE_DIR}"
log "Running: cargo ${*:-check}"

cd "${ROOT_DIR}"
cargo "${@:-check}"
