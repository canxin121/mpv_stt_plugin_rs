#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${ROOT_DIR}/target/mpv-headers"
MPV_REPO="${MPV_REPO:-https://github.com/mpv-player/mpv.git}"

if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "Cloning mpv headers (depth=1) into ${CACHE_DIR}..."
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

echo "Using MPV_INCLUDE_DIR=${MPV_INCLUDE_DIR}"
echo "Running: cargo ${*:-check}"

cd "${ROOT_DIR}"
cargo "${@:-check}"
