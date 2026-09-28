#!/usr/bin/env bash
# Build the plugin for every desktop platform/arch in one go:
#
#   linux-x86_64   x86_64-unknown-linux-gnu   (BtbN FFmpeg)
#   darwin-arm64   aarch64-apple-darwin       (brew FFmpeg)
#   darwin-x86_64  x86_64-apple-darwin        (brew FFmpeg)
#   windows-x86_64 x86_64-pc-windows-msvc     (BtbN FFmpeg)
#
# Android is a source build of libmpv + FFmpeg and lives in build-android.sh.
#
# FFmpeg is linked dynamically, so nothing here compiles FFmpeg: the script
# resolves a dev prefix per platform and exports FFMPEG_DIR for ffmpeg-sys-next.
#
# On the host's own platform the freshly built plugin is also played once
# through scripts/e2e-media-matrix.sh, when MPV_STT_PLUGIN_RS_CONFIG is set (see
# that script; it needs a reachable STT service and never opens the speakers).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
BUILD_LOG="${DIST_DIR}/build.log"
MPV_REPO="${MPV_REPO:-https://github.com/mpv-player/mpv.git}"
MPV_HEADERS_DIR="${MPV_HEADERS_DIR:-${ROOT_DIR}/target/mpv-headers}"
FFMPEG_CACHE_DIR="${ROOT_DIR}/target/ffmpeg"
# BtbN publishes a stable alias per FFmpeg release line; the aliases move with
# the line, never across it. See ensure_ffmpeg_btbn.
FFMPEG_BTBN_LINE="${FFMPEG_BTBN_LINE:-n9.0}"
FEATURES="${MPV_STT_PLUGIN_RS_FEATURES:-}"

PLATFORMS=(linux-x86_64 darwin-arm64 darwin-x86_64 windows-x86_64)
CLEAN_DIST=0
SHOW_LIST=0
SELECTED=()

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "${BUILD_LOG}"
}

warn() { log "[warn] $*"; }
error() { log "[error] $*"; }

usage() {
    cat <<'EOF'
Usage: ./scripts/build-all.sh [options]

Options:
  -p, --platform <list>  Comma-separated platforms to build (default: all).
                         linux-x86_64, darwin-arm64, darwin-x86_64, windows-x86_64
  -f, --features <list>  Build a single STT backend (--no-default-features):
                         stt_ferrum or stt_openai. Default: both in one artifact.
      --clean            Remove dist/ first
  -l, --list             List the supported platforms and exit
  -h, --help             Show this help

Environment:
  FFPREFIX             macOS only: FFmpeg prefix (default: brew --prefix ffmpeg)
  FFMPEG_DIR           Use this dev prefix as-is on every platform
  FFMPEG_BTBN_URL      Linux/Windows: override the resolved BtbN asset URL
  MPV_STT_PLUGIN_RS_CONFIG
                       Config file for the host e2e run; unset skips it
  MPV_STT_PLUGIN_RS_SKIP_E2E=1
                       Never run the e2e pass

Artifacts land in dist/<platform>/ (runtime/ carries the FFmpeg shared libraries
for Linux and Windows; macOS links the brew ones in place).
EOF
}

target_of() {
    case "$1" in
        linux-x86_64)   echo "x86_64-unknown-linux-gnu" ;;
        darwin-arm64)   echo "aarch64-apple-darwin" ;;
        darwin-x86_64)  echo "x86_64-apple-darwin" ;;
        windows-x86_64) echo "x86_64-pc-windows-msvc" ;;
        *) return 1 ;;
    esac
}

# Cargo's own cdylib filename for a target triple.
cargo_artifact_of() {
    case "$1" in
        *windows*) echo "mpv_stt_plugin_rs.dll" ;;
        *apple*)   echo "libmpv_stt_plugin_rs.dylib" ;;
        *)         echo "libmpv_stt_plugin_rs.so" ;;
    esac
}

# Installable plugin filename: mpv picks its C-plugin backend by extension and
# takes only .so on every non-Windows platform, macOS included (Cargo still
# emits a Mach-O file named .dylib).
artifact_of() {
    case "$1" in
        *windows*) echo "mpv_stt_plugin_rs.dll" ;;
        *)         echo "libmpv_stt_plugin_rs.so" ;;
    esac
}

host_platform() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  echo "linux-x86_64" ;;
        Darwin-arm64)  echo "darwin-arm64" ;;
        Darwin-x86_64) echo "darwin-x86_64" ;;
        *)             echo "" ;;
    esac
}

# --- mpv headers -----------------------------------------------------------
# ffmpeg-sys-next and mpv-client-sys are bindgen-only: they need mpv/client.h,
# not a libmpv to link (the host process provides the symbols).
ensure_mpv_headers() {
    if [[ ! -d "${MPV_HEADERS_DIR}" ]]; then
        log "[setup] cloning mpv headers (depth=1) into ${MPV_HEADERS_DIR}"
        git clone --depth 1 "${MPV_REPO}" "${MPV_HEADERS_DIR}" >> "${BUILD_LOG}" 2>&1
    fi
    local p
    for p in "${MPV_HEADERS_DIR}/include" "${MPV_HEADERS_DIR}/libmpv" "${MPV_HEADERS_DIR}"; do
        if [[ -f "${p}/mpv/client.h" ]]; then
            MPV_INCLUDE_DIR="${p}"
            break
        fi
    done
    if [[ -z "${MPV_INCLUDE_DIR:-}" ]]; then
        error "mpv/client.h not found under ${MPV_HEADERS_DIR}"
        return 1
    fi
    export MPV_INCLUDE_DIR
    # mpv-client-sys' bindgen wrapper includes <mpv/client.h> by name, and
    # upstream mpv's own headers carry deprecated-marked declarations.
    export BINDGEN_EXTRA_CLANG_ARGS="-I${MPV_INCLUDE_DIR}${BINDGEN_EXTRA_CLANG_ARGS:+ ${BINDGEN_EXTRA_CLANG_ARGS}}"
    export RUSTFLAGS="${RUSTFLAGS:-} -A deprecated"
}

ensure_rust_target() {
    local target="$1"
    if ! rustup target list --installed | grep -qx "${target}"; then
        rustup target add "${target}" >> "${BUILD_LOG}" 2>&1
    fi
}

# --- FFmpeg dev prefix ------------------------------------------------------
ensure_ffmpeg_darwin() {
    local prefix="${FFPREFIX:-${FFMPEG_DIR:-}}"
    if [[ -z "${prefix}" ]] && command -v brew >/dev/null 2>&1; then
        prefix="$(brew --prefix ffmpeg 2>/dev/null || true)"
    fi
    if [[ -z "${prefix}" || ! -f "${prefix}/include/libavcodec/avcodec.h" ]]; then
        error "no FFmpeg dev prefix found (brew install ffmpeg, or set FFPREFIX)"
        return 1
    fi
    export FFMPEG_DIR="${prefix}"
}

# BtbN/FFmpeg-Builds shared package, cached under target/ffmpeg/<platform>.
# The asset name is a stable release alias:
#   https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/<asset>
ensure_ffmpeg_btbn() {
    local platform="$1" asset cache archive top
    case "${platform}" in
        linux-x86_64)   asset="ffmpeg-${FFMPEG_BTBN_LINE}-latest-linux64-lgpl-shared-9.0.tar.xz" ;;
        windows-x86_64) asset="ffmpeg-${FFMPEG_BTBN_LINE}-latest-win64-lgpl-shared-9.0.zip" ;;
        *) error "no prebuilt FFmpeg package for ${platform}"; return 1 ;;
    esac
    cache="${FFMPEG_CACHE_DIR}/${platform}"

    if [[ ! -f "${cache}/.ready" ]]; then
        log "[setup] downloading prebuilt FFmpeg (${asset})"
        archive="${FFMPEG_CACHE_DIR}/${asset}"
        mkdir -p "${FFMPEG_CACHE_DIR}"
        curl -fL --retry 3 -o "${archive}" \
            "${FFMPEG_BTBN_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/${asset}}" \
            >> "${BUILD_LOG}" 2>&1 || { error "failed to download ${asset}"; return 1; }
        rm -rf "${cache}"
        mkdir -p "${cache}"
        if [[ "${asset}" == *.zip ]]; then
            unzip -q "${archive}" -d "${cache}"
            # The zip nests everything under a versioned top-level directory;
            # hoist it so the cache dir itself is the FFmpeg prefix.
            top="$(find "${cache}" -mindepth 1 -maxdepth 1 -type d | head -1)"
            if [[ -n "${top}" && "${top}" != "${cache}" ]]; then
                mv "${top}"/* "${cache}"/ 2>/dev/null || true
                rmdir "${top}" 2>/dev/null || true
            fi
        else
            tar -xJf "${archive}" -C "${cache}" --strip-components=1
        fi
        rm -f "${archive}"
        touch "${cache}/.ready"
    fi
    export FFMPEG_DIR="${cache}"
}

resolve_ffmpeg() {
    local platform="$1"
    if [[ -n "${FFMPEG_DIR:-}" ]]; then
        log "[setup] using FFMPEG_DIR=${FFMPEG_DIR}"
        return 0
    fi
    case "${platform}" in
        darwin-*) ensure_ffmpeg_darwin ;;
        *)        ensure_ffmpeg_btbn "${platform}" ;;
    esac
    log "[setup] FFMPEG_DIR=${FFMPEG_DIR}"
}

# Windows needs the FFmpeg DLLs on the DLL search path, Linux needs the .so via
# LD_LIBRARY_PATH; ship both next to the plugin. macOS links the brew dylibs at
# their absolute install paths, so there is nothing to copy.
collect_ffmpeg_runtime() {
    local platform="$1" rt
    [[ "${platform}" == darwin-* ]] && return 0
    [[ -n "${FFMPEG_DIR:-}" ]] || return 0
    rt="${DIST_DIR}/${platform}/runtime"
    mkdir -p "${rt}"
    case "${platform}" in
        linux-x86_64)   cp -P "${FFMPEG_DIR}"/lib/lib*.so* "${rt}"/ 2>/dev/null || true ;;
        windows-x86_64) cp "${FFMPEG_DIR}"/bin/*.dll "${rt}"/ 2>/dev/null || true ;;
    esac
    if [[ -z "$(ls -A "${rt}")" ]]; then
        warn "no FFmpeg runtime libraries collected for ${platform}"
    fi
}

# --- e2e ---------------------------------------------------------------------
# Play the container matrix once through the plugin that was just built. Only
# meaningful on the host's own platform, and only with a config that points at
# a reachable STT service.
run_e2e() {
    local platform="$1"
    if [[ "${MPV_STT_PLUGIN_RS_SKIP_E2E:-}" == "1" ]]; then
        return 0
    fi
    if [[ "${platform}" != "$(host_platform)" ]]; then
        return 0
    fi
    if [[ -z "${MPV_STT_PLUGIN_RS_CONFIG:-}" ]]; then
        warn "MPV_STT_PLUGIN_RS_CONFIG unset; skipping the e2e pass"
        return 0
    fi
    # A failing pass here is reported, never fatal: it needs a network service,
    # and the artifacts are already correct by this point.
    if ! MPV_STT_PLUGIN_RS_TEST_PLUGIN="${DIST_DIR}/${platform}/$(artifact_of "$(target_of "${platform}")")" \
        bash "${ROOT_DIR}/scripts/e2e-media-matrix.sh" 2>&1 | tee -a "${BUILD_LOG}"; then
        warn "e2e pass failed for ${platform}; artifacts are still in dist/"
    fi
}

# --- build -------------------------------------------------------------------
build_desktop() {
    local platform="$1" target cargo_art dist_art out_dir
    target="$(target_of "${platform}")" || { error "unknown platform '${platform}'"; return 1; }

    log "[build] ${platform} (${target})"
    ensure_rust_target "${target}"
    resolve_ffmpeg "${platform}" || return 1

    local args=(build --release --target "${target}")
    if [[ -n "${FEATURES}" ]]; then
        args+=(--no-default-features --features "${FEATURES}")
    fi

    if ! (cd "${ROOT_DIR}" && cargo "${args[@]}") >> "${BUILD_LOG}" 2>&1; then
        error "cargo build failed for ${platform} (see ${BUILD_LOG})"
        return 1
    fi

    out_dir="${DIST_DIR}/${platform}"
    mkdir -p "${out_dir}"
    cargo_art="$(cargo_artifact_of "${target}")"
    dist_art="$(artifact_of "${target}")"
    cp "${ROOT_DIR}/target/${target}/release/${cargo_art}" "${out_dir}/${dist_art}"
    collect_ffmpeg_runtime "${platform}"
    log "  -> dist/${platform}/${dist_art}"

    run_e2e "${platform}"
}

generate_manifest() {
    local manifest="${DIST_DIR}/MANIFEST.txt" platform f
    {
        echo "mpv_stt_plugin_rs"
        echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""
        for platform in "${PLATFORMS[@]}"; do
            [[ -d "${DIST_DIR}/${platform}" ]] || continue
            echo "${platform} ($(target_of "${platform}"))"
            for f in "${DIST_DIR}/${platform}"/*; do
                [[ -f "${f}" ]] && echo "  $(basename "${f}") ($(du -h "${f}" | cut -f1))"
            done
            if [[ -d "${DIST_DIR}/${platform}/runtime" ]]; then
                echo "  runtime/: $(find "${DIST_DIR}/${platform}/runtime" -maxdepth 1 -type f | wc -l | tr -d ' ') files"
            fi
        done
    } > "${manifest}"
    log "[dist] ${manifest}"
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--platform)
                [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
                IFS=',' read -r -a _parts <<<"$2"
                SELECTED+=("${_parts[@]}")
                shift 2
                ;;
            -f|--features)
                [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
                FEATURES="$2"
                shift 2
                ;;
            --clean) CLEAN_DIST=1; shift ;;
            -l|--list)
                SHOW_LIST=1
                shift
                ;;
            -h|--help) usage; exit 0 ;;
            *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
        esac
    done

    if ((SHOW_LIST)); then
        echo "platforms: ${PLATFORMS[*]} android"
        echo "features : stt_ferrum stt_openai (omit to build both into one artifact)"
        exit 0
    fi

    if [[ -n "${FEATURES}" ]] \
        && [[ ! "${FEATURES}" =~ ^(stt_ferrum|stt_openai)(,(stt_ferrum|stt_openai))*$ ]]; then
        echo "ERROR: --features takes stt_ferrum and/or stt_openai, comma-separated" >&2
        exit 1
    fi

    if ((CLEAN_DIST)); then
        rm -rf "${DIST_DIR}"
    fi
    mkdir -p "${DIST_DIR}"
    : > "${BUILD_LOG}"

    local platforms=("${SELECTED[@]}")
    if [[ ${#platforms[@]} -eq 0 ]]; then
        platforms=("${PLATFORMS[@]}")
    fi

    log "==> mpv_stt_plugin_rs build"
    log "    platforms: ${platforms[*]}"
    log "    features : ${FEATURES:-stt_ferrum,stt_openai}"
    log "    host     : $(host_platform || echo unknown)"

    ensure_mpv_headers

    local failed=0 platform
    for platform in "${platforms[@]}"; do
        build_desktop "${platform}" || failed=$((failed + 1))
    done

    generate_manifest

    if ((failed > 0)); then
        log "==> ${failed} platform(s) failed; see ${BUILD_LOG}"
        exit 1
    fi
    log "==> done"
}

main "$@"
