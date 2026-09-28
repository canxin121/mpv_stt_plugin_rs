#!/usr/bin/env bash
# Cross-compile the plugin for Android.
#
# Android has no system libmpv/libavcodec to link against, so this script
# builds them: the vendored scripts/android-mpv helper (a trimmed copy of
# mpv-android's own buildscripts) produces a libmpv + FFmpeg prefix for the
# requested ABI under target/android-mpv/prefix/<arch>/usr/local, and the plugin
# links that. The first run therefore spends most of its time in FFmpeg and mpv
# and needs the NDK, meson, ninja and a pkg-config that can cross-compile.
#
#   ./scripts/build-android.sh                      # arm64-v8a (default)
#   ./scripts/build-android.sh -a arm64-v8a,x86_64  # several ABIs
#   ./scripts/build-android.sh --all-abis
#   ./scripts/build-android.sh -f stt_openai        # single STT backend
#
# Artifacts land in dist/android/<abi>/libmpv_stt_plugin_rs.so.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
BUILD_LOG="${DIST_DIR}/build.log"
WORK_DIR="${ANDROID_MPV_WORK_DIR:-${ROOT_DIR}/target/android-mpv}"
BUILDER_DIR="${ROOT_DIR}/scripts/android-mpv"
PREFIX_BASE="${ANDROID_MPV_PREFIX_BASE:-${WORK_DIR}/prefix}"
API="${ANDROID_API:-21}"
SUPPORTED_ABIS=(arm64-v8a armeabi-v7a x86 x86_64)
DEFAULT_ABIS=(arm64-v8a)

SELECTED_ABIS=()
FEATURES=""
DECLARED_TARGETS=()

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "${BUILD_LOG}"
}
warn() { log "[warn] $*"; }
error() { log "[error] $*"; }

usage() {
    cat <<'EOF'
Usage: ./scripts/build-android.sh [options]

Options:
  -a, --abi <list>       Comma-separated ABIs (default: arm64-v8a).
                         arm64-v8a, armeabi-v7a, x86, x86_64
      --all-abis         Build every supported ABI
  -f, --features <list>  Single STT backend: stt_ferrum and/or stt_openai.
                         Default: both, in one .so
  -l, --list             List the supported ABIs and exit
  -h, --help             Show this help

Environment:
  ANDROID_NDK_HOME / NDK   Android NDK r29 or newer (required)
  ANDROID_API              API level to target (default: 21)
  ANDROID_MPV_WORK_DIR     Where the dep sources and prefix are built
  ANDROID_MPV_PREFIX_BASE  Where the per-arch libmpv prefix is installed

The 32-bit ABIs are not built by default: ffmpeg-sys-next's Vulkan stub
hardcodes a 64-bit sizeof(VkPhysicalDeviceFeatures2), so bindgen fails there.
EOF
}

# ABI -> arch:rust-target:clang-target (arch is the prefix directory name used
# by scripts/android-mpv).
abi_spec() {
    case "$1" in
        arm64-v8a)   echo "arm64:aarch64-linux-android:aarch64-linux-android" ;;
        armeabi-v7a) echo "armv7l:armv7-linux-androideabi:armv7a-linux-androideabi" ;;
        x86)         echo "x86:i686-linux-android:i686-linux-android" ;;
        x86_64)      echo "x86_64:x86_64-linux-android:x86_64-linux-android" ;;
        *)           return 1 ;;
    esac
}

restore_env() {
    local name
    for name in "${DECLARED_TARGETS[@]}"; do
        unset -v "${name}" 2>/dev/null || true
    done
}

resolve_ndk() {
    local ndk="${ANDROID_NDK_HOME:-${NDK:-${CMAKE_ANDROID_NDK:-}}}"
    if [[ -z "${ndk}" ]]; then
        # Local convenience: the android-mpv helper keeps its own NDK copy here.
        ndk="${WORK_DIR}/android-ndk-r29"
    fi
    echo "${ndk}"
}

host_tag() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  echo "linux-x86_64" ;;
        Darwin-arm64)  echo "darwin-arm64" ;;
        Darwin-x86_64) echo "darwin-x86_64" ;;
        *)             echo "linux-x86_64" ;;
    esac
}

ensure_rust_target() {
    local target="$1"
    if ! rustup target list --installed | grep -qx "${target}"; then
        rustup target add "${target}" >> "${BUILD_LOG}" 2>&1
    fi
}

# Build libmpv + FFmpeg for one arch if the prefix is not there yet. The helper
# runs under `env -i`: its configure/make/meson output would otherwise be
# captured into the caller's environment and blow past ARG_MAX for every later
# command, so it also gets its output redirected into the build log.
ensure_mpv_prefix() {
    local arch="$1" ndk="$2"
    local prefix="${PREFIX_BASE}/${arch}/usr/local"

    if [[ ! -f "${prefix}/lib/libmpv.so" ]]; then
        log "[deps] building libmpv + FFmpeg for ${arch} (first run takes a while)"
        (
            cd "${BUILDER_DIR}" && env -i \
                PATH="${PATH}" HOME="${HOME:-/tmp}" TERM="${TERM:-xterm}" \
                ANDROID_MPV_WORK_DIR="${WORK_DIR}" \
                ANDROID_MPV_PREFIX_BASE="${PREFIX_BASE}" \
                ANDROID_NDK_HOME="${ndk}" \
                ANDROID_API="${API}" \
                ./buildall.sh --arch "${arch}" mpv
        ) >> "${BUILD_LOG}" 2>&1
    fi

    if [[ ! -f "${prefix}/lib/libmpv.so" ]]; then
        error "no libmpv.so under ${prefix} after the helper run (see ${BUILD_LOG})"
        return 1
    fi
}

# Export everything cargo, bindgen and the linkers need for one ABI. Deletes the
# target-specific variables it sets, so the next ABI starts from a clean slate.
setup_android_env() {
    local abi="$1" rust_target="$2" clang_target="$3" prefix="$4" sysroot="$5" toolchain="$6"

    export ANDROID_ABI="${abi}"
    export ANDROID_API="${API}"
    # opusic-sys (bundled, default) compiles libopus through cmake against the
    # NDK's own toolchain file; these are the knobs it reads from the env.
    export ANDROID_PLATFORM="android-${API}"
    export ANDROID_STL="c++_shared"
    export ANDROID_SYSROOT="${sysroot}"
    export MPV_PREFIX="${prefix}"
    export MPV_LIB_DIR="${prefix}/lib"
    export MPV_INCLUDE_DIR="${prefix}/include"
    export FFMPEG_DIR="${prefix}"
    export CMAKE_TOOLCHAIN_FILE="${ROOT_DIR}/toolchains/android.cmake"
    export PKG_CONFIG_ALLOW_CROSS=1
    export PKG_CONFIG_PATH="${prefix}/lib/pkgconfig"
    export PKG_CONFIG_LIBDIR="${prefix}/lib/pkgconfig"
    export BINDGEN_EXTRA_CLANG_ARGS="--target=${clang_target} --sysroot=${sysroot} -I${prefix}/include -I${MPV_INCLUDE_DIR}"
    export CC="${toolchain}/bin/${clang_target}${API}-clang"
    export CXX="${toolchain}/bin/${clang_target}${API}-clang++"
    export AR="${toolchain}/bin/llvm-ar"
    export RANLIB="${toolchain}/bin/llvm-ranlib"
    export STRIP="${toolchain}/bin/llvm-strip"

    local target_env="${rust_target//-/_}"
    # bash 3.2 (what macOS still ships as /bin/bash) has no ${var^^}, so upper-
    # case the triple with tr.
    local upper_target
    upper_target="$(printf '%s' "${rust_target}" | tr '[:lower:]' '[:upper:]')"
    local linker_var="CARGO_TARGET_${upper_target//-/_}_LINKER"
    DECLARED_TARGETS=("CC_${target_env}" "CFLAGS_${target_env}" "CXXFLAGS_${target_env}" "${linker_var}")
    export "CC_${target_env}"="${CC}"
    export "CFLAGS_${target_env}"="--sysroot=${sysroot}"
    export "CXXFLAGS_${target_env}"="--sysroot=${sysroot}"
    export "${linker_var}"="${CC}"
}

build_abi() {
    local abi="$1" spec arch rust_target clang_target ndk toolchain sysroot prefix out_dir
    spec="$(abi_spec "${abi}")" || { error "unknown ABI '${abi}'"; return 1; }
    IFS=":" read -r arch rust_target clang_target <<<"${spec}"

    ndk="$(resolve_ndk)"
    if [[ ! -d "${ndk}" ]]; then
        error "Android NDK not found at ${ndk}; set ANDROID_NDK_HOME"
        return 1
    fi
    toolchain="${ndk}/toolchains/llvm/prebuilt/$(host_tag)"

    restore_env
    ensure_rust_target "${rust_target}"
    ensure_mpv_prefix "${arch}" "${ndk}" || return 1

    prefix="${PREFIX_BASE}/${arch}/usr/local"
    setup_android_env "${abi}" "${rust_target}" "${clang_target}" "${prefix}" "${toolchain}/sysroot" "${toolchain}"

    log "[build] android ${abi} (${rust_target})"
    local args=(build --release --target "${rust_target}")
    if [[ -n "${FEATURES}" ]]; then
        args+=(--no-default-features --features "${FEATURES}")
    fi
    if ! (cd "${ROOT_DIR}" && cargo "${args[@]}") >> "${BUILD_LOG}" 2>&1; then
        error "cargo build failed for android ${abi} (see ${BUILD_LOG})"
        return 1
    fi

    out_dir="${DIST_DIR}/android/${abi}"
    mkdir -p "${out_dir}"
    cp "${ROOT_DIR}/target/${rust_target}/release/libmpv_stt_plugin_rs.so" "${out_dir}/"
    log "  -> dist/android/${abi}/libmpv_stt_plugin_rs.so"
}

main() {
    local all_abis=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -a|--abi)
                [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
                IFS=',' read -r -a _parts <<<"$2"
                SELECTED_ABIS+=("${_parts[@]}")
                shift 2
                ;;
            --all-abis|--all) all_abis=1; shift ;;
            -f|--features)
                [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
                FEATURES="$2"
                shift 2
                ;;
            -l|--list)
                echo "abis    : ${SUPPORTED_ABIS[*]}"
                echo "default : ${DEFAULT_ABIS[*]}"
                echo "features: stt_ferrum stt_openai"
                exit 0
                ;;
            -h|--help) usage; exit 0 ;;
            *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
        esac
    done

    if ((all_abis)); then
        SELECTED_ABIS=("${SUPPORTED_ABIS[@]}")
    elif [[ ${#SELECTED_ABIS[@]} -eq 0 ]]; then
        SELECTED_ABIS=("${DEFAULT_ABIS[@]}")
    fi

    if [[ -n "${FEATURES}" ]] \
        && [[ ! "${FEATURES}" =~ ^(stt_ferrum|stt_openai)(,(stt_ferrum|stt_openai))*$ ]]; then
        echo "ERROR: --features takes stt_ferrum and/or stt_openai, comma-separated" >&2
        exit 1
    fi

    mkdir -p "${DIST_DIR}"
    : > "${BUILD_LOG}"

    log "==> mpv_stt_plugin_rs android build"
    log "    abis    : ${SELECTED_ABIS[*]}"
    log "    features: ${FEATURES:-stt_ferrum,stt_openai}"
    log "    ndk     : $(resolve_ndk)"

    local failed=0 abi
    for abi in "${SELECTED_ABIS[@]}"; do
        build_abi "${abi}" || failed=$((failed + 1))
    done

    if ((failed > 0)); then
        log "==> ${failed} ABI(s) failed; see ${BUILD_LOG}"
        exit 1
    fi
    log "==> done: dist/android"
}

main "$@"
