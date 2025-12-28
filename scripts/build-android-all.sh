#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPV_ANDROID="${MPV_ANDROID:-/mnt/disk1/shared/git/mpv-android}"
NDK_DEFAULT="$MPV_ANDROID/buildscripts/sdk/android-ndk-r29"
NDK="${ANDROID_NDK_HOME:-${NDK:-$NDK_DEFAULT}}"
API="${ANDROID_API:-21}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"

if [[ ! -d "$NDK" ]]; then
  echo "NDK not found: $NDK" >&2
  echo "Set ANDROID_NDK_HOME or NDK to your NDK path." >&2
  exit 1
fi

# Vulkan loader is only available in NDK sysroot from API 24+,
# and Vulkan 1.2 symbols used by ggml-vulkan appear from API 31+.
if [[ "${WHISPERSUBS_GPU:-}" == "1" || "${WHISPERSUBS_VULKAN:-}" == "1" ]]; then
  if [[ "$API" -lt 31 ]]; then
    echo "WHISPERSUBS_GPU=1 requires ANDROID_API >= 31; bumping API $API -> 31"
    API=31
  fi
fi

if [[ ! -d "$MPV_ANDROID" ]]; then
  echo "mpv-android not found: $MPV_ANDROID" >&2
  echo "Set MPV_ANDROID to your mpv-android repo." >&2
  exit 1
fi

ensure_vulkan_headers() {
  local headers_root=""

  if [[ -n "${VULKAN_HEADERS:-}" ]]; then
    if [[ -f "$VULKAN_HEADERS/include/vulkan/vulkan.hpp" || -f "$VULKAN_HEADERS/vulkan/vulkan.hpp" ]]; then
      headers_root="$VULKAN_HEADERS"
    fi
  fi

  if [[ -z "$headers_root" && -f "/usr/include/vulkan/vulkan.hpp" ]]; then
    headers_root="/usr"
  fi

  if [[ -z "$headers_root" ]]; then
    local cache_dir="$ROOT_DIR/toolchains/vulkan-headers"
    if [[ -f "$cache_dir/include/vulkan/vulkan.hpp" ]]; then
      headers_root="$cache_dir"
    else
      echo "Vulkan-Headers not found; downloading to $cache_dir"
      mkdir -p "$cache_dir"
      local url="https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.3.275.0.tar.gz"
      if command -v curl >/dev/null 2>&1; then
        curl -L "$url" | tar -xz -C "$cache_dir" --strip-components=1
      elif command -v wget >/dev/null 2>&1; then
        wget -O- "$url" | tar -xz -C "$cache_dir" --strip-components=1
      else
        echo "Neither curl nor wget is available; please install one or set VULKAN_HEADERS." >&2
        exit 1
      fi
      headers_root="$cache_dir"
    fi
  fi

  export VULKAN_HEADERS="$headers_root"
}

if [[ "${WHISPERSUBS_GPU:-}" == "1" || "${WHISPERSUBS_VULKAN:-}" == "1" ]]; then
  ensure_vulkan_headers
  if ! command -v glslc >/dev/null 2>&1; then
    echo "glslc not found in PATH; please install Vulkan SDK or provide glslc." >&2
    exit 1
  fi
fi

# Rust targets needed for Android ABIs.
ensure_rust_target() {
  local target="$1"
  if ! rustup target list --installed | rg -q "^${target}$"; then
    rustup target add "$target"
  fi
}

# Build mpv-android for a given arch if libmpv.so is missing.
ensure_mpv_prefix() {
  local arch="$1"
  local prefix="$MPV_ANDROID/buildscripts/prefix/$arch/usr/local"
  if [[ ! -f "$prefix/lib/libmpv.so" ]]; then
    echo "libmpv.so not found for $arch, building mpv-android..."
    (cd "$MPV_ANDROID/buildscripts" && ./buildall.sh --arch "$arch" mpv)
  fi
}

# Map ABI -> (arch, rust_target, clang_target)
abis=(
  "arm64-v8a:arm64:aarch64-linux-android:aarch64-linux-android"
  "armeabi-v7a:armv7l:armv7-linux-androideabi:armv7a-linux-androideabi"
  "x86:x86:i686-linux-android:i686-linux-android"
  "x86_64:x86_64:x86_64-linux-android:x86_64-linux-android"
)

# Ensure Rust targets.
for spec in "${abis[@]}"; do
  IFS=":" read -r abi arch rust_target clang_target <<<"$spec"
  ensure_rust_target "$rust_target"
  ensure_mpv_prefix "$arch"
  echo ""
  echo "=== Building $abi ($rust_target) ==="

  PREFIX="$MPV_ANDROID/buildscripts/prefix/$arch/usr/local"

  export ANDROID_ABI="$abi"
  export ANDROID_API="$API"
  export NDK="$NDK"
  export CC="$TOOLCHAIN/bin/${clang_target}${API}-clang"
  export CXX="$TOOLCHAIN/bin/${clang_target}${API}-clang++"
  export AR="$TOOLCHAIN/bin/llvm-ar"
  export RANLIB="$TOOLCHAIN/bin/llvm-ranlib"
  export STRIP="$TOOLCHAIN/bin/llvm-strip"

  # Cargo linker env var is target-specific.
  linker_var="CARGO_TARGET_$(echo "$rust_target" | tr '[:lower:]' '[:upper:]' | tr '-' '_')_LINKER"
  export "$linker_var"="$CC"

  export BINDGEN_EXTRA_CLANG_ARGS="--target=${clang_target} --sysroot=$TOOLCHAIN/sysroot -I$PREFIX/include"
  export CMAKE_TOOLCHAIN_FILE="$ROOT_DIR/toolchains/android.cmake"
  export FFMPEG_DIR="$PREFIX"
  export MPV_PREFIX="$PREFIX"
  export RUSTFLAGS="-C link-arg=-Wl,-z,defs"

  cargo_features=()
  if [[ "${WHISPERSUBS_GPU:-}" == "1" || "${WHISPERSUBS_VULKAN:-}" == "1" ]]; then
    cargo_features+=(--features whisper-vulkan)
  fi

  (cd "$ROOT_DIR" && cargo build --target "$rust_target" --release "${cargo_features[@]}")

  OUT_DIR="$ROOT_DIR/dist/android/$abi"
  mkdir -p "$OUT_DIR"
  cp "$ROOT_DIR/target/$rust_target/release/libwhispersubs_rs.so" "$OUT_DIR/"
  echo "Output: $OUT_DIR/libwhispersubs_rs.so"

done

echo "All done. Outputs under: $ROOT_DIR/dist/android"
