#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPV_ANDROID="${MPV_ANDROID:-/mnt/disk1/shared/git/mpv-android}"
NDK_DEFAULT="$MPV_ANDROID/buildscripts/sdk/android-ndk-r29"
NDK="${ANDROID_NDK_HOME:-${NDK:-$NDK_DEFAULT}}"
API="${ANDROID_API:-21}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
OPENCL_HEADERS_DIR="$ROOT_DIR/toolchains/opencl-headers"
OPENCL_LIB_DIR="$ROOT_DIR/toolchains/opencl-android"
OPENCL_HEADERS_URL="${OPENCL_HEADERS_URL:-https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/heads/main.tar.gz}"
OPENCL_ANDROID_LIB_URL="${OPENCL_ANDROID_LIB_URL:-}"
ADB_BIN="${ADB_BIN:-$MPV_ANDROID/buildscripts/sdk/android-sdk-linux/platform-tools/adb}"

if [[ ! -d "$NDK" ]]; then
  echo "NDK not found: $NDK" >&2
  echo "Set ANDROID_NDK_HOME or NDK to your NDK path." >&2
  exit 1
fi

if [[ ! -d "$MPV_ANDROID" ]]; then
  echo "mpv-android not found: $MPV_ANDROID" >&2
  echo "Set MPV_ANDROID to your mpv-android repo." >&2
  exit 1
fi

ENABLE_OPENCL=0
if [[ "${WHISPERSUBS_OPENCL:-}" == "1" || "${WHISPERSUBS_GPU:-}" == "1" ]]; then
  ENABLE_OPENCL=1
fi
OPENCL_ABIS="${WHISPERSUBS_OPENCL_ABIS:-arm64-v8a}"
BUILD_ABIS="${WHISPERSUBS_BUILD_ABIS:-arm64-v8a,armeabi-v7a}"

opencl_enabled_for_abi() {
  local abi="$1"
  if [[ "$ENABLE_OPENCL" != "1" ]]; then
    return 1
  fi
  if [[ "$OPENCL_ABIS" == "all" ]]; then
    return 0
  fi
  local IFS=","
  for item in $OPENCL_ABIS; do
    if [[ "$item" == "$abi" ]]; then
      return 0
    fi
  done
  return 1
}

ensure_opencl_headers() {
  if [[ -f "$OPENCL_HEADERS_DIR/CL/cl.h" ]]; then
    return
  fi

  echo "OpenCL headers not found; downloading to $OPENCL_HEADERS_DIR"
  mkdir -p "$OPENCL_HEADERS_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$OPENCL_HEADERS_URL" | tar -xz -C "$OPENCL_HEADERS_DIR" --strip-components=1
  elif command -v wget >/dev/null 2>&1; then
    wget -O- "$OPENCL_HEADERS_URL" | tar -xz -C "$OPENCL_HEADERS_DIR" --strip-components=1
  else
    echo "Neither curl nor wget is available; please install one or set OPENCL_INCLUDE_DIR." >&2
    exit 1
  fi
}

download_opencl_lib() {
  local abi="$1"
  if [[ -z "${OPENCL_ANDROID_LIB_URL}" ]]; then
    return 1
  fi

  local out_dir="$OPENCL_LIB_DIR/$abi"
  local out_lib="$out_dir/libOpenCL.so"
  local url="${OPENCL_ANDROID_LIB_URL//\{abi\}/$abi}"
  mkdir -p "$out_dir"

  echo "Downloading OpenCL library for $abi from $url..."
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out_lib" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out_lib" "$url" || return 1
  else
    echo "Neither curl nor wget is available; please install one or provide adb." >&2
    return 1
  fi
  return 0
}

ensure_opencl_lib() {
  local abi="$1"
  local device_path=""
  case "$abi" in
    arm64-v8a)
      device_path="/vendor/lib64/libOpenCL.so"
      ;;
    armeabi-v7a)
      device_path="/vendor/lib/libOpenCL.so"
      ;;
    *)
      return 1
      ;;
  esac

  local out_dir="$OPENCL_LIB_DIR/$abi"
  local out_lib="$out_dir/libOpenCL.so"
  if [[ -f "$out_lib" ]]; then
    return 0
  fi

  if download_opencl_lib "$abi"; then
    return 0
  fi

  if ! command -v adb >/dev/null 2>&1; then
    if [[ -x "$ADB_BIN" ]]; then
      ADB="$ADB_BIN"
    else
      echo "adb not found; cannot pull OpenCL library for $abi." >&2
      return 1
    fi
  else
    ADB="adb"
  fi
  mkdir -p "$out_dir"
  echo "Pulling OpenCL library for $abi from device ($device_path)..."
  "$ADB" pull "$device_path" "$out_lib" || return 1
}

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
abi_spec() {
  case "$1" in
    arm64-v8a)
      echo "arm64-v8a:arm64:aarch64-linux-android:aarch64-linux-android"
      ;;
    armeabi-v7a)
      echo "armeabi-v7a:armv7l:armv7-linux-androideabi:armv7a-linux-androideabi"
      ;;
    x86)
      echo "x86:x86:i686-linux-android:i686-linux-android"
      ;;
    x86_64)
      echo "x86_64:x86_64:x86_64-linux-android:x86_64-linux-android"
      ;;
    *)
      return 1
      ;;
  esac
}

IFS="," read -r -a abis <<<"$BUILD_ABIS"

# Ensure Rust targets.
for abi in "${abis[@]}"; do
  spec="$(abi_spec "$abi")" || { echo "Unknown ABI: $abi" >&2; exit 1; }
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
  enable_opencl_for_abi=0
  if opencl_enabled_for_abi "$abi"; then
    enable_opencl_for_abi=1
  fi

  if [[ "$enable_opencl_for_abi" == "1" ]]; then
    ensure_opencl_headers
    if ! ensure_opencl_lib "$abi"; then
      echo "OpenCL lib not available for $abi." >&2
      echo "Set OPENCL_ANDROID_LIB_URL or connect a device to pull libOpenCL.so." >&2
      exit 1
    fi
    export OPENCL_INCLUDE_DIR="$OPENCL_HEADERS_DIR"
    export OPENCL_LIBRARY="$OPENCL_LIB_DIR/$abi/libOpenCL.so"
    cargo_features+=(--features whisper-opencl)
  else
    unset OPENCL_INCLUDE_DIR OPENCL_LIBRARY OPENCL_LIB OPENCL_ROOT
  fi

  (cd "$ROOT_DIR" && cargo build --target "$rust_target" --release "${cargo_features[@]}")

  OUT_DIR="$ROOT_DIR/dist/android/$abi"
  mkdir -p "$OUT_DIR"
  cp "$ROOT_DIR/target/$rust_target/release/libwhispersubs_rs.so" "$OUT_DIR/"
  echo "Output: $OUT_DIR/libwhispersubs_rs.so"

done

echo "All done. Outputs under: $ROOT_DIR/dist/android"
