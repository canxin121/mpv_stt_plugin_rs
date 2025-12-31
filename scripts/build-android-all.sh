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

if [[ ! -d "$MPV_ANDROID" ]]; then
  echo "mpv-android not found: $MPV_ANDROID" >&2
  echo "Set MPV_ANDROID to your mpv-android repo." >&2
  exit 1
fi

BUILD_ABIS="${MPV_STT_PLUGIN_RS_BUILD_ABIS:-arm64-v8a,armeabi-v7a}"

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

  (cd "$ROOT_DIR" && cargo build --target "$rust_target" --release "${cargo_features[@]}")

  OUT_DIR="$ROOT_DIR/dist/android/$abi"
  mkdir -p "$OUT_DIR"
  cp "$ROOT_DIR/target/$rust_target/release/libmpv_stt_plugin_rs.so" "$OUT_DIR/"
  echo "Output: $OUT_DIR/libmpv_stt_plugin_rs.so"

done

echo "All done. Outputs under: $ROOT_DIR/dist/android"
