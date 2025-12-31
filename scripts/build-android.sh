#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPV_ANDROID="${MPV_ANDROID:-/mnt/disk1/shared/git/mpv-android}"
NDK_DEFAULT="$MPV_ANDROID/buildscripts/sdk/android-ndk-r29"
NDK="${ANDROID_NDK_HOME:-${NDK:-$NDK_DEFAULT}}"
API="${ANDROID_API:-21}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
ALL_ABIS=(arm64-v8a armeabi-v7a x86 x86_64)

usage() {
  cat <<'EOF'
Usage: scripts/build-android-all.sh [options]

Options:
  -a, --abi <abi>         Build a single ABI (repeatable, or comma-separated).
  --abis <list>           Build ABIs from a comma-separated list.
  --all-abis              Build all supported ABIs.
  -f, --features <list>   Cargo features (comma-separated).
  --no-default-features   Disable default features.
  --all-features          Enable all features (not supported on Android).
  -h, --help              Show this help message.

Env:
  MPV_STT_PLUGIN_RS_BUILD_ABIS=arm64-v8a,armeabi-v7a|all
  MPV_STT_PLUGIN_RS_FEATURES=stt_local_cpu
  MPV_STT_PLUGIN_RS_NO_DEFAULT_FEATURES=1
  MPV_STT_PLUGIN_RS_ALL_FEATURES=1
EOF
}

BUILD_ABIS_ENV="${MPV_STT_PLUGIN_RS_BUILD_ABIS:-arm64-v8a,armeabi-v7a}"
FEATURES="${MPV_STT_PLUGIN_RS_FEATURES:-}"
NO_DEFAULT_FEATURES="${MPV_STT_PLUGIN_RS_NO_DEFAULT_FEATURES:-}"
ALL_FEATURES="${MPV_STT_PLUGIN_RS_ALL_FEATURES:-}"
ABI_LIST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--abi)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      IFS=',' read -r -a _abis <<<"$2"
      ABI_LIST+=("${_abis[@]}")
      shift 2
      ;;
    --abis)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      IFS=',' read -r -a _abis <<<"$2"
      ABI_LIST+=("${_abis[@]}")
      shift 2
      ;;
    --all-abis|--all)
      ABI_LIST=("${ALL_ABIS[@]}")
      shift
      ;;
    -f|--features)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      FEATURES="$2"
      shift 2
      ;;
    --no-default-features)
      NO_DEFAULT_FEATURES=1
      shift
      ;;
    --all-features)
      ALL_FEATURES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

FEATURES="${FEATURES//[[:space:]]/}"

if [[ ${#ABI_LIST[@]} -eq 0 ]]; then
  if [[ "${BUILD_ABIS_ENV,,}" == "all" ]]; then
    ABI_LIST=("${ALL_ABIS[@]}")
  else
    IFS=',' read -r -a ABI_LIST <<<"$BUILD_ABIS_ENV"
  fi
fi

for abi in "${ABI_LIST[@]}"; do
  if [[ "${abi,,}" == "all" ]]; then
    ABI_LIST=("${ALL_ABIS[@]}")
    break
  fi
done

declare -A _seen_abis
abis=()
for abi in "${ABI_LIST[@]}"; do
  abi="${abi//[[:space:]]/}"
  if [[ -z "$abi" ]]; then
    continue
  fi
  if [[ -z "${_seen_abis[$abi]:-}" ]]; then
    _seen_abis[$abi]=1
    abis+=("$abi")
  fi
done

if [[ ${#abis[@]} -eq 0 ]]; then
  echo "No ABIs selected." >&2
  usage
  exit 1
fi

if [[ -n "${ALL_FEATURES:-}" && -n "${NO_DEFAULT_FEATURES:-}" ]]; then
  echo "Cannot combine --all-features with --no-default-features." >&2
  exit 1
fi

if [[ -n "${ALL_FEATURES:-}" && -n "${FEATURES:-}" ]]; then
  echo "Both --all-features and --features specified; ignoring --features." >&2
  FEATURES=""
fi

if [[ -n "${NO_DEFAULT_FEATURES:-}" && -z "${FEATURES:-}" && -z "${ALL_FEATURES:-}" ]]; then
  echo "No backend selected. Use --features stt_local_cpu or omit --no-default-features." >&2
  exit 1
fi

if [[ -n "${ALL_FEATURES:-}" ]]; then
  echo "Android does not support stt_local_cuda; --all-features is not supported (backends are mutually exclusive)." >&2
  exit 1
fi

if [[ -n "${FEATURES:-}" ]]; then
  if [[ "$FEATURES" =~ (^|,)(stt_local_cuda)(,|$) ]]; then
    echo "Android does not support the stt_local_cuda backend; requested features: $FEATURES" >&2
    exit 1
  fi
fi

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
  export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-z,defs"

  cargo_features=()
  if [[ -n "${NO_DEFAULT_FEATURES:-}" ]]; then
    cargo_features+=(--no-default-features)
  fi
  if [[ -n "${ALL_FEATURES:-}" ]]; then
    cargo_features+=(--all-features)
  fi
  if [[ -n "${FEATURES:-}" ]]; then
    cargo_features+=(--features "$FEATURES")
  fi

  (cd "$ROOT_DIR" && cargo build --target "$rust_target" --release "${cargo_features[@]}")

  OUT_DIR="$ROOT_DIR/dist/android/$abi"
  mkdir -p "$OUT_DIR"
  cp "$ROOT_DIR/target/$rust_target/release/libmpv_stt_plugin_rs.so" "$OUT_DIR/"
  echo "Output: $OUT_DIR/libmpv_stt_plugin_rs.so"

done

echo "All done. Outputs under: $ROOT_DIR/dist/android"
