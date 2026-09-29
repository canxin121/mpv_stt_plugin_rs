#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_DIR="${ANDROID_MPV_WORK_DIR:-${ROOT_DIR}/target/android-mpv}"
BIN_DIR="${WORK_DIR}/bin"

. "$SCRIPT_DIR/include/depinfo.sh"

os=linux
[[ "$OSTYPE" == "darwin"* ]] && os=mac
export os

if [ "$os" == "mac" ]; then
  [ -z "$cores" ] && cores=$(sysctl -n hw.ncpu)
  # Both names are resolved to absolute paths, because the helper's `env -i`
  # re-exec keeps only PATH: a bare name would be looked up by every configure
  # script that inherited it, and the value also lands in libtool as a relative
  # path (`../ginstall`) that does not resolve from the build directory.
  # Only export when present. An exported `SED=gsed` that does not exist is
  # worse than not setting it: autoconf's `AC_PROG_LD` runs `$SED` to locate the
  # linker, gets "command not found" through the shell, and reports it as
  # `no acceptable ld found in $PATH` — naming the linker rather than the
  # missing sed. BSD sed is enough everywhere the tree uses it; the one caller
  # that needs GNU syntax asks with `${SED:-sed}` and gets the fallback.
  gsed_path=$(command -v gsed) && export SED="$gsed_path"
  ginstall_path=$(command -v ginstall) && export INSTALL="$ginstall_path"
else
  [ -z "$cores" ] && cores=$(grep -c ^processor /proc/cpuinfo)
fi
cores=${cores:-4}

# Ensure gas-preprocessor for ffmpeg asm
if ! command -v gas-preprocessor.pl >/dev/null 2>&1; then
  mkdir -p "$BIN_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "https://github.com/FFmpeg/gas-preprocessor/raw/master/gas-preprocessor.pl" -o "$BIN_DIR/gas-preprocessor.pl"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$BIN_DIR/gas-preprocessor.pl" "https://github.com/FFmpeg/gas-preprocessor/raw/master/gas-preprocessor.pl"
  else
    echo "gas-preprocessor.pl missing and neither curl nor wget available" >&2
  fi
  chmod +x "$BIN_DIR/gas-preprocessor.pl"
  export PATH="$BIN_DIR:$PATH"
fi

# Resolve NDK/toolchain path: prefer env, else local sdk download layout
resolve_ndk_path() {
  if [ -n "${ANDROID_NDK_HOME:-}" ]; then
    echo "$ANDROID_NDK_HOME"
  elif [ -n "${NDK:-}" ]; then
    echo "$NDK"
  elif [ -n "${CMAKE_ANDROID_NDK:-}" ]; then
    echo "$CMAKE_ANDROID_NDK"
  else
    echo "${WORK_DIR}/android-ndk-${v_ndk}"
  fi
}

toolchain_root=$(echo "$(resolve_ndk_path)"/toolchains/llvm/prebuilt/*)
if [ -d "$toolchain_root" ]; then
  export PATH="$toolchain_root/bin:$(resolve_ndk_path):$WORK_DIR/sdk/bin:$PATH"
fi

export ANDROID_HOME="${ANDROID_HOME:-$WORK_DIR/sdk/android-sdk-$os}"
unset ANDROID_SDK_ROOT ANDROID_NDK_ROOT

# When cross compiling, pkg-config should look into the prefix
if [ -n "${ndk_triple:-}" ] && [ -n "${prefix_dir:-}" ]; then
  export PKG_CONFIG_SYSROOT_DIR="$prefix_dir"
  export PKG_CONFIG_LIBDIR="$PKG_CONFIG_SYSROOT_DIR/lib/pkgconfig"
  unset PKG_CONFIG_PATH
fi
