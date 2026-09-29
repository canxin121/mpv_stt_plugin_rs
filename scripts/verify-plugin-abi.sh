#!/usr/bin/env bash
# Check whether a built Android plugin can actually dlopen against the libmpv and
# FFmpeg set a host player ships.
#
# Android's linker resolves the plugin's *versioned* references — the
# `avcodec_send_packet@LIBAVCODEC_63` recorded in `.gnu.version_r` — against the
# libraries already loaded in the process, which are the host player's. A plugin
# built against a different FFmpeg major names a version that library does not
# define, so dlopen fails with `cannot locate symbol "avcodec_send_packet"` even
# though the symbol itself exists in both. Nothing in a normal build output says
# so: the link succeeds against the prefix, and the failure only shows up on the
# device.
#
# Run it against the FFmpeg/libmpv set the player actually ships. For mpvEx that
# is the arm64-v8a directory of the `io.github.marlboro-advance:mpv-android` AAR:
#
#   unzip -o mpv-android-1.0.0.aar 'jni/arm64-v8a/*' -d /tmp/aar
#   ./scripts/verify-plugin-abi.sh dist/android/arm64-v8a/libmpv_stt_plugin_rs.so /tmp/aar/jni/arm64-v8a
#
# usage: verify-plugin-abi.sh <plugin.so> <libdir>
set -uo pipefail

# The NDK ships the only readelf reliably present on a checkout; override for
# another one.
READELF="${READELF:-${ANDROID_NDK_HOME:-${NDK:-}}/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/llvm-readelf}"
PLUGIN="$1"
LIBDIR="$2"

if [[ ! -x "$READELF" ]]; then
  echo "no llvm-readelf at $READELF; set READELF or ANDROID_NDK_HOME" >&2
  exit 2
fi
if [[ ! -f "$PLUGIN" ]]; then
  echo "no such plugin: $PLUGIN" >&2
  exit 2
fi
if [[ ! -d "$LIBDIR" ]]; then
  echo "no such libdir: $LIBDIR" >&2
  exit 2
fi

status=0

echo "== plugin =="
echo "  $PLUGIN"
echo "  sha256 $(shasum -a 256 "$PLUGIN" | cut -d' ' -f1)"
echo
echo "== DT_NEEDED =="
"$READELF" -d "$PLUGIN" | sed -n 's/.*NEEDED.*\[\(.*\)\].*/  \1/p'
echo

# 1. Every library the plugin names must be next to the host player's, or the
#    loader cannot satisfy the dependency at all.
echo "== library presence =="
while read -r need; do
  case "$need" in libc.so|libm.so|libdl.so|liblog.so|libz.so|libandroid.so) continue ;; esac
  if [[ -f "$LIBDIR/$need" ]]; then
    echo "  ok       $need"
  else
    echo "  MISSING  $need"
    status=1
  fi
done < <("$READELF" -d "$PLUGIN" | sed -n 's/.*NEEDED.*\[\(.*\)\].*/\1/p')

# 2. The version names the plugin asks for have to be the ones the host libraries
#    define — this is the check that catches an FFmpeg major mismatch.
#
#    --version-info prints a needs block as
#      0x0000: Version: 1  File: libavcodec.so  Cnt: 1
#      0x0080:   Name: LIBAVCODEC_62  Flags: none  Version: 3
#    so the file comes from the `File:` line and each following `Name:` belongs
#    to it until the next `File:`. The keyword is located by token rather than by
#    field position, because the address column shifts the numbering between the
#    two line shapes.
echo
echo "== versioned reference check =="
"$READELF" --version-info "$PLUGIN" | python3 -c '
import sys
file = None
for line in sys.stdin:
    tokens = line.split()
    if "File:" in tokens:
        file = tokens[tokens.index("File:") + 1]
    elif "Name:" in tokens and file:
        name = tokens[tokens.index("Name:") + 1]
        if name.startswith("LIB"):
            print(f"{name}:{file}")
' | sort -u > /tmp/_plugin_need_ver.txt

if [[ ! -s /tmp/_plugin_need_ver.txt ]]; then
  echo "  (the plugin records no versioned references)"
fi

fail=0
while IFS=: read -r ver file; do
  [[ -z "${ver:-}" ]] && continue
  # libc/libm/libdl/liblog come from the platform, not from what a player ships,
  # so their version names are never the player's business.
  case "$file" in libc.so|libm.so|libdl.so|liblog.so|libz.so|libandroid.so) continue ;; esac
  if [[ ! -f "$LIBDIR/$file" ]]; then
    echo "  MISSING  $ver (file $file absent)"
    fail=1
    continue
  fi
  if "$READELF" --version-info "$LIBDIR/$file" 2>/dev/null | grep -q "Name: $ver$"; then
    echo "  ok       $ver  <- $file"
  else
    have=$("$READELF" --version-info "$LIBDIR/$file" 2>/dev/null |
      sed -n 's/.*Name: \(LIB[A-Z0-9_]*\).*/\1/p' | sort -u | tr '\n' ' ')
    echo "  MISMATCH $ver  <- $file exports: ${have:-none}"
    fail=1
  fi
done < /tmp/_plugin_need_ver.txt
[[ $fail -eq 1 ]] && status=1

# 3. Every FFmpeg and libmpv symbol the plugin imports has to exist in the host
#    set. Names outside those two namespaces resolve from the platform, so
#    reporting them would be noise.
echo
echo "== unresolved symbol check =="
"$READELF" --dyn-syms --wide "$PLUGIN" | sed -n 's/.*UND *\([^ ]*\)$/\1/p' | sed 's/@.*//' |
  grep -E '^(av|swr|sws)_|^mpv_' | sort -u > /tmp/_plugin_need_sym.txt
: > /tmp/_plugin_have_sym.txt
for lib in "$LIBDIR"/lib*.so; do
  "$READELF" --dyn-syms --wide "$lib" 2>/dev/null | awk '$7!="UND"{print $8}' | sed 's/@.*//'
done | sort -u > /tmp/_plugin_have_sym.txt

if comm -23 /tmp/_plugin_need_sym.txt /tmp/_plugin_have_sym.txt > /tmp/_plugin_unresolved.txt &&
   [[ -s /tmp/_plugin_unresolved.txt ]]; then
  echo "  NOT PROVIDED by the host library set:"
  sed 's/^/    /' /tmp/_plugin_unresolved.txt
  status=1
else
  echo "  ok  all $(wc -l < /tmp/_plugin_need_sym.txt | tr -d ' ') FFmpeg/libmpv symbols the plugin imports are provided"
fi

echo
echo "checked against: ${LIBDIR}"
[[ $status -eq 0 ]] && echo "RESULT: compatible" || echo "RESULT: INCOMPATIBLE"
exit $status
