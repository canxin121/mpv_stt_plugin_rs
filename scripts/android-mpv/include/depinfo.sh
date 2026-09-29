#!/bin/bash -e

# Versions pulled from mpv-android buildscripts (trimmed for mpv/libmpv build)
v_ndk=r29
v_unibreak=6.1
v_harfbuzz=12.2.0
v_fribidi=1.0.16
v_freetype=2.14.1

# Dependency tree (minimal for libmpv)
dep_ffmpeg=()
dep_freetype2=()
dep_fribidi=()
dep_harfbuzz=()
dep_unibreak=()
dep_libplacebo=()
dep_libass=(freetype2 fribidi harfbuzz unibreak)
dep_mpv=(ffmpeg libass libplacebo)

# Pinned ffmpeg revision.
#
# This must be an FFmpeg release line that still records
#
#     upstream/libavcodec/libavcodec.map: LIBAVCODEC_<major>
#
# in its version script, because Android's dynamic linker resolves the plugin's
# versioned FFmpeg references (avcodec_send_packet@LIBAVCODEC_63 and friends)
# against the libavcodec.so the host player ships. n8.0 produces 62.x libraries
# and the plugin then only satisfies a libavcodec that exports LIBAVCODEC_62.
# mpv-android, and therefore mpvEx, ships FFmpeg 9 / LIBAVCODEC_63.
v_ci_ffmpeg=n9.0
