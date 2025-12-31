# mpv_stt_plugin_rs - Pure Rust MPV Plugin

MPV 实时字幕生成插件，使用 Rust 实现为原生 MPV C 插件。

## 安装

1. **构建插件**
   ```bash
   cd ~/.config/mpv/scripts/mpv_stt_plugin_rs
   # 推荐：自动拉取 mpv 头文件并设置环境变量
   ./scripts/cargo-with-mpv.sh build --release

   # 若脚本无法访问 GitHub，可手动设置：
   # export MPV_INCLUDE_DIR="/path/to/mpv/include"  # 需包含 mpv/client.h
   # cargo build --release
   ```

2. **安装插件**
   ```bash
   cp target/release/libmpv_stt_plugin_rs.so ~/.config/mpv/scripts/
   ```

3. **配置键绑定**

   添加以下内容到 `~/.config/mpv/input.conf`：
   ```
   Ctrl+. script-message toggle-stt
   ```

   或使用其他按键，例如 `F8`。

4. **配置 Whisper 和翻译设置**

   现在支持运行时配置文件（无需改代码/重新编译）：

   - 默认配置文件路径：`~/.config/mpv/mpv_stt_plugin_rs.toml`
   - 也可以用环境变量指定：`MPV_STT_PLUGIN_RS_CONFIG=/path/to/mpv_stt_plugin_rs.toml`

   示例（TOML，带注释说明）：
```toml
[stt.local_whisper]
model_path = "/path/to/ggml-base.bin"
threads = 8              # CPU 线程数
language = "en"
gpu_device = 0           # 仅 cuda 版有效
flash_attn = false       # 仅 cuda 版有效
timeout_ms = 120000

[stt.remote_udp]
server_addr = "127.0.0.1:9000"
timeout_ms = 120000
max_retry = 3
enable_encryption = false
encryption_key = ""
auth_secret = ""

[translate]
from_lang = "en"
to_lang = "zh"
concurrency = 4

[chunk]
local_ms = 15000    # 本地文件模式：每次转写的媒体片段长度
network_ms = 15000  # 网络流模式：每次转写的媒体片段长度

[timeout]
ffmpeg_ms = 30000
ffprobe_ms = 10000
stt_ms = 120000
translate_ms = 30000

[playback]
show_progress = true   # 在 mpv OSD 显示进度
save_srt = true        # 保存 SRT 到文件
auto_start = false     # 自动启动（默认关闭）

[prefetch]
lookahead_chunks = 2           # 预读块数：提前处理的片段数量

[network]
# 网络流缓存（可选，单位字节）
demuxer_max_bytes = 536870912
```

   CUDA 支持说明（纯编译期选择，无运行时回退）：
   - 编译时开启：`cargo build --release --features stt_local_cuda`
   - 运行时需确保系统能找到 CUDA 运行库（例如配置 `LD_LIBRARY_PATH` 或系统动态链接器路径）

   STT 后端编译选项（Linux 可用，Android 支持 `stt_local_cpu` / `stt_remote_udp`，不支持 CUDA）：
   - `stt_local_cpu`：本地 whisper.cpp CPU 后端
   - `stt_local_cuda`：本地 whisper.cpp CUDA 后端
   - `stt_remote_udp`：远端 UDP STT 服务端

   示例：
   - `cargo build --release`（默认 `stt_local_cpu`）
   - `cargo build --release --no-default-features --features stt_local_cuda`
   - `cargo build --release --no-default-features --features stt_remote_udp`


## Android 构建

### 一键构建（全部 ABI）
脚本位置：`scripts/build-android-all.sh`

```bash
cd ~/.config/mpv/scripts/mpv_stt_plugin_rs
./scripts/build-android-all.sh
```

产物输出：
```
dist/android/arm64-v8a/libmpv_stt_plugin_rs.so
dist/android/armeabi-v7a/libmpv_stt_plugin_rs.so
dist/android/x86/libmpv_stt_plugin_rs.so
dist/android/x86_64/libmpv_stt_plugin_rs.so
```

### 依赖说明（Android 动态链接）
每个 ABI 的 `libmpv_stt_plugin_rs.so` 依赖以下动态库：

需要随包（放进 APK 的 `jniLibs/<abi>/` 或等效位置）：
- `libmpv.so`
- `libavcodec.so`
- `libavdevice.so`
- `libavformat.so`
- `libavutil.so`
- `libswresample.so`
- `libc++_shared.so`

系统自带（无需拷贝）：
- `libc.so`
- `libm.so`
- `libdl.so`

> 说明：**一个 `.so` 无法同时运行在所有 ABI 上**，必须按 ABI 分别编译与打包。

### 中间产物 / 依赖库获取位置（Android）

这些依赖由 `mpv-android/buildscripts` 构建并安装到 prefix 目录：

- 目录模板：
  - `mpv-android/buildscripts/prefix/<arch>/usr/local/lib/`
  - `<arch>` 为：`arm64` / `armv7l` / `x86` / `x86_64`

常用库文件位置示例（以 arm64 为例）：
- `libmpv.so`：`mpv-android/buildscripts/prefix/arm64/usr/local/lib/libmpv.so`
- `libavcodec.so`：`mpv-android/buildscripts/prefix/arm64/usr/local/lib/libavcodec.so`
- `libavdevice.so`：`mpv-android/buildscripts/prefix/arm64/usr/local/lib/libavdevice.so`
- `libavformat.so`：`mpv-android/buildscripts/prefix/arm64/usr/local/lib/libavformat.so`
- `libavutil.so`：`mpv-android/buildscripts/prefix/arm64/usr/local/lib/libavutil.so`
- `libswresample.so`：`mpv-android/buildscripts/prefix/arm64/usr/local/lib/libswresample.so`

`libc++_shared.so` 来自 NDK：
- `.../android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/<triple>/libc++_shared.so`
  - 例如 arm64：`.../sysroot/usr/lib/aarch64-linux-android/libc++_shared.so`

### 关键环境变量（脚本已自动处理）
- `ANDROID_NDK_HOME`/`NDK`：NDK 路径
- `MPV_ANDROID`：mpv-android 仓库路径（默认 `/mnt/disk1/shared/git/mpv-android`）
- `ANDROID_API`：API level（默认 21）

## 使用方法

1. 启动 MPV 播放视频
2. 插件会自动加载并显示欢迎消息
3. 按 `Ctrl+.` 切换实时字幕生成
4. 再次按 `Ctrl+.` 停止
