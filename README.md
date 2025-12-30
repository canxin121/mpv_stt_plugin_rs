# WhisperSubs - Pure Rust MPV Plugin

MPV 实时字幕生成插件，使用 Rust 实现为原生 MPV C 插件。

## 架构

完全 Rust 实现：

- **Rust MPV C 插件** - 使用 `mpv-client` 库直接集成到 MPV
- **核心模块** - 音频处理、Whisper、SRT、翻译
- **零 Lua 依赖** - 纯 Rust 实现

## 功能特性

- ✅ 原生 MPV C 插件
- ✅ 高性能 Rust 实现
- ✅ 内存安全保证
- ✅ 类型安全的 MPV API 调用
- ✅ 实时语音识别（网络流支持）
- ✅ 自动翻译（双语字幕）
- ✅ 持续处理（自动跟随播放进度）
- ✅ Seek 支持（快进/后退自动适应）
- ✅ 智能字幕管理（内存中统一管理，单一文件）

## 依赖

### 运行时依赖
- MPV 播放器
- FFmpeg
- 说明：
  - **桌面版**：通过 Rust 绑定静态链接，无需外部 ffmpeg/ffprobe 可执行文件
  - **Android**：动态链接系统/打包内的 FFmpeg 动态库（见下方 Android 构建与依赖）
- Whisper.cpp
- Crow Translate CLI（可选）

### 构建依赖
- Rust 工具链（1.70+）
- Cargo
- libmpv 开发头文件（或 MPV 源码）
- Clang/LLVM（用于 bindgen）

## 安装

1. **构建插件**
   ```bash
   cd ~/.config/mpv/scripts/whispersubs_rs

   # 设置 MPV 头文件路径（如果未安装 mpv-devel）
   export BINDGEN_EXTRA_CLANG_ARGS="-I/mnt/disk1/shared/git/mpv/include"

   cargo build --release
   ```

2. **安装插件**
   ```bash
   cp target/release/libwhispersubs_rs.so ~/.config/mpv/scripts/
   ```

3. **配置键绑定**

   添加以下内容到 `~/.config/mpv/input.conf`：
   ```
   Ctrl+. script-message toggle-whisper
   ```

   或使用其他按键，例如 `F8`。

4. **配置 Whisper 和翻译设置**

   现在支持运行时配置文件（无需改代码/重新编译）：

   - 默认配置文件路径：`~/.config/mpv/whispersubs.toml`
   - 也可以用环境变量指定：`WHISPERSUBS_CONFIG=/path/to/whispersubs.toml`

   示例（TOML，带注释说明）：
   ```toml
   # whisper.cpp 配置
   model_path = "/path/to/ggml-base.bin"
   threads = 8              # CPU 线程数
   language = "ja"
   gpu_device = 0           # 仅 cuda 版有效
   flash_attn = false       # 仅 cuda 版有效
   whisper_timeout_ms = 120000

   # 翻译设置（内置 Google 翻译）
   from_lang = "ja"       # 源语言
   to_lang = "zh"         # 目标语言
   # crow_engine 仅在 external-translate 特性和外部翻译模式下使用
   crow_engine = "google"

   # 分段设置（毫秒）
   local_chunk_size_ms = 15000    # 本地文件模式：每次转写的媒体片段长度
   network_chunk_size_ms = 15000  # 网络流模式：每次转写的媒体片段长度
   wav_chunk_size_ms = 16000  # 提供给 Whisper 的 wav 片段长度

   # 行为开关
   show_progress = true   # 在 mpv OSD 显示进度
   start_at_zero = true   # 从 0 开始转写（本地文件）
   save_srt = true        # 保存 SRT 到文件
   auto_start = false     # 自动启动（默认关闭）

   # 超时设置（毫秒）
   ffmpeg_timeout_ms = 30000
   ffprobe_timeout_ms = 10000
   whisper_timeout_ms = 120000
   translate_timeout_ms = 30000
   translate_concurrency = 4   # 翻译并发数

   # 延迟与预处理策略（以下功能强制启用，只能配置参数）
   catchup_threshold_ms = 30000   # 追赶模式阈值：落后超过此时间则跳转
   lookahead_chunks = 2           # 预读块数：提前处理的片段数量
   lookahead_limit_ms = 60000     # 预读限制：最多提前处理的时间

   # 网络流缓存（可选，单位字节）
   demuxer_max_bytes = 536870912
   min_network_chunk_ms = 5000   # 网络流最短处理时长，不足则等待更多缓存
   ```

   CUDA 支持说明（纯编译期选择，无运行时回退）：
   - 编译时开启：`cargo build --release --features whisper_cpp_cuda`
   - 运行时需确保系统能找到 CUDA 运行库（例如配置 `LD_LIBRARY_PATH` 或系统动态链接器路径）

   Whisper 后端编译选项（Linux 可用，Android 仅支持 `whisper_cpp_cpu`）：
   - `whisper_cpp_cpu`：Whisper.cpp CPU 后端
   - `whisper_cpp_cuda`：Whisper.cpp CUDA 后端

   示例：
   - `cargo build --release`（默认 `whisper_cpp_cpu`）
   - `cargo build --release --no-default-features --features whisper_cpp_cuda`
   - `cargo build --release --no-default-features --features fast_whisper_cpu`

   fast_whisper 运行时说明：
   - 需要系统安装 Python 3.9+ 与 `faster-whisper`（`pip install faster-whisper`）
   - `model_path` 需为 **faster-whisper** 模型名（如 `large-v3`）或 **CTranslate2 模型目录**
   - 可用环境变量：
     - `WHISPERSUBS_PYTHON=/path/to/python`
     - `WHISPERSUBS_FAST_WHISPER_COMPUTE_TYPE`（默认 `default`）
     - `WHISPERSUBS_FAST_WHISPER_BEAM_SIZE`（默认 `5`）


## Android 构建

### 一键构建（全部 ABI）
脚本位置：`scripts/build-android-all.sh`

```bash
cd ~/.config/mpv/scripts/whispersubs_rs
./scripts/build-android-all.sh
```

产物输出：
```
dist/android/arm64-v8a/libwhispersubs_rs.so
dist/android/armeabi-v7a/libwhispersubs_rs.so
dist/android/x86/libwhispersubs_rs.so
dist/android/x86_64/libwhispersubs_rs.so
```

### 依赖说明（Android 动态链接）
每个 ABI 的 `libwhispersubs_rs.so` 依赖以下动态库：

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

## 为什么要 patch ffmpeg-next（Android）

Android NDK 自带的 FFmpeg 头文件在不同版本中可能**新增枚举值**（例如色彩、side-data、codec id 等）。
`ffmpeg-next` 上游在某些匹配分支里是**穷举 match**，当遇到这些新增值时会触发编译错误（non‑exhaustive patterns）。

为保证 Android 构建稳定，本项目做了两点改动：

1) **Android-only 兜底分支**  
   在 `vendor/ffmpeg-next` 中为相关 `match` 增加 Android 专用兜底分支，避免因新增枚举值导致编译失败。

2) **Unknown 枚举保留**  
   对 `AVFrameSideDataType` / `AVPacketSideDataType` 加 `Unknown(...)` 变体（仅 Android）用于兼容未来新增值。

这些改动通过 `Cargo.toml` 的 `[patch.crates-io]` 指向 `vendor/ffmpeg-next`，**仅影响本项目**且只在 Android 目标生效；桌面构建保持原行为。

## 使用方法

1. 启动 MPV 播放视频
2. 插件会自动加载并显示欢迎消息
3. 按 `Ctrl+.` 切换实时字幕生成
4. 再次按 `Ctrl+.` 停止

## 项目结构

```
whispersubs_rs/
├── Cargo.toml
└── src/
    ├── lib.rs           # 库入口
    ├── plugin.rs        # MPV 插件入口和事件循环
    ├── audio.rs         # 音频处理
    ├── whisper.rs       # Whisper 集成
    ├── srt.rs           # SRT 处理
    ├── translate.rs     # 翻译功能
    ├── error.rs         # 错误类型
    └── ffi.rs           # C FFI 接口（备用）
```

## Rust 模块说明

### plugin.rs
MPV C 插件入口点，实现：
- `mpv_open_cplugin` - MPV 插件入口函数
- 事件循环处理
- 键绑定和用户交互
- 插件状态管理

### 其他模块
与原设计相同，提供核心功能：
- `audio.rs` - FFmpeg 音频提取
- `whisper.rs` - Whisper.cpp 集成
- `srt.rs` - SRT 文件处理
- `translate.rs` - 翻译功能

## 优势

相比 Lua 实现：

1. **性能**: 原生代码，无解释器开销
2. **安全**: Rust 的内存安全保证
3. **类型安全**: 编译时类型检查
4. **直接 API 访问**: 无需 Lua FFI 桥接
5. **单一二进制**: 一个 .so 文件包含所有功能

## 开发

### 运行测试
```bash
cargo test
```

### 开发构建
```bash
env BINDGEN_EXTRA_CLANG_ARGS="-I/path/to/mpv/include" cargo build
```

### Release 构建
```bash
env BINDGEN_EXTRA_CLANG_ARGS="-I/path/to/mpv/include" cargo build --release
```

## 当前状态

✅ 已完成：
- 项目结构
- 核心模块实现（audio, whisper, srt, translate）
- MPV 插件框架
- 编译系统
- **持续转录逻辑** (网络流)
- **翻译集成** (双语字幕)
- **字幕同步** (实时更新单一文件)
- **Seek 处理** (快进/后退自动适应)
- **智能字幕管理** (内存中 BTreeMap 统一管理)

⚠️ 待实现：
- 本地文件模式
- 字幕保存到媒体文件目录

## 故障排除

### 编译失败：找不到 mpv/client.h
安装 mpv-devel 或设置环境变量：
```bash
export BINDGEN_EXTRA_CLANG_ARGS="-I/path/to/mpv/include"
```

### 插件未加载
检查 MPV 日志：
```bash
mpv --msg-level=all=debug yourfile.mp4
```

### 键绑定不工作
当前使用 `script-message` 机制，可能需要调整实现。

## 许可

本项目基于 GPL-3.0 许可（继承自 mpv-client 库）

## 贡献

欢迎提交 Issue 和 Pull Request！

## 鸣谢

- [mpv-client](https://github.com/TheCactusVert/mpv-client) - Rust MPV 客户端库
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Whisper 模型实现
- [MPV](https://mpv.io/) - 媒体播放器
