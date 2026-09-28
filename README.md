# mpv_stt_plugin_rs

MPV 实时字幕插件(Rust 原生 C 插件)。插件是**纯远程客户端**:音频抽取后送到远程
STT 服务转写,再送远程翻译服务翻译。不内置任何本地推理引擎、不直连 Google 网页接口。

## 架构(单 crate,多 mod)

```
mpv_stt_plugin_rs/
├── Cargo.toml               # 单 crate (mpv_stt_plugin_rs, cdylib+rlib)
├── build.rs                 # 链接处理(macOS dynamic_lookup / Windows FORCE:UNRESOLVED / Android -lmpv)
├── src/
│   ├── lib.rs               # mod 声明 + re-export
│   ├── common.rs            # 错误类型 / Result
│   ├── crypto.rs            # AES-256-GCM / AuthToken(ferrum 协议)
│   ├── srt.rs               # SRT 字幕解析/偏移
│   ├── audio.rs             # FFmpeg 音频抽取(链接预编译 FFmpeg)
│   ├── config.rs            # 配置 + 后端选择
│   ├── plugin.rs            # mpv cplugin 入口(mpv_open_cplugin)、worker、缓存
│   ├── ffi.rs               # C 导出(翻译 / 音频 / SRT)
│   ├── process.rs           # 子进程管理
│   ├── subtitle_manager.rs  # 字幕管理
│   ├── translate.rs         # 远程翻译客户端(DeepL 兼容 + LibreTranslate)
│   └── stt/
│       ├── mod.rs           # SttRunner / SttBackend 调度
│       ├── ferrum.rs        # ferrum 协议后端(stt_ferrum)
│       └── openai.rs        # OpenAI 协议后端(stt_openai)
├── scripts/
│   ├── cargo-with-deps.sh   # 宿主构建(自动拉 mpv 头文件 + 设 FFMPEG_DIR)
│   └── build-android.sh     # Android 交叉编译
└── toolchains/android*.cmake
```

### STT 后端

两个远程 STT 后端**同时编译、运行时选择**(`config.stt.backend`):

| Cargo feature | 配置字段 | 协议 |
|---|---|---|
| `stt_ferrum` | `[stt.ferrum]` | 自定义 ferrum 协议:raw-body POST `/transcribe`,支持 Opus 压缩 / AES-256-GCM 加密 / 鉴权 / 模型选择(`x-model`)/ 语言提示(`x-language`) |
| `stt_openai` | `[stt.openai]` | 标准 OpenAI `POST /v1/audio/transcriptions`(multipart),任何兼容服务端可用:本地 subtitle-gateway(`model = "sensevoice"`)、OpenAI(`whisper-1`)、Groq(`whisper-large-v3`) |

> `model` 必须是服务端实际提供的 id,写错会在第一个音频块上报
> `Server error (404 Not Found): model_not_found`。

两个 feature 默认同时开启;需要单后端专用构建时用 `--no-default-features --features stt_openai`。

`openai` 后端只发**标准字段**:`file` / `model` / `language` / `response_format=verbose_json`
/ `timestamp_granularities[]=segment`。分段靠后两个字段(OpenAI 只在 `verbose_json` 下
返回 `segments` 数组);服务端若忽略它们、只回 `{"text": ...}`,或返回空 `segments`,
插件就退化成**每个音频块一条字幕**,而不是写一个空 SRT 覆盖已有字幕。

### 翻译后端

翻译同样走**远程接口**,两种协议**同时编译、运行时选择**(`config.translate.backend`):

| backend | 配置字段 | 协议 |
|---|---|---|
| `deepl`(默认) | `[translate]` 平铺 `server_addr`/`api_key` | `POST {server}/v1/translate`,key 走 `Authorization: DeepL-Auth-Key`,target 大写 |
| `libretranslate` | `[translate.libretranslate]` | `POST {server}/translate`,key 走 body `api_key`,target 小写,`auto` 可显式/省略 |

## 编译

### 系统依赖

日志平台**动态链接**预编译 FFmpeg(不从源码编译):

- macOS: `brew install ffmpeg`(插件链接其 dylib),需 `clang`(bindgen 用,brew 自带)
- Linux: `sudo apt-get install clang pkg-config` + 预编译 FFmpeg(`FFMPEG_DIR` 指向 dev 前缀)
- Windows: MSVC 工具链 + 预编译 FFmpeg 共享包
- Android(交叉): NDK + mpv-android 的 libmpv/libffmpeg 前缀

**不需要安装 `libmpv-dev`**:`scripts/cargo-with-deps.sh` 会自动 `git clone --depth 1`
mpv 仓库到 `target/mpv-headers` 并导出 `MPV_INCLUDE_DIR` / `BINDGEN_EXTRA_CLANG_ARGS`。

### 宿主构建

```bash
./scripts/cargo-with-deps.sh build --release
./scripts/cargo-with-deps.sh test          # 离线单测
```

脚本会自动准备 mpv 头文件,并在 macOS 上把 `FFMPEG_DIR` 设为 `brew --prefix ffmpeg`;
其他平台需自行 `export FFMPEG_DIR=/path/to/ffmpeg-dev-prefix`。

产物(Cargo 原名,以 macOS 为例):

```bash
ls target/release/libmpv_stt_plugin_rs.dylib
```

> macOS 上若 mpv 头文件不在 `/opt/homebrew/include`,需
> `BINDGEN_EXTRA_CLANG_ARGS="-I/opt/homebrew/include" cargo build`。

### Android 构建

```bash
export ANDROID_NDK_HOME=~/Android/Sdk/ndk/26.1.10909125
export MPV_ANDROID=/path/to/mpv-android     # 提供 libmpv.so / libffmpeg 前缀
./scripts/build-android.sh -a arm64-v8a
# 多 ABI:./scripts/build-android.sh --all-abis
# 单后端:./scripts/build-android.sh -a arm64-v8a -f stt_openai
```

输出在 `dist/android/<abi>/libmpv_stt_plugin_rs.so`。

## 安装

```bash
cp target/release/libmpv_stt_plugin_rs.dylib ~/.config/mpv/scripts/libmpv_stt_plugin_rs.so
```

> mpv 按**文件名后缀**选择 C 插件后端,非 Windows 平台只认 `.so`,所以 macOS 上也要
> 把 Mach-O 产物改名成 `.so`。

macOS 更新正在被 IINA/mpv 加载的动态库时,不要直接覆盖原文件。先完全退出 IINA,
再通过临时文件原子替换,避免系统把映射中的 Mach-O 判定为签名页失效:

```bash
mkdir -p ~/.config/mpv/scripts
cp target/release/libmpv_stt_plugin_rs.dylib \
  ~/.config/mpv/scripts/.libmpv_stt_plugin_rs.so.new
mv ~/.config/mpv/scripts/.libmpv_stt_plugin_rs.so.new \
  ~/.config/mpv/scripts/libmpv_stt_plugin_rs.so
```

替换后重新启动 IINA。

## 快捷键

插件加载后会直接向当前 mpv/IINA 播放器实例注册以下强绑定,不需要再修改
IINA 的 `input.conf`:

| 快捷键 | 功能 |
|---|---|
| `Ctrl+Shift+S` | 开启/停止实时字幕;停止后可再次开启 |
| `Ctrl+Shift+T` | 开启/停止新字幕的自动翻译 |
| `Ctrl+Shift+C` | 清除当前媒体的字幕与翻译缓存 |

音频抽取和远程 STT 在独立 worker 中执行,不会占用 mpv/IINA 的事件线程。即使
服务端正在推理或失去响应,上述快捷键、切换文件和退出仍会立即处理;停止、seek 或
退出会取消旧请求,并用任务代次隔离迟到结果,避免旧视频字幕写进新会话。

关闭一个视频、停止一次字幕或一次远程请求失败只会结束当前转写会话,不会终止整个
插件;后续打开视频或再次按 `Ctrl+Shift+S` 会创建新会话。

也可以在 `input.conf` 里用消息路由(把 `<client>` 换成插件实例名):

```
Ctrl+Shift+S script-message-to <client> toggle-stt
Ctrl+Shift+T script-message-to <client> toggle-translate
Ctrl+Shift+C script-message-to <client> clear-cache
```

## 配置

配置文件为 `mpv_stt_plugin_rs.toml`,默认路径:

- macOS: `~/Library/Application Support/mpv/mpv_stt_plugin_rs.toml`
- Linux: `~/.config/mpv/mpv_stt_plugin_rs.toml`

可用环境变量 `MPV_STT_PLUGIN_RS_CONFIG=/path/to/file.toml` 覆盖路径;任何键都可用
`MPV_STT_PLUGIN_RS_<键>` 形式的环境变量覆盖。

### STT

```toml
[stt]
backend = "openai"           # openai(默认) | ferrum

[stt.openai]
server_addr = "https://api.groq.com/openai"   # 任意 OpenAI 兼容 /v1/audio/transcriptions
api_key = "..."              # 可选;设置后发 Authorization: Bearer {key}
model = "whisper-large-v3"   # multipart form 里的 model;必须是服务端提供的模型
language = "ja"              # 可选语言提示(ja/zh/en...);省略 = 服务端自动检测
timeout_ms = 120000
max_retry = 3

# 分段时间戳不需要配置:插件固定请求 response_format=verbose_json +
# timestamp_granularities[]=segment(标准 OpenAI 字段);服务端不支持时自动退化成
# 每块一条字幕。
# [stt.ferrum]
# server_addr = "http://127.0.0.1:8000"
# model = "sensevoice"       # 通过 x-model header 传给服务端
# language = "ja"            # 通过 x-language header 传;省略 = 自动检测
# use_opus = true
# enable_encryption = false
# encryption_key = "..."
# auth_secret = "..."
# timeout_ms = 120000
# max_retry = 3
```

ferrum 协议的服务端由 [subtitle-gateway](https://github.com/canxin121/subtitle-gateway)
(FunASR ASR + 翻译统一网关)实现,同一端点复用同一套 FunASR 引擎。

### 翻译

```toml
[translate]
backend = "deepl"             # deepl(默认) | libretranslate
from_lang = "ja"              # 内容语言(建议显式指定,避免 auto 把日文误判成中文)
to_lang = "zh"
concurrency = 4
server_addr = "http://127.0.0.1:8000"   # DeepL 兼容基址
api_key = ""                            # 网关 key(DeepL-Auth-Key)

[translate.libretranslate]
server_addr = "http://127.0.0.1:8000"
api_key = ""
```

`deepl` 协议期望 `POST {server}/v1/translate`,body
`{"text": [...], "target_lang": "ZH", "source_lang": "JA"}`(source_lang 省略 = auto),
响应 `{"translations": [{"text": "..."}]}`。

### 其他

```toml
[chunk]
local_ms = 15000              # 本地文件每个转写分片时长
network_ms = 15000            # 网络流分片时长

[playback]
show_progress = true
save_srt = true
auto_start = false            # 打开文件自动开始

[prefetch]
lookahead_chunks = 2

[network]
demuxer_max_bytes = 0         # 可选;网络流缓存上限
```

## 测试

```bash
./scripts/cargo-with-deps.sh test
```

`translate.rs` / `stt/openai.rs` 里标了 `#[ignore]` 的端到端测试需要本地跑一个
subtitle-gateway:

```bash
cargo test --lib -- --ignored translate_against_live_gateway

# STT 那条要真实语音(静音 WAV 转写不出内容),用 bench 语料或自己的录音:
MPV_STT_PLUGIN_RS_LIVE_AUDIO=/path/to/speech.wav \
  cargo test --lib -- --ignored openai_backend_against_live_server
```

## License

MIT
