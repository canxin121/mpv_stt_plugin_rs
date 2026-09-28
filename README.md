# mpv_stt_plugin_rs

MPV 实时字幕插件(Rust 原生 C 插件)。插件**不跑任何本地推理**:音频抽取后送到远程
STT 服务转写,再送翻译服务翻译。翻译既可以接自建/外部服务,也可以**零配置直接用内置的
免费源**(Google / 微软 Edge / 阿里的网页接口,见下文)。

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
│   ├── translate.rs         # 翻译客户端(内置免费源 + DeepL 兼容 + LibreTranslate)
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

所有后端**同时编译、运行时选择**(`config.translate.backend`)。三个内置免费源不需要
任何配置或 key,`auto`(默认)按 `google → edge → alibaba` 顺序回退,第一个成功为止:

| backend | 配置字段 | 协议 |
|---|---|---|
| `auto`(默认) | — | 依次试下面三个内置源,第一个成功为止 |
| `google` / `edge` / `alibaba` | `[translate.google]` / `[translate.edge]` / `[translate.alibaba]` | 只用这一个内置源(失败不换源);host 已内置,通常整节都不用写 |
| `deepl` | `[translate]` 平铺 `server_addr`/`api_key` | `POST {server}/v1/translate`,key 走 `Authorization: DeepL-Auth-Key`,target 大写 |
| `libretranslate` | `[translate.libretranslate]` | `POST {server}/translate`,key 走 body `api_key`,target 小写,`auto` 可显式/省略 |

内置源是这些站点**网页前端自己用的接口**,不保证长期可用、也不保证稳定:
它们的形状一旦变了,日志里会出现带 HTTP 状态和响应体摘要的 `warn`,而不是静默给出空译文。

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

**翻译失败不影响字幕本身。** 识别结果先落地并显示,翻译只是在其后追加一行:

- 翻译服务不可用(内置源被限速、网关没起、key 不对、上游 503…)时,原文照常显示、
  照常写盘,只是没有译文;屏幕上提示一次失败原因,不会每块都弹。
  `backend = "auto"` 会先自己换一个内置源试,三个都不行才提示。
- 失败的那几条不再重复投递(否则每次 seek 都会重试一遍),恢复服务后按
  `Ctrl+Shift+T` 关再开(或 `Ctrl+Shift+C` 清缓存)即会重新翻译已有字幕。
- 翻译相关的问题永远不会结束转写会话,`Ctrl+Shift+S` 的开关状态不受影响。

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
`MPV_STT_PLUGIN_RS_<键>` 形式的环境变量覆盖(键里的 `.` 写成 `_`,如 `MPV_STT_PLUGIN_RS_LOG_FILE=off`)。

`MPV_STT_PLUGIN_RS_LOG` 是个例外:它不是配置键,而是日志过滤指令,详见[日志](#日志)。

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
backend = "auto"              # auto(默认) | google | edge | alibaba | deepl | libretranslate
from_lang = "ja"              # 内容语言(建议显式指定,避免 auto 把日文误判成中文)
to_lang = "zh"
concurrency = 4
server_addr = "http://127.0.0.1:8000"   # DeepL 兼容基址
api_key = ""                            # 网关 key(DeepL-Auth-Key)

[translate.libretranslate]
server_addr = "http://127.0.0.1:8000"
api_key = ""

# 三个内置免费源:host 已内置,整节不写就是下面这些值
[translate.google]
server_addr = "https://clients5.google.com"
api_key = ""                  # 预留,当前查询串里不带

[translate.edge]
server_addr = "https://edge.microsoft.com"
api_key = ""                  # 填了就走微软官方通道

[translate.alibaba]
server_addr = "https://translate.alibaba.com"
api_key = ""                  # 预留
```

`deepl` 协议期望 `POST {server}/v1/translate`(`server_addr` 写基址 `https://api-free.deepl.com`,
路径由插件补),body
`{"text": [...], "target_lang": "ZH", "source_lang": "JA"}`(source_lang 省略 = auto),
响应 `{"translations": [{"text": "..."}]}`。

### 内置免费源

`backend = "auto"`(默认)就能直接翻:三个网页接口都在插件里用 Rust 实现,不需要外部进程、
不需要注册、不需要 key。它是插件里唯一"连出去"的地方,发出去的只有待译的字幕文本。

| backend | 端点 | 批量 | 说明 |
|---|---|---|---|
| `google`(回退第 1) | `GET clients5.google.com/translate_a/t` | 原生多 `q` | 质量与速度均衡;限速按出口 IP 算,重度使用会吃到 |
| `edge`(回退第 2) | `POST edge.microsoft.com/translate/translatetext` | 原生 JSON 数组 | 实测最稳定、几乎不限速;带 `api_key` 即走微软官方通道 |
| `alibaba`(回退第 3) | `POST translate.alibaba.com` | 无 | 每条字幕一次 token + 一次请求,链路最重,故排最后 |

选中单个源时**失败不换源**(不会悄悄降级到质量更差的引擎还让你以为用的是它);
只有 `auto` 会按上表顺序回退,全失败才走"翻译放弃"提示。

三者都是站点前端自己用的接口,`to_lang` 都能直接写 `zh`(阿里只认它自己的码集,
写 `zh-Hans` 会被拒,所以插件只发语言主标签)。

### 外部翻译服务

改 `server_addr`/`api_key`/`from_lang`/`to_lang` 即可,不需要改代码 —— 前提是对方讲的是
DeepL 或 LibreTranslate 这两种形状之一(2026-09 实测核对,服务端随时可能变)。

| 服务 | 怎么连 | 说明 |
|---|---|---|
| [subtitle-gateway](https://github.com/canxin121/subtitle-gateway) | `backend = "deepl"`,`server_addr = "http://127.0.0.1:8000"` | 本仓库配套网关,ASR 与翻译同一端点 |
| [DeepL API Free](https://www.deepl.com/en/signup?cta=checkout&is_api=true&productId=api-developer) | `backend = "deepl"`,`server_addr = "https://api-free.deepl.com"`,`api_key = "<xxx:fx>"` | 免费档叫 **API Developer**:100 万字符/月、1 个 key;免费 key 带 `:fx` 后缀,所以 endpoint 是 `api-free` 而不是 `api`;日译中质量最好;国内可直连 |
| [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) | `backend = "libretranslate"`,`server_addr = "http://127.0.0.1:5000"` | 自建:不限量、不出内网。`pip install libretranslate` 或官方 Docker 镜像;默认监听 `127.0.0.1:5000`;AGPL-3.0;Argos 引擎,日译中绕英语 |
| 公共 LibreTranslate 镜像 | `backend = "libretranslate"`,`server_addr = "https://translate.hostux.net"` | 无需 key,但**必须显式写 `from_lang`**(镜像不接受省略 `source`);`to_lang` 只能写 `zh` 或 `zh-Hans`,写 `zh-CN` 会 400;Argos 引擎,质量明显低于内置源 |

需要绑定信用卡才能开通、或整条链路要额外跑一个服务端(Google Cloud / Azure Translator /
[MTranServer](https://github.com/xxnuo/MTranServer))的,请求形状既不是 DeepL 也不是
LibreTranslate,光改 `server_addr` 接不上 —— 内置源已经覆盖了这几家的免费档。

已失效、不要再配的:**`libretranslate.com` 官方站**(已无免费 key,最低 $14/月)、
**Lingva / SimplyTranslate 公共实例**(要么被 Cloudflare 拦,要么返回空译文)、
**MyMemory**(整个 IP 共享每日 5000 字符额度,单次查询上限 500 字符,极易耗尽)。

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

## 日志

日志走 `tracing`,装配点只有 `src/logging.rs`。三个通道同时收到同一条记录:

| 通道 | 用途 | 格式 |
|---|---|---|
| stderr | 终端里手敲 `mpv` 时盯着看 | compact 单行,仅 TTY 上色 |
| 文件 | 从 Finder 启动 IINA 时唯一能事后翻的记录(`stderr` 会被丢弃) | 默认 compact,可换 full/json,带轮转 |
| mpv OSD | 出错时不用翻日志就能看见 | 只画 `info` 及以上、且带 `display` 字段的记录 |

### 级别

- **error** —— 用户要的事做不成、只能收摊:分片失败终止会话、字幕落盘失败、worker 不可用、FFI 调用失败。
- **warn** —— 降级但还能继续:重试、翻译放弃、某个内置免费源失败后换下一个、空结果、
  缓存文件读写失败、manifest 损坏后重新转写。
- **info** —— 每个会话/每次操作一条的里程碑:生效配置、插件加载、进入本地/网络模式、字幕路径、缓存命中、设备提示、翻译后端不响应。
- **debug** —— 每块/每请求的生命周期:调度、提交、HTTP 结果摘要、seek 判定、mpv 事件、后端选择。
- **trace** —— 热循环里的空转与逐条判定:等缓存、等播放追上、look-ahead 上限、迟到结果丢弃、单条文本翻译。

### 上下文与字段

一次媒体会话是一个 `session` span(`session` 自增 id、`media`、`duration_ms`、`mode`),每块音频是它下面的
`chunk` span(`seq`、`start_ms`、`dur_ms`、`gen`)。所以一块音频从调度、抽音频、HTTP 请求到字幕落地,
跨线程也带着同一份身份:

```
DEBUG session{session=1 media=…}:chunk{session=1 seq=0 start_ms=0 dur_ms=15000 gen=0}: mpv_stt_plugin_rs::stt::openai: chunk transcribed segments=2 entries=2
```

`start_ms` / `dur_ms` / `entries` / `bytes` / `status` / `wall_ms` / `gen` 都是字段而非句子的一部分,
方便 `grep`、聚合和事后按值过滤。`format = "full"` 时每个 span 还会单独打一行开/闭,带
`time.busy` / `time.idle`(由 `tracing` 计时,不用手写 `Instant::now()`)。

### 配置

```toml
[log]
level = "info"            # EnvFilter 语法;裸级别(如 "debug")= 本插件,不含依赖
format = "compact"        # compact(默认) | full | json
file = "auto"             # "auto" = 与配置文件同目录的 mpv_stt_plugin_rs.log;"" 关闭
file_level = "debug"      # 文件里保留到哪一级(比终端更详细,便于事后排查)
file_max_files = 5        # 轮转保留份数(按天)
ansi = ""                 # "" = 自动;true/false 强制
osd = true                # 是否把带 display 字段的记录送到 mpv OSD
```

- **默认写文件**:这正是"GUI 里看不到日志"的解药。文件按天轮转,保留 `file_max_files` 份,
  写不进去(只读目录)时只打印一行提示并跳过,不影响终端与 OSD。`file = ""` 一行关掉。
- **环境变量优先**:`MPV_STT_PLUGIN_RS_LOG` 覆盖 `log.level`,且对三个通道一律生效(包括文件),
  所以 `MPV_STT_PLUGIN_RS_LOG=debug mpv …` 一个词就能整体开到 debug。它同时接受 target 语法,
  例如只打开某个子系统:`MPV_STT_PLUGIN_RS_LOG="mpv_stt_plugin_rs::stt=trace,warn"`。
- **依赖的日志默认丢弃**:裸级别只作用于 `mpv_stt_plugin_rs`。`debug` 是"本插件verbose",不是
  "把进程里链接进来的所有库都打开" —— 否则 hyper 的逐连接日志会把插件自己的行埋掉。要看 HTTP 层
  得显式点名:`MPV_STT_PLUGIN_RS_LOG="hyper_util=trace"`。
- **只记非敏感字段**:生效配置摘要是逐字段手写的;`Config` 里有 api_key / encryption_key / auth_secret,
  任何时候都不会被整体打印。

OSD 那一行由记录里的 `display` 字段决定,而不是日志消息本身:日志说开发者看的话
(`chunk failed; ending the session`),屏幕说用户看的话(`STT failed: …`)。没有 `display` 字段的
`warn` 是过程噪音(正在重试、缓存写失败),只进日志不进屏幕;连续重复的会合并成 `(xN)`,
一次最多画 3 条,不会跟进度文字抢屏。

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
