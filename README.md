# mpv_stt_plugin_rs

mpv 的**实时字幕**插件(Rust 原生 C 插件)。播放时边抽音轨边送远程服务转写,再翻译成目标
语言,字幕实时显示并落盘成 `.srt`。

- **不跑本地推理** —— 音频分片送到任意 OpenAI 兼容或 ferrum 协议的 STT 服务
- **翻译零配置可用** —— 默认走内置免费源(Google / 微软 Edge / 阿里),也能接 DeepL、
  LibreTranslate 或自建网关
- **不阻塞播放** —— 抽取、请求、翻译都在独立 worker,seek / 停止 / 退出立即响应
- **单文件插件** —— 放一个 `.so` / `.dll` 进 mpv 的 `scripts` 目录即可,没有额外进程

[安装](#安装) · [配置](#配置) · [快捷键](#快捷键) · [日志](#日志) · [排错](#排错)

## 安装

### 下载预编译

从 [Releases](https://github.com/canxin121/mpv_stt_plugin_rs/releases) 下载对应平台的文件。
每个文件名都带平台前缀,去掉前缀就是原始文件名:

```bash
mkdir -p ~/.config/mpv/scripts
for f in linux-x86_64-*; do
  cp "$f" ~/.config/mpv/scripts/"${f#linux-x86_64-}"
done
```

| 平台 | 下载 | 怎么放 |
|---|---|---|
| Linux | `linux-x86_64-libmpv_stt_plugin_rs.so` + 同前缀的 `libav*.so.*` | 全部放 `~/.config/mpv/scripts/` |
| macOS | `darwin-arm64-libmpv_stt_plugin_rs.so`(Intel 机器取 `darwin-x86_64`) | 放 `~/.config/mpv/scripts/`;先 `brew install ffmpeg` |
| Windows | `windows-x86_64-mpv_stt_plugin_rs.dll` + 同前缀的 `*.dll` | 放同一目录,并把该目录加进 `PATH` |
| Android | `android-arm64-v8a-libmpv_stt_plugin_rs.so` | 放进 mpv-android 的脚本目录 |

> 文件名别改。mpv 按**后缀**认 C 插件,非 Windows 平台只认 `.so`;它同时用**加载的文件名**
> 当插件的 client 名,而快捷键的消息就是发给这个名字的。

macOS 上更新正在被 IINA 加载的插件时,**先完全退出 IINA**,再用临时文件原子替换 —— 直接覆盖
会让系统把映射中的 Mach-O 判成签名页失效:

```bash
cp 新产物 ~/.config/mpv/scripts/.libmpv_stt_plugin_rs.so.new
mv ~/.config/mpv/scripts/.libmpv_stt_plugin_rs.so.new \
   ~/.config/mpv/scripts/libmpv_stt_plugin_rs.so
```

### 从源码构建

```bash
git clone --recurse-submodules https://github.com/canxin121/mpv_stt_plugin_rs.git
cd mpv_stt_plugin_rs
./scripts/build-all.sh              # 当前能建的桌面平台全建一遍
```

产物在 `dist/<平台>/`。脚本自己解析 FFmpeg 开发前缀(macOS 用 brew,Linux / Windows 下载
[BtbN](https://github.com/BtbN/FFmpeg-Builds) 的 `lgpl-shared` 包到 `target/ffmpeg/`),所以
**不需要预先装 FFmpeg**,也不编 FFmpeg 源码。`-p darwin-arm64` 只建一个平台,`-l` 列出平台。

宿主上快速迭代:

```bash
./scripts/cargo-with-deps.sh build --release   # 自动拉 mpv 头文件、设好 FFMPEG_DIR
./scripts/cargo-with-deps.sh test              # 离线单测
```

Android 上没有现成的 libmpv / FFmpeg,脚本会现编(首次较慢):

```bash
export ANDROID_NDK_HOME=~/Android/Sdk/ndk/29.0.14206865   # NDK r29 或更新
./scripts/build-android.sh -a arm64-v8a
```

输出在 `dist/android/<abi>/`。默认只编 64 位:32 位 ABI 会卡在上游 `ffmpeg-sys-next` 的
Vulkan stub 上(它把 `sizeof(VkPhysicalDeviceFeatures2)` 硬编码成 240,只在 64 位下成立)。

单测不需要任何服务;标了 `#[ignore]` 的那几条要本地跑着 STT / 翻译服务,按需手动触发:

```bash
./scripts/cargo-with-deps.sh test                      # 离线单测
./scripts/gen-test-media.sh                            # 生成容器矩阵(target/testmedia)
./scripts/cargo-with-deps.sh test --lib -- --ignored   # 含端到端(要有在跑的服务)
```

`testdata/ja_all.mp4` 是一段 101.7 s 的日语素材,脚本把它转成 mp4/mkv/avi/webm/ts/flv 等常用
容器,用来验证"换个容器还灵不灵" —— 抽不出音轨的容器用户只会看到没有字幕。真播一遍:

```bash
MPV_STT_PLUGIN_RS_CONFIG=~/"Library/Application Support/mpv/mpv_stt_plugin_rs.toml" \
  ./scripts/e2e-media-matrix.sh
```

脚本只传配置文件路径、不读也不回显内容,每次播放都带 `--ao=null --vo=null`,**不会发出声音**。

推 `v*` tag 会触发 CI 构建四个平台并发 Release:文件**不打包**、每份都带平台前缀(前缀去掉就是
原始文件名,mpv 的 client 名才不会变),Linux 只发插件真正 `NEEDED` 的 SONAME 那一份 FFmpeg 库。

## 配置

配置文件是 `mpv_stt_plugin_rs.toml`:

| 平台 | 默认路径 |
|---|---|
| macOS | `~/Library/Application Support/mpv/mpv_stt_plugin_rs.toml` |
| Linux | `~/.config/mpv/mpv_stt_plugin_rs.toml` |

`MPV_STT_PLUGIN_RS_CONFIG=/path/to/file.toml` 可以换路径。扁平键都能用环境变量覆盖(键里的
`.` 写成 `_`):`MPV_STT_PLUGIN_RS_TRANSLATE_SOURCE=edge_free`、`MPV_STT_PLUGIN_RS_LOG_FILE=off`。

> `[stt.sources.<名字>]` 和 `[translate.sources.<名字>]` 里的字段**不能用环境变量覆盖**:源的
> 名字本身是键的一层,而名字里可能带 `_`(如 `google_free`),和环境变量表示层级的字符撞车。
> 改某个源的 `server_addr` / `api_key` 就直接改 toml。

### 最小配置

STT 必须至少声明一个源;翻译默认 `auto`,不写就能用:

```toml
[stt]
source = "groq"

[stt.sources.groq]
protocol = "openai"
server_addr = "https://api.groq.com/openai"
api_key = "<你的 key>"
model = "whisper-large-v3"
language = "ja"

[translate]
from_lang = "ja"
to_lang = "zh"
```

### STT

同一协议可以声明任意多个源,换服务端只改 `source` 一行:

```toml
[stt]
source = "groq"              # 只声明了一个源时可以留空

[stt.sources.groq]
protocol = "openai"          # openai | ferrum(必填)
server_addr = "https://api.groq.com/openai"   # 任意 OpenAI 兼容 /v1/audio/transcriptions
api_key = "..."              # 可选;设置后发 Authorization: Bearer {key}
model = "whisper-large-v3"   # 必须是服务端提供的模型 id
language = "ja"              # 可选语言提示;省略 = 服务端自动检测
timeout_ms = 120000
max_retry = 3

[stt.sources.gw]             # 同协议的第二个源:本地网关
protocol = "openai"
server_addr = "http://127.0.0.1:8000"
model = "sensevoice"
```

- **分段时间戳不用配**:插件固定请求标准字段 `response_format=verbose_json` +
  `timestamp_granularities[]=segment`;服务端不支持时退化成**每个音频块一条字幕**,而不是
  写一个空 SRT 覆盖已有字幕。
- `source` 留空 = 用唯一声明的那一个;声明了 0 个或多个而没选,启动时直接报错并列出已声明
  的名字,不会静默挑一个。
- `model` 写错会在第一个音频块上报 `Server error (404 Not Found): model_not_found`。

<details>
<summary><code>ferrum</code> 协议:同一套字段,另有几个只为它读的键</summary>

```toml
[stt.sources.local]
protocol = "ferrum"
server_addr = "http://127.0.0.1:9000"
model = "sensevoice"       # 通过 x-model header 传
language = "ja"            # 通过 x-language header 传;省略 = 自动检测
use_opus = true
enable_encryption = false
encryption_key = "..."
auth_secret = "..."
timeout_ms = 120000
max_retry = 3
```

`[stt.sources.<名字>]` 是两种协议的**并集**,扁平一层,`protocol` 决定读哪些:
`use_opus` / `enable_encryption` / `encryption_key` / `auth_secret` 只有 `ferrum` 读,
`api_key` 只有 `openai` 读。服务端由
[subtitle-gateway](https://github.com/canxin121/subtitle-gateway)(FunASR ASR + 翻译统一
网关)实现。

</details>

### 翻译

```toml
[translate]
source = "auto"               # auto(默认)| 任意一个源的名字
from_lang = "ja"              # 内容语言(建议显式指定,避免 auto 把日文误判成中文)
to_lang = "zh"
concurrency = 4
```

**`auto`(默认)零配置就能用**,按 `google_free → edge_free → alibaba_free` 顺序回退,第一个
成功为止。这三个名字是内置的,host 是固定的公共站点:

| `source` | `protocol` | 说明 |
|---|---|---|
| `google_free`(回退第 1) | `google` | 质量与速度均衡;限速按出口 IP 算 |
| `edge_free`(回退第 2) | `edge` | 实测最稳定、几乎不限速;带 `api_key` 即走微软官方通道 |
| `alibaba_free`(回退第 3) | `alibaba` | 每条字幕一次 token + 一次请求,链路最重,故排最后 |

想覆盖某个内置源的字段(比如给 Edge 填官方 key),只写写到的字段即可:

```toml
[translate.sources.edge_free]
api_key = ""
```

**其他服务**要自己声明一个源,写清 `protocol` 和 `server_addr`(名字随你取):

```toml
[translate]
source = "deepl_free"

[translate.sources.deepl_free]
protocol = "deepl"
server_addr = "https://api-free.deepl.com"
api_key = "<xxx:fx>"

[translate.sources.lt]
protocol = "libretranslate"
server_addr = "http://127.0.0.1:5000"
```

| 协议 | 请求形状 |
|---|---|
| `deepl` | `POST {server}/v1/translate`,key 走 `Authorization: DeepL-Auth-Key`,`target` 大写 |
| `libretranslate` | `POST {server}/translate`,key 走 body `api_key`,`target` 小写,`auto` 可显式/省略 |

常见接法:

| 服务 | 配置 |
|---|---|
| [subtitle-gateway](https://github.com/canxin121/subtitle-gateway) | `protocol = "deepl"`,`server_addr = "http://127.0.0.1:8000"`(ASR 与翻译同一端点) |
| [DeepL API Free](https://www.deepl.com/en/signup?cta=checkout&is_api=true&productId=api-developer) | `protocol = "deepl"`,`server_addr = "https://api-free.deepl.com"`,`api_key = "<xxx:fx>"` |
| [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) 自建 | `protocol = "libretranslate"`,`server_addr = "http://127.0.0.1:5000"` |
| 公共 LibreTranslate 镜像 | `protocol = "libretranslate"`,`server_addr = "https://translate.hostux.net"`,并**显式写 `from_lang`** |

> 少写 `protocol` 或 `server_addr` 都是启动错误(前者会在消息里列出可选的协议名),不会猜。
> 选中单个源时**失败不换源**,不会悄悄降级到别的引擎还让你以为用的是它。
> 内置源是这些站点**网页前端自己用的接口**,不保证长期可用;形状变了会在日志里留下带 HTTP
> 状态和响应体摘要的 `warn`,而不是静默给出空译文。

<details>
<summary>怎么注册一个 DeepL API Free(免费档,不绑卡)</summary>

1. 打开 [注册页](https://www.deepl.com/en/signup?cta=checkout&is_api=true&productId=api-developer)。
   **先用无痕窗口,或先退出已登录的 DeepL 账号** —— 已登录时这个链接会退化成普通翻译账号
   注册,注册完在账号页里找不到 API key。
2. 邮箱 + 密码注册,套餐选 **API Developer**。注册完要做一次邮箱验证,不验证 key 用不了。
3. 去 [账号 → API keys](https://www.deepl.com/en/your-account/keys) 复制 key,免费档结尾带 `:fx`。
4. 填进配置(`server_addr` 固定 `https://api-free.deepl.com`,与 key 无关)。

额度 100 万字符/月,超了不扣费、只停到下个月,用量见
[账号 → Usage](https://www.deepl.com/en/your-account/usage)。

</details>

已失效、别浪费时间配的:`libretranslate.com` 官方站(已无免费 key)、Lingva /
SimplyTranslate 公共实例(被 Cloudflare 拦或返回空译文)、MyMemory(整 IP 共享每日 5000
字符)。

### 其他

```toml
[chunk]
local_ms = 15000              # 本地文件每个转写分片时长
network_ms = 15000            # 网络流分片时长

[playback]
show_progress = true          # 屏幕上显示转写进度
save_srt = true               # 字幕自动落盘成 .srt(与媒体同目录)
auto_start = false            # 打开文件自动开始

[prefetch]
lookahead_chunks = 2          # 预取几个后续分片

[network]
demuxer_max_bytes = 0         # 可选;网络流 demuxer 缓存上限
```

### 环境变量速查

| 变量 | 作用 |
|---|---|
| `MPV_STT_PLUGIN_RS_CONFIG` | 换配置文件路径 |
| `MPV_STT_PLUGIN_RS_LOG` | 日志过滤指令,覆盖 `log.level` 且对三个通道一律生效;也接受 target 语法 |
| `MPV_STT_PLUGIN_RS_<键>` | 覆盖任意扁平键,键里的 `.` 写成 `_`,如 `MPV_STT_PLUGIN_RS_LOG_FILE=off` |

## 快捷键

| 快捷键 | 功能 |
|---|---|
| `Ctrl+Shift+S` | 开启/停止实时字幕;停止后可再次开启 |
| `Ctrl+Shift+T` | 开启/停止新字幕的自动翻译 |
| `Ctrl+Shift+C` | 清除当前媒体的字幕与翻译缓存 |

插件加载后会把这些强绑定直接注册到 mpv 的输入引擎,**这只对命令行 mpv 生效**。IINA 不走
mpv 的输入引擎:它自己查 `input_conf` 里的表,命中哪一行就把那行的 mpv 命令原样执行。所以在
IINA 里按键走的始终是**你自己那张表**(设置 → 快捷键),要加上这三行:

```
Ctrl+Shift+S script-message-to libmpv_stt_plugin_rs toggle-stt
Ctrl+Shift+T script-message-to libmpv_stt_plugin_rs toggle-translate
Ctrl+Shift+C script-message-to libmpv_stt_plugin_rs clear-cache
```

目标是插件的 **client 名**,即加载时那个文件去掉 `.so`。改完要**重启 IINA**(`input_conf`
只在启动时读一次)。

> 表里条目**存在但目标名字写错**时,按键会静默失效:命令打到不存在的客户端上,只在 IINA
> 自己的日志里留一行,界面上没有任何提示,插件侧连日志都不会有。
>
> 另外,可打印字符上的 `Shift` 会被折叠,`Ctrl+Shift+S` 和 `Ctrl+S` 是同一个绑定,别再写一份。

### 运行时的行为

- 音频抽取和远程请求在独立 worker 中执行,不占用 mpv / IINA 的事件线程:服务端正在推理或
  失去响应时,快捷键、切换文件、退出仍然立即响应。停止、seek、退出会取消旧请求,迟到的结果
  不会写进新会话。
- 关闭视频、停止字幕、一次请求失败都只结束当前转写会话,不会终止插件;再次打开视频或按
  `Ctrl+Shift+S` 会创建新会话。
- **翻译失败不影响字幕本身**:识别结果先落地并显示,翻译只是在其后追加一行。翻译服务不可用
  时原文照常显示、照常写盘,屏幕上只提示一次失败原因。失败的条目不再重复投递(否则每次 seek
  都会重试一遍),服务恢复后按 `Ctrl+Shift+T` 关再开、或 `Ctrl+Shift+C` 清缓存即会重新翻译。

## 日志

日志走 `tracing`,三个通道同时收到同一条记录:

| 通道 | 用途 | 格式 |
|---|---|---|
| stderr | 终端里手敲 `mpv` 时盯着看 | compact 单行,仅 TTY 上色 |
| 文件 | 从 Finder 启动 IINA 时唯一能事后翻的记录(`stderr` 会被丢弃) | 默认 compact,可换 full/json,带轮转 |
| mpv OSD | 出错时不用翻日志就能看见 | 只画 `info` 及以上、且带 `display` 字段的记录 |

```toml
[log]
level = "info"            # EnvFilter 语法;裸级别(如 "debug")= 本插件,不含依赖
format = "compact"        # compact(默认)| full | json
file = "auto"             # "auto" = 与配置文件同目录的 mpv_stt_plugin_rs.log;"" 关闭
file_level = "debug"      # 文件里保留到哪一级(比终端更详细,便于事后排查)
file_max_files = 5        # 轮转保留份数(按天)
ansi = ""                 # "" = 自动;true/false 强制
osd = true                # 是否把带 display 字段的记录送到 mpv OSD
```

- **默认写文件**,这正是"GUI 里看不到日志"的解药。文件按天轮转,保留 `file_max_files` 份;
  写不进去(只读目录)时只提示一行并跳过,不影响终端与 OSD。`file = ""` 一行关掉。
- **环境变量优先**:`MPV_STT_PLUGIN_RS_LOG` 覆盖 `log.level`,所以 `MPV_STT_PLUGIN_RS_LOG=debug
  mpv …` 一个词就能整体开到 debug;它也接受 target 语法,例如只打开某个子系统:
  `MPV_STT_PLUGIN_RS_LOG="mpv_stt_plugin_rs::stt=trace,warn"`。
- **依赖的日志默认丢弃**:裸级别只作用于 `mpv_stt_plugin_rs`,否则 hyper 的逐连接日志会把
  插件自己的行埋掉。要看 HTTP 层得显式点名:`MPV_STT_PLUGIN_RS_LOG="hyper_util=trace"`。
- **只记非敏感字段**:`Config` 里的 api_key / encryption_key / auth_secret 任何时候都不会被
  整体打印。
- OSD 那一行由记录里的 `display` 字段决定,而不是日志消息本身:日志说开发者看的话,屏幕说
  用户看的话。没有 `display` 的 `warn` 只进日志不进屏幕;连续重复的会合并成 `(xN)`。

## 排错

| 现象 | 先看这里 |
|---|---|
| 完全没有字幕 | 开 `MPV_STT_PLUGIN_RS_LOG=debug`,看配置文件同目录的 `mpv_stt_plugin_rs.log` |
| 按键没反应 | IINA 下 `input_conf` 里的目标名字要和插件文件名对得上;改了要重启 IINA |
| 第一个音频块报 404 | `model` 不是服务端提供的 id |
| 字幕有原文没译文 | 翻译服务的问题(限速、网关没起、key 不对);原文和 SRT 不受影响 |
| Linux / Windows 起不来 | FFmpeg 动态库要和插件放同一目录(Linux 还要在 `LD_LIBRARY_PATH` 里) |
| macOS 覆盖插件后异常 | 退出 IINA 后用临时文件原子替换,别直接覆盖 |
| 日志文件在哪 | 和配置文件同目录;`file = ""` 时没有文件,只走终端与 OSD |

## License

MIT
