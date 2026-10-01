<div align="center">

# mpv_stt_plugin_rs

**边播放，边生成双语字幕。**

为 mpv / IINA 提供语音转写、字幕翻译与 SRT 保存。

[下载插件](https://github.com/canxin121/mpv_stt_plugin_rs/releases/latest) · [快速开始](#快速开始) · [按需配置](#按需配置) · [常见问题](#常见问题)

</div>

---

音频由你配置的转写服务处理，识别出的文本交给翻译服务。插件不在本地运行识别模型；翻译可直接使用内置免费源。

## 快速开始

### 1. 安装插件

从 [Releases](https://github.com/canxin121/mpv_stt_plugin_rs/releases/latest) 下载对应平台的文件，再按下方步骤安装。**安装或更新前，请先退出播放器。**

<details>
<summary><strong>macOS · Apple Silicon</strong></summary>

下载 `darwin-arm64-libmpv_stt_plugin_rs.so`，在下载目录执行：

```bash
brew install ffmpeg
mkdir -p ~/.config/mpv/scripts
cp darwin-arm64-libmpv_stt_plugin_rs.so ~/.config/mpv/scripts/.libmpv_stt_plugin_rs.so.new
mv ~/.config/mpv/scripts/.libmpv_stt_plugin_rs.so.new ~/.config/mpv/scripts/libmpv_stt_plugin_rs.so
```

IINA 用户还需在「设置 → 高级」中开启高级设置，勾选「Use config directory」，并指定为 `~/.config/mpv`。

</details>

<details>
<summary><strong>Linux · x86_64</strong></summary>

下载 `linux-x86_64-libmpv_stt_plugin_rs.so`，在下载目录执行：

```bash
mkdir -p ~/.config/mpv/scripts
cp linux-x86_64-libmpv_stt_plugin_rs.so ~/.config/mpv/scripts/libmpv_stt_plugin_rs.so
```

还需安装 **FFmpeg 9 共享库**，只有 `ffmpeg` 可执行文件不够。可从 [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases/latest) 下载 `n9.0`、`linux64`、`shared` 版本；系统未提供匹配库时，用解压后的 `lib` 目录启动 mpv：

```bash
LD_LIBRARY_PATH="/path/to/ffmpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" mpv "视频文件.mkv"
```

将 `/path/to/ffmpeg/lib` 换成实际路径。

</details>

<details>
<summary><strong>Windows · x86_64</strong></summary>

1. 下载所有以 `windows-x86_64-` 开头的 `.dll` 文件，并去掉这个前缀。
2. 将 `mpv_stt_plugin_rs.dll` 放入 `%APPDATA%\mpv\scripts\`，目录不存在时自行创建。
3. 将其余 FFmpeg DLL 放入 `mpv.exe` 所在目录，**不要放进 `scripts`**。

使用 mpv 的便携配置时，插件放入 `portable_config\scripts\`。

</details>

<details>
<summary><strong>Android · arm64-v8a</strong></summary>

下载 `android-arm64-v8a-libmpv_stt_plugin_rs.so`。

**先把文件重命名为 `libmpv_stt_plugin_rs.so` 再安装。** mpv 用文件名给插件起客户端名（去掉扩展名，其余非字母数字的字符全部换成 `_`），`script-message-to` 按这个名字找插件。保留下载时的完整文件名，插件会以 `android_arm64_v8a_libmpv_stt_plugin_rs` 注册，按文档写出的 `script-message-to` 就匹配不到，而播放器只把这条记成一行 verbose 日志。

需要支持 C 插件、且提供匹配 libmpv / FFmpeg 库的播放器。安装目录与加载方式由宿主决定，不能按普通 Lua 脚本的方式直接套用桌面安装步骤。

**mpvEx**：设置 → 高级 → C 插件里开启开关并选中重命名后的 `.so`，App 会在播放器启动前把它复制到 libmpv 的脚本目录。

加载成功后插件会自己 `define-section` + `enable-section` 注册三组快捷键，自己打印一行 `registered the forced shortcut section`，这一步正常时不需要再配 `input.conf`。若该行没出现，或快捷键与输入法冲突，再在配置目录的 `input.conf` 里写一份；`script-message-to` 的客户端名以 mpvEx 日志里 `Synced C plugin: <文件名> (client name: <客户端名>)` 那条为准：

```text
Ctrl+Shift+S script-message-to libmpv_stt_plugin_rs toggle-stt
Ctrl+Shift+T script-message-to libmpv_stt_plugin_rs toggle-translate
Ctrl+Shift+C script-message-to libmpv_stt_plugin_rs clear-cache
```

插件的加载失败与它自己打印的日志都在同一个日志页面里（设置 → 高级 → 日志）。

Android 上没有插件能自动找到的配置目录（`BaseDirs` 拿不到），所以配置文件必须靠宿主的环境变量 `MPV_STT_PLUGIN_RS_CONFIG` 指定绝对路径；mpvEx 在「设置 → 高级 → 环境变量」里加即可。

</details>

其他架构可参考[源码构建脚本](scripts/build-all.sh)。

### 2. 填写配置

创建 `mpv_stt_plugin_rs.toml`：

| 系统 | 默认位置 |
| :--- | :--- |
| macOS | `~/Library/Application Support/mpv/mpv_stt_plugin_rs.toml` |
| Linux | `~/.config/mpv/mpv_stt_plugin_rs.toml` |
| Windows | `%APPDATA%\mpv\mpv_stt_plugin_rs.toml` |

也可用环境变量 `MPV_STT_PLUGIN_RS_CONFIG` 指定配置文件的完整路径。

以下示例使用 **Groq 转写日语、内置免费源翻译成中文**。填入你自己的 [Groq API Key](https://console.groq.com/keys) 即可：

```toml
[stt]
source = "groq"

[stt.sources.groq]
protocol = "openai"
server_addr = "https://api.groq.com/openai"
api_key = "填入你的 API Key"
model = "whisper-large-v3"
language = "ja"

[translate]
source = "auto"
from_lang = "ja"
to_lang = "zh"
```

- **换原文语言**：同时修改 `language` 和 `from_lang`，例如英语 `en`、日语 `ja`。
- **换译文语言**：修改 `to_lang`，例如中文 `zh`、英语 `en`。
- **使用其他服务**：见[按需配置](#按需配置)。修改配置后重启播放器。

### 3. 开始使用

打开视频，按 **Ctrl + Shift + S** 开始生成字幕。首次出字需要等待第一个音频分片处理完成。

| 快捷键 | 操作 |
| :--- | :--- |
| `Ctrl+Shift+S` | 开始 / 停止转写 |
| `Ctrl+Shift+T` | 开启 / 关闭新字幕的翻译 |
| `Ctrl+Shift+C` | 清除当前媒体的字幕与翻译缓存 |

<details>
<summary><strong>IINA 用户：先配置快捷键</strong></summary>

在「设置 → 快捷键」中添加对应的 mpv 命令，或将以下三行写入当前使用的快捷键配置文件，然后重启 IINA：

```text
Ctrl+Shift+S script-message-to libmpv_stt_plugin_rs toggle-stt
Ctrl+Shift+T script-message-to libmpv_stt_plugin_rs toggle-translate
Ctrl+Shift+C script-message-to libmpv_stt_plugin_rs clear-cache
```

插件文件名须为 `libmpv_stt_plugin_rs.so`。若快捷键有冲突，请更换组合键。

</details>

本地视频的字幕默认保存为同目录、同名的 `.srt` 文件；已有同名字幕请先备份。翻译服务暂时失败时，识别出的原文仍会显示，插件会自动退避重试。

## 按需配置

下面的示例用于**替换对应配置段**，不要重复添加 `[stt]` 或 `[translate]`。源名可以自取，用 `source` 选择。

<details>
<summary><strong>更换转写服务</strong> · OpenAI 兼容接口 / 自建网关</summary>

以本地 [subtitle-gateway](https://github.com/canxin121/subtitle-gateway) 为例：

```toml
[stt]
source = "local"

[stt.sources.local]
protocol = "openai"
server_addr = "http://127.0.0.1:8000"
model = "sensevoice"
language = "ja"
```

使用其他 OpenAI 兼容服务时，替换 `server_addr` 和 `model`；需要鉴权时添加 `api_key`。模型名称必须是服务端实际提供的 ID。为了让每句话按识别时间显示，服务端还必须支持 `verbose_json` 并返回带 `start` / `end` 的 `segments`；只返回整段 `text` 的模型无法提供句子时间，插件会报错而不会猜一个整片时间。

</details>

<details>
<summary><strong>使用 ferrum 转写服务</strong></summary>

```toml
[stt]
source = "local"

[stt.sources.local]
protocol = "ferrum"
server_addr = "http://127.0.0.1:9000"
model = "sensevoice"
language = "ja"
use_opus = true
```

若服务端启用鉴权，填写 `auth_secret`；启用加密时，添加 `enable_encryption = true` 和 `encryption_key`。这些值需与服务端一致。

</details>

<details>
<summary><strong>更换翻译服务</strong> · 免费源 / LibreTranslate / DeepL 兼容网关</summary>

免费翻译无需 API Key，修改 `[translate]` 中的 `source` 即可：

| `source` | 使用方式 |
| :--- | :--- |
| `auto` | 默认；依次尝试 Google → Edge → 阿里 |
| `google_free` | 仅使用 Google |
| `edge_free` | 仅使用微软 Edge |
| `alibaba_free` | 仅使用阿里 |

固定选择一个源时，失败不会切换到其他源。免费网页接口可能限流或暂时不可用。

自建 LibreTranslate 示例：

```toml
[translate]
source = "local"
from_lang = "ja"
to_lang = "zh"

[translate.sources.local]
protocol = "libretranslate"
server_addr = "http://127.0.0.1:5000"
```

自定义源必须填写 `protocol` 和 `server_addr`；需要鉴权时添加 `api_key`。

使用 [subtitle-gateway](https://github.com/canxin121/subtitle-gateway) 等 DeepL 兼容网关时，将上例改为 `protocol = "deepl"`，并填写网关地址，例如 `http://127.0.0.1:8000`。该服务需提供 `/v1/translate` 接口。

</details>

<details>
<summary><strong>自动开始、字幕保存与分片时长</strong></summary>

```toml
[playback]
auto_start = true      # 打开视频后自动开始转写
save_srt = true        # 保存字幕文件
show_progress = true  # 显示处理进度

[chunk]
local_ms = 15000       # 本地文件：每片 15 秒
network_ms = 15000     # 网络媒体：每片 15 秒
```

缩短分片可减少等待，但会增加请求次数。只想查看字幕、不保留 SRT 时，设置 `save_srt = false`。

</details>

## 常见问题

桌面版日志默认位于**配置文件同目录**，文件名格式为 `mpv_stt_plugin_rs.log.YYYY-MM-DD`。排查时先查看当天日志；分享前请检查并移除敏感信息。

| 问题 | 检查方法 |
| :--- | :--- |
| 播放后没有字幕 | 默认不会自动开始，先按 `Ctrl+Shift+S`；再检查配置路径和转写服务是否可用。 |
| OpenAI 兼容服务报“没有有效的分段时间戳” | 确认所选模型和服务端支持 `verbose_json` 的带时间戳 `segments`。纯文本响应无法与每句话对齐。 |
| 旧字幕仍然错位 | 网络字幕缓存会在本版自动重算；本地同名 `.srt` 会作为已有字幕缓存复用，先备份后移走旧文件，再重新转写。 |
| IINA 快捷键无效 | 确认已启用 mpv 配置目录、添加快捷键、保留正确插件文件名，并重启 IINA。 |
| 服务返回 401 / 403 | 检查 API Key、权限与服务端鉴权配置。 |
| 服务返回 404 | 检查服务地址和模型名称，不要把完整请求路径填入 `server_addr`。 |
| 有原文，没有译文 | 检查翻译开关和服务状态；插件会自动重试未翻译字幕，服务恢复后也可关闭再开启 `Ctrl+Shift+T` 立即重试。 |
| 转写请求失败 | 检查服务地址、模型和鉴权；当前分片会自动重试，`429` 限流会使用更长的退避间隔。 |
| 插件无法加载 | 确认播放器支持 C 插件、文件架构正确，且 FFmpeg 共享库已安装并可被加载。 |
| 报 `cannot locate symbol "avcodec_..."` | 插件与宿主播放器的 FFmpeg 主版本不一致。桌面版下载与播放器同版本的 FFmpeg 共享库（Linux 见上文的 `LD_LIBRARY_PATH`）；Android 的库由播放器自带、无法替换，需换用与宿主 FFmpeg 主版本一致的插件版本。 |
| 报 `cannot locate symbol "mpv_..."` | 宿主播放器的 libmpv 比插件编译时更旧。升级播放器，或改用更早的插件版本。 |
| 快捷键无效 / 按了没有反应 | 插件加载后会自己注册一组快捷键（日志里的 `registered the forced shortcut section`），正常情况下不需要用户配置。若仍未响应，先确认插件名是 `libmpv_stt_plugin_rs.so`——mpv 用文件名生成客户端名，文件名不同，`script-message-to` 就匹配不到且不会报错。 |
| mpvEx 上快捷键无效 | 设置 → 高级：开启 C 插件并选中重命名后的 `.so`；插件加载失败或它自己的日志都在设置 → 高级 → 日志里。若插件自带的快捷键与系统输入法冲突，可把那三行写进配置目录的 `input.conf` 取代。 |

---

MIT License
