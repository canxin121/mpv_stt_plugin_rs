# testdata

`ja_all.mp4` —— 一段 101.7 s 的测试素材:640×360 的纯黑画面(10 fps)+ 16 kHz 单声道音轨。
音轨是七段日语语料(短句、长句、绕口令、俳句、专业术语、快语速、长句+数字+外来语)各跟
一段 0.8 s 静音拼起来的,语料由 `subtitle-gateway` 仓库的 `scripts/gen-bench-audio.py`
生成,文本见那边的 `bench/audio/ja/models.json`。

保留它是为了让「换一种容器还灵不灵」可复现:`scripts/gen-test-media.sh` 用它派生出
mp4/mkv/avi/mov/webm/ts/flv/wmv/mpg 与 mp3/m4a,`src/audio.rs` 的
`audio_extraction_covers_every_container` 与 `scripts/e2e-media-matrix.sh` 都跑在这套
派生格式上。派生文件生成到 `target/testmedia/`,不进仓库。