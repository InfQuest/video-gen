# Harry Potter Transformer - 使用说明

## 命令行参数使用方式

`transformer.py` 现已支持完整的命令行参数配置，无需修改代码即可灵活使用。

### 基本用法

```bash
# 默认处理第一章（使用默认设置）
uv run python transformer.py

# 查看所有可用参数
uv run python transformer.py --help
```

### 常用示例

#### 1. 处理单个章节（完整流程 + 视频）

```bash
uv run python transformer.py \
  --chapters "1" \
  --output-dir ./output
```

#### 2. 处理多个章节

```bash
uv run python transformer.py \
  --chapters "1,2,3" \
  --output-dir ./output
```

#### 3. 处理所有章节

```bash
uv run python transformer.py \
  --chapters "all" \
  --output-dir ./output
```

#### 4. 生成 30 秒预览视频

```bash
uv run python transformer.py \
  --chapters "1" \
  --preview-duration 30.0 \
  --output-dir ./output
```

#### 5. 快速模式（跳过校正、翻译、视频）

```bash
uv run python transformer.py \
  --chapters "1" \
  --no-correct \
  --no-translate \
  --no-video \
  --output-dir ./output
```

#### 6. 只做转录和对齐（不生成视频）

```bash
uv run python transformer.py \
  --chapters "1,2,3" \
  --no-scenes \
  --no-scene-images \
  --no-video \
  --output-dir ./output
```

#### 7. 音频编码优化（B站上传）

```bash
# 转换为 2 声道高码率 AAC（防止 B 站二压失真）
uv run python transformer.py \
  --chapters "1" \
  --audio-codec aac \
  --audio-bitrate 320k \
  --audio-channels 2 \
  --output-dir ./output
```

#### 8. 音频编码优化（网络分发）

```bash
# 平衡质量和文件大小
uv run python transformer.py \
  --chapters "1" \
  --audio-codec aac \
  --audio-bitrate 192k \
  --audio-channels 2 \
  --sample-rate 48000 \
  --output-dir ./output
```

#### 9. 自定义视频尺寸

```bash
uv run python transformer.py \
  --chapters "1" \
  --video-width 2560 \
  --video-height 1440 \
  --video-fps 30 \
  --output-dir ./output
```

#### 10. 使用自定义文件路径

```bash
uv run python transformer.py \
  --m4b-path /path/to/custom.m4b \
  --english-reference /path/to/custom_en.txt \
  --chinese-reference /path/to/custom_cn.txt \
  --chapters "1" \
  --output-dir /path/to/output
```

### 完整参数列表

#### 输入文件

- `--m4b-path`: M4B 音频书文件路径（默认: `resource/harrypotter.m4b`）
- `--english-reference`: 英文参考文本路径（默认: `resource/harrypotter.txt`）
- `--chinese-reference`: 中文参考文本路径（默认: `resource/harrypotter_cn.txt`）

#### 章节选择

- `--chapters`: 要处理的章节 ID，逗号分隔（如 `"1,2,3"`）或 `"all"` 处理全部（默认: `"1"`）

#### 输出目录

- `--output-dir`: 输出目录路径（默认: `./output`）

#### 处理步骤开关

- `--no-transcribe`: 跳过 ASR 转录
- `--no-correct`: 跳过转录校正
- `--no-align`: 跳过词级对齐
- `--no-translate`: 跳过中文翻译
- `--no-scenes`: 跳过场景检测
- `--no-scene-images`: 跳过场景图片生成
- `--no-video`: 跳过视频合成

#### 音频编码选项

- `--audio-codec`: 音频编码器（默认: `"copy"` 保持原始质量）
  - 可选值: `copy`, `aac`, `mp3`, `libopus` 等
- `--audio-bitrate`: 音频码率（如 `192k`, `320k`）
  - 仅当 `--audio-codec` 不是 `copy` 时有效
- `--audio-channels`: 音频声道数（默认: 保持原始）
  - `1` = 单声道，`2` = 立体声，`6` = 5.1 环绕声
- `--sample-rate`: 采样率（如 `44100`, `48000`）
  - 默认: 保持原始采样率

**使用场景**:
- **默认 (copy)**: 保持原始音频质量，零损失（适合本地存储）
- **AAC 320k 2ch**: 高质量立体声，适合 B 站上传（防止二压失真）
- **AAC 192k 2ch**: 平衡质量和文件大小，适合网络分发
- **Opus 64k 1ch**: 极致压缩，适合语音学习材料

#### 视频选项

- `--video-width`: 视频宽度（默认: 1920）
- `--video-height`: 视频高度（默认: 1080）
- `--video-fps`: 视频帧率（默认: 24）
- `--preview-duration`: 预览时长（秒），`None` = 完整视频（默认: None）

### 程序化使用（Python API）

如果需要在 Python 代码中使用，可以直接导入类：

```python
from src.materials.videos.harrypotter.transformer import HarryPotterTransformer

# 创建 transformer
transformer = HarryPotterTransformer(
    m4b_path="resource/harrypotter.m4b",
    chapter_ids=[1, 2, 3],  # 或 None 表示全部
    english_reference_path="resource/harrypotter.txt",  # 可选
    chinese_reference_path="resource/harrypotter_cn.txt",  # 可选
    # 音频编码选项（可选）
    audio_codec="aac",  # 默认 "copy"
    audio_bitrate="192k",  # 默认 None
    audio_channels=2,  # 默认 None
    sample_rate=48000,  # 默认 None
)

# 执行转换
materials = transformer.transform(
    output_dir="./output",
    transcribe=True,
    correct=True,
    align=True,
    translate=True,
    detect_scenes=True,
    generate_scene_images=True,
    synthesize_video=True,
    video_width=1920,
    video_height=1080,
    video_fps=24,
    preview_duration=30.0,  # 或 None
)

# 处理结果
for chapter_id, material in materials.items():
    print(f"Chapter {chapter_id}: {material.chapter.title}")
    print(f"  Audio: {material.audio_url}")
    print(f"  Segments: {len(material.transcript)}")
```

### 输出文件

程序会在输出目录生成以下文件：

```
output/
├── chapter1_audio.mp3              # 提取的音频
├── chapter1_material.json          # 完整材料数据（JSON）
├── chapter1_video.mp4              # 合成的视频（如果启用）
└── chapter_1_scenes/               # 场景图片目录（如果启用）
    ├── scene_000.png
    ├── scene_001.png
    └── ...
```

### 注意事项

1. **首次运行**: 会下载 WhisperX 模型，需要一定时间
2. **缓存**: 转录、校正、翻译结果会缓存在 `/tmp/cached`，避免重复计算
3. **内存**: 处理长音频（>30分钟）可能需要较大内存
4. **时间**: 完整处理一章（~30分钟音频）约需 1-2 小时
