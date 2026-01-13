# Video-Gen

Harry Potter 有声书转视频学习材料的处理管线。将 M4B 格式的有声书转换为带字幕、场景图片的学习视频。

## 目录结构

```
video_gen/
├── __init__.py
├── video_material.py              # 核心数据模型 (TranscriptSegment, VideoMaterial 等)
├── common/
│   ├── __init__.py
│   └── tools.py                   # 通用工具函数集合
├── core/
│   ├── __init__.py
│   ├── configs/
│   │   ├── __init__.py
│   │   └── config.py              # 配置管理 (从 .env 读取 API keys)
│   └── tools/
│       ├── __init__.py
│       ├── openai_client.py       # OpenAI API 客户端 (支持代理)
│       └── volcengine_client.py   # 火山引擎图片生成 API 客户端
└── harrypotter/
    ├── __init__.py
    ├── models.py                  # 数据模型定义
    ├── transformer.py             # 主处理流程入口
    ├── audio_extractor.py         # M4B 音频章节提取
    ├── scene_detector.py          # 场景检测与图片生成
    ├── synthesizer.py             # 视频合成 (ffmpeg + ASS 字幕)
    ├── transcript_corrector.py    # ASR 转录校正
    ├── transcript_translator.py   # 英译中翻译
    └── packager.py                # 输出文件打包
```

## 核心模块说明

### `video_gen/video_material.py`

定义核心数据结构：

- `TranscriptWord`: 单词级别时间戳
- `TranscriptSegment`: 字幕片段 (含时间、文本、翻译、词级时间戳)
- `VideoMaterial`: 完整视频学习材料

### `video_gen/common/tools.py`

通用工具函数：

| 函数 | 说明 |
|------|------|
| `transcribe_with_whisperx()` | WhisperX ASR 转录 |
| `align_transcript_segments()` | 词级别强制对齐 |
| `parse_ass()` / `write_ass()` | ASS 字幕解析/生成 |
| `extract_m4b_metadata()` | M4B 元数据提取 |
| `parse_m4b_chapters()` | M4B 章节解析 |
| `cached()` | 函数结果缓存装饰器 |

### `video_gen/core/configs/config.py`

从 `.env` 文件读取配置：

```python
from video_gen.core.configs.config import settings

settings.openai_api_key      # OpenAI API Key
settings.volcengine_api_key  # 火山引擎 API Key (可选)
```

### `video_gen/core/tools/openai_client.py`

OpenAI 兼容的 LLM 客户端，支持：
- 同步/异步调用
- 批量生成
- 自动重试
- 代理服务器

### `video_gen/core/tools/volcengine_client.py`

火山引擎图片生成 API 客户端，用于生成场景背景图。

### `video_gen/harrypotter/transformer.py`

**主入口**，`HarryPotterTransformer` 类实现完整处理流程：

1. 从 M4B 提取章节音频
2. WhisperX ASR 转录
3. LLM 校正转录文本
4. 词级别对齐
5. LLM 翻译成中文
6. 场景检测与图片生成
7. 视频合成

### `video_gen/harrypotter/models.py`

Harry Potter 专用数据模型：

- `M4BChapter`: M4B 章节信息
- `M4BMetadata`: M4B 元数据
- `SceneInfo`: 场景信息
- `AudiobookChapterMaterial`: 章节学习材料

### `video_gen/harrypotter/scene_detector.py`

使用 LLM 分析转录文本，检测场景变化并生成图片提示词，然后调用火山引擎 API 生成场景背景图。

### `video_gen/harrypotter/synthesizer.py`

使用 ffmpeg + ASS 字幕合成最终视频，支持：
- 卡拉 OK 式词高亮
- 场景背景图切换
- 中英双语字幕

## 安装

### 前置要求

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python 包管理器)
- ffmpeg (视频处理)

### 安装 uv

```bash
# macOS
brew install uv

# 或使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装 ffmpeg

```bash
# macOS
brew install ffmpeg
```

### 安装项目依赖

```bash
cd ~/code/video-gen
uv sync
```

### 配置环境变量

创建 `.env` 文件：

```bash
# 必需
OPENAI_API_KEY=your_openai_api_key

# 可选 (场景图片生成)
VOLCENGINE_API_KEY=your_volcengine_api_key

# 可选 (S3 存储)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

## 使用方法

### 命令行

```bash
# 查看帮助
uv run python -m video_gen.harrypotter.transformer --help

# 处理第 1 章 (需要准备 resource/chapter1.m4b)
uv run python -m video_gen.harrypotter.transformer --chapters 1

# 处理多个章节
uv run python -m video_gen.harrypotter.transformer --chapters 1,2,3

# 只转录不生成视频
uv run python -m video_gen.harrypotter.transformer --chapters 1 --no-video

# 生成预览视频 (前 30 秒)
uv run python -m video_gen.harrypotter.transformer --chapters 1 --preview-duration 30

# 指定输出目录
uv run python -m video_gen.harrypotter.transformer --chapters 1 --output-dir ./my_output
```

### Python API

```python
from video_gen.harrypotter.transformer import HarryPotterTransformer

# 初始化
transformer = HarryPotterTransformer(
    m4b_path="path/to/audiobook.m4b",
    chapter_ids=[1, 2],  # 处理第 1、2 章，None 表示全部
    english_reference_path="path/to/harrypotter.txt",  # 英文原文 (用于校正)
    chinese_reference_path="path/to/harrypotter_cn.txt",  # 中文翻译 (用于翻译参考)
)

# 运行完整流程
results = transformer.transform(
    output_dir="output",
    transcribe=True,        # ASR 转录
    correct=True,           # LLM 校正
    translate=True,         # LLM 翻译
    detect_scenes=True,     # 场景检测
    generate_scene_images=True,  # 生成场景图片
    synthesize_video=True,  # 合成视频
    preview_duration=30.0,  # 预览模式 (None 为完整视频)
)

# 结果是 dict[chapter_id, AudiobookChapterMaterial]
for chapter_id, material in results.items():
    print(f"Chapter {chapter_id}: {len(material.transcript)} segments")
```

### 单独使用各模块

#### ASR 转录

```python
from video_gen.common.tools import transcribe_with_whisperx

transcript = transcribe_with_whisperx(
    audio_path="audio.mp3",
    model_name="medium",
    language="en",
    align=True,  # 词级别对齐
)
```

#### 场景检测

```python
from video_gen.harrypotter.scene_detector import SceneDetector

detector = SceneDetector()
result = detector.detect_scenes(transcript)
result = detector.generate_scene_images(result, output_dir="scenes")
```

#### 视频合成

```python
from video_gen.harrypotter.synthesizer import synthesize_video_from_material

synthesize_video_from_material(
    material_json_path="output/chapter1_material.json",
    output_path="output/chapter1_video.mp4",
    enable_karaoke=True,
    preview_duration=30.0,
)
```

## 输入文件要求

需要在 `video_gen/harrypotter/resource/` 目录下准备以下文件：

```
video_gen/harrypotter/resource/
├── harrypotter.m4b      # 完整有声书 (可选，处理多章时使用)
├── chapter1.m4b         # 单章音频 (可选，处理单章时优先使用)
├── chapter2.m4b
├── ...
├── harrypotter.txt      # 英文原文 (必需，用于 ASR 校正)
└── harrypotter_cn.txt   # 中文翻译 (必需，用于翻译参考)
```

### M4B 音频文件

- 格式: M4B (带章节的 AAC 音频容器)
- 可以是完整有声书 `harrypotter.m4b`，或单章文件 `chapter1.m4b`
- 单章文件命名格式: `chapter{N}.m4b`，程序会优先使用

### 参考文本文件

**harrypotter.txt** (英文原文)
- 用于 ASR 转录校正，提高准确率
- 每章以 `CHAPTER ONE`、`CHAPTER TWO` 等格式开头

**harrypotter_cn.txt** (中文翻译)
- 用于翻译参考，确保人名、地名等术语一致
- 每章以 `第１章`、`第２章` 或 `第一章`、`第二章` 格式开头

示例章节格式：
```
CHAPTER ONE
The Boy Who Lived

Mr. and Mrs. Dursley, of number four, Privet Drive...
```

```
第１章　大难不死的男孩

家住女贞路四号的德思礼夫妇总是得意地说...
```

## 输出文件

处理完成后，输出目录结构：

```
output/
├── chapter_001_The Boy Who Lived.mp4   # 提取的章节音频
├── chapter1_material.json              # 章节材料 JSON
├── chapter1_full.mp4                   # 合成的视频 (或 chapter1_preview.mp4)
└── chapter_1_scenes/                   # 场景图片
    ├── scene_000.png
    ├── scene_001.png
    └── ...
```

## 依赖版本说明

以下依赖版本已锁定以确保兼容性：

| 依赖 | 版本 | 说明 |
|------|------|------|
| torch | 2.3.1 | PyTorch (支持 Python 3.12) |
| torchaudio | 2.3.1 | 音频处理 |
| whisperx | 3.3.1 | ASR 转录 |
| pyannote-audio | 3.3.2 | 说话人分离 |
| transformers | 4.44.0 | Hugging Face 模型 |

## 缓存

函数结果缓存在 `/tmp/cached/` 目录，基于参数哈希。重复运行相同参数时会复用缓存结果。

清除缓存：

```bash
rm -rf /tmp/cached
```

## 常见问题

### Q: WhisperX 转录很慢？

A:
- 使用更小的模型: `model_name="small"` 或 `"base"`
- 缓存会自动生效，重复运行会快很多
- 考虑使用 GPU (需修改 device 参数)

### Q: ffmpeg 报错？

A:
- 确认 ffmpeg 已安装: `ffmpeg -version`
- 检查输入文件是否存在且未损坏

### Q: 场景图片生成失败？

A:
- 确认 `VOLCENGINE_API_KEY` 已配置
- 检查网络连接
- 查看日志中的错误信息
