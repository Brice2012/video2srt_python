# Video2SRT Project (Video to Subtitle Project)

## Quick Start

First, install the software required by the project (refer to the next section for detailed steps):

- python >= 3.13.5
- ffmpeg

Install the Video2SRT project package:

```bash
pip install video2srt

# To speed up installation, users in mainland China can append the -i parameter to each command to use domestic pip mirrors. For example:
pip install video2srt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Run the code (Python):

```python
from video2srt import video_to_srt
# Convert video to subtitles
srt_lines = video_to_srt('input.mp4', 'output.srt')
# Print subtitle content (only the first 10 lines are displayed)
print(srt_lines[:10])
```

Run the code (Full Parameter Example)

```python
from video2srt import video_to_srt
# ## Generate SRT (using the video_to_srt function)
srt_lines = video_to_srt(
    video_path = 'input.mp4',       
    # Input video path
    srt_output_path = 'output.srt', 
    # Output subtitle path
    model_size = 'base', 
    # Model size, optional values: tiny, base, small, medium, large; default value: base
    language = 'ja',  
    # Japanese, supports en, zh, ko, fr, etc.
    is_translate = True,
    translate_engine = 'api', 
    # Translation engine, optional values: model (local model), api (Google Translate API)
    translate_lang = 'zh',  
    # Translate to Chinese, supports en, zh, ko, fr, etc.
    use_gpu = True  
    # Whether to use GPU acceleration, default value: True. It will automatically switch to CPU mode if CUDA is not installed or the environment is not properly configured
)
print(srt_lines[:10])
Translation Notes
```

## Project Overview

This is a simple project that converts speech from video into subtitles.

### FFmpeg Audio Extraction

The project uses FFmpeg to extract the audio stream from video files:

Input video formats: mp4, mkv, avi, flv, ts, m3u8, mov, wmv, asf, rmvb, vob, webm, and other formats

Output audio formats: wav, mp3, aac, flac, ogg, wma, m4a, aiff, and other formats

FFmpeg must be pre-installed and configured in the system environment variables.For detailed configuration methods, refer to the [FFmpeg official website](https://ffmpeg.org/).

### Whisper Speech Recognition

The project uses the Whisper model for speech recognition, supporting multilingual recognition (90+ languages).

Official Core Models:
**tiny**: Tiny model, supports only core languages with low accuracy but fastest speed (approximately 108MB).
**base**: Base model, supports mainstream languages with high accuracy and fast speed (approximately 1GB). This is the default model.
**small**: Small model, supports multiple languages with high accuracy and medium speed (approximately 4GB).
**medium**: Medium model, supports low-resource languages with high accuracy and slow speed (approximately 10GB).
**large**: Large model, supports 99 languages + dialects (Cantonese, Wu Chinese, etc.) with high accuracy but slow speed (approximately 20GB).

Suitable for high-precision requirements and recognition of minority languages/dialects.Supported Languages List:[https://github.com/openai/whisper/blob/main/whisper/tokenizer.py]

### Multilingual Translation

The project uses the facebook/m2m100 model and Google Translate API for multilingual translation, converting source language to target language (supports 100+ languages).

Optional Parameters:
is_translate:
Default value: False
Optional values: True

translate_engine:
Default value: model: facebook/m2m100 local model (private, free and open-source; the model needs to be downloaded on first use, with slow speed)
Optional values: api: Google Translate API (free, with fast speed)

Supported Languages List:
facebook/m2m100 model: [https://huggingface.co/facebook/m2m100_418M/blob/main/README.md]
Google Translate API: [https://cloud.google.com/translate/docs/languages]

### GPU Acceleration

To use GPU acceleration for steps 2 and 3, CUDA must be installed along with the GPU-supported version of torch compatible with the installed CUDA version.For detailed installation methods, refer to the [PyTorch official website](https://pytorch.org/get-started/locally/).

```bash
nvidia-smi # First, verify whether your graphics card supports CUDA. If the graphics card information is output (including the CUDA Version field, e.g., CUDA Version: 12.6), it means CUDA is supported. If a prompt appears stating that nvidia-smi is not an internal or external command, you need to install the NVIDIA driver first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # CUDA 12.6
```

## Video2SRT Usage Steps

1. Install environment software: Ensure Python 3.13.5 or higher is installed, and FFmpeg is installed and configured in the system environment variables.

2. Install project dependencies (Project dependency file: [pyproject.toml]; project dependencies can be installed using pip or uv)

3. Run the project code (Project main file: [video2srt.py])

## Install Environment Software

Ensure Python 3.13.5 or higher is installed [https://www.python.org/downloads/]
Ensure FFmpeg is installed and configured in the system environment variables [https://ffmpeg.org/download.html]

```bash
### Install FFmpeg
# CentOS
yum install ffmpeg ffmpeg-devel -y

# Ubuntu/MacOS
apt install ffmpeg
brew install ffmpeg

# Windows: Download the FFmpeg installation package, place it in a specific directory, and add the path to the system Path.
# Download URL: https://ffmpeg.org/download.html

# Verify installation
ffmpeg -version # Check FFmpeg version
ffmpeg -formats # View all formats supported by FFmpeg
```

## Install Project Dependencies

Python >= 3.13.5

### Using uv (Recommended)

Project configuration file: [project.toml]

```bash
uv python install 3.14
uv python pin 3.14
uv sync
uv lock
```

### Using Pip

Install required packages (Project dependency file: [requirements.txt])

```bash
pip install deep_translator
pip install openai-whisper
pip install transformers torch
pip install sentencepiece

# To speed up installation, users in mainland China can add -i to each command to use domestic pip mirrors. For example:
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

# You can also install directly using the requirements.txt file
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Run Project Code

The Whisper model (the base model is used by default) needs to be downloaded on the first run. The model is approximately 1GB in size and will take some time. Please wait patiently.

### Usage Example (sample)

```python
    from video2srt import video_to_srt
    srt_lines = video_to_srt(
        video_path = "test_video.mp4", 
        srt_output_path = "test_video.mp4",
        language="zh"
    )
    
    print("Sample SRT 0-7 lines:")
    print("\n".join(srt_lines[:7]))
```

### Usage Example (Full Parameter Configuration)

```python
# ## Import the video_to_srt function
from video2srt import video_to_srt

# ## Parameters Configuration

VIDEO_PATH = input("Please input video file path:")  
# Video file path
SRT_OUTPUT = "test_out.srt"  
# Output SRT path
MODEL_SIZE = "base"  
# Whisper model type (tiny/base/small/medium/large)
# Defaults to base; large offers higher accuracy but requires more VRAM and processing time, and supports minority languages and dialects
LANGUAGE = None       
# Recognition language, auto-detected by default, supports multilingual recognition (90+ languages) (e.g., en/zh/ja/lo/fr/de, etc.; refer to Whisper documentation for more languages)
IS_TRANSLATE = False  
# Whether to translate recognized text, disabled by default
TRANSLATE_ENGINE = "model"  
# Translation engine (model/api); model uses the local facebook/m2m100_418M model by default, api uses Google Translate API by default
TRANSLATE_LANG = "zh"      
# Target translation language (e.g., zh/en/ja/ko/fr), defaults to Chinese, supports translation to 100+ languages. Basically consistent with Whisper but with some differences (e.g., no dialect support: zh/yue/wuu -> zh), the system will automatically convert them to zh
USE_GPU = True      
# Whether to use GPU acceleration, uses GPU by default, if no GPU is available, uses CPU

# ## Generate SRT (using the video_to_srt function)
srt_lines = video_to_srt(
    video_path = VIDEO_PATH,
    srt_output_path = SRT_OUTPUT,
    model_size = MODEL_SIZE,
    language = LANGUAGE,
    is_translate = IS_TRANSLATE,
    translate_engine = TRANSLATE_ENGINE,
    translate_lang = TRANSLATE_LANG,
    use_gpu = USE_GPU
)

print("Sample SRT 0-7 lines:")
print("\n".join(srt_lines[:7]))
```

### View Help and Samples

```python
from video2srt import sample, hello_video2srt

hello_video2srt()

srt_lines = sample()
print("Sample SRT 0-7 lines:")
print("\n".join(srt_lines[:7]))
```
