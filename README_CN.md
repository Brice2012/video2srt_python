# Video2SRT Project (视频转字幕项目)

## 快速开始

首先安装项目需要的软件（详细步骤见下一节）：
    - python >=3.13.5
    - ffmpeg

安装Video2SRT项目包：
    ```bash
    pip install video2srt

    # 为了提高安装速度，中国大陆地区用户可在每条命令后添加 -i 以使用国内的pip源。如：
    pip install video2srt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

运行代码(python)：

    ```python
    from video2srt import video_to_srt
    # 转换视频为字幕
    srt_lines = video_to_srt('input.mp4', 'output.srt')
    # 打印字幕内容（仅显示前10行）
    print(srt_lines[:10])
    ```

运行代码(完整参数示例)

    ```python
    from video2srt import video_to_srt
    # ## 生成SRT（使用video_to_srt函数）
    srt_lines = video_to_srt(
        video_path = 'input.mp4',       
        # 输入视频路径
        srt_output_path = 'output.srt', 
        # 输出字幕路径
        model_size = 'base', 
        # 模型大小，可选值：tiny, base, small, medium, large, 默认值：base
        language = 'ja',  
        # 日语，支持 en, zh, ko, fr 等
        is_translate = True,
        translate_engine = 'api', 
        # 翻译引擎，可选值：model（本地模型）, api（Google Translate API）
        translate_lang = 'zh',  
        # 翻译为中文，支持 en, zh, ko, fr 等
        use_gpu = True  
        # 是否使用GPU加速，默认值：True, 若未安装CUDA或未配置好环境，会自动切换为CPU模式
    )
    print(srt_lines[:10])
    ```

## 项目简介

这是一个简单的将视频中的语音转换为字幕的项目。

### ffmpeg音频剥离

项目使用ffmpeg剥离视频中的音频流
    输入视频格式： mp4, mkv, avi, flv, ts, m3u8, mov, wmv, asf, rmvb, vob, webm 等格式
    输出音频格式： wav, mp3, aac, flac, ogg, wma, m4a, aiff 等格式
需要预安装ffmpeg并配置到环境变量中。
具体配置方法请参考[ffmpeg官网](https://ffmpeg.org/)

### whisper语音识别

项目使用whisper模型进行语音识别，支持多语言识别（90+种语言）。
**官方核心模型：**：
    tiny: 微小模型，仅支持核心语种，精度较低，速度最快，约108MB。
    base: 基础模型，支持主流语种, 精度较高，速度较快, 约1GB。默认使用该模型。
    small: 小型模型，支持多语种, 精度较高，速度中等, 约4GB。
    medium: 中型模型，支持低资源语种, 精度较高，速度较慢, 约10GB。
    large: 大型模型，支持99种语言+方言(粤语、吴语等)，精度高, 速度较慢，约20GB, 适合处理高精度要求及小众语言/方言识别。
**支持的语言列表：**
[https://github.com/openai/whisper/blob/main/whisper/tokenizer.py]

### 多语言翻译

项目使用facebook/m2m100模型和Google Translate API进行多语言翻译。将源语言翻译成目标语言（支持100+种语言）。
**可选参数：**
    is_translate:
        默认值：False
        可选值：True
    translate_engine:
        默认值：model: facebook/m2m100 本地模型（私密,免费开源，第一次使用需要下载模型,速度较慢）
        可选值：api:  Google Translate API（免费，速度较快）
**支持的语言列表：**
    facebook/m2m100模型：[https://huggingface.co/facebook/m2m100_418M/blob/main/README.md]
    Google Translate API：[https://cloud.google.com/translate/docs/languages]

### GPU加速

第2、3步若使用GPU加速，需要安装cuda，并安装对应CUDA版本的支持GPU的torch。
具体安装方法请参考[torch官网](https://pytorch.org/get-started/locally/)

    ```bash
    nvidia-smi # 首先验证显卡是否支持 CUDA
    # 若输出显卡信息（含 CUDA Version 字段，如  CUDA Version: 12.6），说明支持 CUDA；若提示 nvidia-smi 不是内部或外部命令，需先安装 NVIDIA 驱动。
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 #需要适应CUDA版本
    ```

## video2srt 使用步骤

1. 安装环境软件：确保已安装python3.13.5或以上版本，安装ffmpeg并配置到环境变量中。
2. 安装项目依赖 (项目的包依赖文件：[pyproject.toml]，可使用pip或uv安装项目依赖)
3. 运行项目代码 (项目的主文件：[video2srt.py])

## 安装环境软件

请确保已安装python3.13.5或以上版本[https://www.python.org/downloads/]

请确保已安装ffmpeg并配置到环境变量中。[https://ffmpeg.org/download.html]

    ```bash
    ### 安装ffmpeg
    # Centos
    yum install ffmpeg ffmpeg-devel -y

    # Ubuntu/MacOS
    apt install ffmpeg
    brew install ffmpeg

    # windows 下载ffmpeg安装包，放到特定的目录下，将路径添加到系统Path中。
    # 下载地址：https://ffmpeg.org/download.html

    # 验证是否安装成功
    ffmpeg -version # 查看ffmpeg版本
    ffmpeg -formats # 查看ffmpeg支持的所有格式
    ```

## 安装项目依赖

python >= 3.13.5

### 使用uv管理（推荐）

项目配置文件：[project.toml]

    ```bash
    uv python install 3.14
    uv python pin 3.14
    uv sync
    uv lock
    ```

### 使用Pip管理

安装需要的包(项目的包依赖文件：[requirements.txt])

    ```bash
    pip install video2srt

    # 为了提高安装速度，中国大陆地区用户可在每条命令后添加 -i 以使用国内的pip源。如：
    pip install video2srt -i https://pypi.tuna.tsinghua.edu.cn/simple

    # 也可直接使用requirements.txt文件安装
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## 运行项目代码

首次运行时需要下载whisper模型（默认使用base模型），模型大小约1GB。需要一点时间。请耐心等待。

### 使用示例（sample）

    ```python
    from video2srt import video_to_srt
    srt_lines = video_to_srt(
        video_path = "test_video.mp4",      # 输入视频文件路径
        srt_output_path = "test_video.mp4",  # 输出SRT路径
        language="zh"
    )
    
    print("Sample SRT 0-7 lines:")
    print("\n".join(srt_lines[:7]))
    ```

### 使用示例（完整参数配置）

    ```python
    # ## 导入video_to_srt函数
    from video2srt import video_to_srt
    
    # ## 配置参数（Parameters Configuration）

    VIDEO_PATH = input("Please input video file path:")  
    # 视频文件路径
    SRT_OUTPUT = "test_out.srt"  
    # 输出SRT路径
    MODEL_SIZE = "base"  
    # Whisper模型类型（tiny/base/small/medium/large）
    # 默认使用base, large精度更高，需更多显存，处理时间更长，支持小众语言和方言
    LANGUAGE = None       
    # 识别语言，默认自动识别, 支持多语言（90多种）识别（如en/zh/ja/lo/fr/de等, 更多语言请参考Whisper文档）
    IS_TRANSLATE = False  
    # 是否翻译识别的文本，默认不翻译
    TRANSLATE_ENGINE = "model"  
    # 翻译引擎（model/api），model默认使用本地模型facebook/m2m100_418M，api默认使用Google翻译API
    TRANSLATE_LANG = "zh"      
    # 翻译目标语言（如zh/en/ja/ko/fr）, 默认翻译为中文，支持100+种语言翻译，j基本与whisper一致，但有一些差异,如没有对方言的支持（zh/yue/wuu->zh)，系统会自动将其转换为zh
    USE_GPU = True     
    # 是否使用GPU加速，默认会自动检测是否有可用的GPU，若有则使用GPU加速，否则使用CPU
    
    # ## 生成SRT（使用video_to_srt函数）
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

### 查看帮助(help)和示例(sample)

    ```python
    from video2srt import sample, hello_video2srt
    hello_video2srt()

    srt_lines = sample()
    print("Sample SRT 0-7 lines:")
    print("\n".join(srt_lines[:7]))
    ```
