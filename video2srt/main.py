# ####################################
# function: generate_srt_from_video
# Author: YeJunjie (nickname: Brice)
# EMail: ye@okwords.cn
# Date: 2025-12-17
# #####################################

#%%
# hello_video2srt function
def hello_video2srt():
    print("""
    This is a sample code to generate SRT file from video using Whisper model.
    
    Steps to run:
    1. Make sure you have installed the required packages: 
        pip install deep_translator
        pip install openai-whisper
        pip install transformers torch
        pip install sentencepiece
        
    2. Prepare your video file and note its path.
    
        video(input)： mp4, mkv, avi, flv, ts, m3u8, mov, wmv, asf, rmvb, vob, webm etc.
        Audio(output)： wav, mp3, aac, flac, ogg, wma, m4a, aiff etc.
    
    3. Run this python script and input the video file path when prompted.
    
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
        USE_GPU = False      
        # Whether to use GPU acceleration, uses CPU by default

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

        print(f"\nThe SRT file generated from video {VIDEO_PATH} has been saved as {SRT_OUTPUT}, source language: {LANGUAGE}.")

        print("Sample SRT 0-7 lines:")
        print("\n".join(srt_lines[:7]))
    
    4. The generated SRT file will be saved in the current directory as 'test_out.srt'.
    
    You can modify parameters such as MODEL_SIZE, LANGUAGE, IS_TRANSLATE, etc. in the sample() function.
        from video2srt import sample, help
        help()
        srt_lines = sample()
        
    """)
    
# 运行, 从视频生成 SRT 文件
def sample():
    from .video2srt import video_to_srt
    from importlib.resources import files
    from importlib.resources.abc import Traversable
    # 配置参数
    VIDEO_PATH = input("Please input video file path (请输入视频文件路径):")
    if VIDEO_PATH.strip() == "":
        video_test_path: Traversable = files("video2srt") / "test_video.mp4"
        VIDEO_PATH = video_test_path  # 测试视频
    SRT_OUTPUT = "test_out.srt"  # 输出SRT路径

    # 生成SRT
    srt_lines = video_to_srt(
        video_path = VIDEO_PATH,
        srt_output_path = SRT_OUTPUT,
        language="zh"
    )
    
    return srt_lines

#%%