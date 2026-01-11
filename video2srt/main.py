# ####################################
# function: generate_srt_from_video
# Author: YeJunjie (nickname: Brice)
# EMail: ye@okwords.cn
# Date: 2025-12-17
# #####################################

#%%
# hello_video2srt function
def hello_video2srt():
    from .__init__ import __version__
    
    print("""
    Hello video2srt! Version: {__version__}
    
    This is a sample code to generate SRT file from video using Whisper model.
    
    Steps to run:
    1. Make sure you have installed the required packages: 
        pip install video2srt
        
    2. Prepare your input(video/audio/livecaptions) file and note its path.
    
        video： mp4, mkv, avi, flv, ts, m3u8, mov, wmv, asf, rmvb, vob, webm etc.
        audio： wav, mp3, aac, flac, ogg, wma, m4a, aiff etc.
        live captions： txt.
    
    3. Run this python script and input the video file path when prompted.
    
        # ## Import the video_to_srt or livecaptions_to_srt function
        from video2srt import video_to_srt livecaptions_to_srt
    
    You can modify parameters such as MODEL_SIZE, LANGUAGE, IS_TRANSLATE, etc. in the sample() function.
        from video2srt import sample, hello_video2srt
        hello_video2srt()
        srt_lines = sample()
        
        yours blue(ye@okwords.cn).
        
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