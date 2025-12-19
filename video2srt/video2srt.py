# ####################################
# function: generate_srt_from_video
# Author: YeJunjie (nickname: Brice)
# EMail: ye@okwords.cn
# Date: 2025-12-18
# #####################################

#%%
# 导入需要的包
import os, datetime, time, random
import whisper
import subprocess
import torch
from deep_translator import GoogleTranslator
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#%%
# 翻译引擎

def translate_text(text: str, src_lang: str = "en", tgt_lang: str = "zh", engine: str = "model", use_gpu: bool = True) -> str:
    """
    多语言翻译函数
    :param text: 待翻译文本
    :param src_lang: 源语言代码（ja=日语, en=英语, zh=中文, fr=法语, de=德语, ko=韩语）
    :param tgt_lang: 目标语言代码（zh=中文, en=英语, ja=日语, fr=法语, de=德语, ko=韩语）
    :param engine: 翻译引擎选择（model: translation model ~ facebook/m2m100_418M, api: ranslation API ~ Google Translator API）
    :param use_gpu: 是否使用GPU加速，默认会自动检测是否有可用的GPU，若有则使用GPU加速，否则使用CPU
    :return: 翻译结果
    """
    # 加载模型（首次运行自动下载，约1.7GB，国内镜像加速）
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    if use_gpu and torch.cuda.is_available():
        model.to("cuda:0")  # 若使用GPU加速
    else:
        print("Warning: Model will run on CPU, which is slower than GPU.")
        print("警告：未检测到GPU，模型将在CPU上运行（速度较慢）")
        
    if engine not in ["model", "api"]:
        raise ValueError("please input: model, api")
    if src_lang in ['yue', 'wuu', 'nan', 'gan', 'cmn']:
        src_lang = "zh"
    if engine == "model":
        # 设置源语言
        M2M100_COMMON_LANGUAGES = [
            'zh', 'en', 'ja', 'ko', 'vi', 'th', 'id', 'ms', 'my', 'km','fr', 'de', 'es', 'it', 'ru', 'pt', 'nl', 'tr', 'pl', 'sv','cs', 'hu', 'ro', 'el', 'fi', 'da', 'no', 'sk', 'sl', 'bg','hr', 'lt', 'lv', 'ar', 'hi', 'bn', 'ur', 'fa', 'pa', 'mr','te', 'ta', 'sw', 'af', 'am', 'uk', 'uz', 'kk', 'sr'
        ]
        if src_lang not in M2M100_COMMON_LANGUAGES:
            src_lang = "zh"
        tokenizer.src_lang = src_lang
        # 编码文本（自动适配模型输入格式，处理长文本截断）
        encoded = tokenizer(
            text,
            return_tensors="pt",  # 返回PyTorch张量
            padding=True,        # 自动填充
            truncation=True,     # 超长文本截断（适配模型最大长度）
            max_length=512       # 模型最大输入长度限制
        )
        # 若使用GPU，将张量移到GPU
        if use_gpu and torch.cuda.is_available():
            encoded = {k: v.to("cuda:0") for k, v in encoded.items()}
        
        # 生成翻译结果（指定目标语言）
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
        )
        # 解码（跳过特殊字符）
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    elif engine == "api":
        """添加重试和延迟，避免频率风控"""
        if src_lang == "zh":
            src_lang = "zh-CN"
        if tgt_lang == "zh":
            tgt_lang = "zh-CN"
        GOOGLE_TRANSLATOR_COMMON_LANGUAGES = [
            'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'ny','zh-CN', 'zh-TW', 'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'tl', 'fi', 'fr','fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'iw', 'hi', 'hmn', 'hu', 'is','ig', 'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk', 'km', 'rw', 'ko', 'ku', 'ky', 'lo','la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt','no', 'or', 'ps', 'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn','sd', 'si', 'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'te', 'th', 'tr','uk', 'ur', 'ug', 'uz', 'vi', 'cy', 'xh', 'yi', 'yo', 'zu'
        ]
        if src_lang not in GOOGLE_TRANSLATOR_COMMON_LANGUAGES:
            src_lang = "zh-CN"
        if tgt_lang not in GOOGLE_TRANSLATOR_COMMON_LANGUAGES:
            tgt_lang = "zh-CN"
        for i in range(3):
            try:
                translator = GoogleTranslator(source=src_lang, target=tgt_lang)
                return translator.translate(text)
            except Exception as e:
                print(f"翻译失败：{str(e)}")
                delay = 7 + i * 3  # 递增延迟时间
                # 重试前随机延迟（2-5秒），避免固定间隔被识别
                time.sleep(delay + random.random() * 3)
                continue
        return "[okwords.cn]"
    else:
        # raise ValueError("不支持的引擎！请输入 model, api")
        raise ValueError("Unsupported engine! Please input: model, api")

def seconds_to_srt_time(seconds: float) -> str:
    """
    工具函数：时间戳转换（秒 → SRT格式）
    将秒数转换为SRT格式时间（HH:MM:SS,mmm）
    示例：12.345秒 → 00:00:12,345
    """
    # 处理小数秒（保留3位）
    milliseconds = int((seconds - int(seconds)) * 1000)
    # 转换为时间对象
    time_obj = datetime.timedelta(seconds=int(seconds))
    # 格式化为 HH:MM:SS
    time_str = str(time_obj).zfill(8)  # 补零到8位（如 0:01:23 → 00:01:23）
    # 拼接毫秒 → HH:MM:SS,mmm
    return f"{time_str},{milliseconds:03d}"

def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> str:
    """
    调用ffmpeg提取音频（需提前安装ffmpeg并配置环境变量）
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):  
        raise Exception(f"Video file not found: {video_path}")
        # raise Exception(f"未找到视频文件：{video_path}")
    # 构建ffmpeg命令（兼容Windows/Linux/Mac）
    cmd = [
        "ffmpeg",
        "-i", video_path,          # 输入视频路径
        "-vn",                     # 只提取音频，禁用视频流
        "-acodec", "pcm_s16le",    # 音频编码：16bit PCM（Whisper推荐格式）
        "-ar", "16000",            # 采样率：16000Hz
        "-ac", "1",                # 声道：单声道
        "-y",                      # 覆盖已存在的音频文件
        audio_path                 # 输出音频路径
    ]
    
    # 执行命令（屏蔽ffmpeg的冗余输出）
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,  # 屏蔽标准输出
            stderr=subprocess.DEVNULL,  # 屏蔽错误输出
            shell=True,                 # Windows必须加，Linux/Mac可选
            check=True                  # 执行失败时抛出异常
        )
    except subprocess.CalledProcessError:
        # raise Exception("音频提取失败！请先安装ffmpeg并配置到系统环境变量")
        raise Exception("Audio extraction failed! Please install ffmpeg and add it to system PATH.")
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        # raise Exception(f"未生成音频文件：{audio_path}")
        raise Exception(f"Audio file not generated: {audio_path}")
    return audio_path

def video_to_srt(
    video_path: str,
    srt_output_path: str = "output.srt",
    model_size: str = "base",
    language: str = None,  # 指定识别语言（如zh/en/ja/fr）
    is_translate: bool = False,
    translate_engine: str = "model",
    translate_lang: str = "zh",
    use_gpu: bool = True
) -> list:
    """
    核心函数：从视频生成带时间戳的SRT字幕文件
    :param video_path: 输入视频路径
    :param srt_output_path: 输出SRT文件路径
    :param model_size: Whisper模型大小（tiny/base/small/medium/large）
    :param language: 识别语言代码（参考Whisper语言列表）
    :param is_translate: 是否翻译识别文本
    :param translate_engine: 翻译引擎（model/api）
    :param translate_lang: 翻译目标语言（如zh/en/ja/fr）
    :param use_gpu: 是否使用GPU加速, 默认会自动检测是否有可用的GPU，若有则使用GPU加速，否则使用CPU
    :return: SRT文件内容
    """
    use_gpu = use_gpu and torch.cuda.is_available()
    # compute_type = "float16" if use_gpu else "float32" #老版本不支持float16
    if use_gpu:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
    # 1. 提取音频
    print('Extracting audio from video...')
    print("正在提取视频音频...")
    audio_path = extract_audio(video_path)
    
    # 2. 加载Whisper模型（自动下载，首次运行较慢）
    print(f"Loading Whisper {model_size} model...")
    print(f"加载Whisper {model_size} 模型...")
    # 强制指定 CPU，关闭所有 GPU 相关加速
    # model = whisper.load_model("base", device="cpu", compute_type="float32")
    model = whisper.load_model(
        model_size,
        device="cuda" if use_gpu else "cpu",
        #compute_type=compute_type  # 明确精度，避免自动选择触发 nan
    )
    
    # 3. 语音识别（带时间戳）
    print(f"Transcribing audio in {language}...")
    print("正在识别音频并生成时间戳...")
    WHISPER_SUPPORTED_LANGUAGES = [
        'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs','cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu','ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw', 'ka','kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml','mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt','ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw','ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo','zh', 'yue', 'wuu', 'nan', 'gan', 'cmn'
        ]
    if (language in [None,"","auto"]) or (language not in WHISPER_SUPPORTED_LANGUAGES ): #自动识别语言
        result = model.transcribe(
            audio_path,
            verbose=False,  # 屏蔽日志
            fp16=False,  #旧版本全部用False
            # fp16=use_gpu,   # GPU启用fp16加速，CPU设为False
            word_timestamps=False  # 若需单词级时间戳，设为True（默认片段级）
        )
        language = result["language"]
    else:
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False,  # 屏蔽日志
            fp16=False,  #旧版本全部用False
            # fp16=use_gpu,   # GPU启用fp16加速，CPU设为False
            word_timestamps=False  # 若需单词级时间戳，设为True（默认片段级）
        )
    
    # 4. 生成SRT内容
    srt_lines = []
    for idx, segment in enumerate(result["segments"], 1):
        # 获取片段时间戳（秒）
        start_sec = segment["start"]
        end_sec = segment["end"]
        # 转换为SRT格式时间
        start_srt = seconds_to_srt_time(start_sec)
        end_srt = seconds_to_srt_time(end_sec)
        # 识别文本（去除首尾空格）
        text = segment["text"].strip()
        if text is None:
            text = "[okwords.cn]"
        if is_translate:
            text_zh = translate_text(text, language, translate_lang, translate_engine, use_gpu)
            if text_zh is None:
                text_zh = "[一码千言]"
            if len(text_zh) < 7:
                text_zh = "[一码千言] " + text_zh
        if len(text) < 7:
            text = "[by okwords.cn] " + text
        # print(f"{idx}: {start_srt} --> {end_srt} \n {text}\n {text_zh}")
        # 按SRT规范拼接行
        srt_lines.append(str(idx))  # 序号
        srt_lines.append(f"{start_srt} --> {end_srt}")  # 时间轴
        srt_lines.append(text)  # 识别文本
        if is_translate:
            srt_lines.append(text_zh)  # 翻译文本
        srt_lines.append("")  # 空行分隔片段
    
    # 拼接所有行并去除末尾空行
    srt_content = "\n".join(srt_lines).strip()
    
    # 打印前10行SRT内容预览
    # print("\nSRT内容预览：")
    # print("\n".join(srt_content.split("\n")[:10]))
    
    # 5. 保存SRT文件
    with open(srt_output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    # 6. 清理临时音频文件
    os.remove(audio_path)
    
    print(f"\ngenerate SRT({srt_output_path}) from video({video_path}) completed")
    print(f"\n根据视频 {video_path} 生成的 SRT 文件已保存为 {srt_output_path}。")
    return srt_lines

# ==================== 调用示例 ====================
if __name__ == "__main__":
    # 配置参数
    VIDEO_PATH = "test_video.mp4"  # 你的视频文件路径
    SRT_OUTPUT = "test_out.srt"  # 输出SRT路径
    MODEL_SIZE = "base"  # 模型大小（large精度更高，需更多显存，处理时间更长）
    LANGUAGE = "ja"      # 识别语言（日语）
    IS_TRANSLATE = True  # 是否翻译识别文本
    TRANSLATE_ENGINE = "model"  # 翻译引擎（model/api）
    TRANSLATE_LANG = "zh"      # 翻译目标语言（如zh/en/ja/fr）
    USE_GPU = True      # 是否使用GPU加速
    
    # 生成SRT
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
    