# ###################################################
# 使用windows得实时字幕功能获取多语言文本并生成中文字幕
# Blue 2026-01-07
# ###################################################

#%%
# 导入必要的包
from operator import is_
import re, time, random, os, tqdm
from datetime import datetime, timedelta
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from .video2srt import translate_text

# import os
# # 关键：配置 Hugging Face 国内镜像，避免访问被拦截
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# # 自定义缓存目录（可选，避免C盘占用）
# os.environ["HUGGINGFACE_HUB_CACHE"] = r"G:\HuggingFace\Cache"  # 替换为你的路径

#%%
# 读取save-live-captions保存的通过实时字幕功能生成的多语言文本文件，并除掉换行和空格
def read_live_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取文件的全部内容; 文件内含save_live_captions保存的实时字幕生成的原始文本字符串（含[时间戳]和对应文本）
        content = file.read()
        # 移除所有换行符（\n、\r、\r\n）和空格（普通空格、制表符）
        # 替换换行符：\n（Linux/macOS）、\r\n（Windows）、\r（旧系统）
        segments = content.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
        # 按[时间]分隔
        # segments = re.split(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', content)
        # 装换为字符串
        # segments = ''.join(segments)
        # 在形如[12:21:18]的时间之前后之后各加一个换行
        segments = re.sub(r'(\[\d{2}:\d{2}:\d{2}\])', r'\n\1\n', segments)
    return segments
    
#%%
# 将整理好的文本转换为SRT格式（并根据参数选择是否翻译）
def livecaptions_to_srt(
    live_file_name: str,
    srt_output_path: str = None,
    language: str = None,  # 指定识别语言（如日语ja, 英语en, 韩语ko , 法语 fr, 意大利语 it, 西班牙语es, 德语de, 俄语 ru）
    is_translate: bool = False,
    translate_engine: str = "model",
    translate_lang: str = "zh",
    use_gpu: bool = True
    ) -> list:
    """
    核心函数：从视频生成带时间戳的SRT字幕文件
    :param live_file_name: 输入视频的多语言文本文件路径和文件名, 如：   E:\EV\2026-01-07_11-15-51_captions.txt
    :param srt_output_path: 输出SRT文件路径和文件名, 默认根据文件名自动识别
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
        print("尝试使用GPU加速...")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()

    file_name_source = live_file_name.split("\\")[-1]
    file_path = live_file_name
    if srt_output_path in [None,""]:
        if is_translate:
            srt_output_path = file_name_source.replace(".txt", "")+"_"+language+"_"+translate_lang+".srt"
        else:
            srt_output_path = file_name_source.replace(".txt", "")+"_"+language+".srt"

    # 处理文本并重新保存到本地
    live_text = read_live_txt(file_path)
    with open(file_name_source, 'w', encoding='utf-8') as file:
        file.write(live_text)
    print(f"整理后的实时字幕文本已保存到：{file_name_source}")
    # 自定义起始基准时间（例如：12:23:00）
    # base_start_time = "12:20:48"
    # 时间为file_name_source中间的部分转换过来
    base_start_time = file_name_source.split("_")[1].replace("-", ":")

    # 1. 解析起始基准时间为datetime对象（方便时间计算）
    try:
        start_time = datetime.strptime(base_start_time, "%H:%M:%S")
    except ValueError:
        raise ValueError("起始时间格式错误！请确认文件名包含 HH:MM:SS 格式的时间，例如 '12:23:55'")

    # 2. 正则匹配所有[HH:MM:SS]时间戳和对应的文本内容
    # 匹配规则：[时间戳] + 换行 + 文本内容（非空行）
    pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]\s*\n([^\[\n]+)')
    matches = pattern.findall(live_text)  # 结果为列表：[(时间戳1, 文本1), (时间戳2, 文本2), ...]

    if not matches:
        raise ValueError("未匹配到任何[HH:MM:SS]时间戳和文本内容！")

    # 3. 处理每个时间戳和文本，生成SRT内容
    srt_lines = []
    total_matches = len(matches)
    
    # 为for循环添加进度条
    # 进度条配置
    progress_bar = tqdm.tqdm(total=total_matches, desc="处理进度", unit="条")
    for idx, (time_str, content) in enumerate(matches):
        # 清理文本内容（去除首尾空白、多余换行）
        text = content.strip()
        #print(idx,'\n', text)
        if text is None:
            text = "[by okwords.cn generate SRT]"
        if is_translate:
            text_zh = translate_text(text, language, translate_lang, translate_engine, use_gpu)
            if text_zh is None:
                text_zh = "[by 一码千言 提供字幕生成]"
            if len(text_zh) < 7:
                text_zh = "[by 一码千言 提供字幕生成] " + text_zh
        if len(text) < 7:
            text = "[by okwords.cn generate SRT] " + text
        # print(text_zh)
        
        # 3.1 解析当前时间戳为datetime对象
        current_time = datetime.strptime(time_str, "%H:%M:%S")
        
        # 3.2 计算相对起始时间的偏移（当前时间 - 起始时间）→ SRT开始时间
        start_delta = current_time - start_time
        # 转换为 SRT 格式（HH:MM:SS,000），不足24小时补0
        srt_start = (datetime(1900, 1, 1) + start_delta).strftime("%H:%M:%S,000")
        
        # 3.3 计算SRT结束时间（下一个时间的前10秒；最后一条则延长10秒）
        if idx < total_matches - 1:
            # 取下一个时间戳
            next_time_str = matches[idx + 1][0]
            next_time = datetime.strptime(next_time_str, "%H:%M:%S")
            # 下一个时间 - 10秒
            end_time = next_time - timedelta(seconds=10)
        else:
            # 最后一条：当前时间 + 10秒（兜底）
            end_time = current_time + timedelta(seconds=10)
        
        # 计算结束时间相对起始时间的偏移
        end_delta = end_time - start_time
        # 处理偏移为负的情况（若end_time < start_time，强制设为start_delta + 10秒）
        if end_delta.total_seconds() < 0:
            end_delta = start_delta + timedelta(seconds=10)
        
        srt_end = (datetime(1900, 1, 1) + end_delta).strftime("%H:%M:%S,000")
        
        # 3.4 拼接SRT单行内容（序号 + 时间轴 + 文本 + 空行分隔）
        srt_index = idx + 1  # SRT序号从1开始
        srt_lines.extend([
            str(srt_index),
            f"{srt_start} --> {srt_end}",
            text,
            text_zh,
            ""  # 空行分隔不同字幕
        ])
        
        # 更新进度条
        progress_bar.update(1)
    progress_bar.close()
    # 4. 拼接所有行，生成最终SRT字符串
    srt_content = '\n'.join(srt_lines).strip()
    # 保存到SRT文件
    with open(srt_output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
        print(f"SRT 文件已保存到：{srt_output_path}")
    return srt_lines

#%%
if __name__ == "__main__":
    # 配置参数
    LIVE_FILE_PATH = 'E:\\EV\\2025-12-17_11-41-41_captions.txt'  # 你的Live文件路径
    LANGUAGE = "ja"      # 识别语言（日语）
    IS_TRANSLATE = True  # 是否翻译识别文本
    TRANSLATE_ENGINE = "model"  # 翻译引擎（model/api）
    TRANSLATE_LANG = "zh"      # 翻译目标语言（如zh/en/ja/fr）
    USE_GPU = True      # 是否使用GPU加速
    
    # 生成SRT
    srt_lines = livecaptions_to_srt(
        live_file_name = LIVE_FILE_PATH,
        language = LANGUAGE,
        is_translate = IS_TRANSLATE,
        translate_engine = TRANSLATE_ENGINE,
        translate_lang = TRANSLATE_LANG,
        use_gpu = USE_GPU
    )
    
    print("Sample SRT 0-7 lines:")
    print("\n".join(srt_lines[:7]))