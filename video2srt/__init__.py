# findaok/__init__.py
__version__ = "0.2.2"  # 版本号（与setup.cfg一致）

from .main import hello_video2srt, sample
from .video2srt import video_to_srt

__all__ = ["hello_video2srt", "sample", "video_to_srt"]  # 导出的公开接口