# video2srt/__init__.py
import importlib.metadata
# 从已安装的包中读取版本号（对应pyproject.toml中的version）
try:
    __version__ = importlib.metadata.version("financeDA")
except importlib.metadata.PackageNotFoundError:
    # 未安装时（本地开发）的兜底版本
    __version__ = "0.0.0-dev (local or not installed)"
# __version__ = "0.2.4"  # 版本号（与setup.cfg一致）

from .main import hello_video2srt, sample
from .video2srt import video_to_srt
from .livecaptions2srt import livecaptions_to_srt

__all__ = ["hello_video2srt", "sample", "video_to_srt", "livecaptions_to_srt"]  # 导出的公开接口