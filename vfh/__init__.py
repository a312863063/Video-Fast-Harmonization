# Author: bupt_gwy
# Date: 2026-10-01
"""vfh —— Video Fast Harmonization：零生成换背景（光照标定 + 逐像素套用）。

| 模块 | 干什么 |
|---|---|
| `paths` | 路径出处（项目内部：产物目录、背景图目录） |
| `imgio` | 图像/视频 IO（兼容非 ASCII 路径，视频走 ffmpeg 管道流式） |
| `relight` | 核心：标定低频光照场 T / 逐帧套用 |
| `shadow` | 接触阴影（从 alpha 生成，非物理投影） |
| `composite` | 合成 `人×T×a + 背景×(1−a)` |
| `alpha` | alpha 来源（读灰度 alpha 视频） |
| `pipeline` | 标定 + 流式合成，两个入口脚本共用 |

不依赖任何深度学习框架，只用 numpy / OpenCV / ffmpeg。

用法速览：

    from vfh import pipeline
    p = pipeline.prepare('demo', 'bg.jpg', (720, 1270), 'video.mp4', 'alpha.mp4')
    print(p)
"""
from . import alpha, composite, imgio, paths, pipeline, relight, shadow  # noqa: F401
from .alpha import AlphaFixer, AlphaVideo, PairedReader  # noqa: F401
from .composite import compose, compose_frame  # noqa: F401
from .imgio import FrameReader, FrameWriter, fit_bg, imread_rgb, imread_u, imwrite_u  # noqa: F401
from .relight import apply_transform, calibrate_from_bg, sigma_for, tint_img  # noqa: F401
from .pipeline import Prepared, prepare, stream  # noqa: F401

__all__ = [
    'AlphaFixer', 'AlphaVideo', 'FrameReader', 'FrameWriter', 'PairedReader',
    'Prepared', 'apply_transform', 'calibrate_from_bg', 'compose', 'compose_frame',
    'fit_bg', 'imread_rgb', 'imread_u', 'imwrite_u', 'paths', 'pipeline',
    'prepare', 'sigma_for', 'stream', 'tint_img',
]
