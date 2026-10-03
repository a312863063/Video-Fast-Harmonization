# Author: bupt_gwy
# Date: 2026-10-01
"""alpha 的来源：灰度 alpha 视频，与源视频按帧对齐读取。"""
import os

import cv2
import numpy as np

from .imgio import FrameReader


class AlphaVideo:
    """读灰度 alpha 视频，逐帧给出 [0,1] 的 float alpha。"""

    def __init__(self, alpha_path, scale=None):
        self.path = alpha_path
        self.reader = FrameReader(alpha_path, scale=scale, pix_fmt='gray')

    def __iter__(self):
        for fr in self.reader:
            yield fr.astype(np.float32) / 255.0

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PairedReader:
    """源视频 + alpha 视频按帧对齐读取，两块都缩放到同一面板尺寸。

    读完较短的一条就停。`scale` 是面板 (宽, 高)。
    """

    def __init__(self, video_path, alpha_path, scale=None):
        self.video = FrameReader(video_path, scale=scale)
        self.alpha = AlphaVideo(alpha_path, scale=scale)
        self.pw = self.video.width
        self.ph = self.video.height
        self.n = min(self.video.n, self.alpha.reader.n)
        self.fps = self.video.fps

    def __iter__(self):
        for fr, al in zip(self.video, self.alpha):
            yield fr.astype(np.uint8), al

    def close(self):
        self.video.close()
        self.alpha.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AlphaFixer:
    """把 alpha 用逐帧 RMBG 软 alpha 收一收：

        a_final = min( a_input, dilate(smooth(a_RMBG), d) )

    需要逐帧 RMBG 的 alpha 目录（每帧一张 png）。目录为空时 `enabled=False`，
    调用时原样返回输入 alpha。
    """

    def __init__(self, rmbg_dir, dilate=12, smooth=2):
        self.smooth = smooth
        self.k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * dilate + 1, 2 * dilate + 1))
        self.files = (sorted(os.path.join(rmbg_dir, f) for f in os.listdir(rmbg_dir)
                             if f.endswith('.png')) if rmbg_dir
                      and os.path.isdir(rmbg_dir) else [])
        self.cache = {}
        self.enabled = bool(self.files)

    def _rmbg(self, i):
        if not self.enabled or i < 0 or i >= len(self.files):
            return None
        if i not in self.cache:
            self.cache[i] = cv2.imdecode(np.fromfile(self.files[i], np.uint8),
                                         cv2.IMREAD_GRAYSCALE)
        return self.cache[i]

    def __call__(self, i, alpha_f):
        """alpha_f: [H,W] float32 in [0,1]。"""
        if not self.enabled:
            return alpha_f
        ws = [self._rmbg(j) for j in range(i - self.smooth, i + self.smooth + 1)
              if 0 <= j < len(self.files)]
        if not ws:
            return alpha_f
        sm = np.mean(ws, axis=0)
        cm = cv2.dilate((sm > 127).astype(np.uint8) * 255, self.k)
        out = np.minimum(alpha_f * 255.0, cm.astype(np.float32))
        for j in [j for j in self.cache if j < i - self.smooth - 1]:
            del self.cache[j]
        return out / 255.0
