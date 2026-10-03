# Author: bupt_gwy
# Date: 2026-10-01
"""合成：人物×T×a + 阴影化背景×(1−a)。"""
import cv2
import numpy as np

from .relight import apply_transform
from .shadow import make_shadow

SHADOW_TINT = (1.0, 1.0, 1.10)      # 阴影略偏蓝


def _erode_alpha(alpha, edge_erode):
    if edge_erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * edge_erode + 1, 2 * edge_erode + 1))
        return cv2.erode(alpha, k)
    return alpha


def _apply_gain(src_u8, T, B=None):
    """逐像素套光度变换：`src × T + B`，截断到 [0,255]。"""
    src = src_u8.astype(np.float32)
    off = (B.astype(np.float32)[None, None, :] if B is not None else 0.0)
    return np.clip(src * T + off, 0, 255)


def compose(person_f, bg_f, alpha, shadow=None, shadow_k=0.55,
            shadow_tint=SHADOW_TINT, edge_erode=1):
    """person_f: [H,W,3] float32（已套过 T）；bg_f: [H,W,3] float32；alpha: [H,W] float。

    `edge_erode` 把 alpha 收几像素，压掉抠像边缘残留的原背景色。
    """
    a = np.clip(_erode_alpha(alpha, edge_erode), 0, 1)[..., None]
    bg = bg_f
    if shadow is not None and shadow_k > 0:
        tint = np.array(shadow_tint, np.float32)[None, None, :]
        bg = bg * (1.0 - shadow_k * shadow[..., None] * tint)
    return np.clip(person_f * a + bg * (1 - a), 0, 255)


def compose_frame(src_u8, alpha, bg_f, T, B=None, shadow=None, shadow_k=0.55,
                  shadow_tint=SHADOW_TINT, edge_erode=1):
    """流式管线用：套 T + 合成，一帧到底。返回 [H,W,3] float32。"""
    person = _apply_gain(src_u8, T, B)
    return compose(person, bg_f, alpha, shadow, shadow_k, shadow_tint, edge_erode)


def run(frames_u8, alphas, bg_u8, T, B=None, shadow_k=0.55, shadow_params=None,
        shadow_tint=SHADOW_TINT, edge_erode=1, shadow_ref=None):
    """整段端到端（数据能全放进内存时用；长片走流式路径）。"""
    person = apply_transform(frames_u8, T, B)
    bgf = bg_u8.astype(np.float32)
    sh = (make_shadow(alphas, ref=shadow_ref, **(shadow_params or {}))
          if shadow_k > 0 else [None] * len(alphas))
    return [compose(p, bgf, a, s, shadow_k, shadow_tint, edge_erode)
            for p, a, s in zip(person, alphas, sh)]
