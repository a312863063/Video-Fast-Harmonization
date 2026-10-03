# Author: bupt_gwy
# Date: 2026-10-01
"""接触阴影：把 alpha 下移并乘垂直权重后模糊，整段共用同一个归一化系数。"""
import cv2
import numpy as np


def _blur(x, sigma):
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigma)


def _shifted_profile(H, W, dy_frac, sigma_frac, floor_pow, blur_extra):
    """返回 (下移量 dy, 模糊尺度 sigma, 垂直权重 prof)。"""
    dy = max(1, int(H * dy_frac))
    sigma = max(1.0, H * sigma_frac) + blur_extra
    prof = (np.linspace(0, 1, H, dtype=np.float32) ** floor_pow)[:, None]
    return dy, sigma, prof


def _raw(alpha, dy, prof):
    W = alpha.shape[1]
    s = alpha.astype(np.float32) * prof
    return np.concatenate([np.zeros((dy, W), np.float32), s[:-dy]], axis=0)


def make_shadow(alphas, dy_frac=0.018, sigma_frac=0.030, floor_pow=2.5,
                norm_pct=90, blur_extra=0.0, ref=None):
    """从 alpha 序列生成接触阴影图（0..1）。`ref` 给了就直接当归一化系数用。"""
    H, W = alphas[0].shape
    dy, sigma, prof = _shifted_profile(H, W, dy_frac, sigma_frac, floor_pow, blur_extra)
    outs = [_blur(_raw(a, dy, prof), sigma) for a in alphas]
    if ref is None:
        ref = float(np.percentile([o.max() for o in outs], norm_pct))
    ref = max(ref, 1e-6)
    return [np.clip(o / ref, 0, 1) for o in outs]


def shadow_norm(alphas, dy_frac=0.018, sigma_frac=0.030, floor_pow=2.5,
                norm_pct=90, blur_extra=0.0):
    """只算 `make_shadow` 的归一化系数，不保留结果图（流式 pass1 用）。"""
    H, W = alphas[0].shape
    dy, sigma, prof = _shifted_profile(H, W, dy_frac, sigma_frac, floor_pow, blur_extra)
    mx = [_blur(_raw(a, dy, prof), sigma).max() for a in alphas]
    return max(float(np.percentile(mx, norm_pct)), 1e-6)
