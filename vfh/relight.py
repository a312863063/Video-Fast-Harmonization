# Author: bupt_gwy
# Date: 2026-10-01
"""低频逐像素增益场 T 的标定与套用。
T 是逐像素乘法，不改动像素之间的相对关系，因此人物结构与身份逐像素保留。
"""
import cv2
import numpy as np

# T 的低频尺度，占帧宽的比例。
SIGMA_FRAC = 0.156
# 标定时腐蚀 alpha 的半径，同样占帧宽的比例。
ERODE_FRAC = 0.034


def _blur(x, sigma):
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigma)


def sigma_for(width, frac=SIGMA_FRAC):
    """把低频尺度从「帧宽的几分之几」换成像素，避免换分辨率就要重调参数。"""
    return max(1.0, width * frac)


def erode_for(width, frac=ERODE_FRAC):
    """标定腐蚀半径（像素），按帧宽缩放。"""
    return max(1, int(round(width * frac)))


# --------------------------------------------------------------------- 标定
def _region_mask(alpha, select, erode_px):
    """圈出可信区域：`person`=人物内部，`room`=旧房间。"""
    m = (alpha > 0.5) if select == 'person' else (alpha < 0.1)
    m = m.astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, k)
    m = m.astype(bool)
    if m.sum() < 200:                 # 区域太小/腐蚀没了，退回不腐蚀
        m = (alpha > 0.5) if select == 'person' else (alpha < 0.5)
    return m


def _finish_T(R, m, sigma, strength, clamp, out_clamp=(0.15, 6.0), exposure=1.0,
              chroma=1.0):
    """比值场 → 限幅 → 区域外填中位数 → 低频化 → 亮度分量按 exposure 开方
    → 色度分量按 chroma 幂次收放 → 按 strength 在 1.0 与 T 之间线性插值。
    返回 (T, 区域中位数)。

    `chroma` < 1 把色偏往中性拉（皮肤不再被背景色调带得过红/过黄），= 1 为原样。
    """
    R = np.clip(R, clamp[0], clamp[1])
    med = np.median(R[m], axis=0)
    R[~m] = med                       # 区域外填中位数，别把不该要的比值带进来
    S = _blur(R, sigma)

    if (exposure != 1.0 or chroma != 1.0) and S.ndim == 3:
        luma = (S * np.array([0.299, 0.587, 0.114], np.float32)).sum(2, keepdims=True)
        luma = np.maximum(luma, 1e-3)
        S = (luma ** exposure) * ((S / luma) ** chroma)

    T = 1.0 + (S - 1.0) * strength    # strength=1 全量，=0 完全不改
    return np.clip(T, out_clamp[0], out_clamp[1]), med


def _stats(T, person_mask, mode, global_ratio, B, extra=None):
    info = {
        'mode': mode,
        'global_ratio': [float(x) for x in global_ratio],
        'global_gain_mean': float(np.mean(global_ratio)),
        'B_offset': [float(x) for x in (B if B is not None else np.zeros(3))],
        'T_mean_in_mask': T[person_mask].mean(0).tolist(),
        'T_std_in_mask': T[person_mask].std(0).tolist(),
        'T_left_right': float(T[person_mask][:, 0].mean()
                              - T[person_mask][:, 2].mean()),
        'mask_frac': float(person_mask.mean()),
    }
    return {**info, **(extra or {})}


def calibrate_from_bg(src_f0_u8, alpha_f0, bg_u8, *, sigma, strength=1.0,
                      erode_px=24, mode='gain', clamp=(0.25, 4.0), min_src=8.0,
                      pre_frac=0.34, room_max_alpha=0.1, exposure=0.6, chroma=1.0):
    """无模型后端：用背景图标定光照变换。

        T = blur( 新场景低频 / 旧房间低频 )      比值取在 alpha<room_max_alpha 的房间区

    两侧各先 blur 一次（`pre_frac`×sigma）再相除。返回 (T, B=None, info)。
    """
    pre = max(2.0, sigma * pre_frac)
    src_lf = _blur(src_f0_u8.astype(np.float32), pre)
    bg_lf = _blur(bg_u8.astype(np.float32), pre)

    m = _region_mask(alpha_f0, 'room', erode_px)
    R = bg_lf / np.maximum(src_lf, min_src)
    T, med = _finish_T(R, m, sigma, strength, clamp, exposure=exposure, chroma=chroma)

    return (T.astype(np.float32), None,
            _stats(T, _region_mask(alpha_f0, 'person', erode_px), mode, med, None,
                   extra={'sigma_pre': pre, 'room_frac': float(m.mean()),
                          'exposure': exposure, 'chroma': chroma}))


# --------------------------------------------------------------------- 套用
def apply_transform(frames_u8, T, B=None):
    """逐帧逐像素光度变换。frames_u8: iterable of [H,W,3] uint8 → list。"""
    off = (B.astype(np.float32)[None, None, :] if B is not None else 0.0)
    return [np.clip(f.astype(np.float32) * T + off, 0, 255).astype(np.uint8)
            for f in frames_u8]


def tint_img(T):
    """T 可视化：以 1.0 为基准，偏亮=红、偏暗=蓝。"""
    g = T.mean(axis=2)
    v = np.clip((g - 1.0) * 2.0 + 0.5, 0, 1)
    rgb = np.zeros((*g.shape, 3), np.float32)
    rgb[..., 0] = np.clip((v - 0.5) * 2, 0, 1)
    rgb[..., 2] = np.clip((0.5 - v) * 2, 0, 1)
    rgb[..., 1] = 1 - np.abs(v - 0.5) * 2 * 0.6
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
