# Author: bupt_gwy
# Date: 2026-10-01
"""把「标定 + 流式合成」串起来，两个入口脚本共用。

长片必须流式：整段展开进内存会爆掉。
"""
import os
import time

import numpy as np

from . import composite, relight, shadow
from .alpha import AlphaFixer, PairedReader
from .imgio import FrameWriter, fit_bg
from .paths import background as bg_path


class Prepared:
    """一个 (素材, 背景) 标定完的中间量。"""

    def __init__(self, name, bg_name, panel, bg_f, T, B, info, shadow_ref, fps, n,
                 shadow_params):
        self.name = name
        self.bg_name = bg_name
        self.panel = panel
        self.bg_f = bg_f
        self.T = T
        self.B = B
        self.info = info
        self.shadow_ref = shadow_ref
        self.shadow_params = shadow_params
        self.fps = fps
        self.n = n

    def __repr__(self):
        m = ' '.join(f'{x:.3f}' for x in self.info['T_mean_in_mask'])
        return (f'<Prepared {self.name}/{os.path.basename(self.bg_name)} '
                f'{self.panel[0]}x{self.panel[1]} '
                f'T均值(RGB)={m} 覆盖={self.info["mask_frac"]:.1%}>')


def calibrate_t(name, bg_name, panel, first_src_u8, first_alpha, *, sigma=None,
                strength=1.0, erode_px=None, mode='gain', exposure=0.6, chroma=1.0,
                verbose=True):
    """标定 (素材, 背景) 的 T。返回 (背景图, T, B, info)。"""
    W, H = panel
    bg_u8 = fit_bg(bg_path(bg_name), W, H)
    sig = sigma or relight.sigma_for(W)
    ero = relight.erode_for(W) if erode_px is None else erode_px
    if verbose:
        print(f'  [标定] {os.path.basename(bg_name)}  sigma={sig:.1f}  erode={ero}',
              flush=True)
    T, B, info = relight.calibrate_from_bg(first_src_u8, first_alpha, bg_u8,
                                           sigma=sig, strength=strength,
                                           erode_px=ero, mode=mode,
                                           exposure=exposure, chroma=chroma)
    info['sigma'] = sig
    info['erode_px'] = ero
    info['strength'] = strength
    return bg_u8, T, B, info


def shadow_ref_for(video_path, alpha_path, panel, *, dy_frac=0.022, sigma_frac=0.022,
                   fixer=None, limit=0, verbose=True):
    """扫一遍 alpha 求接触阴影的归一化系数，整段共用一个固定值。

    只跟素材有关、跟背景无关，同一素材的多个背景应复用同一个 ref。
    """
    t0 = time.time()
    m = PairedReader(video_path, alpha_path, scale=panel)
    mx = []
    try:
        for i, (_src, a) in enumerate(m):
            if limit and i >= limit:
                break
            if fixer is not None:
                a = fixer(i, a)
            mx.append(shadow.shadow_norm([a], dy_frac=dy_frac,
                                         sigma_frac=sigma_frac))
    finally:
        m.close()
    ref = max(float(np.percentile(mx, 90)), 1e-6)
    if verbose:
        print(f'  [阴影] {len(mx)} 帧扫完 {time.time() - t0:.0f}s  ref={ref:.4f}',
              flush=True)
    return ref


def prepare(name, bg_name, panel, video_path, alpha_path, *, sigma=None, strength=1.0,
            erode_px=None, mode='gain', exposure=0.6, chroma=1.0,
            dy_frac=0.022, sigma_frac=0.022, rmbg_dir=None, rmbg_dilate=12,
            rmbg_smooth=2, shadow_ref=None, limit=0, verbose=True):
    """读首帧 → 标定 T → 拿阴影系数。返回 `Prepared`。

    `shadow_ref` 给了就跳过扫描（同一素材跑多个背景时传进来复用）。
    """
    m = PairedReader(video_path, alpha_path, scale=panel)
    try:
        src0, a0 = next(iter(m))
    finally:
        m.close()
    fixer = AlphaFixer(rmbg_dir, rmbg_dilate, rmbg_smooth) if rmbg_dir else None
    bg_u8, T, B, info = calibrate_t(name, bg_name, panel, src0, a0, sigma=sigma,
                                    strength=strength, erode_px=erode_px, mode=mode,
                                    exposure=exposure, chroma=chroma, verbose=verbose)
    sp = {'dy_frac': dy_frac, 'sigma_frac': sigma_frac}
    if shadow_ref is None:
        shadow_ref = shadow_ref_for(video_path, alpha_path, panel, fixer=fixer,
                                    limit=limit, verbose=verbose, **sp)
    return Prepared(name, bg_name, panel, bg_u8.astype(np.float32), T, B, info,
                    shadow_ref, m.fps, m.n, sp)


def stream(video_path, alpha_path, prepared, out_path, *, audio_from=None,
           shadow_k=0.55, edge_erode=1, crf=16, preset='veryfast',
           limit=0, fixer=None, progress=500, verbose=True):
    """流式合成一条片子：一帧解码→套 T→合成→编码，内存恒定。返回帧数。

    `limit>0` 只跑前 N 帧（冒烟测试用；此时不回封音轨，帧数对不上）。
    """
    p = prepared
    W, H = p.panel
    t0 = time.time()
    matte = PairedReader(video_path, alpha_path, scale=p.panel)
    audio = None if (limit or not audio_from) else audio_from
    with FrameWriter(out_path, p.fps, (W, H), crf=crf, preset=preset,
                     source_audio=audio) as w:
        for i, (src, a) in enumerate(matte):
            if limit and i >= limit:
                break
            if fixer is not None:
                a = fixer(i, a)
            sh = (shadow.make_shadow([a], ref=p.shadow_ref, **p.shadow_params)[0]
                  if shadow_k > 0 else None)
            out = composite.compose_frame(
                src, a, p.bg_f, p.T, p.B, sh, shadow_k=shadow_k,
                edge_erode=edge_erode)
            w.write(np.clip(out, 0, 255).astype(np.uint8))
            if verbose and progress and i and i % progress == 0:
                el = time.time() - t0
                print(f'    {i}/{p.n}  {el:.0f}s ({i / el:.1f} fps)', flush=True)
        n = w.n
    matte.close()
    if verbose:
        print(f'  [完成] {n} 帧 -> {out_path}  ({time.time() - t0:.0f}s)', flush=True)
    return n
