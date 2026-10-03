# Author: bupt_gwy
# Date: 2026-10-01
"""2×2 对照视频：`原视频 | 背景1 | 背景2 | 背景3`。

    左上 原视频   右上 背景1
    左下 背景2    右下 背景3

    python make_quad_2x2.py --video <原视频> --alpha <alpha视频> \\
                            --bgs <背景1> <背景2> <背景3>

输出默认写到 `outputs/<视频名>_<风格1>_<风格2>_<风格3>.mp4`。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vfh import composite, paths, pipeline, shadow  # noqa: E402
from vfh.alpha import PairedReader  # noqa: E402
from vfh.imgio import FrameWriter, probe_video  # noqa: E402

def auto_panel_h(video, panel_w):
    """按源片宽高比算面板高，避免拉伸变形。"""
    _, sw, sh, _ = probe_video(video)
    h = int(round(panel_w * sh / sw))
    return h - (h % 2)


def assemble(panels, panel, gap):
    """把 4 块面板拼成 2×2 整帧。panels 顺序：原视频, 背景1, 背景2, 背景3。"""
    W, H = panel
    total_w, total_h = W * 2 + gap, H * 2 + gap
    rows = [np.concatenate([panels[2 * r], np.zeros((H, gap, 3), np.float32),
                            panels[2 * r + 1]], axis=1) for r in range(2)]
    return np.concatenate([rows[0], np.zeros((gap, total_w, 3), np.float32), rows[1]],
                          axis=0), (total_w, total_h)


def main():
    ap = argparse.ArgumentParser(description='2×2 对照视频')
    ap.add_argument('--video', required=True, help='原视频')
    ap.add_argument('--alpha', required=True, help='alpha 视频（灰度，与原视频同规格）')
    ap.add_argument('--bgs', nargs=3, required=True, metavar=('BG1', 'BG2', 'BG3'),
                    help='三张背景图（路径，或 examples/背景图/ 下的名字）')
    ap.add_argument('--out', default=None,
                    help='默认 outputs/<视频名>_<风格1>_<风格2>_<风格3>.mp4')
    ap.add_argument('--panel-w', type=int, default=720)
    ap.add_argument('--panel-h', type=int, default=0,
                    help='0=按源片宽高比自动算（不变形）；必须偶数')
    ap.add_argument('--gap', type=int, default=4, help='格子之间的黑分隔线宽度')
    ap.add_argument('--strength', type=float, default=0.90)
    ap.add_argument('--exposure', type=float, default=0.35)
    ap.add_argument('--chroma', type=float, default=0.60,
                    help='色度强度：<1 把背景色偏往中性拉')
    ap.add_argument('--shadow-k', type=float, default=0.55)
    ap.add_argument('--edge-erode', type=int, default=1)
    ap.add_argument('--crf', type=int, default=16)
    ap.add_argument('--preset', default='veryfast')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 帧（冒烟测试）')
    ap.add_argument('--audio-from', default=None,
                    help='音轨来源文件（默认原视频本身）')
    ap.add_argument('--no-audio', action='store_true')
    a = ap.parse_args()

    for f in [a.video, a.alpha] + list(a.bgs):
        if not os.path.exists(f) and not os.path.exists(paths.background(f)):
            raise SystemExit(f'找不到 {f}')

    bgs = [paths.background(b) for b in a.bgs]
    vname = os.path.splitext(os.path.basename(a.video))[0]
    stems = [os.path.splitext(os.path.basename(b))[0] for b in bgs]
    out = a.out or os.path.join(paths.OUTPUTS, f'{vname}_' + '_'.join(stems) + '.mp4')
    panel_w = a.panel_w - (a.panel_w % 2)
    panel_h = a.panel_h or auto_panel_h(a.video, panel_w)
    panel = (panel_w, panel_h)
    if panel[0] % 2 or panel[1] % 2:
        ap.error('--panel-w/--panel-h 必须是偶数')

    # ---- 标定三张背景 ----
    print(f'### {vname} 2x2  {panel[0]}x{panel[1]}×4  gap={a.gap}', flush=True)
    print(f'  原视频 {a.video}\n  alpha  {a.alpha}', flush=True)
    shadow_ref = None
    preps = []
    for k, bg in enumerate(bgs, 1):
        bstem = os.path.splitext(os.path.basename(bg))[0]
        print(f'\n-- bg{k}: {bg}', flush=True)
        p = pipeline.prepare(
            vname, bg, panel, a.video, a.alpha, strength=a.strength,
            exposure=a.exposure, chroma=a.chroma,
            shadow_ref=shadow_ref, limit=a.limit)
        if shadow_ref is None:
            shadow_ref = p.shadow_ref          # 只跟素材有关，三个背景共用
        print(f'   {p!r}', flush=True)
        print(f'   T 左右差={p.info["T_left_right"]:+.3f}', flush=True)
        preps.append(p)

    # ---- 逐帧：一次解码，出 4 格 ----
    H = panel[1]
    gap = a.gap
    mat = PairedReader(a.video, a.alpha, scale=panel)
    fps = mat.fps

    # 先写临时文件，全部写完再改名（中断时不留残片）
    tmp = out + '.part.mp4' if out.endswith('.mp4') else out + '.part'
    print(f'\n合成 -> {out}  2x2 @ {fps:g}fps', flush=True)
    t0 = time.time()
    n = 0
    size = None
    audio = a.audio_from or a.video
    with FrameWriter(tmp, fps, assemble([np.zeros((H, panel[0], 3), np.uint8)] * 4,
                                        panel, gap)[1],
                     crf=a.crf, preset=a.preset,
                     source_audio=None if (a.limit or a.no_audio) else audio) as w:
        for i, (src, alpha) in enumerate(mat):
            if a.limit and i >= a.limit:
                break
            panels = [src.astype(np.float32)]          # 第 1 格：原视频
            for prep in preps:
                sh = (shadow.make_shadow([alpha], ref=prep.shadow_ref,
                                         **prep.shadow_params)[0]
                      if a.shadow_k > 0 else None)
                panels.append(composite.compose_frame(
                    src, alpha, prep.bg_f, prep.T, prep.B, sh,
                    shadow_k=a.shadow_k, edge_erode=a.edge_erode))
            frame, size = assemble(panels, panel, gap)

            w.write(np.clip(frame, 0, 255).astype(np.uint8))
            n = w.n
            if i and i % 500 == 0:
                el = time.time() - t0
                print(f'    {i}/{mat.n}  {el:.0f}s ({i / el:.1f} fps)', flush=True)
    mat.close()
    if tmp != out:
        os.replace(tmp, out)          # 只有跑到这里，成片才算数
    print(f'[完成] {n} 帧 -> {out}  ({time.time() - t0:.0f}s)  '
          f'{size[0]}x{size[1]}', flush=True)


if __name__ == '__main__':
    main()
