# Author: bupt_gwy
# Date: 2026-10-01
"""换背景 CLI：标定 T + 流式合成 + 回封原音轨。

    python run.py --video <原视频> --alpha <alpha视频> --bg <背景图>

输出默认写到 `outputs/<原视频名>_<背景名>.mp4`。

"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vfh import paths, pipeline  # noqa: E402
from vfh.imgio import probe_video  # noqa: E402

DEF = dict(panel_w=720, panel_h=0, strength=0.90, exposure=0.35, chroma=0.60,
           sigma=None, erode=None, shadow_k=0.55, dy=0.022,
           sigma_s=0.022, edge_erode=1, crf=16, preset='veryfast', limit=0)


def auto_panel_h(video, panel_w):
    """按源片宽高比算面板高，避免拉伸变形。"""
    _, sw, sh, _ = probe_video(video)
    h = int(round(panel_w * sh / sw))
    return h - (h % 2)


def names_of(job):
    """产出名：<原视频名>_<背景名>。"""
    return (os.path.splitext(os.path.basename(job['video']))[0],
            os.path.splitext(os.path.basename(job['bg']))[0])


def process(job, defaults, verbose=True):
    p = {**defaults, **job}
    vname, bname = names_of(p)
    bg = paths.background(p['bg'])
    panel = (p['panel_w'] - p['panel_w'] % 2,
             p['panel_h'] or auto_panel_h(p['video'], p['panel_w']))
    out = p.get('out') or os.path.join(paths.OUTPUTS, f'{vname}_{bname}.mp4')
    audio = None if p.get('no_audio') else p['video']

    print(f'\n### {vname} / {bname}  {panel[0]}x{panel[1]}', flush=True)
    print(f'  原视频 {p["video"]}', flush=True)
    print(f'  alpha  {p["alpha"]}', flush=True)
    print(f'  背景图 {bg}', flush=True)

    prep = pipeline.prepare(
        vname, bg, panel, p['video'], p['alpha'], sigma=p['sigma'],
        strength=p['strength'], erode_px=p['erode'], exposure=p['exposure'],
        chroma=p['chroma'], dy_frac=p['dy'],
        sigma_frac=p['sigma_s'], limit=p['limit'], verbose=verbose)
    print(f'  {prep!r}', flush=True)
    print(f'  T 左右差={prep.info["T_left_right"]:+.3f}  '
          f'腐蚀={prep.info["erode_px"]}px  阴影ref={prep.shadow_ref:.4f}', flush=True)

    n = pipeline.stream(p['video'], p['alpha'], prep, out, audio_from=audio,
                        shadow_k=p['shadow_k'], edge_erode=p['edge_erode'],
                        crf=p['crf'], preset=p['preset'], limit=p['limit'],
                        progress=500, verbose=verbose)
    return out, n


def build_parser():
    ap = argparse.ArgumentParser(description='换背景：标定 T + 流式合成 + 回封原音轨')
    ap.add_argument('--video', help='原视频')
    ap.add_argument('--alpha', help='alpha 视频（灰度，与原视频同分辨率/帧率/帧数）')
    ap.add_argument('--bg', help='背景图（路径，或 examples/背景图/ 下的名字）')
    ap.add_argument('--out', help='默认 outputs/<原视频名>_<背景名>.mp4')
    ap.add_argument('--no-audio', action='store_true', help='不回封源片音轨')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--panel-w', type=int, default=DEF['panel_w'])
    ap.add_argument('--panel-h', type=int, default=DEF['panel_h'],
                    help='0=按源片宽高比自动算；必须偶数（yuv420p）')
    ap.add_argument('--strength', type=float, default=DEF['strength'],
                    help='光照变换强度，1.0=全量。整体偏色过头就降到 0.7~0.85')
    ap.add_argument('--exposure', type=float, default=DEF['exposure'],
                    help='T 的亮度分量开几次方。1=亮度跟着新场景走，0=完全保留人物原亮度')
    ap.add_argument('--chroma', type=float, default=DEF['chroma'],
                    help='色度强度：<1 把背景色偏往中性拉，皮肤不再被带得过红/过黄')
    ap.add_argument('--sigma', type=float, default=None,
                    help='T 的低频尺度(px)。默认按帧宽算，见 relight.SIGMA_FRAC')
    ap.add_argument('--erode', type=int, default=None,
                    help='标定时腐蚀 alpha 的半径(px)。默认按帧宽算')
    ap.add_argument('--shadow-k', type=float, default=DEF['shadow_k'],
                    help='接触阴影强度，0=不要阴影')
    ap.add_argument('--dy', type=float, default=DEF['dy'], help='阴影下移量(占帧高)')
    ap.add_argument('--sigma-s', type=float, default=DEF['sigma_s'],
                    help='阴影模糊尺度(占帧高)')
    ap.add_argument('--edge-erode', type=int, default=DEF['edge_erode'],
                    help='合成时把 alpha 收几 px，压掉抠像边缘残留')
    ap.add_argument('--crf', type=int, default=DEF['crf'])
    ap.add_argument('--preset', default=DEF['preset'])
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 帧（冒烟测试）')
    return ap


def main():
    ap = build_parser()
    a = ap.parse_args()

    defaults = {k: v for k, v in vars(a).items()
                if k in DEF or k == 'no_audio'}

    if not (a.video and a.alpha and a.bg):
        ap.error('需要 --video / --alpha / --bg 三个输入')
    for f in (a.video, a.alpha):
        if not os.path.exists(f):
            raise SystemExit(f'找不到 {f}')

    job = {'video': a.video, 'alpha': a.alpha, 'bg': a.bg, 'out': a.out}
    out, n = process(job, defaults, verbose=not a.quiet)
    print(f'  -> {out}  {n} 帧', flush=True)


if __name__ == '__main__':
    main()
