# Author: bupt_gwy
# Date: 2026-10-01
"""路径出处：默认指向本项目内部。"""
import os

# 项目根目录（vfh/ 的上一层）
HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 产物目录
OUTPUTS = os.path.join(HOME, 'outputs')
# 背景图目录
BG_DIR = os.path.join(HOME, 'examples', '背景图')


def background(name):
    """找背景图：先当路径试，再在 `examples/背景图/` 里按名字找（扩展名可省略）。"""
    if os.path.exists(name):
        return name
    stem, ext = os.path.splitext(name)
    for e in ((ext,) if ext else ()) + ('.jpg', '.png', '.webp'):
        p = os.path.join(BG_DIR, stem + e)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'找不到背景图 {name!r}（找过：{name} 与 {BG_DIR}）')
