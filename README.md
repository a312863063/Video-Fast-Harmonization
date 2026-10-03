# Video-Fast-Harmonization

自动分析前后景光照与色调差异，在换视频背景时，快速让前后景调融为一体。

---

# 直出运行效果
#### 测试素材收集自短视频平台，仅作实际场景下效果测试。测试素材不随仓库公开，请自备测试数据。本项目允许商业化。
<p align="center">
<img src="docs/test2_neutral_white_studio_neutral_living_room_warm_brick_corner.gif" alt="演示">
<br /><br />
<img src="docs/test2_cool_neon_street_neutral_white_niche_warm_sunset_window.gif" alt="演示">
<br /><br />
<img src="docs/test1_cool_neon_dusk_room_cool_neon_teal_room_cool_neon_violet_room.gif" alt="演示">

</p>

---



# 安装配置

## 1. 依赖与安装

```bash
pip install -r requirements.txt
ffmpeg -version      # 确认可执行
```


## 2. 输入数据


| 输入 | 说明                                                                                |
|---|-------------------------------------------------------------------------------------|
| 原视频 | 要换背景的视频                                                                      |
| alpha 视频 | 灰度视频，与原视频同分辨率、同帧率、同帧数                                          |
| 背景图 | 任意图片（你自己的照片，给路径即可）|



## 3. 快速开始

```bash
# 单条换背景：原视频 + alpha 视频 + 背景图
python run.py --video <你的视频.mp4> --alpha <你的alpha.mp4> \
           --bg <你的背景图.jpg>

# 2×2 对照：左上原视频，右上/左下/右下 三张背景
python make_quad_2x2.py --video <你的视频.mp4> --alpha <你的alpha.mp4> \
                     --bgs <背景1.jpg> <背景2.jpg> <背景3.jpg>
```



<details>
<summary><b>4. 技术原理</b>（点击展开）</summary>

### 4.1 标定低频光照场

以**新背景**为参考，比的是**旧房间 → 新场景**的光照变化：

```
T = blur( 新场景低频 / 旧房间低频 )          比值取在 alpha < 0.1 的「旧房间」区域
```

实现上两侧先各模糊一次再相除（抑制霓虹灯牌、窗框这类高频纹理把比值踢到极端）：

```
pre   = max(2, 0.34·sigma)
R     = blur(bg, pre) / max( blur(src0, pre), 8.0 )
```

得到比值场后进行后处理：

```
R ← clip(R, 0.25, 4.0)                        限幅
R[房间外] ← median(R[房间内])                  区域外填中位数
S  = blur(R, sigma)                           低频化，只留光照场
luma = 0.299·S_r + 0.587·S_g + 0.114·S_b
S  = luma^exposure · (S / luma)^chroma        亮度开 exposure 次方、色度开 chroma 次方
T  = 1 + (S − 1)·strength                     强度插值
T ← clip(T, 0.15, 6.0)
```


### 4.2 接触阴影

从 alpha 生成（**不是物理正确的投影**）：把 alpha 下移一点模拟光从上方来，乘一个「越靠下越强」的垂直权重，再大幅模糊：

```
dy    = max(1, 0.022·H)            sigma = max(1, 0.022·H)
prof  = linspace(0, 1, H)^2.5
raw   = shift_down( alpha · prof , dy )
shadow = blur(raw, sigma) / ref      ref = 整段 max 的 90 分位数
```

归一化系数 `ref` **整段只算一次**（流式合成前扫一遍），逐帧归一化会让阴影强度逐帧抖 = 闪。

### 4.3 合成

```
person = clip( src · T + B , 0, 255 )            B 为可选环境光偏置（当前后端恒为 None）
a      = clip( erode(alpha, edge_erode) , 0, 1)  收 1px 压掉抠像边缘残留色
bg'    = bg · (1 − shadow_k · shadow · tint)     tint = (1, 1, 1.10) 阴影略偏蓝
out    = clip( person·a + bg'·(1−a) , 0, 255 )
```

</details>

## 5. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--strength` | 0.90 | 光照变换强度，1.0 = 全量。整体偏色过头就降到 0.7~0.85 |
| `--exposure` | 0.35 | `T` 亮度分量开几次方。1 = 亮度跟着新场景走，0 = 完全保留人物原亮度 |
| `--chroma` | 0.60 | 色度强度。`<1` 把背景色偏往中性拉——暖/霓虹背景不会把脸带得发红发黄。1.0 = 完全套用（肤色会被背景带过头） |
| `--sigma` | 帧宽×0.156 | `T` 的低频尺度(px)。↑ = 更接近全局色偏、更抗姿态变化 |
| `--erode` | 帧宽×0.034 | 标定时腐蚀 alpha 的半径(px)，避免边界把背景比值混进来 |
| `--shadow-k` | 0.55 | 接触阴影强度，0 = 不要阴影 |
| `--dy` / `--sigma-s` | 0.022 / 0.022 | 阴影下移量 / 模糊尺度（占帧高比例） |
| `--edge-erode` | 1 | 合成时把 alpha 收几像素 |
| `--panel-w` / `--panel-h` | 720 / 0 | 面板尺寸。`--panel-h 0` = 按源片宽高比自动算；**必须偶数**（yuv420p） |
| `--crf` / `--preset` | 16 / veryfast | 编码参数 |
| `--limit` | 0 | 只跑前 N 帧，冒烟测试用 |



## 6. 目录结构

```
Video-Fast-Harmonization/
├─ README.md          ← 本文件
├─ LICENSE            ← MIT
├─ requirements.txt
├─ run.py             ← 入口：原视频 + alpha + 背景图 → 换背景融合视频
├─ make_quad_2x2.py   ← 入口：原视频 + alpha + 三张背景图 → 2×2 对照视频
├─ outputs/           ← 产物（运行时创建）
└─ vfh/               ← 库
   ├─ paths.py        ← 路径出处（项目内部：产物目录、背景图目录）
   ├─ imgio.py        ← 图像/视频 IO；兼容非 ASCII 路径，视频走 ffmpeg 管道流式
   ├─ relight.py      ← 核心：标定低频光照场 T / 逐帧套用
   ├─ shadow.py       ← 接触阴影
   ├─ composite.py    ← 合成 人×T×a + 阴影化背景×(1−a)
   ├─ alpha.py        ← alpha 来源：读灰度 alpha 视频
   ├─ pipeline.py     ← 标定 + 流式合成串起来，两个入口共用
```



## 7. License

MIT © 2026 bupt_gwy

