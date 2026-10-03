# Author: bupt_gwy
# Date: 2026-10-01
"""图像与视频 IO：读图兼容非 ASCII 路径，视频走 ffmpeg 管道逐帧流式处理。"""
import os
import subprocess

import cv2
import numpy as np


# --------------------------------------------------------------------- 单张图
def imread_u(path, flags=cv2.IMREAD_COLOR):
    """读图，兼容非 ASCII 路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f'读不了 {path}（空文件？）')
    return cv2.imdecode(data, flags)


def imread_rgb(path, flags=cv2.IMREAD_COLOR):
    """读成 RGB（管线内部统一用 RGB）。"""
    return cv2.cvtColor(imread_u(path, flags), cv2.COLOR_BGR2RGB)


def imwrite_u(path, img):
    """写图，兼容非 ASCII 路径。"""
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f'编码失败 {path}')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf.tofile(path)


def fit_bg(path, W, H):
    """背景图 cover 缩放 + 居中裁到 W×H。返回 [H,W,3] uint8 RGB。"""
    im = imread_rgb(path)
    bh, bw = im.shape[:2]
    s = max(W / bw, H / bh)
    im = cv2.resize(im, (max(W, int(bw * s + .5)), max(H, int(bh * s + .5))),
                    interpolation=cv2.INTER_LANCZOS4)
    bh, bw = im.shape[:2]
    x0, y0 = (bw - W) // 2, (bh - H) // 2
    return im[y0:y0 + H, x0:x0 + W]


# --------------------------------------------------------------------- 视频
def probe_video(path):
    """不解码就拿到 (fps, w, h, n_frames)。"""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True, check=True).stdout
    d = dict(line.split('=', 1) for line in out.strip().splitlines() if '=' in line)
    num, den = d['r_frame_rate'].split('/')
    w, h = int(d['width']), int(d['height'])
    n = int(d.get('nb_frames') or 0)
    if n <= 0:                      # 容器没记帧数时数 packet，仍然不解码
        n = int(subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
             '-show_entries', 'stream=nb_read_packets',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, check=True).stdout.strip())
    return float(num) / float(den), w, h, n


class FrameReader:
    """ffmpeg 管道顺序解码成 RGB。只能前进，回退就重开解码器。"""

    def __init__(self, path, scale=None, pix_fmt='rgb24'):
        self._proc = None
        self._pos = 0
        self.path = path
        self.pix_fmt = pix_fmt
        f, w, h, n = probe_video(path)
        self.fps, self.src_w, self.src_h, self.n = f, w, h, n
        self.scale = scale                       # (W, H) 或 None
        self.width, self.height = scale or (w, h)
        self.channels = 3 if pix_fmt == 'rgb24' else 1
        self.frame_bytes = self.width * self.height * self.channels

    def _spawn(self, start):
        self.close()
        vf = []
        if self.scale:
            vf.append(f'scale={self.width}:{self.height}:force_original_aspect_ratio=disable')
        cmd = ['ffmpeg', '-v', 'error']
        if start:
            cmd += ['-ss', f'{start / self.fps:.6f}']
        cmd += ['-i', self.path, '-fps_mode', 'passthrough']
        if vf:
            cmd += ['-vf', ','.join(vf)]
        cmd += ['-f', 'rawvideo', '-pix_fmt', self.pix_fmt, '-']
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL,
                                      bufsize=self.frame_bytes * 4)
        self._pos = start

    def _read_one(self):
        buf = self._proc.stdout.read(self.frame_bytes)
        if len(buf) < self.frame_bytes:
            return None
        a = np.frombuffer(buf, np.uint8)
        if self.channels == 1:
            return a.reshape(self.height, self.width)
        return a.reshape(self.height, self.width, self.channels)

    def get(self, idx):
        if idx < 0 or idx >= self.n:
            raise IndexError(f'frame {idx} 越界 (n={self.n})')
        if self._proc is None or idx < self._pos:
            self._spawn(idx)
        frame = None
        while self._pos <= idx:
            frame = self._read_one()
            if frame is None:
                raise RuntimeError(f'{self.path} 在帧 {self._pos} 提前结束')
            self._pos += 1
        return frame

    def __iter__(self):
        self.close()
        self._spawn(0)
        return self

    def __next__(self):
        if self._pos >= self.n:
            raise StopIteration
        frame = self._read_one()
        if frame is None:
            raise StopIteration
        self._pos += 1
        return frame

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class FrameWriter:
    """把 RGB 帧写进 ffmpeg。`source_audio` 给了就把它的音轨 copy 过去。"""

    def __init__(self, path, fps, size, crf=16, preset='veryfast', pix_fmt='yuv420p',
                 source_audio=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        W, H = size
        cmd = ['ffmpeg', '-v', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', f'{fps}', '-i', '-']
        if source_audio:
            cmd += ['-i', source_audio, '-map', '0:v', '-map', '1:a',
                    '-c:a', 'aac', '-b:a', '160k', '-shortest']
        cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
                '-pix_fmt', pix_fmt, path]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stderr=subprocess.PIPE, bufsize=10 ** 8)
        self.path = path
        self.n = 0

    def write(self, rgb):
        self._proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.n += 1

    def close(self):
        if self._proc is None:
            return
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode('utf-8', 'replace')
        rc = self._proc.wait()
        self._proc = None
        if rc != 0:
            raise RuntimeError(f'ffmpeg 退出码 {rc}：{err.strip()}')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
