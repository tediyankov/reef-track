"""Video probing, proxy creation and random-access frame reading.

Why a *proxy*? Raw GoPro footage here is 4K HEVC at 119.88 fps (~11 GB / 20 min). Every
downstream step works at ~1080p / 30 fps, so we decode the raw file exactly once into a
small H.264 file with a short keyframe interval (fast, frame-exact seeking for the GUI).
The proxy <-> source frame mapping is stored in the run metadata, so every result can be
reported in *original* frame numbers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    n_frames: int
    codec: str

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f"`{tool}` not found on PATH (it is installed by environment.yml).")
    return exe


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty (0 bytes) - likely a failed copy/compression.")
    cmd = [
        _require("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name,duration",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not json.loads(res.stdout or "{}").get("streams"):
        raise ValueError(f"ffprobe could not read {path}: {res.stderr.strip()}")
    s = json.loads(res.stdout)["streams"][0]
    fps = Fraction(s["r_frame_rate"])
    n = int(s.get("nb_frames") or round(float(s["duration"]) * fps))
    return VideoInfo(path, int(s["width"]), int(s["height"]), fps, n, s["codec_name"])


@dataclass(frozen=True)
class FrameMap:
    """Maps proxy frame index <-> source frame index: src = start + i * stride."""

    start: int
    stride: int
    n_proxy: int
    source_fps: float

    @property
    def proxy_fps(self) -> float:
        return self.source_fps / self.stride

    def to_source(self, i: int | np.ndarray) -> int | np.ndarray:
        return self.start + np.asarray(i) * self.stride if isinstance(i, np.ndarray) else self.start + i * self.stride

    def to_proxy(self, src: int) -> int:
        return int(round((src - self.start) / self.stride))

    def time_s(self, i: int | np.ndarray) -> float | np.ndarray:
        """Seconds since the start of the *source* video."""
        return np.asarray(self.to_source(i)) / self.source_fps


def plan_frame_map(info: VideoInfo, start: int, end: int | None, proxy_fps: float) -> FrameMap:
    end = info.n_frames - 1 if end is None else end
    if not 0 <= start <= end < info.n_frames:
        raise ValueError(f"frame range [{start}, {end}] outside video with {info.n_frames} frames")
    stride = max(1, round(info.fps_float / proxy_fps))
    n_proxy = (end - start) // stride + 1
    return FrameMap(start=start, stride=stride, n_proxy=n_proxy, source_fps=info.fps_float)


def build_proxy_command(
    src: Path, dst: Path, fmap: FrameMap, height: int, rotate: int = 0, crf: int = 18, gop: int = 15
) -> list[str]:
    """ffmpeg command producing an exact, frame-accurate proxy.

    `trim=start_frame/end_frame` counts decoded frames, so the first proxy frame is exactly
    source frame `start` (input-side `-ss` seeking can be off by a frame on some files).
    """
    end_excl = fmap.start + (fmap.n_proxy - 1) * fmap.stride + 1
    filters = [
        f"trim=start_frame={fmap.start}:end_frame={end_excl}",
        "setpts=PTS-STARTPTS",
        f"select='not(mod(n\\,{fmap.stride}))'",
    ]
    rate = Fraction(fmap.source_fps).limit_denominator(100000) / fmap.stride  # e.g. 30000/1001
    rot = {90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}.get(rotate)
    if rot:
        filters.append(rot)
    filters.append(f"scale=-2:{height}:flags=area")
    filters.append(f"setpts=N/({rate.numerator}/{rate.denominator}*TB)")
    # Hardware HEVC decoding on macOS (VideoToolbox) is ~3-5x faster than software for 4K.
    hw = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
    return [
        _require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-stats", "-y",
        *hw, "-i", str(src), "-an", "-sn", "-dn",
        "-vf", ",".join(filters), "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-g", str(gop),
        "-pix_fmt", "yuv420p", "-r", f"{rate.numerator}/{rate.denominator}",
        str(dst),
    ]


def make_proxy(src: Path, dst: Path, fmap: FrameMap, height: int, rotate: int, crf: int, gop: int) -> VideoInfo:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    subprocess.run(build_proxy_command(src, tmp, fmap, height, rotate, crf, gop), check=True)
    tmp.replace(dst)
    out = probe(dst)
    if out.n_frames != fmap.n_proxy:
        raise RuntimeError(f"proxy has {out.n_frames} frames, expected {fmap.n_proxy}")
    return out


class FrameReader:
    """Random-access RGB frame reader with a small LRU cache (GUI scrubbing, tracking)."""

    def __init__(self, path: str | Path, cache_size: int = 64):
        import cv2

        self._cv2 = cv2
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"cannot open {self.path}")
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._next = 0  # index the decoder will return on the next read()
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> np.ndarray:
        if not 0 <= idx < self.n_frames:
            raise IndexError(idx)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        if idx != self._next:  # sequential reads are cheap, seeks are not
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self._cap.read()
        if not ok:
            raise OSError(f"failed to decode frame {idx} of {self.path}")
        self._next = idx + 1
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        self._cache[idx] = rgb
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return rgb

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
