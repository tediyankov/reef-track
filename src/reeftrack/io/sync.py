"""Temporal synchronisation of two cameras from their audio tracks.

Two GoPros on one rig hear the same sounds (breathing, fin kicks, bubbles) but are started
by hand, so their clocks differ by up to a few seconds. Cross-correlating the audio finds
that offset to well under one video frame (audio is sampled at 48 kHz; one 120 fps frame is
8.3 ms). We use GCC-PHAT (generalised cross-correlation with phase transform), which whitens
the spectrum so sharp transients dominate rather than loud low-frequency rumble - the
standard choice for time-delay estimation in reverberant, noisy conditions.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reeftrack.io.video import _require


def read_audio(path: Path, start_s: float, dur_s: float, sr: int = 16000) -> np.ndarray:
    """Mono float32 audio [start_s, start_s + dur_s) resampled to `sr`."""
    cmd = [_require("ffmpeg"), "-v", "error", "-ss", f"{start_s:.6f}", "-t", f"{dur_s:.6f}", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_lag_s: float) -> tuple[float, float]:
    """Delay d (s) such that b(t) ~= a(t - d), i.e. events appear d seconds *later* in b.

    Returns (delay, peak sharpness = peak / median |correlation|) as a confidence measure.
    """
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, nfft)
    max_lag = int(max_lag_s * sr)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])  # lags -max..+max
    k = int(np.argmax(np.abs(cc)))
    # parabolic interpolation for sub-sample precision
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = np.abs(cc[k - 1 : k + 2])
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        frac = 0.0
    lag = (k - max_lag + frac) / sr
    sharp = float(np.abs(cc[k]) / (np.median(np.abs(cc)) + 1e-12))
    return lag, sharp


@dataclass
class SyncResult:
    offset_s: float  # right_time = left_time - offset_s  (event at left t happens at right t - offset)
    offset_frames: float  # same, in source frames
    confidence: float  # median peak sharpness over windows
    spread_ms: float  # disagreement between windows (robustness check)
    per_window: list[tuple[float, float, float]]  # (window start, offset_s, sharpness)
    n_inliers: int = 0


def estimate_offset(left: Path, right: Path, fps: float, windows: list[float], dur_s: float = 20.0,
                    max_lag_s: float = 5.0, sr: int = 16000, inlier_ms: float = 5.0) -> SyncResult:
    """Offset between two recordings, estimated independently in several windows.

    A sound at time t in the left file appears at time t - offset in the right file.
    """
    res = []
    for w in windows:
        a = read_audio(left, w, dur_s, sr)
        # read the right file with margin so the true lag lies inside the window
        b = read_audio(right, max(0.0, w - max_lag_s), dur_s + 2 * max_lag_s, sr)
        pad_s = min(max_lag_s, w)  # right excerpt starts this much earlier than the left one
        d, sharp = gcc_phat(a, b, sr, 2 * max_lag_s)
        # event at left time w+tau is at right time (w - pad_s) + tau + d  =>  offset = pad_s - d
        res.append((w, pad_s - d, sharp))
    offs = np.array([r[1] for r in res])
    sharp = np.array([r[2] for r in res])
    # Windows can lock onto a false peak (repetitive sounds, lag at the search edge), so take
    # the consensus: windows within `inlier_ms` of the median, then re-average those.
    good = np.abs(offs - np.median(offs)) < inlier_ms / 1000
    if good.sum() < max(2, len(offs) // 2):
        raise RuntimeError(f"no consensus between audio windows: {np.round(offs, 4).tolist()}")
    off = float(np.median(offs[good]))
    return SyncResult(off, off * fps, float(np.median(sharp[good])), float(np.ptp(offs[good]) * 1000), res,
                      int(good.sum()))
