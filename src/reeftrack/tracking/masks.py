"""Binary-mask utilities: run-length encoding (compact storage) and simple geometry.

A 1080p boolean mask is 2 MB raw; its RLE is typically < 1 KB for a fish, so a whole
tracklet's masks fit in one small `masks.npz` file.
"""

from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> np.ndarray:
    """Row-major RLE. Returns uint32 run lengths, alternating 0-runs and 1-runs, starting with 0s."""
    flat = np.asarray(mask, dtype=bool).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint32)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds)
    if flat[0]:  # must start with a (possibly empty) run of zeros
        runs = np.concatenate([[0], runs])
    return runs.astype(np.uint32)


def rle_decode(runs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    ends = np.cumsum(runs)
    starts = ends - runs
    for s, e in zip(starts[1::2], ends[1::2]):
        flat[s:e] = True
    return flat.reshape(shape)


def area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) inclusive, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def shift(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The mask moved by (dx, dy) pixels (rounded); what moves off the frame is dropped."""
    out = np.zeros_like(mask, dtype=bool)
    dx, dy = int(round(dx)), int(round(dy))
    h, w = mask.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def between(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """A mask a fraction t (0..1) of the way from mask a to mask b: the nearer of the two, moved
    along the line between their centroids. For frames the tracker skipped (`frame_step`)."""
    ca, cb = centroid(a), centroid(b)
    if ca is None or cb is None:
        return (a if t <= 0.5 else b).astype(bool)
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    return shift(a, t * dx, t * dy) if t <= 0.5 else shift(b, (t - 1) * dx, (t - 1) * dy)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected blob (SAM sometimes adds small far-away fragments)."""
    import cv2

    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == keep


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    out = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    return out if out.any() else mask  # tiny masks: keep rather than vanish
