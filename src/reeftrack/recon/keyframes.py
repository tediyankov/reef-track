"""Choose which frames go into the 3D model, and how to split them into chunks.

Feed-forward 3D models (VGGT/Wat3R) need *parallax*: two frames 1/30 s apart are almost
identical and add compute but little geometry. ~2-4 frames/s of a slow swim-through gives
good overlap. Within each slot we take the sharpest frame (variance of the Laplacian),
because motion blur is the main failure mode for underwater footage.
"""

from __future__ import annotations

import numpy as np


def sharpness(rgb: np.ndarray, width: int = 480) -> float:
    import cv2

    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_sharpest(scores: np.ndarray, slot: int) -> np.ndarray:
    """Index of the max score in each consecutive window of `slot` frames."""
    picks = []
    for s in range(0, len(scores), slot):
        picks.append(s + int(np.argmax(scores[s : s + slot])))
    return np.asarray(picks, dtype=int)


def select_keyframes(frames, proxy_fps: float, keyframe_fps: float, use_sharpness: bool = True,
                     progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Returns (keyframe indices, sharpness of every frame or NaN when not computed)."""
    n = len(frames)
    slot = max(1, round(proxy_fps / keyframe_fps))
    scores = np.full(n, np.nan)
    if not use_sharpness:
        return np.arange(0, n, slot), scores
    it = range(n)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="sharpness", unit="fr")
    for i in it:
        scores[i] = sharpness(frames[i])
    return pick_sharpest(scores, slot), scores


def make_chunks(n: int, size: int, overlap: int) -> list[np.ndarray]:
    """Overlapping windows over range(n); consecutive chunks share `overlap` items.

    The last chunk is shifted back so it is full-size (better geometry than a stub).
    """
    if size <= overlap:
        raise ValueError("chunk size must exceed overlap")
    if n <= size:
        return [np.arange(n)]
    step = size - overlap
    starts = list(range(0, n - size + 1, step))
    if starts[-1] + size < n:
        starts.append(n - size)
    return [np.arange(s, s + size) for s in starts]
