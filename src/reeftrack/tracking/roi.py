"""Regions of interest for tiny fish: a fixed-size window that follows the fish.

Used in two places:

* **Zoom view (GUI, display only).** The annotator drags a box around a small fish; the
  window keeps that size, re-centres on the fish every frame and fills the screen. Where the
  fish has a tracked mask the window centres on the mask; elsewhere a small template matcher
  (`FishFollower`) follows it, and a light smoothing removes the camera shake, as in the
  behaviour GIFs. Nothing is written to the video or the annotations.
* **Cropped SAM input.** When tracking from the zoom view, SAM only sees the window: it is
  re-centred on the last mask each frame, so a 30-px fish fills SAM's 1024-px input instead
  of being a speck in a 1920-px frame. Masks are pasted back into full-frame coordinates, so
  everything downstream (tracklet, CSV, 3D fusion) is unchanged.

All coordinates are pixels of the (unrotated) video frame; a rect is (x0, y0, w, h).
"""

from __future__ import annotations

import numpy as np

Rect = tuple[int, int, int, int]
MIN_SIZE = 96  # px: caps the zoom at ~x20 on 1080p (more is blur) and keeps context for SAM


def fit_box(x0: float, y0: float, x1: float, y1: float, aspect: float, shape: tuple[int, ...],
            min_size: int = MIN_SIZE) -> Rect:
    """Grow a dragged box (any two corners) to `aspect` (= w / h) around its centre, inside the frame."""
    H, W = shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(abs(x1 - x0), 1.0), max(abs(y1 - y0), 1.0)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if min(w, h) < min_size:  # keep the aspect, scale up
        k = min_size / min(w, h)
        w, h = w * k, h * k
    if w > W or h > H:  # bigger than the frame: shrink to fit
        k = min(W / w, H / h)
        w, h = w * k, h * k
    return rect_at((cx, cy), (int(round(w)), int(round(h))), shape)


def rect_at(centre: tuple[float, float], size: tuple[int, int], shape: tuple[int, ...]) -> Rect:
    """Window of `size` (w, h) centred on `centre`, slid back inside the frame if needed."""
    H, W = shape[:2]
    w, h = min(size[0], W), min(size[1], H)
    x0 = int(np.clip(round(centre[0] - w / 2), 0, W - w))
    y0 = int(np.clip(round(centre[1] - h / 2), 0, H - h))
    return x0, y0, w, h


def crop(img: np.ndarray, rect: Rect) -> np.ndarray:
    x0, y0, w, h = rect
    return np.ascontiguousarray(img[y0 : y0 + h, x0 : x0 + w])


def paste(mask: np.ndarray, rect: Rect, shape: tuple[int, ...]) -> np.ndarray:
    """A mask computed on a crop, placed back into an empty full-frame mask."""
    x0, y0, w, h = rect
    out = np.zeros(shape[:2], bool)
    out[y0 : y0 + h, x0 : x0 + w] = mask[:h, :w]
    return out


def centre_of(rect: Rect) -> tuple[float, float]:
    x0, y0, w, h = rect
    return x0 + w / 2, y0 + h / 2


class FishFollower:
    """Keeps a fixed-size window on a fish across frames (display stabilisation).

    `update(frame_idx, frame, anchor)`:
      * `anchor` (e.g. the tracked mask's centroid) wins when given, and refreshes the template;
      * otherwise, when stepping to a neighbouring frame, the fish's appearance (centre half of
        the window) is searched for near its last position (normalised cross-correlation);
      * after a jump (seek) with no anchor, the window stays where it was.
    The centre is smoothed between consecutive frames so small shakes don't jolt the view.
    Positions are cached per frame, so stepping back shows exactly what was shown before.
    """

    def __init__(self, rect: Rect, frame: np.ndarray, frame_idx: int, smooth: float = 0.5,
                 min_score: float = 0.5):
        self.size = (rect[2], rect[3])
        self.shape = frame.shape[:2]
        self.smooth = smooth
        self.min_score = min_score
        self.centre = centre_of(rect)
        self.cache: dict[int, tuple[float, float]] = {frame_idx: self.centre}
        self.last = frame_idx
        self.template = self._patch(_grey(frame), self.centre)

    @property
    def rect(self) -> Rect:
        return rect_at(self.centre, self.size, self.shape)

    def _tsize(self) -> tuple[int, int]:
        return max(8, self.size[0] // 2), max(8, self.size[1] // 2)

    def _patch(self, grey: np.ndarray, c: tuple[float, float]) -> np.ndarray:
        return crop(grey, rect_at(c, self._tsize(), grey.shape)).astype(np.float32)

    def _search(self, grey: np.ndarray) -> tuple[tuple[float, float], float] | None:
        import cv2

        tw, th = self._tsize()
        if self.template.shape != (th, tw):
            return None
        win = rect_at(self.centre, (tw + self.size[0], th + self.size[1]), grey.shape)  # +/- half a window
        region = crop(grey, win).astype(np.float32)
        if region.shape[0] < th or region.shape[1] < tw:
            return None
        res = cv2.matchTemplate(region, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        return (win[0] + mx + tw / 2, win[1] + my + th / 2), float(score)

    def update(self, frame_idx: int, frame: np.ndarray, anchor: tuple[float, float] | None = None) -> Rect:
        if anchor is None and frame_idx in self.cache:  # revisiting: show the same view as before
            self.centre, self.last = self.cache[frame_idx], frame_idx
            return self.rect
        near = abs(frame_idx - self.last) <= 2
        grey = _grey(frame)
        target = None
        if anchor is not None:
            target = anchor
        elif near:
            found = self._search(grey)
            if found and found[1] >= self.min_score:
                target = found[0]
        if target is not None:
            a = self.smooth if near else 0.0
            self.centre = (a * self.centre[0] + (1 - a) * target[0], a * self.centre[1] + (1 - a) * target[1])
            new = self._patch(grey, target)
            if anchor is not None or new.shape != self.template.shape:
                self.template = new
            else:  # slowly adapt to turning / changing light, without drifting off the fish
                self.template = 0.8 * self.template + 0.2 * new
        self.cache[frame_idx] = self.centre
        self.last = frame_idx
        return self.rect


def _grey(frame: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
