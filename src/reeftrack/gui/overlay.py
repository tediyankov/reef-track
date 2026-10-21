"""Pure-numpy rendering of masks on frames (kept Qt-free so it is unit-testable)."""

from __future__ import annotations

import numpy as np

# Vivid, colour-blind-friendly-ish palette that stands out against blue-green water.
PALETTE = [
    (255, 140, 0),  # orange
    (255, 0, 170),  # magenta
    (0, 230, 255),  # cyan
    (190, 255, 0),  # lime
    (170, 90, 255),  # violet
    (255, 230, 0),  # yellow
    (255, 70, 70),  # red
    (0, 255, 150),  # mint
]
PREVIEW = (255, 255, 255)


def segment_colour(segment_id: int) -> tuple[int, int, int]:
    return PALETTE[segment_id % len(PALETTE)]


def draw_mask(frame: np.ndarray, mask: np.ndarray, colour, alpha: float = 0.45, outline: int = 2,
              dashed: bool = False) -> np.ndarray:
    """Return a copy of `frame` with `mask` tinted and outlined."""
    import cv2

    out = frame.copy()
    if mask is None or not mask.any():
        return out
    c = np.array(colour, np.float32)
    out[mask] = ((1 - alpha) * out[mask] + alpha * c).astype(np.uint8)
    if outline > 0:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if dashed:
            for cnt in contours:
                pts = cnt[:, 0]
                for i in range(0, len(pts) - 1, 8):
                    cv2.polylines(out, [pts[i : i + 4]], False, tuple(int(v) for v in colour), outline, cv2.LINE_AA)
        else:
            cv2.drawContours(out, contours, -1, tuple(int(v) for v in colour), outline, cv2.LINE_AA)
    return out


def draw_marker(frame: np.ndarray, xy, colour, size: int = 14) -> np.ndarray:
    import cv2

    out = frame
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.drawMarker(out, (x, y), (0, 0, 0), cv2.MARKER_CROSS, size + 4, 5, cv2.LINE_AA)
    cv2.drawMarker(out, (x, y), tuple(int(v) for v in colour), cv2.MARKER_CROSS, size, 2, cv2.LINE_AA)
    return out


# 12 well-separated colours for behaviour categories (in category order).
BEHAVIOUR_PALETTE = [
    (102, 194, 255),  # light blue
    (60, 220, 90),  # green
    (255, 105, 180),  # pink
    (205, 133, 63),  # brown
    (190, 190, 190),  # grey
    (255, 215, 0),  # gold
    (255, 69, 0),  # orange-red
    (148, 103, 255),  # purple
    (0, 206, 209),  # turquoise
    (220, 20, 60),  # crimson
    (154, 205, 50),  # yellow-green
    (255, 160, 122),  # salmon
]


def behaviour_colour(name: str, categories: list[str]) -> tuple[int, int, int]:
    i = categories.index(name) if name in categories else len(categories)
    return BEHAVIOUR_PALETTE[i % len(BEHAVIOUR_PALETTE)]


def draw_labels(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], scale: float = 0.9) -> np.ndarray:
    """Top-left list of coloured text labels on a dark backing (active behaviours)."""
    import cv2

    if not lines:
        return frame
    out = frame.copy()
    th = int(34 * scale)
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for t, _ in lines) + 40
    cv2.rectangle(out, (8, 8), (8 + w, 14 + th * len(lines)), (10, 20, 26), -1)
    for i, (text, colour) in enumerate(lines):
        y = 8 + th * (i + 1)
        cv2.circle(out, (24, y - int(9 * scale)), int(8 * scale), tuple(int(c) for c in colour), -1)
        cv2.putText(out, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def draw_pip(frame: np.ndarray, inset: np.ndarray, label: str, frac: float = 0.3) -> np.ndarray:
    """Picture-in-picture (top-right): e.g. the loop's start frame, to compare with 'now'."""
    import cv2

    h, w = frame.shape[:2]
    iw = int(w * frac)
    ih = int(inset.shape[0] * iw / inset.shape[1])
    small = cv2.resize(inset, (iw, ih), interpolation=cv2.INTER_AREA)
    out = frame.copy()
    x0, y0 = w - iw - 12, 12
    cv2.rectangle(out, (x0 - 3, y0 - 3), (x0 + iw + 2, y0 + ih + 30), (0, 229, 255), -1)
    out[y0 : y0 + ih, x0 : x0 + iw] = small
    cv2.putText(out, label, (x0 + 6, y0 + ih + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 20, 26), 2, cv2.LINE_AA)
    return out
