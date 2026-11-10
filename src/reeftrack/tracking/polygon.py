"""Masks as editable polygons: the GUI's hand-correction of a SAM mask.

A mask is turned into one outline with a modest number of vertices (enough to follow a fish,
few enough to drag by hand); the edited outline is rasterised back with sub-pixel precision
and given to SAM as a *mask prompt* (see `TrackingSession.add_mask`). Coordinates are video
pixels, as [[x, y], ...] lists (JSON-friendly: they are saved with the segment).
"""

from __future__ import annotations

import numpy as np

Polygon = list[list[float]]


def mask_to_polygon(mask: np.ndarray, max_vertices: int = 40) -> Polygon:
    """Outline of the mask's largest blob, simplified until it has at most `max_vertices` points."""
    import cv2

    if mask is None or not mask.any():
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    eps = 0.5
    approx = cv2.approxPolyDP(c, eps, True)
    while len(approx) > max_vertices:
        eps *= 1.3
        approx = cv2.approxPolyDP(c, eps, True)
    pts = approx[:, 0].astype(float)
    if len(pts) < 3:  # a 1-2 px blob: use its bounding box
        x, y, w, h = cv2.boundingRect(c)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    return [[float(x) + 0.5, float(y) + 0.5] for x, y in pts]  # pixel centres


def polygon_to_mask(poly: Polygon, shape: tuple[int, ...]) -> np.ndarray:
    import cv2

    out = np.zeros(shape[:2], np.uint8)
    if len(poly) >= 3:
        pts = np.round((np.asarray(poly, float) - 0.5) * 16).astype(np.int32)  # 4 fractional bits
        cv2.fillPoly(out, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return out.astype(bool)


def insert_vertex(poly: Polygon, xy) -> tuple[Polygon, int]:
    """Add a point: appended while drawing (< 3 points), else on the nearest edge. Returns (poly, index)."""
    p = [list(map(float, v)) for v in poly]
    xy = [float(xy[0]), float(xy[1])]
    if len(p) < 3:
        p.append(xy)
        return p, len(p) - 1
    a = np.asarray(p)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    t = np.clip(((np.asarray(xy) - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - np.asarray(xy), axis=1)
    i = int(np.argmin(d)) + 1
    p.insert(i, xy)
    return p, i
