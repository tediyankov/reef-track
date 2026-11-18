"""Label-free appearance model of the focal fish.

A Picasso triggerfish has a very distinctive colour pattern, so a hue/saturation histogram
of the pixels *inside the mask* is a cheap, surprisingly strong identity signature. We keep:
  * the template from the user's click (ground truth identity), and
  * a slowly-updated running template (adapts to lighting/pose changes),
and score a new mask by its best similarity to either. It does not distinguish two Picasso
triggerfish from each other (motion does most of that work), but it does catch the tracker
sliding onto coral, sand or a differently-coloured fish.
"""

from __future__ import annotations

import numpy as np

H_BINS, S_BINS, V_BINS = 18, 8, 4


def colour_signature(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """L1-normalised HSV histogram of masked pixels (None if the mask is empty)."""
    import cv2

    if not mask.any():
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = mask.astype(np.uint8)
    hist = cv2.calcHist([hsv], [0, 1, 2], m, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]).ravel()
    total = hist.sum()
    return hist / total if total > 0 else None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya coefficient in [0, 1] (1 = identical colour distribution)."""
    return float(np.sum(np.sqrt(a * b)))


class AppearanceModel:
    def __init__(self, template: np.ndarray, momentum: float = 0.95):
        self.template = template
        self.running = template.copy()
        self.momentum = momentum

    def score(self, sig: np.ndarray | None) -> float:
        if sig is None:
            return 0.0
        return max(similarity(sig, self.template), similarity(sig, self.running))

    def update(self, sig: np.ndarray | None) -> None:
        """Only call on frames we are confident about, or drift sneaks into the template."""
        if sig is not None:
            self.running = self.momentum * self.running + (1 - self.momentum) * sig
