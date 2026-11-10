"""Real-time image enhancement for murky footage: a view toggle, optionally also fed to SAM.

    get("clahe")(rgb) -> rgb      same shape, uint8; all methods are stateless per frame

Methods (cheapest first):

* ``clahe``    Contrast-Limited Adaptive Histogram Equalisation on the lightness channel (Lab):
               local contrast without touching hue. Classic; a few ms per 1080p frame.
* ``retinex``  Multi-scale Retinex: divide out a smooth estimate of the illumination / water
               veil at three scales, then stretch each channel between robust percentiles,
               which also removes the blue-green cast. The blurs run on a downscaled copy (the
               illumination is smooth anyway), so it stays real-time.
* ``mobileie`` MobileIE (Yan et al., ICCV 2025): a 4k-parameter CNN trained on the UIEB
               underwater benchmark (weights from github.com/AVC2-UESTC/MobileIE, Apache-2.0,
               see LICENSE-MobileIE; converted to .npz). Needs PyTorch (not in the light
               annotator environment); uses the GPU (MPS/CUDA) when there is one.

None of these change the video on disk. Enhancement is applied *after* cropping, so in the
zoom view it adapts to the fish's surroundings rather than the whole frame.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np

Enhancer = Callable[[np.ndarray], np.ndarray]
METHODS = ("clahe", "retinex", "mobileie")
LABELS = {"clahe": "CLAHE", "retinex": "Retinex", "mobileie": "MobileIE"}


def clahe(rgb: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    lab[..., 0] = eq.apply(np.ascontiguousarray(lab[..., 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def retinex(rgb: np.ndarray, sigmas=(0.02, 0.08, 0.25), work: int = 96, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Multi-scale Retinex; sigmas are fractions of the image's longer side (crop-size independent)."""
    import cv2

    h, w = rgb.shape[:2]
    k = min(1.0, work / max(h, w))
    sw, sh = max(1, round(w * k)), max(1, round(h * k))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) + 1.0
    side = max(sh, sw)
    # log illumination, averaged over scales, all at low resolution (it is smooth by definition)
    log_illum = sum(np.log(cv2.GaussianBlur(small, (0, 0), max(0.5, s * side))) for s in sigmas) / len(sigmas)
    # robust per-channel range of the reflectance, estimated on the small image
    r_small = np.log(small) - log_illum
    a, b = np.percentile(r_small.reshape(-1, 3), lo, axis=0), np.percentile(r_small.reshape(-1, 3), hi, axis=0)
    scale = (255 / np.maximum(b - a, 1e-6)).astype(np.float32)
    # out = (log(img) - log_illum - a) * scale  ->  LUT(img) - upsampled offset
    lut = (np.log(np.arange(1, 257, dtype=np.float32))[:, None] * scale)  # (256, 3)
    offset = cv2.resize(((log_illum + a) * scale).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.subtract(cv2.LUT(np.ascontiguousarray(rgb), lut.reshape(256, 1, 3)), offset)
    return cv2.convertScaleAbs(np.maximum(out, 0, out=out))  # saturating cast to uint8


class MobileIE:
    """Re-parameterised ("slim") MobileIE-LLE network, written out with plain conv ops."""

    WEIGHTS = Path(__file__).with_name("mobileie_uieb.npz")

    def __init__(self, device: str | None = None):
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        with np.load(self.WEIGHTS) as z:
            self.w = {k: torch.from_numpy(z[k]).to(device) for k in z.files}

    def _forward(self, x):
        import torch
        import torch.nn.functional as F

        w = self.w

        def fst(y, p):  # feature self-transform: (w1*y)*(w2*y) + b
            return (w[f"{p}.weight1"] * y) * (w[f"{p}.weight2"] * y) + w[f"{p}.bias"]

        y = F.conv2d(x, w["head.block1.0.weight"], w["head.block1.0.bias"], padding=2)
        y = F.prelu(y, w["head.block1.1.weight"])
        x0 = fst(F.conv2d(y, w["head.block1.2.weight"], w["head.block1.2.bias"], padding=1), "head")
        x1 = fst(F.conv2d(x0, w["body.block1.weight"], w["body.block1.bias"], padding=1), "body")
        x2 = torch.sigmoid(F.conv2d(x1.mean((2, 3), keepdim=True), w["att.1.weight"], w["att.1.bias"]))
        x3 = torch.sigmoid(F.conv2d((x2 * x1).amax(1, keepdim=True), w["att1.0.weight"], w["att1.0.bias"]))
        return F.conv2d(x3 * x2 * x1, w["tail.weight"], w["tail.bias"], padding=1)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1)[None].float() / 255
            y = self._forward(x).clamp_(0, 1).mul_(255).to(torch.uint8)[0].permute(1, 2, 0)
            return y.cpu().numpy()


def available(name: str) -> bool:
    if name != "mobileie":
        return name in METHODS
    from importlib.util import find_spec

    return find_spec("torch") is not None and MobileIE.WEIGHTS.exists()  # no slow torch import here


@cache
def get(name: str) -> Enhancer:
    """Shared instance (GUI display thread)."""
    return make(name)


def make(name: str) -> Enhancer:
    """Fresh instance (e.g. for a tracking worker thread, so threads never share GPU state)."""
    if name == "clahe":
        return clahe
    if name == "retinex":
        return retinex
    if name == "mobileie":
        return MobileIE()
    raise ValueError(f"unknown enhancement {name!r}; choose from {METHODS}")
