"""Feed-forward multi-view 3D models: frames in -> per-frame depth, confidence and cameras.

* ``wat3r`` Wat3R (Ren et al. 2026, arXiv 2607.08772): VGGT fine-tuned *without labels* on
  real underwater video (teacher-student + cross-view consistency), so it is more robust to
  attenuation/back-scatter than plain VGGT. Code is vendored by `scripts/fetch_wat3r.sh`
  into third_party/Wat3R (Apache-2.0); weights download from Hugging Face on first use.
* ``vggt``  original VGGT-1B (Wang et al., CVPR 2025) for comparison; needs
  `pip install git+https://github.com/facebookresearch/vggt`.

Both share one interface (same architecture), so swapping is a config change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from reeftrack.config import REPO_ROOT

WAT3R_DIR = REPO_ROOT / "third_party" / "Wat3R"


@dataclass
class ChunkPrediction:
    images: np.ndarray  # (S, h, w, 3) uint8 - exactly what the model saw
    depth: np.ndarray  # (S, h, w) float32
    depth_conf: np.ndarray  # (S, h, w) float32
    extrinsic: np.ndarray  # (S, 3, 4) camera-from-world, world = first frame of chunk
    intrinsic: np.ndarray  # (S, 3, 3) at (h, w)


def model_input_size(h: int, w: int, target: int = 518, patch: int = 14) -> tuple[int, int]:
    """VGGT 'crop' preprocessing: width -> target, height scaled & rounded to the patch size."""
    nh = int(round(h * target / w / patch) * patch)
    return min(nh, target), target


def _layout(h0: int, w0: int, target: int) -> tuple[int, int, int, int]:
    """(model h, model w, resized full height before crop, top crop offset)."""
    nh, nw = model_input_size(h0, w0, target)
    full_h = int(round(h0 * target / w0))
    if full_h < nh:  # rounding up to the patch size: stretch slightly instead of cropping
        return nh, nw, nh, 0
    return nh, nw, full_h, (full_h - nh) // 2


def to_model_pixels(u: float, v: float, h0: int, w0: int, target: int = 518) -> tuple[float, float]:
    """Map a pixel of the original frame to the model-input image (same resize/crop as `preprocess`)."""
    nh, nw, full_h, top = _layout(h0, w0, target)
    return u * nw / w0, v * full_h / h0 - top


def mask_to_model(mask: np.ndarray, target: int = 518) -> np.ndarray:
    """Resize a frame-resolution mask exactly like `preprocess` resizes the image."""
    import cv2

    nh, nw, full_h, top = _layout(*mask.shape[:2], target)
    m = cv2.resize(mask.astype(np.uint8), (nw, full_h), interpolation=cv2.INTER_NEAREST)
    return m[top : top + nh].astype(bool)


def preprocess(frames: list[np.ndarray], target: int = 518) -> np.ndarray:
    """RGB uint8 frames -> (S, h, w, 3) uint8 resized like VGGT (centre-crop if too tall)."""
    import cv2

    nh, nw, full_h, top = _layout(*frames[0].shape[:2], target)
    out = [cv2.resize(f, (nw, full_h), interpolation=cv2.INTER_AREA)[top : top + nh] for f in frames]
    return np.stack(out)


class FeedForwardRecon:
    def __init__(self, backend: str, model_id: str, device: str, image_size: int = 518):
        import torch

        if backend == "wat3r":
            if not (WAT3R_DIR / "wat3r").exists():
                raise RuntimeError("Wat3R code missing: run `bash scripts/fetch_wat3r.sh` first.")
            sys.path.insert(0, str(WAT3R_DIR))
            from wat3r.models.wat3r import Wat3R as Model
            from wat3r.utils.pose_enc import pose_encoding_to_extri_intri
        else:
            from vggt.models.vggt import VGGT as Model
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self._pose_dec = pose_encoding_to_extri_intri
        self.name = backend
        self.device = device
        self.image_size = image_size
        self.model = Model.from_pretrained(model_id).to(device).eval()
        self.torch = torch

    def __call__(self, frames: list[np.ndarray]) -> ChunkPrediction:
        torch = self.torch
        from reeftrack.devices import autocast_dtype

        imgs = preprocess(frames, self.image_size)
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255).to(self.device)
        dtype = autocast_dtype(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.split(":")[0], dtype=dtype,
                                             enabled=dtype != torch.float32):
            pred = self.model(x[None], need_point=False)
        E, K = self._pose_dec(pred["pose_enc"], x.shape[-2:])
        return ChunkPrediction(
            images=imgs,
            depth=pred["depth"][0, ..., 0].float().cpu().numpy(),
            depth_conf=pred["depth_conf"][0].float().cpu().numpy(),
            extrinsic=E[0].float().cpu().numpy(),
            intrinsic=K[0].float().cpu().numpy(),
        )
