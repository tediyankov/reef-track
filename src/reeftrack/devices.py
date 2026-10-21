"""Pick the best available compute device (CUDA on JASMIN, MPS on Apple Silicon, else CPU)."""

from __future__ import annotations

import os


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # Let ops that MPS doesn't implement yet fall back to CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def autocast_dtype(device: str):
    """Mixed precision only where it is reliably faster (CUDA)."""
    import torch

    if device.startswith("cuda"):
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32
