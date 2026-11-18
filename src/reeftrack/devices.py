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


def tracking_available() -> bool:
    """True if the mask-tracking libraries (PyTorch + transformers) are installed. The lightweight
    annotator environment has neither, so the GUI then offers behaviour/loop annotation only.
    Checked without importing them (importing torch takes seconds)."""
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("transformers") is not None
