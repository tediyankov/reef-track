"""Shared fixtures: small synthetic videos with a known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

H, W = 120, 160
FISH = (255, 0, 0)


def draw_disk(img: np.ndarray, cx: float, cy: float, r: float, colour) -> None:
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = colour


@pytest.fixture
def occlusion_video() -> list[np.ndarray]:
    """50 frames; a red 'fish' swims right, hidden behind 'coral' for frames 20-29."""
    frames = []
    for i in range(50):
        f = np.full((H, W, 3), (20, 80, 90), np.uint8)  # blue-green water
        draw_disk(f, 20 + 2.5 * i, 60, 8, FISH)
        if 20 <= i <= 29:
            f[:, :] = (20, 80, 90)  # fish fully hidden
            f[30:90, 60:140] = (120, 110, 90)  # coral block
        frames.append(f)
    return frames


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """A message box in a GUI test would wait forever for a click (hanging the whole run).
    Make every one fail loudly instead, with its text, so the test shows what went wrong."""
    try:
        from PySide6.QtWidgets import QMessageBox
    except ImportError:
        yield
        return

    def fail(*args, **_):
        text = " | ".join(str(a) for a in args[1:3])
        raise RuntimeError(f"unexpected dialog in a test: {text}")

    for name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(fail))
    yield
