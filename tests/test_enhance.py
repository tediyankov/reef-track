"""View enhancement (CLAHE / Retinex / MobileIE) and SAM optionally seeing it."""

import os
import shutil

import cv2
import numpy as np
import pytest

from reeftrack import enhance
from reeftrack.config import load_config
from reeftrack.tracking.backends import FakeBackend
from reeftrack.tracking.roi import crop, rect_at
from reeftrack.tracking.session import TrackingSession


def murky(h=120, w=160, seed=0):
    """Low-contrast texture under a strong blue-green veil."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.uniform(0, 1, (h, w)).astype(np.float32), (0, 0), 2)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    base = np.stack([0.25 + 0.15 * tex, 0.55 + 0.15 * tex, 0.5 + 0.15 * tex], -1)
    return (base * 255).astype(np.uint8)


@pytest.mark.parametrize("name", [m for m in enhance.METHODS if enhance.available(m)])
@pytest.mark.parametrize("shape", [(120, 160), (37, 53), (1, 1)])
def test_shape_dtype_preserved(name, shape):
    out = enhance.get(name)(murky(*shape) if min(shape) > 1 else np.full((*shape, 3), 90, np.uint8))
    assert out.shape == (*shape, 3) and out.dtype == np.uint8


def test_clahe_raises_local_contrast():
    img = murky()
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]
    L2 = cv2.cvtColor(enhance.clahe(img), cv2.COLOR_RGB2LAB)[..., 0]
    assert L2.std() > 1.5 * L.std()


def test_retinex_removes_colour_cast_and_stretches():
    img = murky()
    out = enhance.retinex(img)
    cast = lambda im: np.ptp(im.reshape(-1, 3).mean(0))  # noqa: E731  spread of channel means
    assert cast(out) < 0.3 * cast(img)
    contrast = lambda im: im.reshape(-1, 3).std(0).mean()  # noqa: E731  per-channel contrast
    assert contrast(out) > 2 * contrast(img)


@pytest.mark.skipif(not enhance.available("mobileie"), reason="needs torch")
def test_mobileie_weights_and_determinism():
    with np.load(enhance.MobileIE.WEIGHTS) as z:
        assert sum(z[k].size for k in z.files) == 4075  # the paper's "~4K parameters"
    m = enhance.MobileIE("cpu")
    img = murky()
    a, b = m(img), m(img)
    assert (a == b).all() and np.abs(a.astype(int) - img.astype(int)).mean() > 3  # it does change the image


class Recorder(FakeBackend):
    def __init__(self):
        super().__init__()
        self.seen = []

    def _seg(self, frame):
        self.seen.append(frame.copy())
        return super()._seg(frame)


def test_sam_gets_the_enhanced_crop(occlusion_video):
    rec = Recorder()
    s = TrackingSession(occlusion_video, load_config().tracking, backend=rec)
    roi = rect_at((32.5, 60), (48, 36), s.shape)
    s.preview(5, (32.5, 60), roi, enhance="clahe")
    assert (rec.seen[-1] == enhance.clahe(crop(occlusion_video[5], roi))).all()
    s.preview(5, (32.5, 60))
    assert (rec.seen[-1] == occlusion_video[5]).all()  # off by default


def test_segment_records_enhancement(occlusion_video):
    s = TrackingSession(occlusion_video, load_config().tracking, backend=FakeBackend(tol=120))
    seg = s.add_click(5, (32.5, 60), enhance="clahe")
    assert seg.enhance == "clahe"


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_enhance_toggle(tmp_path, occlusion_video, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from test_gui import _synthetic_run

    from reeftrack.gui import app as gui_app
    from reeftrack.tracking import session as session_mod

    run = _synthetic_run(tmp_path, occlusion_video)
    monkeypatch.setattr(session_mod, "load_backend", lambda tc, device: FakeBackend(tol=120))
    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow()
    win.settings.setValue("enhance", "")
    win.enhance_box.setCurrentIndex(0)
    win.open_run(run.root)
    win.worker.wait(5000)
    qapp.processEvents()
    win.goto(5)
    plain = win.view._item.pixmap().toImage()
    win.cycle_enhance()
    assert win.enhance_box.currentData() == "clahe"
    assert win.view._item.pixmap().toImage() != plain  # the view changed
    win._on_click(32.5, 60, Qt.MouseButton.LeftButton)
    win.worker.wait(5000)
    qapp.processEvents()
    assert win.pending is not None and win.pending[4] == "clahe"
    win.enhance_sam.setChecked(False)
    win.enhance_box.setCurrentIndex(0)  # changing enhancement drops a stale preview
    assert win.pending is None
    win.close()
    qapp.processEvents()
