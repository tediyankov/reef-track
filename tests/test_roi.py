"""Zoom window for tiny fish: geometry, display follower, and SAM seeing only the window."""

import os
import shutil

import numpy as np
import pytest
from conftest import FISH, draw_disk

from reeftrack.config import load_config
from reeftrack.tracking.backends import FakeBackend
from reeftrack.tracking.roi import FishFollower, centre_of, crop, fit_box, paste, rect_at
from reeftrack.tracking.session import TrackingSession
from reeftrack.tracking.tracklet import Tracklet


def test_fit_box_aspect_minimum_and_frame():
    x0, y0, w, h = fit_box(100, 100, 120, 110, aspect=16 / 9, shape=(1080, 1920))
    assert abs(w / h - 16 / 9) < 0.05 and min(w, h) >= 48  # tiny drag -> min size, view's aspect
    assert abs(centre_of((x0, y0, w, h))[0] - 110) <= 1
    x0, y0, w, h = fit_box(1900, 1060, 1700, 900, aspect=1.0, shape=(1080, 1920))  # corners reversed, at the edge
    assert x0 + w <= 1920 and y0 + h <= 1080 and w == h
    assert fit_box(0, 0, 5000, 5000, 4 / 3, (120, 160)) == (0, 0, 160, 120)


def test_rect_at_slides_inside_and_crop_paste_roundtrip():
    assert rect_at((5, 5), (40, 30), (120, 160)) == (0, 0, 40, 30)
    assert rect_at((158, 119), (40, 30), (120, 160)) == (120, 90, 40, 30)
    m = np.zeros((120, 160), bool)
    m[50:60, 70:75] = True
    r = (60, 40, 40, 30)
    assert (paste(crop(m, r), r, m.shape) == m).all()


def _textured(i, fish_x):
    rng = np.random.default_rng(0)
    f = (rng.integers(0, 60, (240, 320, 1)) + np.array([20, 80, 90])).clip(0, 255).astype(np.uint8)
    f = np.roll(f, i, axis=1)  # camera pans slowly
    draw_disk(f, fish_x, 120, 6, (250, 220, 40))
    return f


def test_follower_tracks_fish_without_masks_and_caches():
    xs = [100 + 4 * i for i in range(20)]
    frames = [_textured(i, x) for i, x in enumerate(xs)]
    fol = FishFollower(rect_at((xs[0], 120), (64, 48), frames[0].shape), frames[0], 0, smooth=0.0)
    for i in range(1, 20):
        r = fol.update(i, frames[i])
        assert abs(centre_of(r)[0] - xs[i]) <= 4, i
    shown = fol.update(5, frames[5])  # stepping back: same view as before
    assert shown == rect_at(fol.cache[5], fol.size, fol.shape)
    r = fol.update(19, frames[19], anchor=(200.0, 100.0))  # a tracked mask wins over the matcher
    assert centre_of(r) == pytest.approx((200, 100), abs=1)


@pytest.fixture
def distracted_video(occlusion_video):
    """The occlusion clip plus a *bigger* red blob far from the fish (another red thing on the reef)."""
    out = []
    for f in occlusion_video:
        f = f.copy()
        draw_disk(f, 140, 15, 11, FISH)
        out.append(f)
    return out


def test_sam_only_sees_the_window(distracted_video):
    tc = load_config().tracking
    full = TrackingSession(distracted_video, tc, backend=FakeBackend())
    assert full.preview(5, (32.5, 60))[15, 140]  # full frame: picks the bigger blob, not the fish
    zoomed = TrackingSession(distracted_video, tc, backend=FakeBackend())
    roi = rect_at((32.5, 60), (48, 36), zoomed.shape)
    m = zoomed.preview(5, (32.5, 60), roi)
    assert m.shape == (120, 160) and m[60, 32] and not m[15, 140]  # full-frame mask, fish only

    seg = zoomed.add_click(5, (32.5, 60), roi=roi)
    assert (seg.start, seg.end, seg.end_reason) == (0, 19, "lost")  # window followed the swimming fish
    tr = zoomed.tracklet
    for f in (0, 10, 19):
        assert abs(tr.obs[f].centroid[0] - (20 + 2.5 * f)) < 1.5 and not tr.mask(f)[15, 140]
    assert seg.roi == [48, 36]


def test_segment_roi_roundtrip(tmp_path, distracted_video):
    s = TrackingSession(distracted_video, load_config().tracking, backend=FakeBackend())
    s.add_click(5, (32.5, 60), roi=rect_at((32.5, 60), (48, 36), s.shape))
    s.tracklet.save(tmp_path)
    assert Tracklet.load(tmp_path).segments[0].roi == [48, 36]


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_zoom_view(tmp_path, occlusion_video, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from test_gui import _synthetic_run

    from reeftrack.gui import app as gui_app
    from reeftrack.tracking import session as session_mod

    run = _synthetic_run(tmp_path, occlusion_video)
    monkeypatch.setattr(session_mod, "load_backend", lambda tc, device: FakeBackend(tol=60))
    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow()
    win.open_run(run.root)
    win.worker.wait(5000)
    qapp.processEvents()
    win.goto(5)
    win._on_box(15, 50, 50, 70)  # drag around the fish (full view: screen px = video px)
    assert win.zoom_on and win.view._item.pixmap().width() == gui_app.ZOOM_W
    x0, y0, w, h = win._shown_rect
    assert x0 <= 32.5 <= x0 + w and y0 <= 60 <= y0 + h
    sc = gui_app.ZOOM_W / w
    assert win._to_video((32.5 - x0) * sc, (60 - y0) * sc) == pytest.approx((32.5, 60))

    win._on_view_click((32.5 - x0) * sc, (60 - y0) * sc, Qt.MouseButton.LeftButton)
    win.worker.wait(5000)
    qapp.processEvents()
    assert win.pending is not None and win.pending[3] == win._shown_rect
    win.commit_click()
    win.worker.wait(10000)
    qapp.processEvents()
    seg = win.session.tracklet.segments[0]
    assert seg.roi == [w, h] and seg.end_reason == "lost"

    win.goto(15)  # view follows the tracked mask (a tiny test frame, so the window may sit against the edge)
    want = rect_at(win.session.tracklet.obs[15].centroid, (w, h), win.session.shape)
    assert np.allclose(win._shown_rect, want, atol=2)
    win._escape()  # back to the full view
    assert not win.zoom_on and win.view._item.pixmap().width() == 160
    win.toggle_zoom()  # and back in
    assert win.zoom_on
    win.close()
    qapp.processEvents()
