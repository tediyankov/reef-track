"""Hand-corrected masks: polygon editing, SAM mask prompts, and the GUI editor."""

import os
import shutil

import numpy as np
import pytest
from conftest import draw_disk

from reeftrack.config import load_config
from reeftrack.gui.overlay import rotate_xy, unrotate_xy
from reeftrack.tracking.backends import FakeBackend
from reeftrack.tracking.polygon import insert_vertex, mask_to_polygon, polygon_to_mask
from reeftrack.tracking.roi import crop, rect_at
from reeftrack.tracking.session import TrackingSession
from reeftrack.tracking.tracklet import Tracklet


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def disk_mask(cx, cy, r, shape=(120, 160)):
    m = np.zeros((*shape, 3), np.uint8)
    draw_disk(m, cx, cy, r, (1, 1, 1))
    return m[..., 0].astype(bool)


def test_polygon_roundtrip_and_vertex_budget():
    m = disk_mask(60, 50, 20)
    poly = mask_to_polygon(m, max_vertices=40)
    assert 3 <= len(poly) <= 40
    assert iou(polygon_to_mask(poly, m.shape), m) > 0.9
    tiny = np.zeros((20, 20), bool)
    tiny[5, 5:7] = True  # 2-pixel fish
    assert len(mask_to_polygon(tiny)) == 4 and polygon_to_mask(mask_to_polygon(tiny), tiny.shape)[5, 5]
    assert mask_to_polygon(np.zeros((5, 5), bool)) == []


def test_insert_vertex():
    p, i = insert_vertex([], (1, 1))
    p, i = insert_vertex(p, (5, 1))
    assert p == [[1, 1], [5, 1]] and i == 1  # still drawing: append
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    p, i = insert_vertex(sq, (11, 5))  # right edge
    assert i == 2 and p[2] == [11.0, 5.0]
    p, i = insert_vertex(sq, (5, -1))  # top edge
    assert i == 1


@pytest.fixture
def session(occlusion_video):
    return TrackingSession(occlusion_video, load_config().tracking, backend=FakeBackend())


def test_mask_prompt_is_used_and_recorded(session):
    hand = disk_mask(32.5, 60, 9)  # the user's outline, a bit bigger than what colour matching finds
    poly = mask_to_polygon(hand)
    seg = session.add_mask(5, poly)
    tr = session.tracklet
    assert seg.prompt_polygon == poly and seg.prompt_frame == 5
    assert (tr.mask(5) == polygon_to_mask(poly, hand.shape)).all()  # stored exactly as drawn
    assert (session.backend.start_mask == polygon_to_mask(poly, hand.shape)).all()  # SAM got it as the prompt
    assert (seg.start, seg.end, seg.end_reason) == (0, 19, "lost")  # and tracking continued from it


def test_mask_prompt_in_zoom_window(session):
    poly = mask_to_polygon(disk_mask(32.5, 60, 8))
    roi = rect_at((40, 60), (48, 36), session.shape)
    seg = session.add_mask(5, poly, roi=roi)
    given = polygon_to_mask(poly, session.shape)
    rect = rect_at(session.tracklet.obs[5].centroid, (48, 36), session.shape)  # re-centred on the mask
    assert session.backend.start_mask.shape == (36, 48)
    assert (session.backend.start_mask == crop(given, rect)).all()
    assert seg.roi == [48, 36]


def test_mask_correction_cuts_existing_segment(session):
    first = session.add_click(5, (32.5, 60))
    seg = session.add_mask(12, mask_to_polygon(disk_mask(50, 60, 8)))
    assert first.end == 11 and first.end_reason == "corrected"
    assert seg.start == 12 and seg.start_reason == "met_segment"


def test_empty_outline_rejected(session):
    with pytest.raises(ValueError):
        session.add_mask(5, [[1, 1], [2, 2]])
    assert not session.tracklet.segments


def test_polygon_saved_for_replay(session, tmp_path):
    poly = mask_to_polygon(disk_mask(32.5, 60, 8))
    session.add_mask(5, poly, enhance="clahe")
    session.tracklet.save(tmp_path)
    import json

    p = json.loads((tmp_path / "tracklet.json").read_text())["prompts"][0]
    assert p["polygon"] == poly and p["enhance"] == "clahe"
    assert Tracklet.load(tmp_path).segments[0].prompt_polygon == poly


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_rotate_xy_inverts_unrotate(deg):
    shape = (120, 160, 3)
    xd, yd = rotate_xy(30, 20, deg, shape)
    assert unrotate_xy(xd, yd, deg, shape) == pytest.approx((30, 20))


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_edit_mask_drag_and_retrack(tmp_path, occlusion_video, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from test_gui import _synthetic_run

    from reeftrack.gui import app as gui_app
    from reeftrack.tracking import session as session_mod

    run = _synthetic_run(tmp_path, occlusion_video)
    monkeypatch.setattr(session_mod, "load_backend", lambda tc, device: FakeBackend(tol=60))
    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow()
    win.enhance_box.setCurrentIndex(0)
    win.resize(1200, 800)
    win.show()
    win.open_run(run.root)
    win.worker.wait(5000)
    qapp.processEvents()

    def wait():
        win.worker.wait(10000)
        qapp.processEvents()

    win.goto(5)
    win._on_click(32.5, 60, Qt.MouseButton.LeftButton)
    wait()
    win.commit_click()
    wait()
    assert len(win.session.tracklet.segments) == 1

    win.goto(10)
    win.toggle_edit()  # tracked frame -> its outline becomes editable points
    assert win.edit and len(win.edit["poly"]) >= 3 and win.view.editing
    assert win.btn_track.text().startswith("Track from my mask")
    n0 = len(win.edit["poly"])

    # drag vertex 0 by +6 video px to the right, with real mouse events on the canvas
    v = win.view
    x, y = win.edit["poly"][0]
    a = v.mapFromScene(QPointF(*win._to_display(x, y)))
    b = v.mapFromScene(QPointF(*win._to_display(x + 6, y)))

    def send(kind, pt, buttons):
        ev = QMouseEvent(kind, QPointF(pt), QPointF(v.mapToGlobal(pt)), Qt.MouseButton.LeftButton, buttons,
                         Qt.KeyboardModifier.NoModifier)
        {QEvent.Type.MouseButtonPress: v.mousePressEvent, QEvent.Type.MouseMove: v.mouseMoveEvent,
         QEvent.Type.MouseButtonRelease: v.mouseReleaseEvent}[kind](ev)

    send(QEvent.Type.MouseButtonPress, a, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseMove, b, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, b, Qt.MouseButton.NoButton)
    assert win.edit["poly"][0][0] == pytest.approx(x + 6, abs=0.6)  # screen -> video mapping round-trips
    assert len(win.edit["poly"]) == n0

    win._on_click(55, 60, Qt.MouseButton.LeftButton)  # click near the outline: adds a point
    assert len(win.edit["poly"]) == n0 + 1

    win.rotate_view(90)  # editing survives a view change; handles follow
    assert v.polygon()[0] == pytest.approx(list(win._to_display(*win.edit["poly"][0])))
    win.rotate_view(-90)

    poly = [list(p) for p in win.edit["poly"]]
    win.commit_click()  # Enter
    wait()
    tr = win.session.tracklet
    seg = max(tr.segments, key=lambda s: s.segment_id)
    assert seg.prompt_frame == 10 and np.allclose(seg.prompt_polygon, poly)
    assert win.edit is None and not v.editing
    assert (run.tracking_dir / "tracklet.json").exists()

    win.toggle_edit()  # Esc cancels; moving to another frame discards
    win._escape()
    assert win.edit is None
    win.toggle_edit()
    win.goto(11)
    assert win.edit is None
    win.close()
    qapp.processEvents()
