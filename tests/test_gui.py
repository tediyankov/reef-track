"""Overlay rendering + an offscreen smoke test of the GUI workflow (fake tracker, synthetic run)."""

import os
import shutil

import numpy as np
import pytest

from reeftrack.gui.overlay import PALETTE, draw_marker, draw_mask, rotate, segment_colour, unrotate_xy


def test_draw_mask_tints_only_inside():
    f = np.zeros((40, 40, 3), np.uint8)
    m = np.zeros((40, 40), bool)
    m[10:20, 10:20] = True
    out = draw_mask(f, m, (255, 0, 0), alpha=0.5, outline=0)
    assert out[15, 15, 0] == 127 and out[30, 30].sum() == 0
    assert f.sum() == 0  # input untouched
    assert draw_mask(f, np.zeros_like(m), (1, 1, 1)).sum() == 0
    assert draw_mask(f, m, (0, 255, 0), outline=2, dashed=True).sum() > 0
    assert draw_marker(f.copy(), (20, 20), (255, 255, 255)).sum() > 0


@pytest.mark.parametrize("deg", [0, 90, 180, 270, -90])
def test_rotate_and_unrotate_click(deg):
    img = np.zeros((40, 60, 3), np.uint8)
    img[10, 50] = 255  # one bright pixel at (x=50, y=10)
    r = rotate(img, deg)
    assert r.shape == ((40, 60, 3) if deg % 180 == 0 else (60, 40, 3))
    ry, rx = np.argwhere(r[..., 0])[0]
    x, y = unrotate_xy(rx + 0.5, ry + 0.5, deg, img.shape)  # click the centre of that pixel on screen
    assert (int(x), int(y)) == (50, 10)


def test_segment_colours_cycle():
    assert segment_colour(0) != segment_colour(1)
    assert segment_colour(len(PALETTE)) == segment_colour(0)


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_click_track_undo(tmp_path, occlusion_video, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import cv2
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from reeftrack.config import load_config
    from reeftrack.gui import app as gui_app
    from reeftrack.io.run import prepare_run
    from reeftrack.tracking import session as session_mod
    from reeftrack.tracking.backends import FakeBackend

    src = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for fr in occlusion_video:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    cfg = load_config({"paths": {"data_root": str(tmp_path / "runs")}, "video": {"proxy_height": 120, "crf": 0}})
    run = prepare_run(src, cfg=cfg, name="t", log=lambda *_: None)
    monkeypatch.setattr(session_mod, "load_backend", lambda tc, device: FakeBackend(tol=60))

    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow()
    win.open_run(run.root)
    win.worker.wait(5000)
    qapp.processEvents()

    win.goto(5)
    win._on_click(20 + 2.5 * 5, 60, Qt.MouseButton.LeftButton)
    win.worker.wait(5000)
    qapp.processEvents()
    assert win.pending is not None
    win.commit_click()
    win.worker.wait(10000)
    qapp.processEvents()
    tr = win.session.tracklet
    assert len(tr.segments) == 1 and tr.segments[0].end_reason == "lost"
    assert (run.tracking_dir / "tracklet.json").exists()  # autosaved
    assert win.timeline.breaks  # break marked on the timeline
    win.jump_break(+1)
    assert win.cur == tr.segments[0].end + 1
    win.undo()
    assert not win.session.tracklet.segments
    win.close()


def _synthetic_run(tmp_path, occlusion_video):
    import cv2

    from reeftrack.config import load_config
    from reeftrack.io.run import prepare_run

    src = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for fr in occlusion_video:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    cfg = load_config({"paths": {"data_root": str(tmp_path / "runs")}, "video": {"proxy_height": 120, "crf": 0}})
    return prepare_run(src, cfg=cfg, name="t", log=lambda *_: None)


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_behaviour_annotation(tmp_path, occlusion_video, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import csv

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from reeftrack.gui import app as gui_app
    from reeftrack.gui.behaviour_panel import BehaviourDialog

    run = _synthetic_run(tmp_path, occlusion_video)
    qapp = QApplication.instance() or QApplication([])

    # the dialog itself: OK only once a behaviour is chosen; shows the ethogram definition
    d = BehaviourDialog(["Calm swimming", "Foraging"], 3, 9, 50, 30.0,
                        definitions={"Foraging": "Fish approaches the substrate head-first."})
    ok = d.buttons.button(d.buttons.StandardButton.Ok)
    assert not ok.isEnabled()
    d.list.setCurrentRow(1)
    d.comment.setPlainText("nibbling algae")
    assert ok.isEnabled() and d.values() == ("Foraging", 3, 9, "nibbling algae")
    assert "head-first" in d.definition.text()

    # simulate the user picking a behaviour in the modal dialog
    choice = {"row": 0, "comment": "near bommie, 2nd fish visible"}  # row 0 = Foraging

    def fake_exec(self):
        self.list.setCurrentRow(choice["row"])
        self.comment.setPlainText(choice["comment"])
        return BehaviourDialog.DialogCode.Accepted

    monkeypatch.setattr(BehaviourDialog, "exec", fake_exec)
    win = gui_app.MainWindow(behaviour_only=True)
    win.annotator_name = "tester"
    win.open_run(run.root)
    qapp.processEvents()
    assert win.worker is None  # behaviour-only: SAM never loaded
    win._on_click(50, 60, Qt.MouseButton.LeftButton)
    assert win.pending is None  # clicks don't start mask tracking

    win.goto(5)
    win.mark_in()
    win.goto(15)
    win.mark_out()  # opens the dialog -> event saved
    path = run.root / "annotations" / "behaviours.csv"
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 1
    r = rows[0]
    assert (r["behaviour"], r["start_frame"], r["end_frame"]) == ("Foraging", "5", "15")
    assert r["comment"] == "near bommie, 2nd fish visible" and r["annotator"] == "tester"
    assert r["source_video"].endswith("clip.mp4") and win.marks == [None, None]

    # overlapping second event, marked backwards (end before start)
    choice.update(row=11, comment="")  # row 11 = Calm swimming
    win.goto(20)
    win.mark_in()
    win.goto(10)
    win.mark_out()
    assert [e.behaviour for e in win.blog.active_at(12)] == ["Foraging", "Calm swimming"]
    assert len(win.timeline.behaviours) == 2 and win.bpanel.table.rowCount() == 2

    # edit + delete through the table
    win.bpanel.table.selectRow(0)
    choice.update(row=0, comment="edited")
    win.edit_behaviour()
    win.bpanel.table.selectRow(1)
    win.delete_behaviour(confirm=False)
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 1 and rows[0]["comment"] == "edited"
    win.close()


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_loop_annotation(tmp_path, occlusion_video, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import csv

    from PySide6.QtWidgets import QApplication

    from reeftrack.gui import app as gui_app
    from reeftrack.gui.behaviour_panel import LoopDialog

    run = _synthetic_run(tmp_path, occlusion_video)
    # a stereo trajectory: fish goes out 1 m along x and comes back to 0.1 m from the start
    (run.root / "fusion_stereo").mkdir()
    lines = ["proxy_frame,x,y,z,kind"]
    for f in range(0, 50):
        x = f / 20 if f <= 20 else max(0.1, 1 - (f - 20) / 20)
        lines.append(f"{f},{x},0,0,measured")
    (run.root / "fusion_stereo" / "trajectory.csv").write_text("\n".join(lines) + "\n")

    qapp = QApplication.instance() or QApplication([])
    seen = {}

    def fake_exec(self):
        seen["thumbs"] = self.findChildren(type(self.duration))  # labels incl. the two thumbnails
        self.confidence.setCurrentText(seen.get("conf", "probable"))
        self.landmark.setText("big bommie")
        self.comment.setPlainText("came back from the left")
        return LoopDialog.DialogCode.Accepted

    monkeypatch.setattr(LoopDialog, "exec", fake_exec)
    win = gui_app.MainWindow(behaviour_only=True)
    win.annotator_name = "tester"
    win.open_run(run.root)
    win.tabs.setCurrentWidget(win.lpanel)
    qapp.processEvents()

    win.goto(2)
    win._dispatch("in")  # I on the Loops tab -> loop start
    assert win.loop_marks == [2, None] and win.marks == [None, None]  # behaviour marks untouched
    win.goto(40)
    win.render()  # draws the picture-in-picture of the start frame without error
    win._dispatch("out")  # O -> dialog -> saved
    rows = list(csv.DictReader(open(run.root / "annotations" / "loops.csv")))
    assert len(rows) == 1
    r = rows[0]
    assert (r["start_frame"], r["end_frame"], r["confidence"], r["landmark"]) == ("2", "40", "probable", "big bommie")
    assert r["units"] == "m" and float(r["return_distance"]) == pytest.approx(0.0, abs=1e-6)  # back at x=0.1
    assert float(r["max_excursion"]) == pytest.approx(0.9, abs=0.01)  # from x=0.1 at frame 2 out to x=1
    assert len(win.timeline.loops) == 1 and win.lpanel.table.rowCount() == 1
    assert "Loop 1" in win.lpanel.active.text() or "loop 1" in win.lpanel.active.text()

    # too short a loop is rejected (dialog OK disabled) -> nothing saved; edit + delete
    d = LoopDialog(5, 5, 50, 30.0)
    assert not d.buttons.button(d.buttons.StandardButton.Ok).isEnabled()
    win.lpanel.table.selectRow(0)
    seen["conf"] = "possible"
    win.edit_loop()
    assert win.loops.events[0].confidence == "possible"
    win.lpanel.table.selectRow(0)
    win.delete_loop(confirm=False)
    assert list(csv.DictReader(open(run.root / "annotations" / "loops.csv"))) == []
    win.close()


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_library_annotation(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import csv

    from PySide6.QtWidgets import QApplication

    from reeftrack.annotation import events
    from reeftrack.gui import app as gui_app
    from reeftrack.gui.behaviour_panel import BehaviourDialog
    from reeftrack.library import Library, LibrarySettings
    from tests.test_library import REL, _write_video

    monkeypatch.setenv("REEFTRACK_HOME", str(tmp_path / "home"))
    drive, rfs = tmp_path / "drive" / "originaldata", tmp_path / "RFS"
    for d in (drive, rfs / "compressed_videos", rfs / "annotations"):
        d.mkdir(parents=True)
    LibrarySettings(video_root=str(drive), compressed_root=str(rfs / "compressed_videos"),
                    annotations_root=str(rfs / "annotations"), cache_dir=str(tmp_path / "cache"),
                    height=240, bitrate="2M").save()
    video = drive / REL
    _write_video(video)

    monkeypatch.setattr(BehaviourDialog, "exec", lambda self: (self.list.setCurrentRow(1),
                                                               BehaviourDialog.DialogCode.Accepted)[1])
    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow(library=True)
    win.annotator_name = "Anna"
    win.open_video(video, choice="compress")  # compress -> copy to cache -> open, in the background
    win.worker.wait(60000)
    qapp.processEvents()
    lib = Library()
    rel = lib.rel(video)
    assert win.run is not None and win.run.root == lib.annotation_dir(rel) and not win.run.video_is_raw
    assert lib.compressed_info(rel) is not None
    assert win.tabs.currentWidget() is win.bpanel and not win.tabs.isTabEnabled(0)

    win.goto(2)
    win.mark_in()
    win.goto(8)
    win.mark_out()
    shared = win.run.root / "behaviours_anna.csv"
    assert len(list(csv.DictReader(open(shared)))) == 1 and not win.save_banner.isVisible()

    real = events.IntervalLog._atomic_write
    monkeypatch.setattr(events.IntervalLog, "_atomic_write",
                        staticmethod(lambda p, t: (_ for _ in ()).throw(OSError("Host is down")) if p == shared
                                     else real(p, t)))
    win.goto(12)
    win.mark_in()
    win.goto(14)
    win.mark_out()
    assert not win.save_banner.isHidden()  # red warning while RFS is unreachable
    monkeypatch.setattr(events.IntervalLog, "_atomic_write", staticmethod(real))
    win.save()  # Ctrl+S once RFS is back
    assert win.save_banner.isHidden() and len(list(csv.DictReader(open(shared)))) == 2
    win.close()


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_gap_filling_and_timeline_selection(tmp_path, occlusion_video, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from reeftrack.gui import app as gui_app
    from reeftrack.gui.behaviour_panel import BehaviourDialog

    run = _synthetic_run(tmp_path, occlusion_video)  # 50 frames
    pick = {"row": 11}  # Calm swimming
    seen = {}

    def fake_exec(self):
        seen["range"] = (self.start.value(), self.end.value())
        self.list.setCurrentRow(pick["row"])
        return BehaviourDialog.DialogCode.Accepted

    monkeypatch.setattr(BehaviourDialog, "exec", fake_exec)
    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow(behaviour_only=True)
    win.annotator_name = "t"
    win.open_run(run.root)
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()

    # interval annotation first
    win.goto(10)
    win.mark_in()
    win.goto(20)
    win.mark_out()
    assert win.blog.gaps() == [(0, 9), (21, 49)] and "2 gaps" in win.bpanel.coverage.text()

    # G from frame 0 selects the gap we're in; B opens the dialog pre-filled with the whole gap
    win.goto(0)
    win.jump_gap(+1)
    assert win.marks == [0, 9]
    win._dispatch("add")
    assert seen["range"] == (0, 9) and win.blog.gaps() == [(21, 49)]

    # clicking the remaining gap selects it; double-clicking opens the dialog for it
    pick["row"] = 13  # Out of view
    win._on_lane_clicked("behaviour", 30, None, False)
    assert win.marks == [21, 49]
    win._on_lane_clicked("behaviour", 30, None, True)
    assert seen["range"] == (21, 49) and win.blog.gaps() == []
    assert "Every frame" in win.bpanel.coverage.text()

    # a real mouse click on a coloured chunk highlights its row in the table (and the chunk)
    qapp.processEvents()
    win.timeline.repaint()
    ltop, lh, rows, _ = win.timeline._layout["behaviour"]
    first = next(r for r in rows if r[1] == 10)  # the original 10-20 event
    QTest.mouseClick(win.timeline, Qt.MouseButton.LeftButton, pos=QPoint(int(win.timeline._x(15)), int(ltop + 2)))
    qapp.processEvents()
    assert win.bpanel.selected_event_id() == first[4]
    assert win.timeline.highlight.get("behaviour") == first[4]
    win.close()


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_offline_compressed_video(tmp_path, monkeypatch):
    """No hard drive, no RFS: open the compressed copy, annotations saved to a local folder."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import csv

    from PySide6.QtWidgets import QApplication

    from reeftrack.gui import app as gui_app
    from reeftrack.gui.behaviour_panel import BehaviourDialog
    from reeftrack.library import Library, LibrarySettings
    from tests.test_library import REL, _write_video

    monkeypatch.setenv("REEFTRACK_HOME", str(tmp_path / "home"))
    drive, rfs = tmp_path / "drive" / "originaldata", tmp_path / "RFS"
    for d in (drive, rfs / "compressed_videos", rfs / "annotations"):
        d.mkdir(parents=True)
    server = Library(LibrarySettings(video_root=str(drive), compressed_root=str(rfs / "compressed_videos"),
                                     annotations_root=str(rfs / "annotations"), cache_dir=str(tmp_path / "c"),
                                     height=240, bitrate="2M"))
    _write_video(drive / REL)
    out = server.compress(drive / REL)
    got = tmp_path / "Downloads"
    got.mkdir()
    shutil.copy(out, got / out.name)
    shutil.copy(server.sidecar_path(server.rel(drive / REL)), got)

    local = tmp_path / "laptop_annotations"
    local.mkdir()
    LibrarySettings(video_root="", compressed_root="", annotations_root=str(local)).save()  # her settings
    monkeypatch.setattr(BehaviourDialog, "exec", lambda self: (self.list.setCurrentRow(0),
                                                               BehaviourDialog.DialogCode.Accepted)[1])
    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow(library=True)
    win.annotator_name = "Anna"
    win.open_compressed(got / out.name)
    qapp.processEvents()
    assert win.run.root == local / "Maldives/2024_02_14/AM/Cait/Left/GX010703" and win.n == 30
    win.goto(3)
    win.mark_in()
    win.goto(6)
    win.mark_out()
    rows = list(csv.DictReader(open(win.run.root / "behaviours_anna.csv")))
    assert rows[0]["behaviour"] == "Foraging" and rows[0]["source_video"] == REL
    win.close()


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_gui_rotate_view(tmp_path, occlusion_video):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from reeftrack.gui import app as gui_app

    run = _synthetic_run(tmp_path, occlusion_video)
    qapp = QApplication.instance() or QApplication([])
    win = gui_app.MainWindow(behaviour_only=True)
    win.annotator_name = "tester"
    win.open_run(run.root)
    h, w = win.reader[0].shape[:2]
    pm = win.view._item.pixmap()
    assert (pm.width(), pm.height()) == (w, h)
    win.rotate_view(90)
    pm = win.view._item.pixmap()
    assert (pm.width(), pm.height()) == (h, w)  # portrait on screen
    assert run.meta()["view"]["rotation"] == 90  # remembered for this video
    seen = []
    win._on_click = lambda x, y, b: seen.append((x, y))
    win._on_view_click(h - 20, 30, Qt.MouseButton.LeftButton)  # screen point -> video pixel
    assert seen == [(30, 20)]
    win.rotate_view(-90)
    assert win.rotation == 0
    win.rotate_view(-90)
    win.close()

    again = gui_app.MainWindow(behaviour_only=True)
    again.annotator_name = "tester"
    again.open_run(run.root)
    assert again.rotation == 270
    again.close()
    qapp.processEvents()
