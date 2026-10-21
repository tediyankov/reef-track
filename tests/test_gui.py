"""Overlay rendering + an offscreen smoke test of the GUI workflow (fake tracker, synthetic run)."""

import os
import shutil

import numpy as np
import pytest

from reeftrack.gui.overlay import PALETTE, draw_marker, draw_mask, segment_colour


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

    # the dialog itself: OK only once a behaviour is chosen
    d = BehaviourDialog(["Calm Swimming", "Foraging"], 3, 9, 50, 30.0)
    ok = d.buttons.button(d.buttons.StandardButton.Ok)
    assert not ok.isEnabled()
    d.list.setCurrentRow(1)
    d.comment.setPlainText("nibbling algae")
    assert ok.isEnabled() and d.values() == ("Foraging", 3, 9, "nibbling algae")

    # simulate the user picking a behaviour in the modal dialog
    choice = {"row": 1, "comment": "near bommie, 2nd fish visible"}

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
    choice.update(row=0, comment="")
    win.goto(20)
    win.mark_in()
    win.goto(10)
    win.mark_out()
    assert [e.behaviour for e in win.blog.active_at(12)] == ["Foraging", "Calm Swimming"]
    assert len(win.timeline.behaviours) == 2 and win.bpanel.table.rowCount() == 2

    # edit + delete through the table
    win.bpanel.table.selectRow(0)
    choice.update(row=1, comment="edited")
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
