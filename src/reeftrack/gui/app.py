"""reef-track GUI: play the clip, click the focal fish, patch breaks.

Workflow
  1. Play / scrub to a frame where the fish is clearly visible, left-click it.
     -> a white dashed preview mask appears (SAM, single frame).
  2. Enter (or "Track") commits: the tracker runs forward (and back-fills) live.
     The view follows the tracker; Stop interrupts.
  3. When the fish is lost the segment ends (red tick on the timeline). Press N to jump to
     the next break, scrub until the fish is back, click it again -> new segment that also
     tracks *backwards* to close the gap from the other side.
  4. Clicking a frame that is already tracked corrects it from that frame on.
  Tracklets autosave to <run>/tracking/ after every change.

Behaviour annotation (Behaviour tab; can run alongside tracking, or alone with --behaviour-only)
  I at the first frame of a behaviour, O at the last frame -> a dialog asks for the
  behaviour (required; click or press its number key) and an optional comment -> one row in
  <run>/annotations/behaviours.csv (saved immediately). Events may overlap.

Loops (Loops tab): the fish returns to a place it has already been. I at the loop's start,
  O when it is back at that spot (a picture-in-picture of the start frame helps you compare)
  -> confidence, landmark, comment -> one row in <run>/annotations/loops.csv, with 3D return
  distance / path length / max excursion filled in from the fish trajectory when available.
  I / O / B act on whichever annotation tab is open.

Keys: Space play/pause · ←/→ ±1 frame · Shift+←/→ ±1 s · N/P next/prev break · Enter track
      Esc cancel · Ctrl+Z undo segment · M toggle masks · Ctrl+S save
      I / O mark behaviour start / end · B add behaviour for the marked range
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from reeftrack.gui.behaviour_panel import LOOP_COLOUR, BehaviourDialog, BehaviourPanel, LoopDialog, LoopPanel
from reeftrack.gui.overlay import (
    PREVIEW,
    behaviour_colour,
    draw_labels,
    draw_marker,
    draw_mask,
    draw_pip,
    segment_colour,
)
from reeftrack.gui.widgets import Timeline, VideoView

log = logging.getLogger("reeftrack.gui")

STYLE = """
QMainWindow, QWidget { background: #0f1a21; color: #dde6ea; font-size: 13px; }
QGroupBox { border: 1px solid #24343f; border-radius: 6px; margin-top: 14px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #7fd6e8; font-weight: 600; }
QPushButton { background: #1d2d38; border: 1px solid #2c4352; border-radius: 5px; padding: 6px 10px; }
QPushButton:hover { background: #26404f; }
QPushButton:disabled { color: #5d6f79; border-color: #1f2c35; }
QPushButton#primary { background: #0f7c8c; border-color: #13a3b8; font-weight: 600; }
QPushButton#primary:hover { background: #1395a8; }
QPushButton#danger { background: #7c1f2a; border-color: #a83240; }
QListWidget { background: #13222b; border: 1px solid #24343f; border-radius: 5px; }
QListWidget::item:selected { background: #26404f; }
QProgressBar { border: 1px solid #24343f; border-radius: 4px; text-align: center; height: 14px; }
QProgressBar::chunk { background: #13a3b8; }
QTableWidget { background: #13222b; border: 1px solid #24343f; gridline-color: #24343f; }
QHeaderView::section { background: #1d2d38; color: #9fb3bd; border: none; padding: 4px; }
QTabWidget::pane { border: 1px solid #24343f; border-radius: 6px; }
QTabBar::tab { background: #1d2d38; padding: 6px 14px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
QTabBar::tab:selected { background: #0f7c8c; }
QLabel#hint { color: #ffd479; font-size: 13px; padding: 6px; background: #1a2730; border-radius: 5px; }
QSlider::groove:horizontal { height: 4px; background: #2c4352; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px; background: #13a3b8; }
"""


class Worker(QThread):
    """Run `fn(emit, stop)` off the GUI thread; `emit(obj)` -> progress signal."""

    progress = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.done.emit(self._fn(self.progress.emit, lambda: self._stop))
        except Exception as e:  # surfaced in a dialog, full trace in the terminal
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path | None = None, behaviour_only: bool = False):
        super().__init__()
        self.config_path = config_path
        self.behaviour_only = behaviour_only
        self.blog = None  # BehaviourLog of the open run
        self.marks: list[int | None] = [None, None]  # behaviour start / end being marked
        self.loops = None  # LoopLog of the open run
        self.loop_marks: list[int | None] = [None, None]
        self.settings = QSettings("reeftrack", "reeftrack-gui")
        self.annotator_name = str(self.settings.value("annotator", "") or "")
        self.run = None
        self.cfg = None
        self.session = None
        self.reader = None
        self.fmap = None
        self.cur = 0
        self.pending: tuple[int, tuple[float, float], np.ndarray] | None = None
        self.worker: Worker | None = None
        self._track_t0 = 0.0
        self._track_n = 0

        self.setWindowTitle("reef-track")
        self.resize(1500, 900)
        self._build_ui()
        self._build_shortcuts()
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick)
        self._set_enabled(False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.view = VideoView()
        self.view.clicked.connect(self._on_click)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.goto)

        self.btn_prev10 = QPushButton("⏪")
        self.btn_prev = QPushButton("◀")
        self.btn_play = QPushButton("▶  Play")
        self.btn_next = QPushButton("▶")
        self.btn_next10 = QPushButton("⏩")
        for b, d in [(self.btn_prev10, -30), (self.btn_prev, -1), (self.btn_next, 1), (self.btn_next10, 30)]:
            b.clicked.connect(lambda _=False, d=d: self.goto(self.cur + d))
            b.setFixedWidth(44)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setFixedWidth(90)
        self.speed = QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._update_timer)
        self.frame_box = QSpinBox()
        self.frame_box.setKeyboardTracking(False)
        self.frame_box.valueChanged.connect(self.goto)
        self.time_label = QLabel("–")
        self.time_label.setStyleSheet("color:#9fb3bd")

        transport = QHBoxLayout()
        for w in (self.btn_prev10, self.btn_prev, self.btn_play, self.btn_next, self.btn_next10):
            transport.addWidget(w)
        transport.addSpacing(12)
        transport.addWidget(QLabel("speed"))
        transport.addWidget(self.speed)
        transport.addSpacing(12)
        transport.addWidget(QLabel("frame"))
        transport.addWidget(self.frame_box)
        transport.addWidget(self.time_label, 1)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.addWidget(self.view, 1)
        lv.addWidget(self.timeline)
        lv.addLayout(transport)

        # --- right panel
        self.hint = QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)

        self.btn_track = QPushButton("Track  ⏎")
        self.btn_track.setObjectName("primary")
        self.btn_track.clicked.connect(self.commit_click)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_click)
        self.btn_stop = QPushButton("Stop tracking")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop_worker)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.track_status = QLabel("")
        self.track_status.setStyleSheet("color:#9fb3bd")
        act = QGroupBox("Tracking")
        av = QVBoxLayout(act)
        row = QHBoxLayout()
        row.addWidget(self.btn_track, 2)
        row.addWidget(self.btn_cancel, 1)
        av.addLayout(row)
        av.addWidget(self.btn_stop)
        av.addWidget(self.progress)
        av.addWidget(self.track_status)
        self.follow = QCheckBox("Follow tracker live")
        self.follow.setChecked(True)
        av.addWidget(self.follow)

        self.seg_list = QListWidget()
        self.seg_list.itemDoubleClicked.connect(lambda it: self.goto(it.data(Qt.ItemDataRole.UserRole)))
        self.btn_undo = QPushButton("Undo last segment")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_nextbreak = QPushButton("Next break  N")
        self.btn_nextbreak.clicked.connect(lambda: self.jump_break(+1))
        self.btn_prevbreak = QPushButton("Prev break  P")
        self.btn_prevbreak.clicked.connect(lambda: self.jump_break(-1))
        segs = QGroupBox("Tracklet")
        sv = QVBoxLayout(segs)
        self.coverage = QLabel("")
        sv.addWidget(self.coverage)
        sv.addWidget(self.seg_list, 1)
        r2 = QHBoxLayout()
        r2.addWidget(self.btn_prevbreak)
        r2.addWidget(self.btn_nextbreak)
        sv.addLayout(r2)
        sv.addWidget(self.btn_undo)

        disp = QGroupBox("Display")
        dv = QFormLayout(disp)
        self.show_masks = QCheckBox()
        self.show_masks.setChecked(True)
        self.show_masks.toggled.connect(lambda _: self.render())
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 90)
        self.opacity.setValue(45)
        self.opacity.valueChanged.connect(lambda _: self.render())
        dv.addRow("Masks (M)", self.show_masks)
        dv.addRow("Opacity", self.opacity)

        self.show_labels = QCheckBox()
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(lambda _: self.render())
        dv.addRow("Labels", self.show_labels)

        self.bpanel = BehaviourPanel()
        self.bpanel.seek.connect(self.goto)
        self.bpanel.btn_in.clicked.connect(self.mark_in)
        self.bpanel.btn_out.clicked.connect(self.mark_out)
        self.bpanel.btn_clear.clicked.connect(self.clear_marks)
        self.bpanel.btn_add.clicked.connect(self.add_behaviour)
        self.bpanel.btn_edit.clicked.connect(self.edit_behaviour)
        self.bpanel.btn_delete.clicked.connect(self.delete_behaviour)
        self.bpanel.btn_annotator.clicked.connect(lambda: self._ensure_annotator(force=True))
        self.bpanel.table.cellDoubleClicked.connect(lambda *_: self.edit_behaviour())

        self.lpanel = LoopPanel()
        self.lpanel.seek.connect(self.goto)
        self.lpanel.btn_in.clicked.connect(self.loop_in)
        self.lpanel.btn_out.clicked.connect(self.loop_out)
        self.lpanel.btn_clear.clicked.connect(self.clear_loop_marks)
        self.lpanel.btn_add.clicked.connect(self.add_loop)
        self.lpanel.btn_edit.clicked.connect(self.edit_loop)
        self.lpanel.btn_delete.clicked.connect(self.delete_loop)
        self.lpanel.btn_annotator.clicked.connect(lambda: self._ensure_annotator(force=True))
        self.lpanel.table.cellDoubleClicked.connect(lambda *_: self.edit_loop())

        track_tab = QWidget()
        tv = QVBoxLayout(track_tab)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(act)
        tv.addWidget(segs, 1)
        self.tabs = QTabWidget()
        self.tabs.addTab(track_tab, "Tracking")
        self.tabs.addTab(self.bpanel, "Behaviour")
        self.tabs.addTab(self.lpanel, "Loops")
        self.tabs.currentChanged.connect(lambda _: self.render() if self.session else None)

        self.info = QLabel("No run loaded")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#9fb3bd")
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.info)
        rv.addWidget(self.hint)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(disp)
        right.setMinimumWidth(380)
        right.setMaximumWidth(540)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setSizes([1040, 460])  # room for the behaviour table's comment column
        self.setCentralWidget(split)

        m = self.menuBar().addMenu("&File")
        for text, slot, key in [("New run from video…", self.new_run, "Ctrl+N"), ("Open run…", self.open_run_dialog, "Ctrl+O"),
                                ("Save tracklet", self.save, "Ctrl+S")]:
            a = QAction(text, self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(slot)
            m.addAction(a)
        self.statusBar().showMessage("File → Open run… or New run from video…")

    def _build_shortcuts(self) -> None:
        keys = {
            "Space": self.toggle_play, "Left": lambda: self.goto(self.cur - 1), "Right": lambda: self.goto(self.cur + 1),
            "Shift+Left": lambda: self.goto(self.cur - self._fps_i()), "Shift+Right": lambda: self.goto(self.cur + self._fps_i()),
            "N": lambda: self.jump_break(+1), "P": lambda: self.jump_break(-1), "Return": self.commit_click,
            "Enter": self.commit_click, "Escape": self.cancel_click, "Ctrl+Z": self.undo,
            "M": lambda: self.show_masks.toggle(), "Home": lambda: self.goto(0),
            "End": lambda: self.goto(self.n - 1),
            "I": lambda: self._dispatch("in"), "O": lambda: self._dispatch("out"),
            "[": lambda: self._dispatch("in"), "]": lambda: self._dispatch("out"), "B": lambda: self._dispatch("add"),
        }
        for k, fn in keys.items():
            QShortcut(QKeySequence(k), self, activated=fn)

    def _set_enabled(self, loaded: bool) -> None:
        busy = self.worker is not None and self.worker.isRunning()
        for w in (self.btn_prev10, self.btn_prev, self.btn_play, self.btn_next, self.btn_next10, self.frame_box,
                  self.btn_nextbreak, self.btn_prevbreak):
            w.setEnabled(loaded)
        self.btn_undo.setEnabled(loaded and not busy)
        for pnl in (self.bpanel, self.lpanel):
            for w in (pnl.btn_in, pnl.btn_out, pnl.btn_clear, pnl.btn_add, pnl.btn_edit, pnl.btn_delete):
                w.setEnabled(loaded)
        self.btn_track.setEnabled(loaded and self.pending is not None and not busy)
        self.btn_cancel.setEnabled(self.pending is not None and not busy)
        self.btn_stop.setVisible(busy)

    # ------------------------------------------------------------------ run handling
    @property
    def n(self) -> int:
        return len(self.reader) if self.reader else 1

    def _fps_i(self) -> int:
        return int(round(self.fmap.proxy_fps)) if self.fmap else 30

    def open_run_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open run directory", "data/runs")
        if d:
            self.open_run(Path(d))

    def new_run(self) -> None:
        from reeftrack.config import load_config
        from reeftrack.io.run import prepare_run
        from reeftrack.io.video import probe

        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", "", "Video (*.mp4 *.MP4 *.mov *.MOV *.avi)")
        if not path:
            return
        try:
            info = probe(path)
        except Exception as e:
            QMessageBox.critical(self, "Cannot read video", str(e))
            return
        start, ok = QInputDialog.getInt(self, "Start frame", f"First source frame (0–{info.n_frames - 1}):", 0, 0,
                                        info.n_frames - 1)
        if not ok:
            return
        end, ok = QInputDialog.getInt(self, "End frame", "Last source frame:", info.n_frames - 1, start, info.n_frames - 1)
        if not ok:
            return
        name, ok = QInputDialog.getText(self, "Run name", "Run name:", text=f"{Path(path).stem}_{start}-{end}")
        if not ok:
            return
        cfg = load_config(self.config_path)
        self.statusBar().showMessage("Building proxy (decoding the source once; a few minutes for 4K)…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._start_worker(lambda emit, stop: prepare_run(Path(path), start, end, name, cfg),
                           done=lambda run: (self.progress.setVisible(False), self.open_run(run.root)))

    def open_run(self, root: Path) -> None:
        from reeftrack.io.run import Run
        from reeftrack.io.video import FrameReader
        from reeftrack.tracking.session import TrackingSession

        try:
            self.run = Run.open(root)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Not a run", str(e))
            return
        self.cfg = self.run.config(self.config_path)
        self.fmap = self.run.frame_map()
        if self.reader:
            self.reader.close()
        self.reader = FrameReader(self.run.proxy)
        self.session = TrackingSession.from_run(self.run, self.cfg)
        from reeftrack.annotation.behaviour import BehaviourLog

        self.blog = BehaviourLog.for_run(self.run, self.cfg.annotation.behaviours)
        if not self.annotator_name and self.cfg.annotation.annotator:
            self.annotator_name = self.cfg.annotation.annotator
        self.marks = [None, None]
        from reeftrack.annotation.loops import LoopLog

        self.loops = LoopLog.for_run(self.run)
        self.loop_marks = [None, None]
        self.timeline.set_length(self.n)
        kf_path = self.run.recon_dir / "keyframes.json"
        if kf_path.exists():
            import json

            self.timeline.keyframes = np.asarray(json.loads(kf_path.read_text())["keyframes"])
        self.frame_box.setRange(0, self.n - 1)
        meta = self.run.meta()
        self.info.setText(f"<b>{self.run.root.name}</b><br>{Path(meta['source']['path']).name} · source frames "
                          f"{meta['frames']['source_start']}–{meta['frames']['source_end']}<br>{self.n} frames @ "
                          f"{self.fmap.proxy_fps:.2f} fps · tracker: {self.cfg.tracking.backend}")
        self.setWindowTitle(f"reef-track — {self.run.root.name}")
        self._set_enabled(True)
        self.refresh_tracklet()
        self.refresh_behaviours()
        self.refresh_loops()
        self.goto(0, force=True)
        self.statusBar().showMessage(f"Opened {root}")
        if self.behaviour_only:
            self.tabs.setCurrentWidget(self.bpanel)
            self.tabs.setTabEnabled(0, False)
            self.info.setText(self.info.text().rsplit("· tracker:", 1)[0] + "· behaviour annotation only")
        else:
            self._warmup()

    def _warmup(self) -> None:
        """Load the SAM weights in the background so the first click is snappy."""
        sess = self.session
        self.statusBar().showMessage(f"Loading {self.cfg.tracking.backend} weights in the background…")

        def load(emit, stop):
            return sess.backend.name

        def ok(name):
            note = "" if name == self.cfg.tracking.backend else f"  (fell back from {self.cfg.tracking.backend}: no access)"
            self.statusBar().showMessage(f"Tracker ready: {name}{note}")
            self.info.setText(self.info.text().rsplit("tracker:", 1)[0] + f"tracker: {name}")

        self._start_worker(load, done=ok)

    # ------------------------------------------------------------------ playback
    def goto(self, f: int, force: bool = False) -> None:
        if not self.reader:
            return
        f = int(np.clip(f, 0, self.n - 1))
        if f == self.cur and not force:
            return
        self.cur = f
        self.frame_box.blockSignals(True)
        self.frame_box.setValue(f)
        self.frame_box.blockSignals(False)
        self.timeline.set_pos(f)
        self.render()

    def toggle_play(self) -> None:
        if not self.reader:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶  Play")
        else:
            self._update_timer()
            self.play_timer.start()
            self.btn_play.setText("⏸  Pause")

    def _update_timer(self) -> None:
        mult = [0.25, 0.5, 1, 2, 4][self.speed.currentIndex()]
        fps = (self.fmap.proxy_fps if self.fmap else 30) * mult
        self.play_timer.setInterval(int(1000 / fps))

    def _tick(self) -> None:
        if self.cur >= self.n - 1:
            self.toggle_play()
            return
        self.goto(self.cur + 1)

    # ------------------------------------------------------------------ rendering
    def render(self) -> None:
        if not self.reader:
            return
        frame = self.reader[self.cur]
        tr = self.session.tracklet
        a = self.opacity.value() / 100
        if self.show_masks.isChecked():
            ob = tr.obs.get(self.cur)
            if ob is not None:
                m = tr.mask(self.cur)
                if m is not None:
                    frame = draw_mask(frame, m, segment_colour(ob.segment_id), a, outline=3)
            if self.pending and self.pending[0] == self.cur:
                frame = draw_mask(frame, self.pending[2], PREVIEW, a * 0.6, outline=3, dashed=True)
                frame = draw_marker(frame.copy(), self.pending[1], PREVIEW)
        for s in tr.segments:
            if s.prompt_frame == self.cur:
                frame = draw_marker(frame.copy(), s.prompt_xy, segment_colour(s.segment_id))
        if self.blog and self.show_labels.isChecked():
            cats = self.blog.categories
            labels = [(e.behaviour, behaviour_colour(e.behaviour, cats)) for e in self.blog.active_at(self.cur)]
            for lp in (self.loops.active_at(self.cur) if self.loops else []):
                where = "start" if self.cur == lp.start_frame else "return" if self.cur == lp.end_frame else "in"
                labels.append((f"Loop {lp.event_id} ({where})", LOOP_COLOUR))
            frame = draw_labels(frame, labels)
        if self.loop_marks[0] is not None and self.tabs.currentWidget() is self.lpanel:
            frame = draw_pip(frame, self.reader[self.loop_marks[0]], f"loop start - frame {self.loop_marks[0]}")
        self.view.set_frame(frame)
        self._update_active_label()
        t = self.fmap.time_s(self.cur)
        self.time_label.setText(f"source frame {self.fmap.to_source(self.cur)}  ·  {int(t // 60):02d}:{t % 60:05.2f}")
        self._update_hint()

    def _update_hint(self) -> None:
        tr = self.session.tracklet
        busy = self.worker is not None and self.worker.isRunning()
        if self.tabs.currentWidget() is self.lpanel:
            s, e = self.loop_marks
            if s is None:
                text = "Loops: at the frame where a loop <b>starts</b> (a place you'll recognise), press <b>I</b>."
            elif e is None:
                text = (f"Loop start at frame {s} (shown top-right). When the fish is <b>back at that spot</b>, "
                        "press <b>O</b>.")
            else:
                text = f"Loop {min(s, e)}–{max(s, e)} marked. Press <b>B</b> to save it."
        elif self.tabs.currentWidget() is self.bpanel:
            s, e = self.marks
            if s is None:
                text = "Behaviour: go to the <b>first</b> frame of a behaviour and press <b>I</b>."
            elif e is None:
                text = (f"Start marked at frame {s}. Go to the <b>last</b> frame of the behaviour and press <b>O</b> "
                        "(then pick the behaviour and add a comment).")
            else:
                text = f"Range {min(s, e)}–{max(s, e)} marked. Press <b>B</b> to assign a behaviour."
        elif busy and self.pending is None:
            text = "Tracking… the view follows the tracker. Press Stop to interrupt."
        elif self.pending:
            text = "Preview shown (white). Press <b>Enter</b> to track from here, or click again / Esc to cancel."
        elif not tr.segments:
            text = "Find a frame where the focal fish is clearly visible and <b>left-click it</b>."
        elif self.cur in tr.obs:
            text = (f"Tracked · segment {tr.obs[self.cur].segment_id}. If the mask is wrong here, click the fish to "
                    "<b>correct</b> from this frame.")
        else:
            text = ("Fish not tracked here. When it is back in view, <b>click it to re-acquire</b>; the tracker "
                    "also runs backwards to close the gap.")
        self.hint.setText(text)

    def refresh_tracklet(self) -> None:
        tr = self.session.tracklet
        owner = np.full(self.n, -1, np.int32)
        for f, o in tr.obs.items():
            if 0 <= f < self.n:
                owner[f] = o.segment_id
        breaks = sorted({s.end + 1 for s in tr.segments if s.end_reason == "lost" and s.end + 1 < self.n}
                        | {s.start - 1 for s in tr.segments if s.start_reason == "lost" and s.start > 0})
        self.timeline.set_state(owner, [(s.prompt_frame, s.segment_id) for s in tr.segments], breaks)
        self.seg_list.clear()
        for s in sorted(tr.segments, key=lambda s: s.start):
            n_obs = sum(1 for o in tr.obs.values() if o.segment_id == s.segment_id)
            it = QListWidgetItem(f"● seg {s.segment_id}:  {s.start}–{s.end}   ({n_obs} fr)   "
                                 f"{s.start_reason} → {s.end_reason}")
            it.setForeground(QColor(*segment_colour(s.segment_id)))
            it.setData(Qt.ItemDataRole.UserRole, s.prompt_frame)
            self.seg_list.addItem(it)
        cov = len(tr.obs) / self.n
        gaps = tr.gaps(self.n)
        self.coverage.setText(f"coverage {100 * cov:.1f}% · {len(tr.segments)} segments · {len(gaps)} gaps")

    def jump_break(self, direction: int) -> None:
        if not self.session:
            return
        gaps = self.session.tracklet.gaps(self.n)
        starts = [g[0] for g in gaps]
        cands = [f for f in starts if (f > self.cur if direction > 0 else f < self.cur)]
        if cands:
            self.goto(min(cands) if direction > 0 else max(cands))

    # ------------------------------------------------------------------ clicking & tracking
    def _on_click(self, x: float, y: float, button) -> None:
        if not self.session or (self.worker and self.worker.isRunning()):
            return
        if self.behaviour_only:
            self.statusBar().showMessage("Mask tracking is disabled in behaviour-only mode.")
            return
        if button == Qt.MouseButton.RightButton:
            self.cancel_click()
            return
        if self.play_timer.isActive():
            self.toggle_play()
        f, sess = self.cur, self.session
        self.statusBar().showMessage(f"Segmenting click at ({x:.0f}, {y:.0f}) on frame {f}…")

        def done(mask):
            if not mask.any():
                self.statusBar().showMessage("Nothing segmented there – click closer to the fish's centre.")
                return
            self.pending = (f, (x, y), mask)
            self._set_enabled(True)
            self.statusBar().showMessage(f"Preview: {int(mask.sum())} px. Enter to track.")
            self.render()

        self._start_worker(lambda emit, stop: sess.preview(f, (x, y)), done=done)

    def cancel_click(self) -> None:
        self.pending = None
        self._set_enabled(self.reader is not None)
        self.render()

    def commit_click(self) -> None:
        if not self.pending or (self.worker and self.worker.isRunning()):
            return
        f, xy, _ = self.pending
        self.pending = None
        sess = self.session
        self.progress.setVisible(True)
        self.progress.setRange(0, self.n)
        self.progress.setValue(0)
        self._track_t0, self._track_n = time.time(), 0

        def fn(emit, stop):
            return sess.add_click(f, xy, callback=lambda fr, m, v: emit((fr, m is not None, v.flags)),
                                  should_stop=stop)

        self._start_worker(fn, progress=self._on_track_progress, done=self._on_track_done)
        self.render()

    def _on_track_progress(self, payload) -> None:
        fr, ok, flags = payload
        self._track_n += 1
        if ok:
            self.timeline.mark(fr, self.session.tracklet.owner(fr) if self.session.tracklet.owner(fr) is not None else -1)
        rate = self._track_n / max(1e-6, time.time() - self._track_t0)
        self.progress.setValue(min(self.n, self._track_n))
        self.track_status.setText(f"frame {fr} · {rate:.1f} fr/s · {'ok' if ok else ', '.join(flags)}")
        if self.follow.isChecked() and (self._track_n % 2 == 0 or not ok):
            self.goto(fr)
        else:
            self.timeline.update()

    def _on_track_done(self, seg) -> None:
        self.progress.setVisible(False)
        self.refresh_tracklet()
        self.save()
        msg = {"lost": "fish lost", "met_segment": "joined an existing segment", "video_end": "reached end of clip",
               "user": "stopped", "video_start": "reached start of clip"}
        self.track_status.setText(f"segment {seg.segment_id}: {seg.start}–{seg.end} · start: "
                                  f"{msg.get(seg.start_reason, seg.start_reason)} · end: {msg.get(seg.end_reason, seg.end_reason)}")
        if seg.end_reason == "lost":
            self.goto(seg.end + 1)
            self.statusBar().showMessage("Fish lost. Scrub forward until it is back and click it to re-acquire.")
        self.render()

    # ------------------------------------------------------------------ behaviour annotation
    def _ensure_annotator(self, force: bool = False) -> bool:
        if self.annotator_name and not force:
            return True
        name, ok = QInputDialog.getText(self, "Annotator", "Your name or initials (saved with every event):",
                                        text=self.annotator_name)
        if ok and name.strip():
            self.annotator_name = name.strip()
            self.settings.setValue("annotator", self.annotator_name)
        self._update_active_label()
        return bool(self.annotator_name)

    def mark_in(self) -> None:
        if not self.blog:
            return
        self.marks = [self.cur, None]
        self.tabs.setCurrentWidget(self.bpanel)
        self.statusBar().showMessage(f"Behaviour start marked at frame {self.cur}. Press O at its last frame.")
        self.refresh_behaviours()

    def mark_out(self) -> None:
        if not self.blog:
            return
        if self.marks[0] is None:
            self.statusBar().showMessage("Press I at the first frame of the behaviour first.")
            return
        if self.play_timer.isActive():
            self.toggle_play()
        self.marks[1] = self.cur
        self.refresh_behaviours()
        self.add_behaviour()

    def clear_marks(self) -> None:
        self.marks = [None, None]
        self.refresh_behaviours()

    def _dialog(self, start, end, behaviour=None, comment="", title="New behaviour event"):
        return BehaviourDialog(self.blog.categories, start, end, self.n, self.fmap.proxy_fps, behaviour, comment,
                               title, self)

    def add_behaviour(self) -> None:
        if not self.blog:
            return
        s, e = self.marks
        if s is None:
            self.statusBar().showMessage("Mark the behaviour first: I at its first frame, O at its last frame.")
            return
        e = self.cur if e is None else e
        if not self._ensure_annotator():
            return
        dlg = self._dialog(min(s, e), max(s, e))
        if dlg.exec() == BehaviourDialog.DialogCode.Accepted:
            self.save_behaviour(*dlg.values())

    def save_behaviour(self, behaviour: str, start: int, end: int, comment: str = "",
                       event_id: int | None = None) -> None:
        """Validate + write one event (new or edited) to the CSV."""
        try:
            if event_id is None:
                ev = self.blog.add(behaviour, start, end, comment, self.annotator_name)
                self.marks = [None, None]
            else:
                ev = self.blog.update(event_id, behaviour=behaviour, start_frame=start, end_frame=end,
                                      comment=comment, annotator=self.annotator_name)
        except ValueError as err:
            QMessageBox.warning(self, "Behaviour not saved", str(err))
            return
        self.refresh_behaviours()
        self.render()
        self.statusBar().showMessage(f"Saved event {ev.event_id}: {ev.behaviour} {ev.start_frame}–{ev.end_frame} "
                                     f"→ {self.blog.path}", 5000)

    def edit_behaviour(self) -> None:
        eid = self.bpanel.selected_event_id()
        if eid is None or not self.blog:
            return
        ev = self.blog.get(eid)
        dlg = self._dialog(ev.start_frame, ev.end_frame, ev.behaviour, ev.comment, f"Edit event {eid}")
        if dlg.exec() == BehaviourDialog.DialogCode.Accepted:
            self.save_behaviour(*dlg.values(), event_id=eid)

    def delete_behaviour(self, confirm: bool = True) -> None:
        eid = self.bpanel.selected_event_id()
        if eid is None or not self.blog:
            return
        ev = self.blog.get(eid)
        if confirm and QMessageBox.question(self, "Delete event",
                                            f"Delete event {eid} ({ev.behaviour}, {ev.start_frame}–{ev.end_frame})?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.blog.delete(eid)
        self.refresh_behaviours()
        self.render()

    # ------------------------------------------------------------------ loops
    def _dispatch(self, action: str) -> None:
        """I / O / B act on the open annotation tab (Behaviour by default)."""
        if self.tabs.currentWidget() is self.lpanel:
            {"in": self.loop_in, "out": self.loop_out, "add": self.add_loop}[action]()
        else:
            {"in": self.mark_in, "out": self.mark_out, "add": self.add_behaviour}[action]()

    def loop_in(self) -> None:
        if not self.loops:
            return
        self.loop_marks = [self.cur, None]
        self.tabs.setCurrentWidget(self.lpanel)
        self.statusBar().showMessage(f"Loop start at frame {self.cur}. Press O when the fish is back at this spot.")
        self.refresh_loops()
        self.render()

    def loop_out(self) -> None:
        if not self.loops:
            return
        if self.loop_marks[0] is None:
            self.statusBar().showMessage("Press I at the start of the loop first.")
            return
        if self.play_timer.isActive():
            self.toggle_play()
        self.loop_marks[1] = self.cur
        self.refresh_loops()
        self.add_loop()

    def clear_loop_marks(self) -> None:
        self.loop_marks = [None, None]
        self.refresh_loops()
        self.render()

    def _thumb(self, f: int, width: int = 260):
        import cv2

        img = self.reader[int(f)]
        return cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)

    def _loop_dialog(self, start, end, title="New loop", **kw):
        return LoopDialog(start, end, self.n, self.fmap.proxy_fps, (self._thumb(start), self._thumb(end)),
                          title=title, parent=self, **kw)

    def add_loop(self) -> None:
        if not self.loops:
            return
        s, e = self.loop_marks
        if s is None:
            self.statusBar().showMessage("Mark the loop first: I at its start, O when the fish is back.")
            return
        e = self.cur if e is None else e
        if not self._ensure_annotator():
            return
        dlg = self._loop_dialog(min(s, e), max(s, e))
        if dlg.exec() == LoopDialog.DialogCode.Accepted:
            self.save_loop(**dlg.values())

    def save_loop(self, start: int, end: int, confidence: str = "certain", landmark: str = "", comment: str = "",
                  loop_id: int | None = None) -> None:
        try:
            if loop_id is None:
                lp = self.loops.add(start, end, comment, self.annotator_name, confidence=confidence, landmark=landmark)
                self.loop_marks = [None, None]
            else:
                lp = self.loops.update(loop_id, start_frame=start, end_frame=end, confidence=confidence,
                                       landmark=landmark, comment=comment, annotator=self.annotator_name)
        except ValueError as err:
            QMessageBox.warning(self, "Loop not saved", str(err))
            return
        self.refresh_loops()
        self.render()
        row = self.loops.row(lp)
        extra = f", returned within {float(row['return_distance']):.2f} {row['units']}" if row["return_distance"] else ""
        self.statusBar().showMessage(f"Saved loop {lp.event_id}: {lp.start_frame}–{lp.end_frame}{extra} "
                                     f"→ {self.loops.path}", 6000)

    def edit_loop(self) -> None:
        lid = self.lpanel.selected_event_id()
        if lid is None or not self.loops:
            return
        lp = self.loops.get(lid)
        dlg = self._loop_dialog(lp.start_frame, lp.end_frame, f"Edit loop {lid}", confidence=lp.confidence,
                                landmark=lp.landmark, comment=lp.comment)
        if dlg.exec() == LoopDialog.DialogCode.Accepted:
            self.save_loop(**dlg.values(), loop_id=lid)

    def delete_loop(self, confirm: bool = True) -> None:
        lid = self.lpanel.selected_event_id()
        if lid is None or not self.loops:
            return
        lp = self.loops.get(lid)
        if confirm and QMessageBox.question(self, "Delete loop", f"Delete loop {lid} ({lp.start_frame}–{lp.end_frame})?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.loops.delete(lid)
        self.refresh_loops()
        self.render()

    def refresh_loops(self) -> None:
        if not self.loops:
            return
        self.timeline.set_loops([(lp.start_frame, lp.end_frame, LOOP_COLOUR) for lp in self.loops.events],
                                tuple(self.loop_marks))
        self.lpanel.set_loops([self.loops.row(lp) for lp in self.loops.sorted()],
                              str(self.loops.path.relative_to(self.run.root)))
        s, e = self.loop_marks
        self.lpanel.range_label.setText("No loop marked" if s is None else
                                        f"Start {s}" + ("" if e is None else f"  ·  Return {e}  ({abs(e - s) + 1} frames)"))
        traj = self.loops.trajectory
        note = f" · 3D from {traj.source} ({traj.units})" if traj else " · no 3D trajectory yet (run `fuse`)"
        self.lpanel.count.setText(self.lpanel.count.text() + note)
        self._update_active_label()
        self._update_hint()

    def refresh_behaviours(self) -> None:
        if not self.blog:
            return
        cats = self.blog.categories
        self.timeline.set_behaviours([(e.start_frame, e.end_frame, behaviour_colour(e.behaviour, cats))
                                      for e in self.blog.events], tuple(self.marks))
        self.bpanel.set_events([self.blog.row(e) for e in self.blog.sorted()], cats,
                               str(self.blog.path.relative_to(self.run.root)))
        s, e = self.marks
        self.bpanel.range_label.setText("No range marked" if s is None else
                                        f"Start {s}" + ("" if e is None else f"  ·  End {e}  ({abs(e - s) + 1} frames)"))
        self._update_active_label()
        self._update_hint()

    def _update_active_label(self) -> None:
        self.bpanel.annotator.setText(f"Annotator: <b>{self.annotator_name or '–'}</b>")
        self.lpanel.annotator.setText(f"Annotator: <b>{self.annotator_name or '–'}</b>")
        if self.loops:
            act = self.loops.active_at(self.cur)
            self.lpanel.active.setText("At this frame: " + (", ".join(f"loop {lp.event_id}" for lp in act) if act else "–"))
        if self.blog:
            act = self.blog.active_at(self.cur)
            self.bpanel.active.setText("At this frame: " + (", ".join(e.behaviour for e in act) if act else "–"))

    def stop_worker(self) -> None:
        if self.worker:
            self.worker.stop()

    def undo(self) -> None:
        if self.session and not (self.worker and self.worker.isRunning()):
            seg = self.session.undo()
            if seg:
                self.statusBar().showMessage(f"Removed segment {seg.segment_id}")
                self.refresh_tracklet()
                self.save()
                self.render()

    def save(self) -> None:
        if self.session and self.run:
            self.session.save()
            self.statusBar().showMessage(f"Saved → {self.run.tracking_dir}", 4000)

    def _start_worker(self, fn, done=None, progress=None) -> None:
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("Busy – wait for the current job or press Stop.")
            return
        self.worker = Worker(fn, self)
        if progress:
            self.worker.progress.connect(progress)
        if done:
            self.worker.done.connect(done)
        self.worker.failed.connect(self._on_fail)
        self.worker.finished.connect(lambda: (self._set_enabled(self.reader is not None), self._update_hint()))
        self.worker.start()
        self._set_enabled(self.reader is not None)

    def _on_fail(self, msg: str) -> None:
        self.progress.setVisible(False)
        self.refresh_tracklet() if self.session else None
        QMessageBox.warning(self, "reef-track", msg)
        self.statusBar().showMessage(msg)

    def closeEvent(self, e):  # noqa: N802
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        if self.session and self.session.tracklet.segments:
            self.save()
        super().closeEvent(e)


def _dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, c in [(QPalette.ColorRole.Window, "#0f1a21"), (QPalette.ColorRole.WindowText, "#dde6ea"),
                    (QPalette.ColorRole.Base, "#13222b"), (QPalette.ColorRole.Text, "#dde6ea"),
                    (QPalette.ColorRole.Button, "#1d2d38"), (QPalette.ColorRole.ButtonText, "#dde6ea"),
                    (QPalette.ColorRole.Highlight, "#13a3b8")]:
        pal.setColor(role, QColor(c))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def main(run_dir: Path | None = None, config: Path | None = None, behaviour_only: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("reef-track")
    _dark_palette(app)
    win = MainWindow(config, behaviour_only=behaviour_only)
    win.show()
    if run_dir:
        QTimer.singleShot(0, lambda: win.open_run(Path(run_dir)))
    sys.exit(app.exec())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
