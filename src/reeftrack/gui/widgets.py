"""Custom Qt widgets: the video canvas and the tracklet timeline."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QSizePolicy, QWidget

from reeftrack.gui.overlay import segment_colour


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class VideoView(QGraphicsView):
    """Letter-boxed frame display; emits clicks in *image* pixel coordinates."""

    clicked = Signal(float, float, object)  # x, y, Qt.MouseButton

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setBackgroundBrush(QColor(10, 18, 24))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, rgb: np.ndarray) -> None:
        first = self._item.pixmap().isNull()
        self._item.setPixmap(QPixmap.fromImage(to_qimage(rgb)))
        if first:
            self.scene().setSceneRect(QRectF(0, 0, rgb.shape[1], rgb.shape[0]))
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, e):  # noqa: N802 (Qt naming)
        super().resizeEvent(e)
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, e):  # noqa: N802
        p = self.mapToScene(e.position().toPoint())
        pm = self._item.pixmap()
        if not pm.isNull() and 0 <= p.x() < pm.width() and 0 <= p.y() < pm.height():
            self.clicked.emit(p.x(), p.y(), e.button())


class Timeline(QWidget):
    """Whole-clip overview: tracked segments (coloured), breaks (red), clicks (diamonds), playhead."""

    seek = Signal(int)
    # (lane name, frame, event id under the cursor or None, double-click?) for clicks on annotation lanes
    lane_clicked = Signal(str, int, object, bool)

    def __init__(self, n_frames: int = 1, parent=None):
        super().__init__(parent)
        self.n = max(1, n_frames)
        self.owner = np.full(self.n, -1, np.int32)
        self.prompts: list[tuple[int, int]] = []  # (frame, segment_id)
        self.breaks: list[int] = []
        self.keyframes: np.ndarray = np.zeros(0, int)
        self.pos = 0
        # annotation lanes (drawn under the tracking bar, in this order): per lane a list of
        # (start, end, rgb) events and the (start, end) range currently being marked
        self.lanes: dict[str, list] = {"behaviour": [[], None], "loops": [[], None]}
        self.lane_gaps: dict[str, list[tuple[int, int]]] = {}  # drawn hatched (frames still to label)
        self.highlight: dict[str, object] = {}  # lane -> event id outlined (selected in the table)
        self._layout: dict[str, tuple] = {}  # lane -> (top, height, stacked rows, n rows), from the last paint
        self.setMinimumHeight(96)
        self.setMouseTracking(True)
        self._hover: int | None = None

    def set_length(self, n: int) -> None:
        self.n = max(1, n)
        self.owner = np.full(self.n, -1, np.int32)
        self.update()

    def set_state(self, owner: np.ndarray, prompts, breaks) -> None:
        self.owner, self.prompts, self.breaks = owner, prompts, breaks
        self.update()

    def mark(self, frame: int, segment_id: int) -> None:
        if 0 <= frame < self.n:
            self.owner[frame] = segment_id

    def set_lane(self, name: str, events, pending=None, gaps=None) -> None:
        """events: (start, end, rgb[, event_id]) tuples; gaps: (start, end) ranges to show hatched."""
        self.lanes[name] = [events, pending]
        if gaps is not None:
            self.lane_gaps[name] = gaps
        self.update()

    def set_highlight(self, name: str, event_id) -> None:
        self.highlight[name] = event_id
        self.update()

    def set_behaviours(self, events, pending=None, gaps=None) -> None:
        self.set_lane("behaviour", events, pending, gaps)

    def set_loops(self, events, pending=None) -> None:
        self.set_lane("loops", events, pending)

    @property
    def behaviours(self):
        return self.lanes["behaviour"][0]

    @property
    def loops(self):
        return self.lanes["loops"][0]

    @staticmethod
    def _stack(events):
        """Assign overlapping events to rows (greedy), so co-occurring behaviours stay visible."""
        ends: list[int] = []
        rows = []
        for ev in sorted(events, key=lambda t: t[0]):
            s, e, c = ev[:3]
            eid = ev[3] if len(ev) > 3 else None
            for r, last in enumerate(ends):
                if s > last:
                    ends[r] = e
                    rows.append((r, s, e, c, eid))
                    break
            else:
                ends.append(e)
                rows.append((len(ends) - 1, s, e, c, eid))
        return rows, max(1, len(ends))

    def set_pos(self, f: int) -> None:
        self.pos = f
        self.update()

    def _x(self, f: float) -> float:
        return 8 + (self.width() - 16) * f / max(1, self.n - 1)

    def _f(self, x: float) -> int:
        return int(np.clip(round((x - 8) / max(1, self.width() - 16) * (self.n - 1)), 0, self.n - 1))

    def paintEvent(self, _):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(18, 28, 36))
        lane = 20  # height of each annotation lane
        top, bh = 14, h - 22 - lane * len(self.lanes)
        p.fillRect(QRectF(8, top, w - 16, bh), QColor(40, 52, 62))
        for li, (name, (events, pending)) in enumerate(self.lanes.items()):
            ltop = top + bh + 4 + li * lane
            lh = lane - 4
            p.fillRect(QRectF(8, ltop, w - 16, lh), QColor(28, 38, 46))
            for gs, ge in self.lane_gaps.get(name, []):  # still-unlabelled frames: hatched
                gx0, gx1 = self._x(gs), max(self._x(gs) + 2.0, self._x(ge + 1))
                p.fillRect(QRectF(gx0, ltop, gx1 - gx0, lh), QBrush(QColor(255, 120, 120, 150), Qt.BrushStyle.BDiagPattern))
            rows, n_rows = self._stack(events)
            self._layout[name] = (ltop, lh, rows, n_rows)
            rh = lh / n_rows
            for r, s_, e_, c, eid in rows:
                x0, x1 = self._x(s_), max(self._x(s_) + 2.0, self._x(e_ + 1))
                p.fillRect(QRectF(x0, ltop + r * rh, x1 - x0, rh - 1), QColor(*c))
                if eid is not None and self.highlight.get(name) == eid:  # selected in the table
                    p.setPen(QPen(QColor(255, 255, 255), 2))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(QRectF(x0 - 1, ltop + r * rh - 1, x1 - x0 + 2, rh + 1))
                if name == "loops":  # loops: emphasise start and return points
                    p.setPen(QPen(QColor(255, 255, 255), 2))
                    p.drawLine(QPointF(x0, ltop + r * rh), QPointF(x0, ltop + (r + 1) * rh - 1))
                    p.drawLine(QPointF(x1, ltop + r * rh), QPointF(x1, ltop + (r + 1) * rh - 1))
            if pending and pending[0] is not None:
                a = pending[0]
                b = pending[1] if pending[1] is not None else self.pos
                lo, hi = min(a, b), max(a, b)
                p.fillRect(QRectF(self._x(lo), ltop - 1, max(2.0, self._x(hi + 1) - self._x(lo)), lh + 2),
                           QColor(255, 255, 255, 70))
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawLine(QPointF(self._x(a), ltop - 2), QPointF(self._x(a), ltop + lh + 1))
        # segments: draw runs of equal owner
        if self.owner.size:
            change = np.flatnonzero(np.diff(self.owner)) + 1
            starts = np.r_[0, change]
            ends = np.r_[change, self.n]
            for s, e in zip(starts, ends):
                sid = int(self.owner[s])
                if sid < 0:
                    continue
                c = QColor(*segment_colour(sid))
                p.fillRect(QRectF(self._x(s), top, max(1.5, self._x(e) - self._x(s)), bh), c)
        p.setPen(QPen(QColor(120, 140, 150), 1))
        for f in self.keyframes:
            x = self._x(f)
            p.drawLine(QPointF(x, top + bh), QPointF(x, top + bh + 3))
        p.setPen(QPen(QColor(255, 60, 60), 2))
        for f in self.breaks:
            x = self._x(f)
            p.drawLine(QPointF(x, top - 3), QPointF(x, top + bh + 3))
        for f, sid in self.prompts:
            x = self._x(f)
            p.setBrush(QColor(*segment_colour(sid)))
            p.setPen(QPen(QColor(255, 255, 255), 1.2))
            p.drawPolygon(QPolygonF([QPointF(x, 2), QPointF(x + 5, 7), QPointF(x, 12), QPointF(x - 5, 7)]))
        x = self._x(self.pos)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, h))
        if self._hover is not None:
            p.setPen(QColor(200, 210, 220))
            p.drawText(QRectF(self._x(self._hover) - 40, h - 11, 80, 11), Qt.AlignmentFlag.AlignCenter,
                       str(self._hover))
        p.end()

    def _lane_hit(self, pos):
        """(lane, frame, event id or None) under a mouse position, or None outside the lanes."""
        f = self._f(pos.x())
        for name, (ltop, lh, rows, n_rows) in self._layout.items():
            if ltop - 2 <= pos.y() <= ltop + lh + 2:
                r = int(np.clip((pos.y() - ltop) / (lh / n_rows), 0, n_rows - 1))
                hit = [eid for (rr, s_, e_, _c, eid) in rows if rr == r and s_ <= f <= e_]
                if not hit:  # thin rows: accept any event at this frame
                    hit = [eid for (_rr, s_, e_, _c, eid) in rows if s_ <= f <= e_]
                return name, f, (hit[0] if hit else None)
        return None

    def mousePressEvent(self, e):  # noqa: N802
        self.seek.emit(self._f(e.position().x()))
        hit = self._lane_hit(e.position())
        if hit:
            self.lane_clicked.emit(*hit, False)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        hit = self._lane_hit(e.position())
        if hit:
            self.lane_clicked.emit(*hit, True)

    def mouseMoveEvent(self, e):  # noqa: N802
        self._hover = self._f(e.position().x())
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.seek.emit(self._hover)
        self.update()

    def leaveEvent(self, _):  # noqa: N802
        self._hover = None
        self.update()
