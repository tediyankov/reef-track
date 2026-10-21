"""Behaviour-annotation widgets: the event dialog and the side panel (table of events)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from reeftrack.gui.overlay import behaviour_colour

# keys shown next to categories; QListWidget type-ahead selects the item starting with that key
KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="]


class BehaviourDialog(QDialog):
    """Pick a behaviour (required), adjust start/end, write an optional comment."""

    def __init__(self, categories: list[str], start: int, end: int, n_frames: int, fps: float,
                 behaviour: str | None = None, comment: str = "", title: str = "New behaviour event", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        self.categories = categories
        self.fps = fps

        self.list = QListWidget()
        for i, c in enumerate(categories):
            key = KEYS[i] if i < len(KEYS) else " "
            it = QListWidgetItem(f"{key}   {c}")
            it.setData(Qt.ItemDataRole.UserRole, c)
            it.setForeground(QColor(*behaviour_colour(c, categories)))
            self.list.addItem(it)
            if c == behaviour:
                self.list.setCurrentItem(it)
        self.list.setMinimumHeight(24 * min(len(categories), 12) + 8)

        self.start = QSpinBox()
        self.end = QSpinBox()
        for sb, v in ((self.start, start), (self.end, end)):
            sb.setRange(0, n_frames - 1)
            sb.setValue(v)
            sb.valueChanged.connect(self._update)
        self.duration = QLabel()
        self.comment = QPlainTextEdit(comment)
        self.comment.setPlaceholderText("Optional comment (context, uncertainty, other fish involved…)")
        self.comment.setFixedHeight(70)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.list.currentItemChanged.connect(self._update)
        self.list.itemDoubleClicked.connect(lambda _: self.accept())

        form = QFormLayout()
        form.addRow("Start frame", self.start)
        form.addRow("End frame", self.end)
        form.addRow("", self.duration)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("<b>Behaviour</b> (click, or press its key):"))
        lay.addWidget(self.list)
        lay.addLayout(form)
        lay.addWidget(QLabel("Comment"))
        lay.addWidget(self.comment)
        lay.addWidget(self.buttons)
        self.list.setFocus()
        self._update()

    def _update(self, *_) -> None:
        s, e = sorted((self.start.value(), self.end.value()))
        self.duration.setText(f"{e - s + 1} frames · {(e - s + 1) / self.fps:.2f} s")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self.list.currentItem() is not None)

    def values(self) -> tuple[str, int, int, str]:
        it = self.list.currentItem()
        return (it.data(Qt.ItemDataRole.UserRole) if it else "", self.start.value(), self.end.value(),
                self.comment.toPlainText())


class EventPanel(QWidget):
    """Side panel shared by interval annotations: annotator, start/end marks, events table."""

    seek = Signal(int)

    COLS: list[str] = []
    STRETCH_COL = -1  # column that takes the remaining width (comment)
    ELIDE_COLS: dict[int, int] = {}  # column -> fixed width (long text elided, full text in tooltip)
    ADD_LABEL = "Add…  [B]"
    NOUN = "events"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotator = QLabel("Annotator: –")
        self.btn_annotator = QPushButton("Change")
        self.btn_in = QPushButton("Start  [I]")
        self.btn_out = QPushButton("End  [O]")
        self.btn_clear = QPushButton("Clear")
        self.range_label = QLabel("No range marked")
        self.range_label.setStyleSheet("color:#9fb3bd")
        self.btn_add = QPushButton(self.ADD_LABEL)
        self.btn_add.setObjectName("primary")
        self.active = QLabel("")
        self.active.setWordWrap(True)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hh = self.table.horizontalHeader()
        hh.setMinimumSectionSize(28)
        for c in range(len(self.COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for c, wdt in self.ELIDE_COLS.items():
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(c, wdt)
        hh.setSectionResizeMode(self.STRETCH_COL % len(self.COLS), QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(lambda r, _c: self.seek.emit(self._start_of_row(r)))
        self.btn_edit = QPushButton("Edit…")
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setObjectName("danger")

        lay = QVBoxLayout(self)
        r0 = QHBoxLayout()
        r0.addWidget(self.annotator, 1)
        r0.addWidget(self.btn_annotator)
        lay.addLayout(r0)
        r1 = QHBoxLayout()
        r1.addWidget(self.btn_in)
        r1.addWidget(self.btn_out)
        r1.addWidget(self.btn_clear)
        lay.addLayout(r1)
        lay.addWidget(self.range_label)
        lay.addWidget(self.btn_add)
        lay.addWidget(self.active)
        lay.addWidget(self.table, 1)
        r2 = QHBoxLayout()
        r2.addWidget(self.btn_edit)
        r2.addWidget(self.btn_delete)
        lay.addLayout(r2)
        self.count = QLabel("")
        self.count.setStyleSheet("color:#9fb3bd")
        lay.addWidget(self.count)

    START_COL = 2

    def _start_of_row(self, row: int) -> int:
        return int(self.table.item(row, self.START_COL).text())

    def selected_event_id(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        return int(self.table.item(rows[0].row(), 0).text()) if rows else None

    def _fill(self, table_rows: list[list], colours: list, path_text: str) -> None:
        """table_rows: cell values per row; colours: per row (col index, rgb) or None."""
        self.table.setRowCount(len(table_rows))
        for i, vals in enumerate(table_rows):
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v).replace("\n", " ⏎ "))
                it.setToolTip(str(v))
                if colours[i] and colours[i][0] == j:
                    it.setForeground(QColor(*colours[i][1]))
                self.table.setItem(i, j, it)
        self.count.setText(f"{len(table_rows)} {self.NOUN} → {path_text}")


class BehaviourPanel(EventPanel):
    COLS = ["#", "Behaviour", "Start", "End", "Dur (s)", "Comment"]
    ELIDE_COLS = {1: 120}
    ADD_LABEL = "Add behaviour…  [B]"
    NOUN = "events"

    def set_events(self, rows: list[dict], categories: list[str], path_text: str) -> None:
        self._fill([[r["event_id"], r["behaviour"], r["start_frame"], r["end_frame"],
                     f"{float(r['duration_s']):.1f}", r["comment"]] for r in rows],
                   [(1, behaviour_colour(r["behaviour"], categories)) for r in rows], path_text)


LOOP_COLOUR = (0, 229, 255)
CONF_COLOURS = {"certain": (0, 229, 255), "probable": (120, 200, 255), "possible": (150, 160, 200)}


class LoopPanel(EventPanel):
    COLS = ["#", "Conf.", "Start", "End", "Dur (s)", "Return", "Notes"]  # notes = landmark · comment
    ADD_LABEL = "Add loop…  [B]"
    NOUN = "loops"

    def set_loops(self, rows: list[dict], path_text: str) -> None:
        def ret(r):
            if r.get("return_distance") in ("", None):
                return "–"
            u = "m" if r.get("units") == "m" else "u"
            return f"{float(r['return_distance']):.2f} {u}"

        self._fill([[r["loop_id"], r["confidence"], r["start_frame"], r["end_frame"], f"{float(r['duration_s']):.1f}",
                     ret(r), " · ".join(x for x in (r["landmark"], r["comment"]) if x)] for r in rows],
                   [(1, CONF_COLOURS.get(r["confidence"], LOOP_COLOUR)) for r in rows], path_text)


class LoopDialog(QDialog):
    """Confirm a loop: start and end frames side by side, confidence, landmark, comment."""

    def __init__(self, start: int, end: int, n_frames: int, fps: float, thumbs=None, confidence: str = "certain",
                 landmark: str = "", comment: str = "", title: str = "New loop", parent=None):
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QComboBox, QLineEdit

        from reeftrack.annotation.loops import CONFIDENCE
        from reeftrack.gui.widgets import to_qimage

        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.fps = fps
        lay = QVBoxLayout(self)
        if thumbs is not None:  # visual check: is the fish really back at the start spot?
            row = QHBoxLayout()
            for label, img in zip(("Loop start", "Return (end)"), thumbs):
                col = QVBoxLayout()
                pic = QLabel()
                pic.setPixmap(QPixmap.fromImage(to_qimage(img)))
                col.addWidget(QLabel(f"<b>{label}</b>"))
                col.addWidget(pic)
                row.addLayout(col)
            lay.addLayout(row)
        self.start = QSpinBox()
        self.end = QSpinBox()
        for sb, v in ((self.start, start), (self.end, end)):
            sb.setRange(0, n_frames - 1)
            sb.setValue(v)
            sb.valueChanged.connect(self._update)
        self.duration = QLabel()
        self.confidence = QComboBox()
        self.confidence.addItems(CONFIDENCE)
        self.confidence.setCurrentText(confidence or "certain")
        self.landmark = QLineEdit(landmark)
        self.landmark.setPlaceholderText("e.g. big bommie, sand patch, coral head with sea fan")
        self.comment = QPlainTextEdit(comment)
        self.comment.setPlaceholderText("Optional comment")
        self.comment.setFixedHeight(60)
        form = QFormLayout()
        form.addRow("Start frame", self.start)
        form.addRow("End (return) frame", self.end)
        form.addRow("", self.duration)
        form.addRow("Confidence", self.confidence)
        form.addRow("Landmark", self.landmark)
        form.addRow("Comment", self.comment)
        lay.addLayout(form)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._update()

    def _update(self, *_) -> None:
        s, e = sorted((self.start.value(), self.end.value()))
        self.duration.setText(f"{e - s + 1} frames · {(e - s + 1) / self.fps:.2f} s")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(e > s)

    def values(self) -> dict:
        return {"start": self.start.value(), "end": self.end.value(), "confidence": self.confidence.currentText(),
                "landmark": self.landmark.text(), "comment": self.comment.toPlainText()}
