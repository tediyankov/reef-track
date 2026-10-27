"""Behaviour-annotation widgets: the event dialog and the side panel (table of events)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "A", "B", "C", "D", "E", "F", "G", "H"]


class BehaviourDialog(QDialog):
    """Pick a behaviour (required), adjust start/end, write an optional comment."""

    def __init__(self, categories: list[str], start: int, end: int, n_frames: int, fps: float,
                 behaviour: str | None = None, comment: str = "", title: str = "New behaviour event", parent=None,
                 definitions: dict[str, str] | None = None):
        super().__init__(parent)
        self.definitions = definitions or {}
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
            if c in self.definitions:
                it.setToolTip(self.definitions[c])
            self.list.addItem(it)
            if c == behaviour:
                self.list.setCurrentItem(it)
        self.list.setMinimumHeight(24 * min(len(categories), 16) + 8)
        self.definition = QLabel("")  # ethogram definition of the highlighted behaviour
        self.definition.setWordWrap(True)
        self.definition.setStyleSheet("color:#ffd479; padding:4px;")
        self.definition.setMinimumHeight(48)

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
        lay.addWidget(self.definition)
        lay.addLayout(form)
        lay.addWidget(QLabel("Comment"))
        lay.addWidget(self.comment)
        lay.addWidget(self.buttons)
        self.list.setFocus()
        self._update()

    def _update(self, *_) -> None:
        s, e = sorted((self.start.value(), self.end.value()))
        self.duration.setText(f"{e - s + 1} frames · {(e - s + 1) / self.fps:.2f} s")
        it = self.list.currentItem()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(it is not None)
        name = it.data(Qt.ItemDataRole.UserRole) if it else None
        self.definition.setText(self.definitions.get(name, "") if name else "Pick a behaviour to see its definition.")

    def values(self) -> tuple[str, int, int, str]:
        it = self.list.currentItem()
        return (it.data(Qt.ItemDataRole.UserRole) if it else "", self.start.value(), self.end.value(),
                self.comment.toPlainText())


class EventPanel(QWidget):
    """Side panel shared by interval annotations: annotator, start/end marks, events table."""

    seek = Signal(int)
    selected = Signal(object)  # event id of the selected row (or None)

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
        self.table.itemSelectionChanged.connect(lambda: self.selected.emit(self.selected_event_id()))
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

    def select_event(self, event_id) -> None:
        """Highlight (and scroll to) the row of `event_id` (e.g. after clicking its chunk on the timeline)."""
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).text() == str(event_id):
                self.table.selectRow(r)
                self.table.scrollToItem(self.table.item(r, 0))
                return
        self.table.clearSelection()

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

    def __init__(self, parent=None):
        super().__init__(parent)
        # full segmentation: every frame should end up with a behaviour
        self.coverage = QLabel("")
        self.coverage.setWordWrap(True)
        self.btn_prev_gap = QPushButton("◀ Prev gap  [Shift+G]")
        self.btn_next_gap = QPushButton("Next gap ▶  [G]")
        # off by default, so clicking the timeline to navigate never replaces a range being marked
        self.gap_mode = QCheckBox("Gap-fill mode: clicking a hatched gap on the timeline selects it")
        self.gap_mode.setToolTip("Off: clicking the timeline only moves the playhead. G / Shift+G work either way.")
        row = QHBoxLayout()
        row.addWidget(self.btn_prev_gap)
        row.addWidget(self.btn_next_gap)
        lay = self.layout()
        i = lay.indexOf(self.btn_add) + 1
        lay.insertWidget(i, self.coverage)
        lay.insertLayout(i + 1, row)
        lay.insertWidget(i + 2, self.gap_mode)

    def set_coverage(self, summary: dict) -> None:
        if summary["complete"]:
            self.coverage.setText("✓ Every frame has a behaviour.")
            self.coverage.setStyleSheet("color:#7ee08a; font-weight:600;")
        else:
            self.coverage.setText(f"Labelled {100 * summary['fraction']:.1f}% of frames · {summary['n_gaps']} gap"
                                  f"{'s' if summary['n_gaps'] != 1 else ''} (largest {summary['largest_gap_s']:.1f} s, "
                                  "hatched on the timeline). Press G (or tick gap-fill mode and click a gap), then B.")
            self.coverage.setStyleSheet("color:#ff9d9d;")

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
