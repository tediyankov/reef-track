"""GUI pieces for the shared video library: folder settings and the 'no compressed copy' choice."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from reeftrack.library import LibrarySettings


class LibraryDialog(QDialog):
    """Pick the three library folders (and the local cache) once per computer."""

    FIELDS = [
        ("video_root", "Original videos (hard drive)",
         "The 'originaldata' folder on the hard drive, e.g. /Volumes/Seagate Backup Plus Drive/originaldata"),
        ("compressed_root", "Compressed videos (RFS)",
         "Shared folder for compressed copies, e.g. RFS/Triggerfish Navigation/fish_in_the_wild/compressed_videos"),
        ("annotations_root", "Annotations (RFS)",
         "Shared folder for annotation CSVs, e.g. RFS/Triggerfish Navigation/fish_in_the_wild/annotations"),
        ("cache_dir", "Local video cache (this computer)",
         "Where compressed videos are copied for smooth playback (old ones are deleted automatically)"),
    ]

    def __init__(self, settings: LibrarySettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Library folders")
        self.setMinimumWidth(720)
        self.s = settings
        self.edits: dict[str, QLineEdit] = {}
        form = QFormLayout()
        for key, label, tip in self.FIELDS:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            e = QLineEdit(getattr(settings, key))
            e.setToolTip(tip)
            e.setPlaceholderText(tip)
            b = QPushButton("Browse…")
            b.clicked.connect(lambda _=False, e=e, label=label: self._browse(e, label))
            h.addWidget(e, 1)
            h.addWidget(b)
            self.edits[key] = e
            form.addRow(label, row)
        self.cache_gb = QDoubleSpinBox()
        self.cache_gb.setRange(2, 2000)
        self.cache_gb.setSuffix(" GB")
        self.cache_gb.setValue(settings.cache_limit_gb)
        form.addRow("Local cache limit", self.cache_gb)
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet("color:#ff8080")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Set these once on this computer. Connect the hard drive and RFS first."))
        lay.addLayout(form)
        lay.addWidget(self.problems)
        lay.addWidget(buttons)

    def _browse(self, edit: QLineEdit, label: str) -> None:
        d = QFileDialog.getExistingDirectory(self, label, edit.text() or str(Path.home()))
        if d:
            edit.setText(d)

    def values(self) -> LibrarySettings:
        s = LibrarySettings(**{**self.s.__dict__})
        for key, e in self.edits.items():
            setattr(s, key, e.text().strip())
        s.cache_limit_gb = self.cache_gb.value()
        return s

    def _accept(self) -> None:
        s = self.values()
        probs = s.problems()
        if probs:
            self.problems.setText("Please fix:\n• " + "\n• ".join(probs))
            return
        s.save()
        self.s = s
        self.accept()


def ask_no_compressed_copy(parent, rel: str, minutes: float) -> str:
    """Returns 'compress', 'original' or 'cancel'."""
    box = QMessageBox(parent)
    box.setWindowTitle("No compressed copy yet")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText(f"<b>{rel}</b> hasn't been compressed yet.")
    box.setInformativeText(
        f"Compress it now (about {minutes:.0f} min, done once, then shared on RFS for everyone and "
        "smooth to annotate), or open the original video straight from the hard drive (works, but "
        "jumping around is slower).")
    b_comp = box.addButton("Compress now", QMessageBox.ButtonRole.AcceptRole)
    b_orig = box.addButton("Open original", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(b_comp)
    box.exec()
    return {b_comp: "compress", b_orig: "original"}.get(box.clickedButton(), "cancel")
