"""Shared core for frame-interval annotations (behaviours, loops, ...).

Every annotation kind is a list of events with an inclusive [start_frame, end_frame] range on
the run's proxy frames, a comment, the annotator and timestamps, plus kind-specific fields
(`EXTRA_COLS`, entered by the annotator) and derived fields (`DERIVED_COLS`, recomputed
on every save, e.g. 3D distances from the fish trajectory). Each kind persists to one CSV
per run, rewritten atomically after every change, so nothing is lost if the GUI closes.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from reeftrack.io.video import FrameMap

log = logging.getLogger(__name__)

FRAME_COLS = ["start_frame", "end_frame", "start_source_frame", "end_source_frame",
              "start_time_s", "end_time_s", "duration_s"]
TAIL_COLS = ["source_video", "run", "comment", "annotator", "created", "modified"]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class Event:
    event_id: int
    start_frame: int
    end_frame: int
    comment: str = ""
    annotator: str = ""
    created: str = field(default_factory=_now)
    modified: str = field(default_factory=_now)
    data: dict = field(default_factory=dict)  # kind-specific fields (behaviour, confidence, ...)

    def __getattr__(self, name):  # e.behaviour, e.confidence, ...
        data = self.__dict__.get("data", {})
        if name in data:
            return data[name]
        raise AttributeError(name)

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


class IntervalLog:
    """Base class. Subclasses set the class attributes and may override the hooks."""

    ID_COL = "event_id"
    EXTRA_COLS: list[str] = []
    DERIVED_COLS: list[str] = []
    MIN_FRAMES = 1  # minimum number of (inclusive) frames an event must span
    FILENAME = "events.csv"

    def __init__(self, path: Path, fmap: FrameMap, source_video: str, run_name: str, n_frames: int | None = None,
                 backup: Path | None = None):
        self.path = Path(path)
        self.fmap = fmap
        self.source_video = source_video
        self.run_name = run_name
        self.n_frames = n_frames if n_frames is not None else fmap.n_proxy
        self.events: list[Event] = []
        # Local safety copy (library runs on network storage): written on every save, before
        # the shared file. If the network drops, nothing is lost; a newer backup is restored
        # (and pushed back to the shared file) the next time the log is opened.
        self.backup = Path(backup) if backup else None
        self.last_save_error: str | None = None
        self.restored_from_backup = False
        src = self._newest_source()
        if src is not None:
            self._load(src)
            if src == self.backup:
                self.restored_from_backup = True
                self.save()

    def _newest_source(self) -> Path | None:
        have = [p for p in (self.path, self.backup) if p is not None and p.exists()]
        if not have:
            return None
        return max(have, key=lambda p: (p.stat().st_mtime, p == self.path))  # tie -> shared file

    @property
    def columns(self) -> list[str]:
        return [self.ID_COL, *self.EXTRA_COLS, *FRAME_COLS, *self.DERIVED_COLS, *TAIL_COLS]

    # ---- hooks -----------------------------------------------------------------
    def validate_extra(self, data: dict) -> dict:
        return data

    def derived(self, e: Event) -> dict:
        return {}

    # ---- validation ------------------------------------------------------------
    def _check_frames(self, start: int, end: int) -> tuple[int, int]:
        start, end = int(start), int(end)
        if start > end:
            start, end = end, start  # marking the end before the start is a natural slip; swap
        if start < 0 or end >= self.n_frames:
            raise ValueError(f"frames [{start}, {end}] outside the clip (0-{self.n_frames - 1})")
        if end - start + 1 < self.MIN_FRAMES:
            raise ValueError(f"an event must span at least {self.MIN_FRAMES} frames")
        return start, end

    # ---- edits -----------------------------------------------------------------
    def add(self, start: int, end: int, comment: str = "", annotator: str = "", **data) -> Event:
        start, end = self._check_frames(start, end)
        data = self.validate_extra({k: data.get(k, "") for k in self.EXTRA_COLS})
        ev = Event(self.next_id(), start, end, str(comment).strip(), annotator, data=data)
        self.events.append(ev)
        self.save()
        return ev

    def update(self, event_id: int, **changes) -> Event:
        ev = self.get(event_id)
        s, e = self._check_frames(changes.get("start_frame", ev.start_frame), changes.get("end_frame", ev.end_frame))
        data = self.validate_extra({k: changes.get(k, ev.data.get(k, "")) for k in self.EXTRA_COLS})
        ev.start_frame, ev.end_frame, ev.data = s, e, data
        if "comment" in changes:
            ev.comment = str(changes["comment"]).strip()
        if "annotator" in changes:
            ev.annotator = changes["annotator"]
        ev.modified = _now()
        self.save()
        return ev

    def delete(self, event_id: int) -> None:
        self.events = [e for e in self.events if e.event_id != event_id]
        self.save()

    # ---- queries ---------------------------------------------------------------
    def get(self, event_id: int) -> Event:
        return next(e for e in self.events if e.event_id == event_id)

    def next_id(self) -> int:
        return max((e.event_id for e in self.events), default=0) + 1

    def active_at(self, frame: int) -> list[Event]:
        return [e for e in self.sorted() if e.contains(frame)]

    def sorted(self) -> list[Event]:
        return sorted(self.events, key=lambda e: (e.start_frame, e.end_frame, e.event_id))

    # ---- persistence -----------------------------------------------------------
    def row(self, e: Event) -> dict:
        fm = self.fmap
        s_src, e_src = fm.to_source(e.start_frame), fm.to_source(e.end_frame)
        t0 = s_src / fm.source_fps
        t1 = (e_src + fm.stride) / fm.source_fps  # end of the last (inclusive) frame
        derived = self.derived(e)
        return {
            self.ID_COL: e.event_id, **{k: e.data.get(k, "") for k in self.EXTRA_COLS},
            "start_frame": e.start_frame, "end_frame": e.end_frame, "start_source_frame": s_src,
            "end_source_frame": e_src, "start_time_s": f"{t0:.3f}", "end_time_s": f"{t1:.3f}",
            "duration_s": f"{t1 - t0:.3f}", **{k: derived.get(k, "") for k in self.DERIVED_COLS},
            "source_video": self.source_video, "run": self.run_name, "comment": e.comment,
            "annotator": e.annotator, "created": e.created, "modified": e.modified,
        }

    def to_csv(self) -> str:
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=self.columns)
        w.writeheader()
        for e in self.sorted():
            w.writerow(self.row(e))
        return buf.getvalue()

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written CSV

    def save(self) -> bool:
        """Write the CSV. Returns False (and sets `last_save_error`) if the shared file could
        not be written; the local backup, if configured, still has every change."""
        text = self.to_csv()
        if self.backup is not None:
            self._atomic_write(self.backup, text)
        try:
            self._atomic_write(self.path, text)
        except OSError as e:
            self.last_save_error = f"{type(e).__name__}: {e}"
            log.warning("could not save %s (%s); backup kept at %s", self.path, e, self.backup)
            return False
        self.last_save_error = None
        if self.backup is not None:  # equal timestamps -> the shared file counts as newest
            st = self.path.stat()
            os.utime(self.backup, (st.st_atime, st.st_mtime))
        return True

    def _load(self, source: Path | None = None) -> None:
        with open(source or self.path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ev = Event(int(r[self.ID_COL]), int(r["start_frame"]), int(r["end_frame"]), r.get("comment", ""),
                           r.get("annotator", ""), r.get("created") or _now(), r.get("modified") or _now(),
                           {k: r.get(k, "") for k in self.EXTRA_COLS})
                self._on_load(ev)
                self.events.append(ev)

    def _on_load(self, ev: Event) -> None:
        pass


def _source_and_name(run) -> tuple[str, str]:
    """(source video to record, run name). Library runs record the path below the originals
    folder (the same on every computer) rather than a machine-specific absolute path."""
    meta = run.meta()
    src = meta["source"]
    return (src.get("rel") or src["path"]), (src.get("rel") or run.root.name)


def backup_path(run, path: Path) -> Path | None:
    """Local safety copy for annotation files on shared storage (library runs only)."""
    meta = run.meta()
    if meta.get("layout") != "library":
        return None
    base = Path(os.environ.get("REEFTRACK_HOME", Path.home() / ".reeftrack")) / "backups"
    rel = Path(meta["source"]["rel"])
    return base.joinpath(*rel.parent.parts, rel.stem, path.name)
