"""Frame-level behaviour annotations: one row per (start frame, end frame, behaviour) event.

Saved as `<run>/annotations/behaviours.csv`, separate from the fish masks, so mask tracking
and behaviour annotation can be done by different people, at different times, on the same
run. The CSV is the source of truth and is rewritten (atomically) after every change.

Frame conventions
-----------------
* `start_frame` / `end_frame` are *proxy* frames (the frame numbers shown in the GUI) and
  are **inclusive**: an event on a single frame has start == end.
* `start_source_frame` / `end_source_frame` and the times refer to the original video, so
  rows can be joined with the raw footage, the fish trajectory or other tools.
* Events may overlap: behaviours can co-occur (e.g. Foraging while Near Other Picasso
  triggerfish).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from reeftrack.annotation.events import (
    FRAME_COLS,
    TAIL_COLS,
    Event,
    IntervalLog,
    _source_and_name,
    backup_path,
)
from reeftrack.io.video import FrameMap

log = logging.getLogger(__name__)

COLUMNS = ["event_id", "behaviour", *FRAME_COLS, *TAIL_COLS]
BehaviourEvent = Event  # backwards-compatible name


class BehaviourLog(IntervalLog):
    """All behaviour events of one run, persisted to `annotations/behaviours.csv`."""

    ID_COL = "event_id"
    EXTRA_COLS = ["behaviour"]
    FILENAME = "behaviours.csv"

    def __init__(self, path: Path, fmap: FrameMap, source_video: str, run_name: str,
                 categories: list[str], n_frames: int | None = None, backup: Path | None = None):
        self.categories = list(categories)
        super().__init__(path, fmap, source_video, run_name, n_frames, backup)

    @classmethod
    def for_run(cls, run, categories: list[str], annotator: str | None = None) -> BehaviourLog:
        path = run.annotation_path("behaviours", annotator)
        return cls(path, run.frame_map(), *_source_and_name(run), categories, backup=backup_path(run, path))

    def validate_extra(self, data: dict) -> dict:
        b = data.get("behaviour", "")
        if not b:
            raise ValueError("every event needs a behaviour")
        if b not in self.categories:
            raise ValueError(f"unknown behaviour {b!r}; allowed: {self.categories}")
        return data

    def add(self, behaviour: str, start: int, end: int, comment: str = "", annotator: str = "") -> Event:
        return super().add(start, end, comment, annotator, behaviour=behaviour)

    # ---- full segmentation (every frame has at least one behaviour) -----------------------
    def coverage(self) -> np.ndarray:
        """Boolean per frame: True if at least one behaviour event covers it."""
        cov = np.zeros(self.n_frames, dtype=bool)
        for e in self.events:
            cov[max(0, e.start_frame): min(self.n_frames, e.end_frame + 1)] = True
        return cov

    def gaps(self) -> list[tuple[int, int]]:
        """Inclusive [start, end] frame ranges that no behaviour covers yet, in time order."""
        cov = self.coverage()
        edges = np.flatnonzero(np.diff(np.r_[1, cov.astype(np.int8), 1]))  # starts/ends of False runs
        return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]

    def gap_at(self, frame: int) -> tuple[int, int] | None:
        return next(((a, b) for a, b in self.gaps() if a <= frame <= b), None)

    def coverage_summary(self) -> dict:
        gaps = self.gaps()
        lengths = [b - a + 1 for a, b in gaps]
        return {"fraction": 1 - sum(lengths) / max(1, self.n_frames), "n_gaps": len(gaps),
                "largest_gap_s": max(lengths, default=0) / self.fmap.proxy_fps, "complete": not gaps}

    def _on_load(self, ev: Event) -> None:
        if ev.behaviour not in self.categories:
            log.warning("event %d has behaviour %r not in the current category list; kept as is",
                        ev.event_id, ev.behaviour)
            self.categories.append(ev.behaviour)
