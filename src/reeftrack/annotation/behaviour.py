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

from reeftrack.annotation.events import FRAME_COLS, TAIL_COLS, Event, IntervalLog
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
                 categories: list[str], n_frames: int | None = None):
        self.categories = list(categories)
        super().__init__(path, fmap, source_video, run_name, n_frames)

    @classmethod
    def for_run(cls, run, categories: list[str]) -> BehaviourLog:
        return cls(run.root / "annotations" / cls.FILENAME, run.frame_map(), run.meta()["source"]["path"],
                   run.root.name, categories)

    def validate_extra(self, data: dict) -> dict:
        b = data.get("behaviour", "")
        if not b:
            raise ValueError("every event needs a behaviour")
        if b not in self.categories:
            raise ValueError(f"unknown behaviour {b!r}; allowed: {self.categories}")
        return data

    def add(self, behaviour: str, start: int, end: int, comment: str = "", annotator: str = "") -> Event:
        return super().add(start, end, comment, annotator, behaviour=behaviour)

    def _on_load(self, ev: Event) -> None:
        if ev.behaviour not in self.categories:
            log.warning("event %d has behaviour %r not in the current category list; kept as is",
                        ev.event_id, ev.behaviour)
            self.categories.append(ev.behaviour)
