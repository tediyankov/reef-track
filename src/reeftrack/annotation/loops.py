"""Loop annotations: the focal fish returns to a place it has already been.

The annotator marks the frame where the loop starts and the frame where they recognise the
fish is back at that spot. One row per loop in `<run>/annotations/loops.csv`.

Columns
-------
Entered by the annotator:
  loop_id, start/end frame, confidence (certain | probable | possible), landmark
  (free text, e.g. "big bommie"), comment, annotator.
Derived from the timing:
  source frames, start/end time and duration (s) in the original video.
Derived from the 3D fish trajectory, if `reeftrack fuse` has been run (stereo preferred,
because it is in metres):
  start_x/y/z, end_x/y/z    fish position at the loop's start and end
  return_distance           |end - start|: how closely the fish really came back
  path_length               distance swum during the loop
  max_excursion             farthest the fish got from the start point during the loop
  units, trajectory_source  "m" for stereo, "recon units" for mono; which file was used
These are recomputed on every save and on GUI start, so they follow the latest trajectory.
Empty if the trajectory has no position near the loop's start/end frame (e.g. fish not tracked).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from reeftrack.annotation.events import Event, IntervalLog, _source_and_name, backup_path
from reeftrack.io.video import FrameMap

CONFIDENCE = ["certain", "probable", "possible"]


class Trajectory:
    """Fish 3D positions by proxy frame, read from a fusion*/trajectory.csv."""

    def __init__(self, frames: np.ndarray, xyz: np.ndarray, units: str, source: str, max_gap: int = 15):
        order = np.argsort(frames)
        self.frames, self.xyz = frames[order], xyz[order]
        self.units, self.source, self.max_gap = units, source, max_gap

    @classmethod
    def for_run(cls, run) -> Trajectory | None:
        """Prefer the metric stereo trajectory, fall back to mono."""
        for d, units in (("fusion_stereo", "m"), ("fusion", "recon units")):
            p = run.root / d / "trajectory.csv"
            if p.exists():
                with open(p) as fh:
                    rows = list(csv.DictReader(fh))
                if rows:
                    f = np.array([int(r["proxy_frame"]) for r in rows])
                    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
                    return cls(f, xyz, units, f"{d}/trajectory.csv")
        return None

    def at(self, frame: int) -> np.ndarray | None:
        """Position at `frame`, or at the nearest tracked frame within `max_gap` frames."""
        if not len(self.frames):
            return None
        i = int(np.clip(np.searchsorted(self.frames, frame), 0, len(self.frames) - 1))
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.frames)]
        j = min(cands, key=lambda k: abs(self.frames[k] - frame))
        return self.xyz[j] if abs(self.frames[j] - frame) <= self.max_gap else None

    def segment(self, start: int, end: int) -> np.ndarray:
        m = (self.frames >= start) & (self.frames <= end)
        return self.xyz[m]


def loop_metrics(traj: Trajectory | None, start: int, end: int) -> dict:
    if traj is None:
        return {}
    p0, p1 = traj.at(start), traj.at(end)
    if p0 is None or p1 is None:
        return {"units": traj.units, "trajectory_source": traj.source}
    seg = traj.segment(start, end)
    seg = np.vstack([p0, seg, p1]) if len(seg) else np.vstack([p0, p1])
    f = lambda v: f"{v:.4f}"  # noqa: E731
    return {
        "start_x": f(p0[0]), "start_y": f(p0[1]), "start_z": f(p0[2]),
        "end_x": f(p1[0]), "end_y": f(p1[1]), "end_z": f(p1[2]),
        "return_distance": f(np.linalg.norm(p1 - p0)),
        "path_length": f(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()),
        "max_excursion": f(np.linalg.norm(seg - p0, axis=1).max()),
        "units": traj.units, "trajectory_source": traj.source,
    }


class LoopLog(IntervalLog):
    """All loops of one run, persisted to `annotations/loops.csv`."""

    ID_COL = "loop_id"
    EXTRA_COLS = ["confidence", "landmark"]
    DERIVED_COLS = ["return_distance", "path_length", "max_excursion", "start_x", "start_y", "start_z",
                    "end_x", "end_y", "end_z", "units", "trajectory_source"]
    MIN_FRAMES = 2  # a loop needs a distinct start and end
    FILENAME = "loops.csv"

    def __init__(self, path: Path, fmap: FrameMap, source_video: str, run_name: str,
                 trajectory: Trajectory | None = None, n_frames: int | None = None, backup: Path | None = None):
        self.trajectory = trajectory
        super().__init__(path, fmap, source_video, run_name, n_frames, backup)

    @classmethod
    def for_run(cls, run, annotator: str | None = None) -> LoopLog:
        path = run.annotation_path("loops", annotator)
        lg = cls(path, run.frame_map(), *_source_and_name(run), Trajectory.for_run(run), backup=backup_path(run, path))
        if lg.path.exists():
            lg.save()  # refresh the 3D columns against the latest trajectory
        return lg

    def validate_extra(self, data: dict) -> dict:
        c = (data.get("confidence") or "certain").strip().lower()
        if c not in CONFIDENCE:
            raise ValueError(f"confidence must be one of {CONFIDENCE}")
        return {"confidence": c, "landmark": str(data.get("landmark", "")).strip()}

    def derived(self, e: Event) -> dict:
        return loop_metrics(self.trajectory, e.start_frame, e.end_frame)
