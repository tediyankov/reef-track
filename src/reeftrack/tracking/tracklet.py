"""The focal-fish tracklet: one identity, made of one or more *segments* separated by gaps.

    frames  ------[ seg 0 ]-----gap-----[ seg 1 ]-------[ seg 2 ]----
                     ^click                ^click           ^click
                  <---- backfill   forward ---->

Each user click creates a segment that grows forward (and, for re-acquisitions, backward)
from the clicked frame until the fish is lost, the video ends, or it runs into frames that
another segment already owns. A frame is owned by at most one segment, so patches never
double-count frames; undoing a click removes exactly that segment's frames.

Persisted as tracking/tracklet.json (human-readable) + tracking/masks.npz (RLE masks), and
exported as tracking/tracklet.csv with *source* frame numbers and timestamps.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from reeftrack.io.video import FrameMap
from reeftrack.tracking.masks import rle_decode, rle_encode


@dataclass
class Observation:
    frame: int
    segment_id: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    presence_logit: float
    appearance_sim: float


@dataclass
class Segment:
    segment_id: int
    prompt_frame: int
    prompt_xy: tuple[float, float]
    start: int
    end: int
    start_reason: str = "prompt"  # prompt | lost | met_segment | video_start
    end_reason: str = "open"  # lost | met_segment | video_end | open

    def __len__(self) -> int:
        return self.end - self.start + 1


@dataclass
class Tracklet:
    shape: tuple[int, int]  # (H, W) of the masks (= proxy resolution)
    segments: list[Segment] = field(default_factory=list)
    obs: dict[int, Observation] = field(default_factory=dict)
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> RLE runs
    meta: dict = field(default_factory=dict)

    # ---- queries -------------------------------------------------------------
    def owner(self, frame: int) -> int | None:
        o = self.obs.get(frame)
        return o.segment_id if o else None

    def mask(self, frame: int) -> np.ndarray | None:
        runs = self.masks.get(frame)
        return None if runs is None else rle_decode(runs, self.shape)

    def frames(self) -> list[int]:
        return sorted(self.obs)

    def segment(self, segment_id: int) -> Segment:
        return next(s for s in self.segments if s.segment_id == segment_id)

    def next_segment_id(self) -> int:
        return max((s.segment_id for s in self.segments), default=-1) + 1

    def gaps(self, n_frames: int) -> list[tuple[int, int]]:
        """Inclusive [start, end] frame ranges with no tracked fish."""
        covered = np.zeros(n_frames, dtype=bool)
        covered[[f for f in self.obs if 0 <= f < n_frames]] = True
        out, start = [], None
        for i, c in enumerate(covered):
            if not c and start is None:
                start = i
            elif c and start is not None:
                out.append((start, i - 1))
                start = None
        if start is not None:
            out.append((start, n_frames - 1))
        return out

    # ---- edits ---------------------------------------------------------------
    def add_observation(self, ob: Observation, mask: np.ndarray) -> None:
        owner = self.owner(ob.frame)
        if owner is not None and owner != ob.segment_id:
            raise ValueError(f"frame {ob.frame} already owned by segment {owner}")
        self.obs[ob.frame] = ob
        self.masks[ob.frame] = rle_encode(mask)

    def drop_frames(self, frames) -> None:
        for f in frames:
            self.obs.pop(f, None)
            self.masks.pop(f, None)

    def refresh_extent(self, segment_id: int) -> None:
        """Recompute a segment's start/end from the frames it actually owns."""
        seg = self.segment(segment_id)
        owned = [f for f, o in self.obs.items() if o.segment_id == segment_id]
        if owned:
            seg.start, seg.end = min(owned), max(owned)

    def remove_segment(self, segment_id: int) -> None:
        self.drop_frames([f for f, o in list(self.obs.items()) if o.segment_id == segment_id])
        self.segments = [s for s in self.segments if s.segment_id != segment_id]

    # ---- persistence ---------------------------------------------------------
    def save(self, folder: Path, fmap: FrameMap | None = None) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        doc = {
            "shape": list(self.shape),
            "meta": self.meta,
            "prompts": [
                {"frame": s.prompt_frame, "xy": list(s.prompt_xy), "segment_id": s.segment_id}
                for s in sorted(self.segments, key=lambda s: s.segment_id)
            ],
            "segments": [asdict(s) for s in self.segments],
            "observations": [asdict(self.obs[f]) for f in self.frames()],
        }
        tmp = folder / "tracklet.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(folder / "tracklet.json")
        np.savez_compressed(folder / "masks.npz", **{f"m{f}": r for f, r in self.masks.items()})
        if fmap is not None:
            self.export_csv(folder / "tracklet.csv", fmap)

    @classmethod
    def load(cls, folder: Path) -> Tracklet:
        doc = json.loads((folder / "tracklet.json").read_text())
        tr = cls(shape=tuple(doc["shape"]), meta=doc.get("meta", {}))
        tr.segments = [Segment(**{**s, "prompt_xy": tuple(s["prompt_xy"])}) for s in doc["segments"]]
        for o in doc["observations"]:
            tr.obs[o["frame"]] = Observation(**{**o, "bbox": tuple(o["bbox"]), "centroid": tuple(o["centroid"])})
        if (folder / "masks.npz").exists():
            with np.load(folder / "masks.npz") as z:
                tr.masks = {int(k[1:]): z[k] for k in z.files}
        return tr

    def export_csv(self, path: Path, fmap: FrameMap) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["proxy_frame", "source_frame", "time_s", "segment_id", "cx", "cy",
                        "x0", "y0", "x1", "y1", "area_px", "presence_logit", "appearance_sim"])
            for f in self.frames():
                o = self.obs[f]
                w.writerow([f, fmap.to_source(f), f"{fmap.time_s(f):.4f}", o.segment_id,
                            f"{o.centroid[0]:.1f}", f"{o.centroid[1]:.1f}", *o.bbox, o.area,
                            f"{o.presence_logit:.3f}", f"{o.appearance_sim:.3f}"])
