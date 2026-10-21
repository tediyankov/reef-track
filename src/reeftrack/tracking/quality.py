"""Decide, frame by frame, whether the tracker is still on the focal fish.

SAM-style video trackers almost never "give up": when the fish swims behind coral they keep
returning *something* (a fragment, a neighbour, nothing). We combine five cheap signals:

  presence    SAM's own object-presence logit (< 0 = "I think the object is not visible")
  area        mask vanished, or its area jumped/collapsed vs. the recent median
  motion      box centre far outside the Kalman prediction (identity switch)
  appearance  colour signature no longer matches the clicked fish
  empty       no mask at all

A frame with any flag is *suspect*. After `patience` consecutive suspect frames the tracklet
breaks, and the segment is trimmed back to the last good frame, so the stored tracklet never
contains the frames where the tracker was drifting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from reeftrack.tracking.motion import BoxKalman, xyxy_to_cxcywh


@dataclass
class FrameSignals:
    area: int
    bbox: tuple[int, int, int, int] | None
    presence_logit: float
    appearance_sim: float
    frame_area: int


@dataclass
class Verdict:
    ok: bool
    flags: list[str]
    motion_sigma: float = 0.0
    broke: bool = False  # True on the frame where patience ran out


@dataclass
class LossDetector:
    presence_logit_threshold: float = 0.0
    min_area_frac: float = 2e-5
    max_area_change: float = 4.0
    min_appearance_sim: float = 0.55
    motion_gate_sigmas: float = 6.0
    patience: int = 3
    history: int = 15

    _areas: deque = field(default_factory=lambda: deque(maxlen=15), init=False, repr=False)
    _kf: BoxKalman | None = field(default=None, init=False, repr=False)
    _bad_streak: int = field(default=0, init=False)

    @classmethod
    def from_config(cls, tc) -> LossDetector:
        return cls(
            presence_logit_threshold=tc.presence_logit_threshold,
            min_area_frac=tc.min_area_frac,
            max_area_change=tc.max_area_change,
            min_appearance_sim=tc.min_appearance_sim,
            motion_gate_sigmas=tc.motion_gate_sigmas,
            patience=tc.patience,
        )

    def initialise(self, s: FrameSignals) -> None:
        """Called with the user-clicked frame: defines what 'good' looks like."""
        self._areas = deque([s.area], maxlen=self.history)
        self._kf = BoxKalman(xyxy_to_cxcywh(s.bbox)) if s.bbox else None
        self._bad_streak = 0

    @property
    def bad_streak(self) -> int:
        return self._bad_streak

    def assess(self, s: FrameSignals) -> Verdict:
        flags: list[str] = []
        sigma = 0.0
        if self._kf is not None:
            self._kf.predict()

        if s.bbox is None or s.area < self.min_area_frac * s.frame_area:
            flags.append("empty")
        else:
            if s.presence_logit < self.presence_logit_threshold:
                flags.append("presence")
            med = float(np.median(self._areas)) if self._areas else s.area
            ratio = s.area / max(med, 1.0)
            if ratio > self.max_area_change or ratio < 1.0 / self.max_area_change:
                flags.append("area")
            if s.appearance_sim < self.min_appearance_sim:
                flags.append("appearance")
            if self._kf is not None:
                sigma = self._kf.mahalanobis(xyxy_to_cxcywh(s.bbox))
                if sigma > self.motion_gate_sigmas:
                    flags.append("motion")

        if flags:
            self._bad_streak += 1
        else:
            self._bad_streak = 0
            self._areas.append(s.area)
            if self._kf is not None:
                self._kf.update(xyxy_to_cxcywh(s.bbox))
            elif s.bbox is not None:
                self._kf = BoxKalman(xyxy_to_cxcywh(s.bbox))
        return Verdict(ok=not flags, flags=flags, motion_sigma=sigma, broke=self._bad_streak >= self.patience)
