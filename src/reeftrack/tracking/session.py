"""Interactive tracking session: clicks in, tracklet out.

One click = one segment:
  1. SAM segments the clicked fish on that frame (the *prompt*).
  2. Forward propagation until the fish is lost / the video ends / we hit another segment.
  3. Backward propagation (``backfill``) the same way. For a re-acquisition this patches the
     gap from the other side: you click when you notice the fish again, the tracker walks
     back to when it actually re-emerged from behind the coral, stopping as soon as it meets
     the previous segment.

Clicking a frame that is already tracked is treated as a *correction*: the owning segment
is cut at that frame (the part after the click, or before it if the click is earlier than
that segment's prompt) and the new click re-tracks from there.

Each propagation direction uses a fresh SAM session seeded only by the click, so a mistake
in one direction never contaminates the other.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from reeftrack.tracking.appearance import AppearanceModel, colour_signature
from reeftrack.tracking.backends import TrackerBackend, load_backend
from reeftrack.tracking.masks import area, bbox, centroid, largest_component
from reeftrack.tracking.quality import FrameSignals, LossDetector, Verdict
from reeftrack.tracking.tracklet import Observation, Segment, Tracklet

log = logging.getLogger(__name__)

# callback(frame_idx, mask or None, verdict) -> called for every processed frame (GUI live view)
FrameCallback = Callable[[int, "np.ndarray | None", Verdict], None]


@dataclass
class _Frames:
    """Anything indexable returning RGB uint8 frames (FrameReader, list of arrays...)."""

    src: object

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.src[i]


class TrackingSession:
    def __init__(
        self,
        frames,
        tracking_cfg,
        tracklet: Tracklet | None = None,
        backend: TrackerBackend | None = None,
        backend_factory: Callable[[], TrackerBackend] | None = None,
        run=None,
    ):
        self.frames = _Frames(frames)
        self.tc = tracking_cfg
        first = self.frames[0]
        self.shape = first.shape[:2]
        self.tracklet = tracklet or Tracklet(shape=self.shape)
        self._backend = backend
        self._factory = backend_factory
        self.run = run

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_run(cls, run, cfg, fresh: bool = False) -> TrackingSession:
        from reeftrack.devices import resolve_device
        from reeftrack.io.video import FrameReader

        reader = FrameReader(run.proxy)
        tracklet = None
        if not fresh and (run.tracking_dir / "tracklet.json").exists():
            tracklet = Tracklet.load(run.tracking_dir)
        device = resolve_device(cfg.tracking.device)
        sess = cls(reader, cfg.tracking, tracklet, backend_factory=lambda: load_backend(cfg.tracking, device), run=run)
        sess.tracklet.meta.update({"config_digest": cfg.digest(), "device": device})
        return sess

    @property
    def backend(self) -> TrackerBackend:
        if self._backend is None:
            if self._factory is None:
                raise RuntimeError("no backend configured")
            self._backend = self._factory()
            self.tracklet.meta["backend"] = self._backend.name
        return self._backend

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    # ---- user actions --------------------------------------------------------
    def preview(self, frame_idx: int, xy: tuple[float, float]) -> np.ndarray:
        """Mask for a click on one frame (no tracking) - lets the user check before committing."""
        res = self.backend.preview(self.frames[frame_idx], [xy])
        return largest_component(res.mask)

    def add_click(
        self,
        frame_idx: int,
        xy: tuple[float, float],
        callback: FrameCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
        progress: bool = False,
    ) -> Segment:
        self._make_room(frame_idx)
        seg = Segment(self.tracklet.next_segment_id(), frame_idx, (float(xy[0]), float(xy[1])), frame_idx, frame_idx)
        self.tracklet.segments.append(seg)
        try:
            seg.end_reason = self._propagate(seg, +1, callback, should_stop, progress)
            if self.tc.backfill:
                seg.start_reason = self._propagate(seg, -1, callback, should_stop, progress)
        except Exception:
            self.tracklet.remove_segment(seg.segment_id)
            raise
        self.tracklet.refresh_extent(seg.segment_id)
        log.info("segment %d: %d-%d (start: %s, end: %s)", seg.segment_id, seg.start, seg.end,
                 seg.start_reason, seg.end_reason)
        return seg

    def undo(self) -> Segment | None:
        """Remove the most recently created segment."""
        if not self.tracklet.segments:
            return None
        seg = max(self.tracklet.segments, key=lambda s: s.segment_id)
        self.tracklet.remove_segment(seg.segment_id)
        return seg

    def save(self) -> None:
        if self.run is None:
            raise RuntimeError("session has no run to save into")
        self.tracklet.save(self.run.tracking_dir, self.run.frame_map())

    # ---- internals -----------------------------------------------------------
    def _make_room(self, frame_idx: int) -> None:
        """Correction semantics: cut whichever segment owns the clicked frame."""
        owner = self.tracklet.owner(frame_idx)
        if owner is None:
            return
        seg = self.tracklet.segment(owner)
        if frame_idx == seg.prompt_frame:
            self.tracklet.remove_segment(owner)
            return
        if frame_idx > seg.prompt_frame:
            drop = [f for f, o in self.tracklet.obs.items() if o.segment_id == owner and f >= frame_idx]
            seg.end_reason = "corrected"
        else:
            drop = [f for f, o in self.tracklet.obs.items() if o.segment_id == owner and f <= frame_idx]
            seg.start_reason = "corrected"
        self.tracklet.drop_frames(drop)
        self.tracklet.refresh_extent(owner)

    def _signals(self, mask: np.ndarray, presence: float, sim: float) -> FrameSignals:
        return FrameSignals(area=area(mask), bbox=bbox(mask), presence_logit=presence,
                            appearance_sim=sim, frame_area=self.shape[0] * self.shape[1])

    def _record(self, seg: Segment, f: int, mask: np.ndarray, s: FrameSignals) -> None:
        self.tracklet.add_observation(
            Observation(frame=f, segment_id=seg.segment_id, area=s.area, bbox=s.bbox,
                        centroid=centroid(mask), presence_logit=s.presence_logit,
                        appearance_sim=s.appearance_sim),
            mask,
        )

    def _propagate(self, seg, direction, callback, should_stop, progress) -> str:
        start = seg.prompt_frame
        frame = self.frames[start]
        res = self.backend.start(frame, [seg.prompt_xy])
        mask = largest_component(res.mask)
        if not mask.any():
            raise ValueError(f"no object segmented at click {seg.prompt_xy} on frame {start}")
        sig0 = colour_signature(frame, mask)
        appearance = AppearanceModel(sig0)
        detector = LossDetector.from_config(self.tc)
        s0 = self._signals(mask, res.presence_logit, 1.0)
        detector.initialise(s0)
        if direction > 0:
            self._record(seg, start, mask, s0)
            if callback:
                callback(start, mask, Verdict(ok=True, flags=[]))

        bar = None
        if progress:
            from tqdm import tqdm

            total = (self.n_frames - 1 - start) if direction > 0 else start
            bar = tqdm(total=total, desc="forward" if direction > 0 else "backward", unit="fr")

        f = start + direction
        reason = "video_end" if direction > 0 else "video_start"
        while 0 <= f < self.n_frames:
            if should_stop and should_stop():
                reason = "user"
                break
            owner = self.tracklet.owner(f)
            if owner is not None and owner != seg.segment_id:
                reason = "met_segment"
                break
            frame = self.frames[f]
            res = self.backend.step(frame)
            mask = largest_component(res.mask)
            sig = colour_signature(frame, mask)
            s = self._signals(mask, res.presence_logit, appearance.score(sig))
            verdict = detector.assess(s)
            if verdict.ok:
                self._record(seg, f, mask, s)
                appearance.update(sig)
            else:
                self.backend.forget_last()
            if callback:
                callback(f, mask if verdict.ok else None, verdict)
            if bar:
                bar.update(1)
                bar.set_postfix_str(",".join(verdict.flags) or "ok")
            if verdict.broke:
                reason = "lost"
                break
            f += direction
        if bar:
            bar.close()
        return reason
