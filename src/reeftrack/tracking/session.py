"""Interactive tracking session: clicks in, tracklet out.

One click = one segment:
  1. SAM segments the clicked fish on that frame (the *prompt*).
  2. Forward propagation until the fish is lost / the video ends / we hit another segment.
  3. Backward propagation (``backfill``) the same way. For a re-acquisition this patches the
     gap from the other side: you click when you notice the fish again, the tracker walks
     back to when it actually re-emerged from behind the coral, stopping as soon as it meets
     the previous segment.

Instead of a click, a segment can start from a hand-corrected mask (`add_mask`, the GUI's
mask editor): SAM takes it as the prompted frame's answer and as memory for the frames after.

Clicking a frame that is already tracked is treated as a *correction*: the owning segment
is cut at that frame (the part after the click, or before it if the click is earlier than
that segment's prompt) and the new click re-tracks from there.

Trimming: `trim(frame, keep="before")` deletes a segment's frames from `frame` on (SAM drifted
off the fish near the end), `keep="after"` deletes them up to `frame`; `delete_segment` removes a
whole one. Every edit (clicks too) can be undone with `undo()`, newest first.

Speed: with `frame_step: 2` SAM sees every 2nd frame only (half the time, the model step is
>90% of it). The frames in between get the neighbouring mask moved halfway along the fish's
motion (`masks.between`), so every tracked frame still has a mask; such segments are listed in
`tracklet.meta["frame_step"]`. Clicks, corrections, ownership and backfill work as before.

Each propagation direction uses a fresh SAM session seeded only by the click, so a mistake
in one direction never contaminates the other.

Tiny fish: pass `roi` (a window x0, y0, w, h, e.g. the GUI's zoom box) and SAM only sees that
window, re-centred on the fish's last mask every frame (see `reeftrack.tracking.roi`). Masks
are pasted back into full-frame coordinates, so the tracklet looks the same either way.

`enhance` (e.g. "clahe", see `reeftrack.enhance`) gives SAM an enhanced image (after the
crop, if any); appearance checks still use the original colours.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from reeftrack.tracking.appearance import AppearanceModel, colour_signature
from reeftrack.tracking.backends import TrackerBackend, load_backend
from reeftrack.tracking.masks import area, bbox, between, centroid, largest_component
from reeftrack.tracking.polygon import Polygon, polygon_to_mask
from reeftrack.tracking.quality import FrameSignals, LossDetector, Verdict
from reeftrack.tracking.roi import Rect, centre_of, crop, paste, rect_at
from reeftrack.tracking.tracklet import Observation, Segment, Tracklet, load_for_run, save_for_run

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
        self._history: list[tuple] = []  # tracklet states before each edit, for undo()
        self.edited = False  # changed since opened (worth saving even if the tracklet is now empty)

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_run(cls, run, cfg, fresh: bool = False, frames=None) -> TrackingSession:
        """`frames`: an already-open reader (e.g. the GUI's, which may be reading the original)."""
        from reeftrack.devices import resolve_device
        from reeftrack.io.video import FrameReader

        reader = frames if frames is not None else FrameReader(run.proxy)
        tracklet = None if fresh else load_for_run(run)[0]
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
    @staticmethod
    def _view(enhance: str | None):
        """frame, rect -> what SAM is shown (cropped and/or enhanced)."""
        enh = None
        if enhance:
            from reeftrack import enhance as _enh

            enh = _enh.make(enhance)

        def view(frame, rect):
            img = frame if rect is None else crop(frame, rect)
            return enh(img) if enh else img

        return view

    def preview(self, frame_idx: int, xy: tuple[float, float], roi: Rect | None = None,
                enhance: str | None = None) -> np.ndarray:
        """Mask for a click on one frame (no tracking) - lets the user check before committing."""
        img = self._view(enhance)(self.frames[frame_idx], roi)
        if roi is None:
            return largest_component(self.backend.preview(img, [xy]).mask)
        res = self.backend.preview(img, [(xy[0] - roi[0], xy[1] - roi[1])])
        return paste(largest_component(res.mask), roi, self.shape)

    def add_click(
        self,
        frame_idx: int,
        xy: tuple[float, float],
        callback: FrameCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
        progress: bool = False,
        roi: Rect | None = None,
        enhance: str | None = None,
        polygon: Polygon | None = None,
        preview_mask: np.ndarray | None = None,
    ) -> Segment:
        """`preview_mask`: the full-frame mask the user saw and confirmed for this click. If SAM's
        restart from the click comes back empty (SAM 3 on a small zoom window can), tracking
        starts from that mask instead of failing."""
        self._checkpoint(f"tracking from frame {frame_idx}")
        self._make_room(frame_idx)
        seg = Segment(self.tracklet.next_segment_id(), frame_idx, (float(xy[0]), float(xy[1])), frame_idx, frame_idx,
                      roi=[int(v) for v in roi[2:]] if roi else None, enhance=enhance,
                      prompt_polygon=[[float(x), float(y)] for x, y in polygon] if polygon else None)
        self.tracklet.segments.append(seg)
        try:
            seg.end_reason = self._propagate(seg, +1, callback, should_stop, progress, roi, enhance, preview_mask)
            if self.tc.backfill:
                seg.start_reason = self._propagate(seg, -1, callback, should_stop, progress, roi, enhance,
                                                   preview_mask)
        except Exception:
            self._restore(self._history.pop())  # also undoes the cut made for a correction
            raise
        self.tracklet.refresh_extent(seg.segment_id)
        log.info("segment %d: %d-%d (start: %s, end: %s)", seg.segment_id, seg.start, seg.end,
                 seg.start_reason, seg.end_reason)
        return seg

    def add_mask(self, frame_idx: int, polygon: Polygon, callback: FrameCallback | None = None,
                 should_stop: Callable[[], bool] | None = None, progress: bool = False,
                 roi: Rect | None = None, enhance: str | None = None) -> Segment:
        """Track from a hand-drawn / corrected mask outline (video pixels) instead of a click."""
        mask = polygon_to_mask(polygon, self.shape)
        if not mask.any():
            raise ValueError("the mask outline is empty (needs at least 3 points around the fish)")
        return self.add_click(frame_idx, centroid(mask), callback, should_stop, progress, roi, enhance, polygon)

    def trim(self, frame_idx: int, keep: str = "before") -> tuple[int, int] | None:
        """Delete part of the segment that owns `frame_idx`, this frame included: everything after
        it (keep="before", e.g. SAM slid onto coral near the end) or everything before it
        (keep="after"). Returns the deleted (first, last) frames, or None if the frame isn't tracked.
        The cut end counts as a break, so re-clicking the fish later tracks back to meet it."""
        if keep not in ("before", "after"):
            raise ValueError(f"keep must be 'before' or 'after', not {keep!r}")
        owner = self.tracklet.owner(frame_idx)
        if owner is None:
            return None
        self._checkpoint(f"trim of segment {owner}")
        seg = self.tracklet.segment(owner)
        drop = sorted(f for f, o in self.tracklet.obs.items() if o.segment_id == owner
                      and (f >= frame_idx if keep == "before" else f <= frame_idx))
        self.tracklet.drop_frames(drop)
        if not any(o.segment_id == owner for o in self.tracklet.obs.values()):
            self.tracklet.remove_segment(owner)
        else:
            if keep == "before":
                seg.end_reason = "trimmed"
            else:
                seg.start_reason = "trimmed"
            self.tracklet.refresh_extent(owner)
        return drop[0], drop[-1]

    def delete_segment(self, segment_id: int) -> Segment | None:
        seg = next((s for s in self.tracklet.segments if s.segment_id == segment_id), None)
        if seg is not None:
            self._checkpoint(f"deleting segment {segment_id}")
            self.tracklet.remove_segment(segment_id)
        return seg

    def undo(self) -> str | None:
        """Undo the last edit (click, correction, trim, delete) and say what it was. With no edits
        this session (e.g. just reopened), remove the most recently created segment instead."""
        self.edited = True
        if self._history:
            what = self._history[-1][-1]
            self._restore(self._history.pop())
            return what
        if not self.tracklet.segments:
            return None
        seg = max(self.tracklet.segments, key=lambda s: s.segment_id)
        self.tracklet.remove_segment(seg.segment_id)
        return f"segment {seg.segment_id}"

    MAX_UNDO = 50

    def _checkpoint(self, what: str) -> None:
        self.edited = True
        tr = self.tracklet  # masks/observations are replaced on edit, never changed in place
        self._history.append((copy.deepcopy(tr.segments), dict(tr.obs), dict(tr.masks), what))
        del self._history[:-self.MAX_UNDO]

    def _restore(self, state) -> None:
        segments, obs, masks, _ = state
        self.tracklet.segments, self.tracklet.obs, self.tracklet.masks = segments, obs, masks

    def save(self) -> None:
        if self.run is None:
            raise RuntimeError("session has no run to save into")
        save_for_run(self.tracklet, self.run)  # library runs: local copy first, then the shared folder

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

    def _next_frame(self, f: int, direction: int, step: int, segment_id: int) -> tuple[int | None, bool]:
        """(the next frame to give SAM, blocked). Normally `step` frames on, but never past the clip's
        end or into another segment: then the last free frame before it (None if there is none;
        blocked = it was another segment, not the clip's end)."""
        nxt, blocked = None, False
        for k in range(1, step + 1):
            g = f + direction * k
            if not 0 <= g < self.n_frames:
                break
            owner = self.tracklet.owner(g)
            if owner is not None and owner != segment_id:
                blocked = True
                break
            nxt = g
        return nxt, blocked and nxt is None

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

    def _propagate(self, seg, direction, callback, should_stop, progress, roi: Rect | None = None,
                   enhance: str | None = None, preview_mask: np.ndarray | None = None) -> str:
        start = seg.prompt_frame
        frame = self.frames[start]
        view = self._view(enhance)

        def segment(res, rect):  # backend output -> full-frame mask
            m = largest_component(res.mask)
            return m if rect is None else paste(m, rect, self.shape)

        rect = roi
        if seg.prompt_polygon:  # hand-corrected mask: SAM's prompt, and recorded exactly as drawn
            given = polygon_to_mask(seg.prompt_polygon, self.shape)
            if rect is not None:
                rect = rect_at(centroid(given), rect[2:], self.shape)
            res = self.backend.start(view(frame, rect), [], mask=given if rect is None else crop(given, rect))
            mask = given
        elif rect is None:
            res = self.backend.start(view(frame, None), [seg.prompt_xy])
            mask = segment(res, rect)
        else:
            res = self.backend.start(view(frame, rect), [(seg.prompt_xy[0] - rect[0], seg.prompt_xy[1] - rect[1])])
            mask = segment(res, rect)
        if not mask.any() and preview_mask is not None and np.any(preview_mask):
            # SAM found nothing this time, but the user confirmed a preview mask: start from it
            log.warning("click on frame %d segmented nothing when tracking started; using the previewed mask", start)
            given = np.asarray(preview_mask, bool)
            res = self.backend.start(view(frame, rect), [], mask=given if rect is None else crop(given, rect))
            mask = given
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

        step = max(1, int(getattr(self.tc, "frame_step", 1)))
        if step > 1:
            self.tracklet.meta.setdefault("frame_step", {})[str(seg.segment_id)] = step
        f, cur_step = start, step
        last_f, last_mask, last_s = start, mask, s0
        reason = "video_end" if direction > 0 else "video_start"
        while True:
            if should_stop and should_stop():
                reason = "user"
                break
            nxt, blocked = self._next_frame(f, direction, cur_step, seg.segment_id)
            if nxt is None:
                if blocked:
                    reason = "met_segment"
                break
            f = nxt
            frame = self.frames[f]
            if rect is not None:  # keep the fish centred in SAM's window
                rect = rect_at(centroid(last_mask) if last_mask.any() else centre_of(rect), rect[2:], self.shape)
            res = self.backend.step(view(frame, rect))
            mask = segment(res, rect)
            sig = colour_signature(frame, mask)
            s = self._signals(mask, res.presence_logit, appearance.score(sig))
            verdict = detector.assess(s)
            if verdict.ok:
                if 1 < abs(f - last_f) <= step:  # the frames SAM skipped: move the masks in between
                    for g in range(last_f + direction, f, direction):
                        m = between(last_mask, mask, (g - last_f) / (f - last_f))
                        sg = self._signals(m, min(last_s.presence_logit, s.presence_logit),
                                           min(last_s.appearance_sim, s.appearance_sim))
                        self._record(seg, g, m, sg)
                        if callback:
                            callback(g, m, Verdict(ok=True, flags=[]))
                self._record(seg, f, mask, s)
                appearance.update(sig)
                last_f, last_mask, last_s = f, mask, s
                cur_step = step
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
            if not verdict.ok and cur_step > 1 and abs(f - last_f) > 1:  # jumped past frames: check them
                f, cur_step = last_f, 1  # one by one (e.g. the fish's last frame before it hides)
        if bar:
            bar.close()
        return reason
