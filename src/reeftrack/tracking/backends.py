"""Promptable video-segmentation backends behind one small streaming interface.

    backend.start(frame, points=[(x, y)])  -> StepResult for the prompted frame
    backend.step(frame)                    -> StepResult for the next frame in the stream
    backend.forget_last()                  -> drop the last frame from the tracker memory

"Streaming" = frames are pushed one at a time (forward *or* reversed order), so we never
hold the whole video in memory and can stop as soon as the fish is lost.

Backends
--------
* ``sam3``  Meta SAM 3 tracker (Hugging Face `facebook/sam3`, gated: request access, then
           `hf auth login`). Successor of SAM 2 with a stronger detector-backbone and
           presence head - best out-of-the-box choice.
* ``sam2``  SAM 2.1 (`facebook/sam2.1-hiera-*`, open). Same API; used automatically if
           SAM 3 weights are not accessible.
* ``FakeBackend`` deterministic stand-in for unit tests (no weights, no GPU).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    mask: np.ndarray  # (H, W) bool at input-frame resolution
    presence_logit: float  # > 0: model thinks the object is visible


class TrackerBackend(Protocol):
    name: str

    def start(self, frame: np.ndarray, points: list[tuple[float, float]], labels: list[int] | None = None) -> StepResult: ...

    def step(self, frame: np.ndarray) -> StepResult: ...

    def forget_last(self) -> None: ...

    def preview(self, frame: np.ndarray, points: list[tuple[float, float]], labels: list[int] | None = None) -> StepResult: ...


class HFVideoTracker:
    """SAM 2 / SAM 3 video tracker via Hugging Face transformers, in streaming mode."""

    OBJ_ID = 1

    def __init__(self, backend: str, model_id: str, device: str, keep_memory_frames: int = 16):
        import torch

        if backend == "sam3":
            from transformers import Sam3TrackerVideoModel as Model
            from transformers import Sam3TrackerVideoProcessor as Processor
        else:
            from transformers import Sam2VideoModel as Model
            from transformers import Sam2VideoProcessor as Processor

        self.name = backend
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Model.from_pretrained(model_id).to(device, dtype=self.dtype).eval()
        self.processor = Processor.from_pretrained(model_id)
        self.keep = keep_memory_frames
        self.session = None
        self._idx = -1
        self._cond: set[int] = set()

    # ---- helpers -------------------------------------------------------------
    def _new_session(self):
        return self.processor.init_video_session(
            inference_device=self.device, video_storage_device="cpu", dtype=self.dtype
        )

    def _encode(self, frame: np.ndarray):
        return self.processor(images=frame, device=self.device, return_tensors="pt")

    def _run(self, session, inputs, idx: int) -> StepResult:
        import torch

        with torch.inference_mode():
            out = self.model(inference_session=session, frame_idx=idx, frame=inputs.pixel_values[0].to(self.dtype))
        masks = self.processor.post_process_masks(
            [out.pred_masks.float()], original_sizes=inputs.original_sizes, binarize=False
        )[0]
        mask = (masks[0, 0] > 0).cpu().numpy()
        logit = out.object_score_logits
        presence = float(logit.flatten()[0]) if logit is not None else (10.0 if mask.any() else -10.0)
        return StepResult(mask=mask, presence_logit=presence)

    def _add_prompt(self, session, idx, inputs, points, labels):
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=idx, obj_ids=self.OBJ_ID,
            input_points=[[[list(map(float, p)) for p in points]]],
            input_labels=[[labels or [1] * len(points)]],
            original_size=inputs.original_sizes[0],
        )

    def _prune(self) -> None:
        """Bound RAM: SAM only attends to the last few frames + prompted (conditioning) frames.

        Old frame tensors are swapped for an empty placeholder rather than deleted: the session
        derives frame counts from this dict, so the keys must stay.
        """
        import torch

        s = self.session
        cutoff = self._idx - self.keep
        try:
            for f, px in (s.processed_frames or {}).items():
                if f < self._idx and f not in self._cond and px.numel():
                    s.processed_frames[f] = torch.empty(0)
            for per_obj in s.output_dict_per_obj.values():
                nc = per_obj.get("non_cond_frame_outputs", {})
                for f in [f for f in nc if f < cutoff]:
                    del nc[f]
        except (AttributeError, KeyError, TypeError) as e:  # internal API changed: degrade gracefully
            log.debug("memory pruning skipped: %s", e)

    # ---- public API ----------------------------------------------------------
    def preview(self, frame, points, labels=None) -> StepResult:
        """Single-frame mask for a click, without touching the tracking session (GUI preview)."""
        session = self._new_session()
        inputs = self._encode(frame)
        self._add_prompt(session, 0, inputs, points, labels)
        return self._run(session, inputs, 0)

    def start(self, frame, points, labels=None) -> StepResult:
        self.session = self._new_session()
        self._idx = 0
        self._cond = {0}
        inputs = self._encode(frame)
        self._add_prompt(self.session, 0, inputs, points, labels)
        return self._run(self.session, inputs, 0)

    def step(self, frame) -> StepResult:
        if self.session is None:
            raise RuntimeError("call start() first")
        self._idx += 1
        res = self._run(self.session, self._encode(frame), self._idx)
        self._prune()
        return res

    def forget_last(self) -> None:
        """SAMURAI-style memory gating: don't let a doubtful frame become a memory."""
        try:
            for per_obj in self.session.output_dict_per_obj.values():
                per_obj.get("non_cond_frame_outputs", {}).pop(self._idx, None)
        except AttributeError:
            pass


class FakeBackend:
    """Test double: 'tracks' whatever pixels match a fixed colour; presence = mask non-empty."""

    name = "fake"

    def __init__(self, colour: tuple[int, int, int] = (255, 0, 0), tol: int = 30):
        self.colour = np.array(colour)
        self.tol = tol
        self.forgotten = 0

    def _seg(self, frame: np.ndarray) -> StepResult:
        m = np.all(np.abs(frame.astype(int) - self.colour) <= self.tol, axis=-1)
        return StepResult(mask=m, presence_logit=5.0 if m.any() else -5.0)

    def preview(self, frame, points, labels=None):
        return self._seg(frame)

    def start(self, frame, points, labels=None):
        return self._seg(frame)

    def step(self, frame):
        return self._seg(frame)

    def forget_last(self):
        self.forgotten += 1


def load_backend(tc, device: str) -> TrackerBackend:
    """Instantiate the configured backend, falling back from gated SAM 3 to open SAM 2.1."""
    order = [tc.backend] + (["sam2"] if tc.backend == "sam3" else [])
    last_err: Exception | None = None
    for name in order:
        try:
            log.info("loading %s (%s) on %s", name, tc.model_ids[name], device)
            return HFVideoTracker(name, tc.model_ids[name], device, tc.keep_memory_frames)
        except Exception as e:  # gated repo / no token / offline
            last_err = e
            log.warning("could not load %s: %s", name, str(e).splitlines()[0][:200])
    raise RuntimeError(f"no tracking backend could be loaded: {last_err}")
