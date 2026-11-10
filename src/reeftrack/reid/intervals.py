"""Find the longest stretch of footage in which the focal fish is not occluded.

A fish's recording is split by the GoPro into chapters (GX01xxxx, GX02xxxx, ...) that follow
each other without a gap, so a stretch may run on from the end of one chapter into the next.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

OCCLUDED = ("Out of view", "Sheltering")


@dataclass(frozen=True)
class Segment:
    """Inclusive proxy-frame range [start, end] inside one video."""

    video: str
    start: int
    end: int

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


def occlusion_mask(n_frames: int, events: Iterable[tuple[str, int, int]],
                   occluded: Sequence[str] = OCCLUDED) -> np.ndarray:
    """True where the fish is occluded. `events` are (behaviour, start_frame, end_frame), inclusive."""
    mask = np.zeros(n_frames, bool)
    for behaviour, start, end in events:
        if behaviour in occluded:
            mask[max(0, int(start)): min(n_frames, int(end) + 1)] = True
    return mask


def longest_clear_run(masks: Sequence[tuple[str, np.ndarray]]) -> list[Segment]:
    """Longest run of un-occluded frames across consecutive chapters, as one segment per video.

    `masks` are (video_id, occlusion mask) in recording order; they are treated as one
    continuous recording. Ties go to the earliest run. Empty list if every frame is occluded.
    """
    if not masks:
        return []
    occ = np.concatenate([m for _, m in masks])
    clear = np.r_[False, ~occ, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(clear))
    starts, ends = edges[0::2], edges[1::2]  # run = [start, end)
    if len(starts) == 0:
        return []
    k = int(np.argmax(ends - starts))
    a, b = int(starts[k]), int(ends[k])
    out, offset = [], 0
    for video, m in masks:
        lo, hi = max(a, offset), min(b, offset + len(m))
        if lo < hi:
            out.append(Segment(video, lo - offset, hi - offset - 1))
        offset += len(m)
    return out


def consecutive_chapters(videos: Sequence[str]) -> list[list[str]]:
    """Group GoPro file names (GX<chapter:2><file:4>) into runs of consecutive chapters.

    A missing chapter (e.g. corrupt or not annotated) breaks the recording into two parts,
    because footage in between is unknown.
    """
    def key(v: str) -> tuple[str, int]:
        return v[4:8], int(v[2:4])

    groups: list[list[str]] = []
    for v in sorted(videos, key=key):
        if groups and key(groups[-1][-1])[0] == key(v)[0] and key(groups[-1][-1])[1] + 1 == key(v)[1]:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups
