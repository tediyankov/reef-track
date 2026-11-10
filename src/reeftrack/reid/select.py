"""Pick the most confident detections while keeping them apart in time.

Neighbouring frames at 30 fps are almost identical, so taking the top N by confidence alone
would give N copies of the same pose. Greedy selection with a minimum gap keeps variety.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pick_spaced(times_s: Sequence[float], scores: Sequence[float], n: int, min_gap_s: float) -> list[int]:
    """Indices of up to `n` items with the highest scores, no two closer than `min_gap_s`.

    Returned in time order. Items with a score of 0 or below (no detection) are never picked.
    """
    t = np.asarray(times_s, float)
    s = np.asarray(scores, float)
    chosen: list[int] = []
    taken = np.empty(0)
    for i in np.argsort(-s, kind="stable"):
        if len(chosen) >= n or s[i] <= 0:
            break
        if taken.size and np.min(np.abs(taken - t[i])) < min_gap_s:
            continue
        chosen.append(int(i))
        taken = np.append(taken, t[i])
    return sorted(chosen, key=lambda i: t[i])
