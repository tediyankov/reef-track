"""Stitch per-chunk reconstructions into one global frame.

Every chunk comes back in its *own* coordinate system (origin = its first camera) and its
*own* scale (a single camera cannot observe absolute scale). Neighbouring chunks share
`overlap` frames, and for those frames both chunks predict a 3D point for every pixel. Those
pixel-wise correspondences give thousands of point pairs, from which we solve for the
similarity transform (rotation, translation, uniform scale) mapping chunk k onto chunk k-1.
Composing these maps everything into chunk 0's frame.

Point-based alignment is more stable than aligning only the 8 overlapping camera centres,
which lie almost on a line and leave the rotation about that line undetermined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from reeftrack.recon.backends import ChunkPrediction
from reeftrack.recon.geometry import robust_umeyama, sim3_matrix, transform_extrinsic, unproject

log = logging.getLogger(__name__)


@dataclass
class GlobalRecon:
    frames: np.ndarray  # (N,) keyframe proxy indices, sorted
    images: np.ndarray  # (N, h, w, 3) uint8
    depth: np.ndarray  # (N, h, w) in global units
    depth_conf: np.ndarray  # (N, h, w)
    extrinsic: np.ndarray  # (N, 3, 4) camera-from-world (global)
    intrinsic: np.ndarray  # (N, 3, 3)
    align_residual: list[float]  # median residual of each chunk-to-chunk alignment (diagnostic)
    chunk_transforms: list[np.ndarray] | None = None  # 4x4 map chunk coords -> global, per chunk


def _overlap_points(pred: ChunkPrediction, idx: np.ndarray, conf_q: float):
    pts, confs = [], []
    for i in idx:
        pts.append(unproject(pred.depth[i], pred.intrinsic[i], pred.extrinsic[i]).reshape(-1, 3))
        confs.append(pred.depth_conf[i].reshape(-1))
    return np.concatenate(pts), np.concatenate(confs)


def align_pair(prev: ChunkPrediction, prev_idx, cur: ChunkPrediction, cur_idx,
               conf_percentile: float = 50, max_points: int = 50_000, seed: int = 0, with_scale: bool = True):
    """Sim3 (or rigid, if `with_scale=False`) map from `cur` chunk coordinates onto `prev`'s."""
    P, cp = _overlap_points(prev, prev_idx, conf_percentile)
    C, cc = _overlap_points(cur, cur_idx, conf_percentile)
    conf = np.minimum(cp, cc)
    good = np.isfinite(P).all(1) & np.isfinite(C).all(1) & (conf >= np.percentile(conf, conf_percentile))
    sel = np.flatnonzero(good)
    if len(sel) > max_points:
        sel = np.random.default_rng(seed).choice(sel, max_points, replace=False)
    s, R, t = robust_umeyama(C[sel], P[sel], with_scale=with_scale)
    res = np.linalg.norm(P[sel] - (s * C[sel] @ R.T + t), axis=1)
    return sim3_matrix(s, R, t), float(np.median(res) / (np.median(np.linalg.norm(P[sel] - P[sel].mean(0), axis=1)) + 1e-9))


def stitch(chunks: list[tuple[np.ndarray, ChunkPrediction]], conf_percentile: float = 50,
           with_scale: bool = True) -> GlobalRecon:
    """`chunks` = list of (keyframe positions, prediction) with consecutive overlaps.

    `with_scale=False` is for chunks that are already metric (stereo): only rotation and
    translation are estimated, so the scale cannot drift from chunk to chunk.
    """
    first_pos, first = chunks[0]
    M_prev = np.eye(4)
    out: dict[int, tuple] = {}
    residuals: list[float] = []
    transforms = [M_prev.copy()]

    def emit(pos_arr, pred, M):
        for j, p in enumerate(pos_arr):
            if p in out:  # overlap frame already provided by the earlier chunk
                continue
            E, s = transform_extrinsic(pred.extrinsic[j], M)
            out[int(p)] = (pred.images[j], pred.depth[j] * s, pred.depth_conf[j], E, pred.intrinsic[j])

    emit(first_pos, first, M_prev)
    for (prev_pos, prev), (cur_pos, cur) in zip(chunks[:-1], chunks[1:]):
        shared = np.intersect1d(prev_pos, cur_pos)
        if len(shared) == 0:
            raise ValueError("consecutive chunks must overlap")
        pi = np.searchsorted(prev_pos, shared)
        ci = np.searchsorted(cur_pos, shared)
        M_local, res = align_pair(prev, pi, cur, ci, conf_percentile, with_scale=with_scale)
        residuals.append(res)
        if res > 0.1:
            log.warning("chunk alignment residual %.3f is high - geometry may be inconsistent here", res)
        M_prev = M_prev @ M_local
        transforms.append(M_prev.copy())
        emit(cur_pos, cur, M_prev)

    keys = sorted(out)
    stack = lambda k: np.stack([out[p][k] for p in keys])  # noqa: E731
    return GlobalRecon(np.asarray(keys), stack(0), stack(1).astype(np.float32), stack(2).astype(np.float32),
                       stack(3), stack(4), residuals, transforms)
