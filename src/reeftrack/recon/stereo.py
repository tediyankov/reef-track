"""Stereo (two-camera rig) reconstruction: metric scale and no scale drift.

What the second camera buys us
------------------------------
A single moving camera reconstructs the scene only *up to scale*, and in our chunked
pipeline that unknown scale can wander from chunk to chunk ("scale drift"). On a rig the
two lenses are a fixed, known distance apart (the *baseline*, here 7 cm). So:

1. Every keyframe is sent to Wat3R as a synchronised **Left + Right pair** (both images in
   the same forward pass), so the model sees two viewpoints of the same instant.
2. Within each chunk we measure the predicted Left-to-Right camera distance: the ratio to
   the true baseline is a (noisy) measurement of that chunk's scale.
3. Neighbouring chunks overlap; aligning them gives a (precise) measurement of their
   *relative* scale. All chunk scales are then solved **jointly** by weighted least squares
   (`reconcile_scales`): overlaps keep neighbours consistent (no seams), while the baseline
   evidence of all chunks together fixes the absolute scale (metres) and prevents drift.
   Rescaling each chunk independently (v1) left seams where single-chunk estimates were off.
4. Chunks are then stitched with **rotation + translation only** (no free scale factor).
5. The point cloud uses depth maps from both cameras (wider coverage).

Diagnostics written to keyframes.json let you judge how well the model respected the rig:
the spread of the predicted baseline within each chunk and the predicted Left-Right
rotation (a rigid rig should give near-constant values).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from reeftrack.devices import resolve_device
from reeftrack.io.video import FrameReader
from reeftrack.recon.backends import ChunkPrediction, FeedForwardRecon, preprocess
from reeftrack.recon.geometry import camera_center, sim3_matrix, transform_extrinsic
from reeftrack.recon.keyframes import make_chunks, select_keyframes
from reeftrack.recon.pipeline import _fish_masks, _FrameRange, _preview, apply_world_transform, save_recon
from reeftrack.recon.pointcloud import build_cloud, camera_centers, drop_above_cameras, levelling_transform
from reeftrack.recon.stitch import GlobalRecon, stitch

STEREO_NAME = "recon_stereo"


def split_pairs(pred: ChunkPrediction) -> tuple[ChunkPrediction, ChunkPrediction]:
    """Interleaved [L0, R0, L1, R1, ...] prediction -> (left, right) predictions."""
    take = lambda sl: ChunkPrediction(pred.images[sl], pred.depth[sl], pred.depth_conf[sl],  # noqa: E731
                                      pred.extrinsic[sl], pred.intrinsic[sl])
    return take(slice(0, None, 2)), take(slice(1, None, 2))


def rig_stats(left: ChunkPrediction, right: ChunkPrediction) -> dict:
    """Predicted baseline lengths, Left->Right rotation angles and baseline direction (left camera frame)."""
    b, ang, dirs = [], [], []
    for EL, ER in zip(left.extrinsic, right.extrinsic):
        cl, cr = camera_center(EL), camera_center(ER)
        b.append(np.linalg.norm(cr - cl))
        R_rel = ER[:3, :3] @ EL[:3, :3].T
        ang.append(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))))
        d = EL[:3, :3] @ (cr - cl)
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    b = np.asarray(b)
    return {"baseline_raw_median": float(np.median(b)), "baseline_cv": float(np.std(b) / (np.mean(b) + 1e-12)),
            "rel_rot_deg_median": float(np.median(ang)), "rel_rot_deg_std": float(np.std(ang)),
            "baseline_dir_left_cam": np.mean(dirs, 0).round(3).tolist()}


def scale_prediction(pred: ChunkPrediction, s: float) -> ChunkPrediction:
    """Uniformly scale a chunk's world (camera positions and depths) by s."""
    M = sim3_matrix(s, np.eye(3), np.zeros(3))
    E = np.stack([transform_extrinsic(e, M)[0] for e in pred.extrinsic])
    return replace(pred, depth=pred.depth * s, extrinsic=E)


def metric_chunk(pred: ChunkPrediction, baseline_m: float):
    """Split an interleaved stereo chunk and rescale it so the rig baseline is `baseline_m` metres."""
    left, right = split_pairs(pred)
    stats = rig_stats(left, right)
    s = baseline_m / stats["baseline_raw_median"]
    stats["scale_to_metres"] = float(s)
    return scale_prediction(left, s), scale_prediction(right, s), stats


def baseline_log_measurement(stats: dict, n_pairs: int, floor: float = 0.03) -> tuple[float, float]:
    """(log scale-to-metres, variance) from one chunk's predicted baselines.

    The variance is the per-pair spread over the number of pairs, plus a floor for model bias
    that averaging pairs within one chunk cannot remove.
    """
    var = stats["baseline_cv"] ** 2 / max(1, n_pairs) + floor**2
    return float(np.log(stats["scale_to_metres"])), float(var)


def reconcile_scales(base_log: np.ndarray, base_var: np.ndarray, join_log: np.ndarray,
                     join_var: np.ndarray, estimate_variances: bool = False, iters: int = 20,
                     return_factors: bool = False):
    """Weighted least squares for per-chunk log-scales lambda_k.

        baseline:  lambda_k                 = base_log[k]     (variance base_var[k])
        overlap:   lambda_k - lambda_{k-1}  = join_log[k-1]   (variance join_var[k-1])

    `join_log[k-1]` is log of the Sim3 scale mapping raw chunk k onto raw chunk k-1.

    With `estimate_variances=True` the *relative* trust in the two kinds of evidence is learned
    from the data (Helmert variance-component estimation): each group's variances are rescaled
    until its weighted residuals match its share of the redundancy. This replaces hand-set
    noise levels, which in v2 over-trusted noisy per-chunk baselines and left seams.
    """
    n = len(base_log)
    A = np.zeros((2 * n - 1, n))
    A[np.arange(n), np.arange(n)] = 1
    for k in range(1, n):
        A[n + k - 1, k], A[n + k - 1, k - 1] = 1, -1
    b = np.r_[base_log, join_log]
    var0 = np.r_[base_var, join_var]
    group = np.r_[np.zeros(n, int), np.ones(n - 1, int)]
    factors = np.ones(2)
    for _ in range(iters if estimate_variances else 1):
        Wd = 1 / (var0 * factors[group])
        N = A.T @ (A * Wd[:, None])
        lam = np.linalg.solve(N, A.T @ (Wd * b))
        if not estimate_variances:
            break
        r = A @ lam - b
        red = 1 - np.einsum("ij,jk,ik->i", A, np.linalg.inv(N), A) * Wd  # redundancy numbers
        new = factors.copy()
        for g in (0, 1):
            m = group == g
            if red[m].sum() > 1e-6:
                new[g] = max(1e-6, float((r[m] ** 2 * Wd[m]).sum() / red[m].sum()) * factors[g])
        if np.allclose(new, factors, rtol=1e-4):
            break
        factors = new
    return (lam, factors) if return_factors else lam


def assemble(chunks: list[tuple[np.ndarray, ChunkPrediction]], transforms: list[np.ndarray]) -> GlobalRecon:
    """Place chunk predictions in the global frame with known per-chunk transforms (first occurrence wins)."""
    out: dict[int, tuple] = {}
    for (pos, pred), M in zip(chunks, transforms):
        for j, p in enumerate(pos):
            if int(p) in out:
                continue
            E, s = transform_extrinsic(pred.extrinsic[j], M)
            out[int(p)] = (pred.images[j], pred.depth[j] * s, pred.depth_conf[j], E, pred.intrinsic[j])
    keys = sorted(out)
    stack = lambda k: np.stack([out[p][k] for p in keys])  # noqa: E731
    return GlobalRecon(np.asarray(keys), stack(0), stack(1).astype(np.float32), stack(2).astype(np.float32),
                       stack(3), stack(4), [])


def save_right(run, right: GlobalRecon) -> None:
    d = run.recon_dir
    np.savez_compressed(d / "cameras_right.npz", frames=right.frames, intrinsic=right.intrinsic,
                        extrinsic=right.extrinsic, image_hw=np.array(right.depth.shape[1:]))
    np.savez_compressed(d / "depth_right.npz", frames=right.frames, depth=right.depth.astype(np.float16),
                        depth_conf=right.depth_conf.astype(np.float16))


def load_right(run, with_images: bool = False) -> GlobalRecon | None:
    d = run.recon_dir
    if not (d / "cameras_right.npz").exists():
        return None
    cams, dep = np.load(d / "cameras_right.npz"), np.load(d / "depth_right.npz")
    images = None
    if with_images:
        with FrameReader(run.proxy_right) as r:
            images = preprocess([r[int(f)] for f in cams["frames"]], int(cams["image_hw"][1]))
    return GlobalRecon(cams["frames"], images, dep["depth"].astype(np.float32), dep["depth_conf"].astype(np.float32),
                       cams["extrinsic"], cams["intrinsic"], [])


def _keyframes(run, rc, reader, fps):
    """Reuse the mono run's keyframes when available, so the two reconstructions are comparable."""
    mono = run.variant("recon").recon_dir / "keyframes.json"
    if mono.exists():
        kf = np.asarray(json.loads(mono.read_text())["keyframes"])
        lo, hi = rc.frame_range or (0, len(reader) - 1)
        kf = kf[(kf >= lo) & (kf <= hi)]
        if len(kf):
            return kf, "reused from recon/keyframes.json"
    lo, hi = rc.frame_range or (0, len(reader) - 1)
    kf, _ = select_keyframes(_FrameRange(reader, lo, hi), fps, rc.keyframe_fps, rc.sharpness_window, progress=True)
    return kf + lo, "selected (sharpest per slot)"


def _cache_path(run, rc, kf, chunks_idx):
    import hashlib

    key = json.dumps([rc.backend, rc.model_ids[rc.backend], rc.image_size, kf.tolist(), [c.tolist() for c in chunks_idx],
                      run.stereo().get("right_source_start")])
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(rc.cache_dir).expanduser() / run.root.name / f"stereo_chunks_{h}.npz"


def _save_chunk_cache(path, preds: list[ChunkPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, depth=np.stack([p.depth for p in preds]).astype(np.float16),
             conf=np.stack([p.depth_conf for p in preds]).astype(np.float16),
             extrinsic=np.stack([p.extrinsic for p in preds]), intrinsic=np.stack([p.intrinsic for p in preds]))


def _load_chunk_cache(path, images: list[np.ndarray]) -> list[ChunkPrediction]:
    z = np.load(path)
    return [ChunkPrediction(images[k], z["depth"][k].astype(np.float32), z["conf"][k].astype(np.float32),
                            z["extrinsic"][k], z["intrinsic"][k]) for k in range(len(z["depth"]))]


def solve_chunk_scales(mode: str, chunk_stats, left_chunks, pair_size: int, baseline: float):
    """Per-chunk scale factors to metres, and a description of how they were obtained."""
    from reeftrack.recon.stitch import align_pair

    base = np.array([baseline_log_measurement(st_, pair_size) for st_ in chunk_stats])
    if mode == "independent":
        return np.exp(base[:, 0]), {}
    join_log, join_res = [], []
    for (pp, prev), (cp, cur) in zip(left_chunks[:-1], left_chunks[1:]):
        shared = np.intersect1d(pp, cp)
        M, res = align_pair(prev, np.searchsorted(pp, shared), cur, np.searchsorted(cp, shared), 50)
        join_log.append(float(np.log(np.cbrt(np.linalg.det(M[:3, :3])))))
        join_res.append(res)
    join_log, join_res = np.array(join_log), np.array(join_res)
    if mode == "global":  # follow the overlaps exactly (mono-like seams), one metric factor overall
        lam_rel = np.r_[0.0, np.cumsum(join_log)]
        offset = np.median(base[:, 0] - lam_rel)
        return np.exp(lam_rel + offset), {"note": "overlap chain + single global scale (keeps any drift)"}
    lam, f = reconcile_scales(base[:, 0], base[:, 1], join_log, (0.005 + join_res) ** 2,
                              estimate_variances=True, return_factors=True)
    return np.exp(lam), {"variance_factor_baseline": float(f[0]), "variance_factor_overlap": float(f[1]),
                         "baseline_sigma_log": float(np.sqrt(f[0] * np.median(base[:, 1]))),
                         "overlap_sigma_log": float(np.sqrt(f[1] * np.median((0.005 + join_res) ** 2)))}


def reconstruct_stereo(run, cfg, out_name: str = STEREO_NAME) -> None:
    import open3d as o3d

    st = run.stereo()
    if not st or not run.proxy_right.exists():
        raise RuntimeError("no right camera: run `reeftrack add-right <run> <right video> --baseline-cm X` first")
    if not st.get("baseline_m"):
        raise RuntimeError("stereo baseline unknown: re-run add-right with --baseline-cm")
    baseline = float(st["baseline_m"])
    out = run.variant(out_name)
    out.recon_dir.mkdir(exist_ok=True)
    rc = cfg.recon
    device = resolve_device(rc.device)
    t0 = time.time()
    left_r, right_r = FrameReader(run.proxy), FrameReader(run.proxy_right)
    fmap = run.frame_map()

    kf, kf_src = _keyframes(run, rc, left_r, fmap.proxy_fps)
    pair_size, pair_overlap = max(2, rc.chunk_size // 2), max(1, rc.chunk_overlap // 2)
    chunks_idx = make_chunks(len(kf), pair_size, pair_overlap)
    print(f"[stereo] {len(kf)} keyframe pairs ({kf_src}) in {len(chunks_idx)} chunks of {pair_size} pairs "
          f"(overlap {pair_overlap}), baseline {baseline * 100:.1f} cm, {rc.backend} on {device}")

    # images exactly as the model sees them (also needed when loading cached predictions)
    images = [preprocess([f for i in idx for f in (left_r[int(kf[i])], right_r[int(kf[i])])], rc.image_size)
              for idx in chunks_idx]
    cache = _cache_path(run, rc, kf, chunks_idx)
    if cache.exists():
        print(f"[stereo] using cached Wat3R chunk predictions: {cache}")
        preds = _load_chunk_cache(cache, images)
    else:
        model = FeedForwardRecon(rc.backend, rc.model_ids[rc.backend], device, rc.image_size)
        preds = []
        for ci, idx in enumerate(chunks_idx):
            tc = time.time()
            frames = [f for i in idx for f in (left_r[int(kf[i])], right_r[int(kf[i])])]
            preds.append(model(frames))
            print(f"[stereo] chunk {ci + 1}/{len(chunks_idx)} ({time.time() - tc:.1f}s)")
            if device == "mps":
                import torch

                torch.mps.empty_cache()
        del model
        _save_chunk_cache(cache, preds)

    left_chunks, right_chunks, chunk_stats = [], [], []
    for idx, pred in zip(chunks_idx, preds):
        L, R = split_pairs(pred)  # raw (arbitrary) chunk scale
        stats = rig_stats(L, R)
        stats["scale_to_metres"] = baseline / stats["baseline_raw_median"]  # this chunk alone
        left_chunks.append((kf[idx], L))
        right_chunks.append((kf[idx], R))
        chunk_stats.append(stats)

    # --- per-chunk metric scale: overlaps (relative, precise) + baselines (absolute, noisy)
    scales, scale_info = solve_chunk_scales(rc.stereo_scale_mode, chunk_stats, left_chunks, pair_size, baseline)
    for k, st_ in enumerate(chunk_stats):
        st_["scale_reconciled"] = float(scales[k])
    left_chunks = [(p, scale_prediction(c, float(scales[k]))) for k, (p, c) in enumerate(left_chunks)]
    right_chunks = [(p, scale_prediction(c, float(scales[k]))) for k, (p, c) in enumerate(right_chunks)]
    print(f"[stereo] scale mode '{rc.stereo_scale_mode}': chunk scales {scales.min():.3f}-{scales.max():.3f} "
          f"(independent: {min(s['scale_to_metres'] for s in chunk_stats):.3f}-"
          f"{max(s['scale_to_metres'] for s in chunk_stats):.3f}) {scale_info}")

    left = stitch(left_chunks, conf_percentile=50, with_scale=False)
    right = assemble(right_chunks, left.chunk_transforms)
    fish = _fish_masks(run, left.frames, rc.image_size) if rc.mask_fish else {}
    pcd = build_cloud(left, rc, fish, extra=[(right, None)] if rc.stereo_right_depth else ())
    T = levelling_transform(pcd, camera_centers(left.extrinsic))
    pcd.transform(T)
    apply_world_transform(left, T)
    apply_world_transform(right, T)
    if rc.drop_above_camera is not None:
        pcd = drop_above_cameras(pcd, camera_centers(left.extrinsic), rc.drop_above_camera, rc.camera_height_percentile)

    save_recon(out, left)
    save_right(out, right)
    o3d.io.write_point_cloud(str(out.recon_dir / "reef.ply"), pcd)
    final_baselines = np.linalg.norm(camera_centers(right.extrinsic) - camera_centers(left.extrinsic), axis=1)
    (out.recon_dir / "keyframes.json").write_text(json.dumps({
        "mode": "stereo", "scale_mode": rc.stereo_scale_mode, "scale_info": scale_info,
        "right_depth_in_cloud": rc.stereo_right_depth, "units": "metres", "baseline_m": baseline, "sync": st,
        "backend": rc.backend, "model_id": rc.model_ids[rc.backend], "device": device, "config_digest": cfg.digest(),
        "keyframes": kf.tolist(), "keyframe_source": kf_src, "chunks": [c.tolist() for c in chunks_idx],
        "align_residuals": left.align_residual, "chunk_stats": chunk_stats,
        "final_baseline_m": {"median": float(np.median(final_baselines)), "p05": float(np.percentile(final_baselines, 5)),
                             "p95": float(np.percentile(final_baselines, 95))},
        "n_points": len(pcd.points), "fish_masked_keyframes": len(fish), "world_transform": T.tolist(),
        "runtime_s": round(time.time() - t0, 1),
    }, indent=1))
    _preview(out)
    print(f"[stereo] {len(pcd.points):,} points -> {out.recon_dir / 'reef.ply'}  ({time.time() - t0:.0f}s)")
