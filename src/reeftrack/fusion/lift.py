"""Lift the 2D tracklet into the 3D reef frame.

For a tracked frame we know (a) the fish's 2D centre (30 fps, from tracking) and (b) the
camera pose (at keyframes from the reconstruction; in between, interpolated). The missing
piece is *how far* the fish is from the camera:

  * at keyframes   depth = robust statistic (median) of the model's depth map inside the
                   eroded fish mask  -> kind = "measured"
  * between them   depth linearly interpolated in time within the same segment
                                     -> kind = "depth_interp"
  * short gaps     (< max_interp_gap_s) between segments: 3D position linearly interpolated
                                     -> kind = "gap_interp"; longer gaps stay empty.

3D point = camera centre + depth along the ray through the fish centre.
Coordinates are in the reconstruction's units (arbitrary scale for monocular video; z up,
seabed ~ z = 0 after levelling). Only *relative* distances/shape are meaningful until a
scale reference (stereo baseline, known object, laser dots) is added.
"""

from __future__ import annotations

import csv
import json

import numpy as np
from scipy.ndimage import median_filter

from reeftrack.recon.backends import mask_to_model, to_model_pixels
from reeftrack.recon.geometry import camera_center, slerp_extrinsics, unproject_pixel
from reeftrack.tracking.masks import erode
from reeftrack.tracking.tracklet import Tracklet


def measure_depth(depth: np.ndarray, conf: np.ndarray, mask_model: np.ndarray, stat: str = "median") -> tuple[float, int]:
    vals = depth[mask_model & np.isfinite(depth) & (depth > 0)]
    if vals.size == 0:
        return np.nan, 0
    return float(np.median(vals) if stat == "median" else np.mean(vals)), int(vals.size)


def interp_pose(frames: np.ndarray, extrinsics: np.ndarray, f: int) -> np.ndarray | None:
    """Camera pose at proxy frame f by SLERP between the surrounding keyframes."""
    if f < frames[0] or f > frames[-1]:
        return None
    j = int(np.searchsorted(frames, f))
    if frames[j] == f:
        return extrinsics[j]
    a = (f - frames[j - 1]) / (frames[j] - frames[j - 1])
    return slerp_extrinsics(extrinsics[j - 1], extrinsics[j], a)


def lift_tracklet(tr: Tracklet, kf_frames, depth, conf, extrinsic, intrinsic, fc, image_size: int = 518,
                  fps: float = 30.0) -> list[dict]:
    H, W = tr.shape
    K = intrinsic[0]
    kf_index = {int(f): i for i, f in enumerate(kf_frames)}
    rows: list[dict] = []

    for seg in sorted(tr.segments, key=lambda s: s.start):
        seg_frames = [f for f in range(seg.start, seg.end + 1) if f in tr.obs]
        # 1) measured depth at keyframes inside this segment
        mf, md, mn = [], [], []
        for f in seg_frames:
            if f in kf_index:
                i = kf_index[f]
                m = mask_to_model(erode(tr.mask(f), fc.mask_erode_px), image_size)
                d, n = measure_depth(depth[i], conf[i], m, fc.depth_stat)
                if np.isfinite(d):
                    mf.append(f)
                    md.append(d)
                    mn.append(n)
        if not mf:
            continue
        md = np.asarray(md)
        if fc.smooth_window > 1 and len(md) >= 3:
            md = median_filter(md, size=min(fc.smooth_window, len(md)), mode="nearest")
        # 2) every tracked frame: interpolated pose + depth, ray through the 2D centroid
        for f in seg_frames:
            E = interp_pose(kf_frames, extrinsic, f)
            if E is None:
                continue
            d = float(np.interp(f, mf, md))
            if (f < mf[0] or f > mf[-1]) and min(abs(f - mf[0]), abs(f - mf[-1])) / fps > fc.max_interp_gap_s:
                continue  # don't extrapolate depth too far beyond the last measurement
            cx, cy = tr.obs[f].centroid
            u, v = to_model_pixels(cx, cy, H, W, image_size)
            X = unproject_pixel(u, v, d, K if f not in kf_index else intrinsic[kf_index[f]], E)
            measured = f in mf
            rows.append(dict(proxy_frame=f, segment_id=seg.segment_id, x=X[0], y=X[1], z=X[2], depth=d,
                             u=cx, v=cy, kind="measured" if measured else "depth_interp",
                             n_px=mn[mf.index(f)] if measured else 0))

    rows.sort(key=lambda r: r["proxy_frame"])
    return rows + _bridge_gaps(rows, fc.max_interp_gap_s * fps)


def _bridge_gaps(rows: list[dict], max_gap_frames: float) -> list[dict]:
    """Linear 3D interpolation across short gaps between consecutive segments."""
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        gap = b["proxy_frame"] - a["proxy_frame"]
        if a["segment_id"] == b["segment_id"] or gap <= 1 or gap > max_gap_frames:
            continue
        pa, pb = np.array([a["x"], a["y"], a["z"]]), np.array([b["x"], b["y"], b["z"]])
        for f in range(a["proxy_frame"] + 1, b["proxy_frame"]):
            t = (f - a["proxy_frame"]) / gap
            p = (1 - t) * pa + t * pb
            out.append(dict(proxy_frame=f, segment_id=-1, x=p[0], y=p[1], z=p[2], depth=np.nan,
                            u=np.nan, v=np.nan, kind="gap_interp", n_px=0))
    return out


def height_above_reef(P: np.ndarray, reef: np.ndarray, radius: float, min_pts: int = 10) -> np.ndarray:
    """Fish height above the reef surface directly beneath it (z up, levelled scene).

    The local reef surface height is the median z of reef points within `radius` (in x-y) of
    the fish; NaN where there is too little reef underneath to tell.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(reef[:, :2])
    out = np.full(len(P), np.nan)
    for i, nb in enumerate(tree.query_ball_point(P[:, :2], r=radius)):
        if len(nb) >= min_pts:
            out[i] = P[i, 2] - np.median(reef[nb, 2])
    return out


def summarise(rows: list[dict], fps: float, n_frames: int, units: str = "reconstruction units") -> dict:
    if not rows:
        return {"n_rows": 0}
    rows = sorted(rows, key=lambda r: r["proxy_frame"])
    P = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    f = np.array([r["proxy_frame"] for r in rows])
    contiguous = np.diff(f) == 1
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)[contiguous]
    kinds = [r["kind"] for r in rows]
    return {
        "n_rows": len(rows),
        "coverage_frac": round(len(set(f)) / n_frames, 4),
        "kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
        "path_length": round(float(steps.sum()), 4),
        "median_speed_per_s": round(float(np.median(steps) * fps), 4) if steps.size else None,
        "height_above_reef_median": (round(float(np.nanmedian([r["h_reef"] for r in rows])), 4)
                                     if any(np.isfinite(r.get("h_reef", np.nan)) for r in rows) else None),
        "units": units,
    }


def fuse(run, cfg) -> list[dict]:
    from reeftrack.fusion.viewer import write_scene
    from reeftrack.recon.pipeline import load_recon

    fmap = run.frame_map()
    tr = Tracklet.load(run.tracking_dir)
    rec = load_recon(run)
    rows = lift_tracklet(tr, rec.frames, rec.depth, rec.depth_conf, rec.extrinsic, rec.intrinsic,
                         cfg.fusion, cfg.recon.image_size, fmap.proxy_fps)
    rows.sort(key=lambda r: r["proxy_frame"])
    import open3d as o3d

    from reeftrack.recon.pointcloud import robust_extent

    reef = np.asarray(o3d.io.read_point_cloud(str(run.recon_dir / "reef.ply")).points)
    if rows and len(reef):
        P = np.array([[r["x"], r["y"], r["z"]] for r in rows])
        for r, h in zip(rows, height_above_reef(P, reef, 0.02 * robust_extent(reef))):
            r["h_reef"] = h
    meta_p = run.recon_dir / "keyframes.json"
    units = "metres" if meta_p.exists() and json.loads(meta_p.read_text()).get("units") == "metres" \
        else "reconstruction units (arbitrary monocular scale)"
    run.fusion_dir.mkdir(exist_ok=True)
    with open(run.fusion_dir / "trajectory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["proxy_frame", "source_frame", "time_s", "segment_id", "x", "y", "z", "fish_depth",
                    "u_px", "v_px", "kind", "n_depth_px", "height_above_reef"])
        for r in rows:
            f = r["proxy_frame"]
            w.writerow([f, fmap.to_source(f), f"{fmap.time_s(f):.4f}", r["segment_id"], f"{r['x']:.5f}",
                        f"{r['y']:.5f}", f"{r['z']:.5f}", f"{r['depth']:.5f}", f"{r['u']:.1f}", f"{r['v']:.1f}",
                        r["kind"], r["n_px"], f"{r.get('h_reef', np.nan):.5f}"])
    summary = summarise(rows, fmap.proxy_fps, fmap.n_proxy, units)
    summary["camera_path_length"] = round(float(np.linalg.norm(
        np.diff(np.stack([camera_center(E) for E in rec.extrinsic]), axis=0), axis=1).sum()), 4)
    (run.fusion_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_scene(run, rows)
    if not run.is_library and (run.root / "annotations" / "loops.csv").exists():  # refresh loop 3D metrics
        from reeftrack.annotation.loops import LoopLog

        LoopLog.for_run(run)
    print(f"[fuse] {len(rows)} trajectory rows -> {run.fusion_dir / 'trajectory.csv'}")
    print(json.dumps(summary, indent=1))
    return rows
