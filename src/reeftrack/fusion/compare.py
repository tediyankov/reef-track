"""`reeftrack compare`: mono vs stereo reconstruction of the same run.

Both reconstructions are built from the same keyframes, so their camera paths correspond
frame by frame:

1. **Global alignment** - one similarity transform (rotation, translation, scale) maps the
   mono camera path onto the stereo one. Its scale converts mono units to metres.
2. **Scale drift** - the same fit on sliding windows (~10 s). A drift-free mono
   reconstruction would give a local/global scale ratio of 1.0 everywhere; deviations show
   how much the mono scale wanders along the swim.
3. **Cloud-to-cloud distance** - after alignment, the distance from each point of one cloud
   to the nearest point of the other (in cm): how much the two geometries disagree.
4. **Fish trajectory** (if both fusions exist) - path length, speed and point-wise distance.

Outputs: compare/report.json, compare/compare.html (side-by-side 3D + drift plot),
compare/overlay.ply (mono tinted blue, stereo tinted orange, in metres).
"""

from __future__ import annotations

import csv
import json

import numpy as np

from reeftrack.recon.geometry import camera_center, sim3_matrix, transform_points, umeyama

MAX_HTML_POINTS = 80_000


def _centres(recon_dir):
    cams = np.load(recon_dir / "cameras.npz")
    return cams["frames"], np.stack([camera_center(E) for E in cams["extrinsic"]])


def align_paths(frames_a, Ca, frames_b, Cb):
    """Sim3 mapping path A onto path B using the keyframes they share."""
    common, ia, ib = np.intersect1d(frames_a, frames_b, return_indices=True)
    if len(common) < 3:
        raise ValueError("the two reconstructions share fewer than 3 keyframes")
    s, R, t = umeyama(Ca[ia], Cb[ib])
    res = np.linalg.norm(Cb[ib] - (s * Ca[ia] @ R.T + t), axis=1)
    return sim3_matrix(s, R, t), common, ia, ib, res


def local_scale_profile(Ca, Cb, s_global: float, window: int, step: int | None = None):
    """Scale of a Sim3 fit on sliding windows of corresponding points, relative to `s_global`."""
    step = step or max(1, window // 2)
    centres, ratios = [], []
    for a in range(0, max(1, len(Ca) - window + 1), step):
        sl = slice(a, a + window)
        if np.ptp(Cb[sl], axis=0).max() < 1e-6: # camera did not move: scale undefined
            continue
        s, _, _ = umeyama(Ca[sl], Cb[sl])
        centres.append(a + window // 2)
        ratios.append(s / s_global)
    return np.asarray(centres, int), np.asarray(ratios)


def _read_traj(path):
    if not path.exists():
        return None
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["kind"] != "gap_interp"]
    return {int(r["proxy_frame"]): np.array([float(r["x"]), float(r["y"]), float(r["z"])]) for r in rows}


def _path_stats(traj: dict, fps: float, scale: float = 1.0):
    f = np.array(sorted(traj))
    P = np.stack([traj[k] for k in f]) * scale
    contiguous = np.diff(f) == 1
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)[contiguous]
    return {"n": len(f), "path_length_m": round(float(steps.sum()), 3),
            "median_speed_m_s": round(float(np.median(steps) * fps), 3) if steps.size else None}


def compare(run, mono: str = "recon", stereo: str = "recon_stereo", window_s: float = 10.0) -> dict:
    import open3d as o3d

    from reeftrack.fusion.viewer import write_compare_html

    dm, ds = run.root / mono, run.root / stereo
    for d in (dm, ds):
        if not (d / "cameras.npz").exists():
            raise FileNotFoundError(f"{d} has no reconstruction")
    out_dir = run.root / "compare"
    out_dir.mkdir(exist_ok=True)
    fmap = run.frame_map()

    fm_, Cm = _centres(dm)
    fs_, Cs = _centres(ds)
    M, common, im, is_, res = align_paths(fm_, Cm, fs_, Cs)
    s_glob = float(np.cbrt(np.linalg.det(M[:3, :3])))
    kf_per_s = len(common) / max(1e-9, (common[-1] - common[0]) / fmap.proxy_fps)
    window = max(4, int(round(window_s * kf_per_s)))
    centre_idx, ratios = local_scale_profile(Cm[im], Cs[is_], s_glob, window)
    t_centres = fmap.time_s(common[centre_idx]) if len(centre_idx) else np.zeros(0)

    pm = o3d.io.read_point_cloud(str(dm / "reef.ply"))
    ps = o3d.io.read_point_cloud(str(ds / "reef.ply"))
    pm.transform(M)  # mono -> stereo frame, metres
    d_ms = np.asarray(pm.compute_point_cloud_distance(ps))
    d_sm = np.asarray(ps.compute_point_cloud_distance(pm))
    Pm, Ps = np.asarray(pm.points), np.asarray(ps.points)

    stereo_meta = json.loads((ds / "keyframes.json").read_text())
    path_len = lambda C: float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())  # noqa: E731
    report = {
        "units": "metres (stereo frame); mono mapped by a global similarity transform",
        "mono_to_metres_scale": round(s_glob, 5),
        "n_common_keyframes": int(len(common)),
        "camera_path_fit_residual_m": {"median": round(float(np.median(res)), 4),
                                       "p95": round(float(np.percentile(res, 95)), 4)},
        "camera_path_length_m": {"stereo": round(path_len(Cs[is_]), 3), "mono_scaled": round(s_glob * path_len(Cm[im]), 3)},
        "mono_scale_drift": {
            "window_s": window_s,
            "ratio_min": round(float(ratios.min()), 3) if ratios.size else None,
            "ratio_max": round(float(ratios.max()), 3) if ratios.size else None,
            "max_over_min": round(float(ratios.max() / ratios.min()), 3) if ratios.size else None,
        },
        "points": {"mono": len(Pm), "stereo": len(Ps)},
        "scene_extent_m": {"mono_scaled": np.ptp(np.percentile(Pm, [1, 99], 0), 0).round(3).tolist(),
                           "stereo": np.ptp(np.percentile(Ps, [1, 99], 0), 0).round(3).tolist()},
        "cloud_distance_cm": {
            "mono_to_stereo_median": round(100 * float(np.median(d_ms)), 2),
            "mono_to_stereo_p90": round(100 * float(np.percentile(d_ms, 90)), 2),
            "stereo_to_mono_median": round(100 * float(np.median(d_sm)), 2),
            "stereo_to_mono_p90": round(100 * float(np.percentile(d_sm, 90)), 2),
        },
        "stereo_rig_check": {
            "final_baseline_m": stereo_meta.get("final_baseline_m"),
            "chunk_baseline_cv_median_pct": round(100 * float(np.median([c["baseline_cv"] for c in stereo_meta["chunk_stats"]])), 2),
            "chunk_LR_rotation_deg_median": round(float(np.median([c["rel_rot_deg_median"] for c in stereo_meta["chunk_stats"]])), 2),
            "chunk_scale_to_metres_range": [round(min(c.get("scale_reconciled", c["scale_to_metres"]) for c in stereo_meta["chunk_stats"]), 4),
                                            round(max(c.get("scale_reconciled", c["scale_to_metres"]) for c in stereo_meta["chunk_stats"]), 4)],
            "stitch_residual_median": round(float(np.median(stereo_meta["align_residuals"])), 4),
            "stitch_residual_max": round(float(np.max(stereo_meta["align_residuals"])), 4),
        },
    }

    # fish trajectories (optional)
    tm = _read_traj(run.root / f"fusion{mono.removeprefix('recon')}" / "trajectory.csv")
    ts = _read_traj(run.root / f"fusion{stereo.removeprefix('recon')}" / "trajectory.csv")
    traj_xyz = {}
    if tm and ts:
        tm_al = {f: transform_points(M, p[None])[0] for f, p in tm.items()}
        shared = sorted(set(tm_al) & set(ts))
        dist = np.array([np.linalg.norm(tm_al[f] - ts[f]) for f in shared]) if shared else np.zeros(0)
        report["fish"] = {"mono_scaled": _path_stats(tm_al, fmap.proxy_fps), "stereo": _path_stats(ts, fmap.proxy_fps),
                          "pointwise_distance_cm": {"median": round(100 * float(np.median(dist)), 2),
                                                    "p90": round(100 * float(np.percentile(dist, 90)), 2)}
                          if dist.size else None}
        traj_xyz = {"mono": np.stack([tm_al[f] for f in sorted(tm_al)]), "stereo": np.stack([ts[f] for f in sorted(ts)])}

    (out_dir / "report.json").write_text(json.dumps(report, indent=1))

    # overlay PLY: mono tinted blue, stereo tinted orange
    tint = lambda pc, c: 0.5 * np.asarray(pc.colors) + 0.5 * np.array(c)  # noqa: E731
    ov = o3d.geometry.PointCloud()
    ov.points = o3d.utility.Vector3dVector(np.vstack([Pm, Ps]))
    ov.colors = o3d.utility.Vector3dVector(np.vstack([tint(pm, [0.1, 0.4, 1.0]), tint(ps, [1.0, 0.55, 0.0])]))
    o3d.io.write_point_cloud(str(out_dir / "overlay.ply"), ov)

    write_compare_html(out_dir / "compare.html", run.root.name, pm, ps, transform_points(M, Cm), Cs,
                       t_centres, ratios, traj_xyz, report)
    print(json.dumps(report, indent=1))
    return report
