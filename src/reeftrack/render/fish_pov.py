"""A fish's-eye swim-through camera path, for rendering a trained reef scene (WaterSplatting)
along the route the fish took.

    python -m reeftrack.render.fish_pov data/runs/valerio_GX020083_loop \
        --export data/render_exports/valerio_GX020083_loop

Writes <export>/fish_pov/:
    fish_pov_path.json   camera-to-world poses (OpenCV axes: x right, y down, z forward) in the
                         reconstruction's world frame (metres), + the export's shared intrinsics
    preview.mp4          the same path flown through the reef point cloud (sanity check, laptop)
    map.png              top view: reef, the whole fish track, the chosen stretch

Camera placement: the stereo reconstruction is reef-aligned (z up, reef around z = 0), so the
camera follows the fish's smoothed track in x/y, looks along its smoothed heading tilted slightly
down, and flies `clearance` metres above the local reef top (the fish itself hugs the reef; a
camera at its exact position would be inside the coral). The JASMIN side
(scripts/jasmin/render_fish_pov.py) maps these poses into the trained model's frame.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 (edge-padded), sigma in samples."""
    if sigma <= 0:
        return x.copy()
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def load_track(path: Path):
    rows = list(csv.DictReader(open(path, newline="")))
    t = np.array([float(r["time_s"]) for r in rows])
    X = np.array([[float(r[k]) for k in "xyz"] for r in rows])
    seg = np.array([int(r["segment_id"]) for r in rows])
    kind = np.array([r["kind"] for r in rows])
    return t, X, seg, kind


def pick_window(t, X, seg, kind, span_s: float) -> tuple[float, float]:
    """Start/end time of the stretch that is one tracking segment, fully tracked (no gap fill),
    and the most purposeful swim (net displacement / path length, times distance covered)."""
    best, best_w = -1.0, None
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + span_s)
        if j >= len(t):
            break
        if len(set(seg[i:j + 1])) > 1 or np.any(kind[i:j + 1] == "gap_interp"):
            continue
        P = _gauss_smooth(X[i:j + 1, :2], 5)
        path = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        net = np.linalg.norm(P[-1] - P[0])
        score = (net / max(path, 1e-6)) * net
        if score > best:
            best, best_w = score, (t[i], t[j])
    if best_w is None:
        raise ValueError(f"no {span_s:.1f} s stretch of continuous tracking")
    return best_w


def reef_top(reef_xy_tree, reef_z, xy: np.ndarray, radius: float) -> np.ndarray:
    """90th-percentile reef height within `radius` of each xy (falls back to wider radii)."""
    out = np.empty(len(xy))
    for i, p in enumerate(xy):
        for r in (radius, 2 * radius, 4 * radius):
            idx = reef_xy_tree.query_ball_point(p, r)
            if len(idx) >= 20:
                out[i] = np.percentile(reef_z[idx], 90)
                break
        else:
            out[i] = np.nan
    if np.isnan(out).all():
        return np.zeros(len(xy))
    return np.where(np.isnan(out), np.nanmedian(out), out)


UP = np.array([0.0, 0.0, 1.0])  # reconstruction is reef-aligned: +z is up


def look_c2w(pos: np.ndarray, fwd: np.ndarray, up: np.ndarray = UP) -> np.ndarray:
    """4x4 camera-to-world, OpenCV axes (x right, y down, z forward)."""
    z = fwd / np.linalg.norm(fwd)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, pos
    return M


def build_path(t, X, reef_P, t0: float, t1: float, fps: float, speed: float, clearance: float,
               pitch_deg: float, smooth_s: float = 0.4, heading_s: float = 1.0):
    from scipy.spatial import cKDTree

    dt = np.median(np.diff(t))
    # smooth over the whole track (no edge effects inside the window), then resample
    Ps = _gauss_smooth(X, smooth_s / dt)
    Vh = np.gradient(_gauss_smooth(X[:, :2], heading_s / dt), axis=0) / dt  # rows are ~evenly spaced
    n = int(round((t1 - t0) / speed * fps))
    tq = t0 + np.arange(n) * speed / fps
    pos = np.stack([np.interp(tq, t, Ps[:, k]) for k in range(3)], 1)
    vh = np.stack([np.interp(tq, t, Vh[:, k]) for k in range(2)], 1)

    tree = cKDTree(reef_P[:, :2])
    ztop = reef_top(tree, reef_P[:, 2], pos[:, :2], radius=0.05)
    zc = _gauss_smooth(np.maximum(ztop, pos[:, 2])[:, None], 0.3 * fps)[:, 0] + clearance

    pitch = np.deg2rad(pitch_deg)
    poses = []
    for p, v, z in zip(pos, vh, zc):
        h = v / (np.linalg.norm(v) + 1e-9)
        fwd = np.array([h[0] * np.cos(pitch), h[1] * np.cos(pitch), -np.sin(pitch)])
        poses.append(look_c2w(np.array([p[0], p[1], z]), fwd))
    return tq, np.stack(poses), pos


def render_points(P, C, c2w, K, W, H, near=0.03):
    """Tiny z-buffered point splat (2x2 px) of the reef cloud: preview only."""
    R, c = c2w[:3, :3], c2w[:3, 3]
    Xc = (P - c) @ R  # world -> camera (R orthonormal)
    ok = Xc[:, 2] > near
    Xc, Cc = Xc[ok], C[ok]
    u = (K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]).astype(int)
    v = (K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]).astype(int)
    img = np.full((H, W, 3), 30, np.uint8)
    zbuf = np.full((H, W), np.inf)
    order = np.argsort(-Xc[:, 2])  # far first, near overwrite
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u[order] + du, v[order] + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[m], uu[m]] = Cc[order][m]
            zbuf[vv[m], uu[m]] = np.minimum(zbuf[vv[m], uu[m]], Xc[order][m, 2])
    return img, zbuf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--export", type=Path, required=True, help="the scene's render export folder")
    ap.add_argument("--seconds", type=float, default=4.0, help="length of the output video")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed vs real time")
    ap.add_argument("--start", type=float, default=None, help="video time (s) to start at; default: auto")
    ap.add_argument("--clearance", type=float, default=0.12, help="metres above the local reef top")
    ap.add_argument("--pitch", type=float, default=15.0, help="degrees the view tilts down")
    a = ap.parse_args(argv)

    import open3d as o3d

    t, X, seg, kind = load_track(a.run / "fusion_stereo" / "trajectory.csv")
    span = a.seconds * a.speed
    t0, t1 = (a.start, a.start + span) if a.start is not None else pick_window(t, X, seg, kind, span)
    pcd = o3d.io.read_point_cloud(str(a.run / "recon_stereo" / "reef.ply"))
    P, C = np.asarray(pcd.points), (np.clip(np.asarray(pcd.colors), 0, 1) * 255).astype(np.uint8)

    tq, poses, fish = build_path(t, X, P, t0, t1, a.fps, a.speed, a.clearance, a.pitch)
    meta = json.loads((a.export / "export.json").read_text())
    K = np.array(meta["shared_intrinsics"])
    W, H = meta["width"], meta["height"]

    out = a.export / "fish_pov"
    out.mkdir(exist_ok=True)
    (out / "fish_pov_path.json").write_text(json.dumps({
        "run": str(a.run), "convention": "camera-to-world, OpenCV axes (x right, y down, z forward), metres",
        "width": W, "height": H, "K": K.tolist(), "fps": a.fps, "speed": a.speed,
        "video_time_s": [round(float(t0), 3), round(float(t1), 3)],
        "clearance_m": a.clearance, "pitch_deg": a.pitch,
        "frames": [{"time_s": round(float(tt), 4), "c2w": M.round(6).tolist()} for tt, M in zip(tq, poses)],
    }, indent=1))

    # checks: how close the camera gets to reef geometry, how much of the view is filled
    from scipy.spatial import cKDTree
    d_min = cKDTree(P).query(poses[:, :3, 3])[0]
    w2, h2 = W // 2, H // 2
    K2 = K.copy()
    K2[:2] /= 2
    import shutil
    import subprocess
    import tempfile

    # encode to a local temp file, then move: writing a video straight into a OneDrive folder
    # can time out while the sync client holds the file
    tmp_mp4 = Path(tempfile.mkdtemp()) / "preview.mp4"
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w2}x{h2}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", str(tmp_mp4)], stdin=subprocess.PIPE)
    fill, close = [], []
    sub = np.random.default_rng(0).choice(len(P), min(len(P), 400_000), replace=False)
    for M in poses:
        img, zb = render_points(P[sub], C[sub], M, K2, w2, h2)
        fill.append(np.isfinite(zb).mean())
        close.append((zb < 0.10).mean())
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the preview")
    shutil.move(str(tmp_mp4), out / "preview.mp4")

    # top-view map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    s2 = sub[:60_000]
    ax.scatter(P[s2, 0], P[s2, 1], c=C[s2] / 255, s=1, linewidths=0)
    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()]  # outline: readable on the reef
    ax.plot(X[:, 0], X[:, 1], color="#ffd400", lw=1.4, label="whole fish track", path_effects=halo)
    ax.plot(poses[:, 0, 3], poses[:, 1, 3], color="#e8112d", lw=2.6, label="rendered stretch",
            path_effects=[pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()])
    for M in poses[:: max(1, len(poses) // 8)]:
        ax.arrow(M[0, 3], M[1, 3], 0.15 * M[0, 2], 0.15 * M[1, 2], color="tab:red", width=0.004)
    ax.set_aspect("equal")
    ax.set_title(f"Fish-POV path: video {t0:.1f}-{t1:.1f} s ({a.speed:g}x)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=130)

    seg_len = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()
    print(f"window: video {t0:.2f}-{t1:.2f} s -> {len(poses)} frames ({a.seconds:g} s at {a.fps:g} fps, {a.speed:g}x)")
    print(f"camera path {seg_len:.2f} m; height above reef top {a.clearance} m; pitch {a.pitch} deg down")
    print(f"nearest reef point: min {d_min.min():.3f} m, median {np.median(d_min):.3f} m")
    print(f"view filled by reef: median {np.median(fill):.0%}; pixels closer than 10 cm: max {max(close):.1%}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
