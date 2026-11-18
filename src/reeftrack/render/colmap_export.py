"""Export a run's reconstruction as a COLMAP scene for Gaussian-splatting / NeRF training.

    <out>/
      images/0000_f00008.jpg ...    keyframes, all warped to ONE shared pinhole camera
      sparse/0/cameras.bin|txt      that camera (PINHOLE, id 1)
      sparse/0/images.bin|txt       camera-from-world pose per image (COLMAP convention)
      sparse/0/points3D.bin|txt     a subsample of reef.ply (initialises the Gaussians)
      test_images.txt               the held-out views (every 8th image, sorted by name)
      export.json                   provenance + I2-NeRF near/far

Why one shared camera: Wat3R estimates the focal length per frame (it varies by about +-10%),
and I2-NeRF's loader only supports a single camera. Changing only the intrinsics of a pinhole
camera is an exact image warp (H = K_shared K_i^-1), so every image is resampled to the run's
*largest* focal length: a slight zoom-in, never black borders. Poses are untouched.

Held-out views: both trainers (nerfstudio's `colmap` dataparser, I2-NeRF's LLFF loader) put
image i of the name-sorted list in the test set when i % 8 == 0; image ids follow name order so
both agree. Those frames have exact Wat3R poses, so renders line up with the real frame.

Binary files are written because I2-NeRF's text reader is Python-2-era code that stops at the
first blank line; nerfstudio reads either. Text copies are kept for humans.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LLFF_HOLD = 8  # nerfstudio eval_interval / I2-NeRF llffhold defaults


# ---------------------------------------------------------------------------- geometry
def full_res_intrinsics(K_model: np.ndarray, model_hw, frame_hw) -> np.ndarray:
    """Model-input intrinsics (after reef-track's resize/crop to Wat3R's input) -> original-frame pixels."""
    from reeftrack.recon.backends import _layout

    H0, W0 = frame_hw
    nw = int(model_hw[1])
    _, _, full_h, top = _layout(H0, W0, nw)
    A = np.array([[nw / W0, 0, 0], [0, full_h / H0, -top], [0, 0, 1.0]])  # full-res pixel -> model pixel
    return np.linalg.inv(A) @ K_model


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> COLMAP quaternion (w, x, y, z), w >= 0."""
    m = R
    K = np.array([
        [m[0, 0] - m[1, 1] - m[2, 2], 0, 0, 0],
        [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], 0, 0],
        [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], 0],
        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]],
    ]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def pca_scale(centres: np.ndarray) -> float:
    """The factor I2-NeRF's `transform_poses_pca` multiplies the scene by (non-forward-facing mode)."""
    t = centres - centres.mean(0)
    _, vecs = np.linalg.eigh(t.T @ t)
    rotated = t @ vecs[:, ::-1]
    return float(1.0 / np.max(np.abs(rotated)))


# ---------------------------------------------------------------------------- COLMAP io
@dataclass
class ColmapImage:
    image_id: int
    name: str
    R: np.ndarray  # camera-from-world rotation
    t: np.ndarray


def write_colmap(sparse: Path, width: int, height: int, K: np.ndarray, images: list[ColmapImage],
                 points: np.ndarray, colours: np.ndarray) -> None:
    sparse.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # --- binary (COLMAP's standard layout, little-endian)
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, width, height))  # camera_id 1, model 1 = PINHOLE
        f.write(struct.pack("<4d", fx, fy, cx, cy))
    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<I4d3dI", im.image_id, *rotmat_to_qvec(im.R), *im.t, 1))
            f.write(im.name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D observations
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, (p, c) in enumerate(zip(points, colours), start=1):
            f.write(struct.pack("<Q3d3Bd", i, *p, *c, 0.0))
            f.write(struct.pack("<Q", 0))  # empty track
    # --- text copies (readable; nerfstudio prefers these when present)
    (sparse / "cameras.txt").write_text(
        f"# Camera list\n1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then an (empty) 2D-points line"]
    for im in images:
        q = rotmat_to_qvec(im.R)
        lines += [f"{im.image_id} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.9f}' for v in im.t)} 1 {im.name}", ""]
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")
    with open(sparse / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for i, (p, c) in enumerate(zip(points, colours), start=1):
            f.write(f"{i} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0\n")


def read_images_bin(path: Path) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """(id, name, R, t) per image: used to check the export round-trips."""
    out = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        s = struct.Struct("<I4d3dI")
        for _ in range(n):
            d = s.unpack(f.read(s.size))
            name = b"".join(iter(lambda: f.read(1), b"\x00")).decode()
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n2d)
            out.append((d[0], name, qvec_to_rotmat(np.array(d[1:5])), np.array(d[5:8])))
    return out


# ---------------------------------------------------------------------------- export
def export_colmap(run, out: Path, width: int = 960, max_points: int = 300_000, jpeg_quality: int = 95,
                  cameras: str = "left", split: str = "interval", log=print) -> dict:
    """`cameras`: "left" or "left+right" (the stereo partner's frames as extra *training* views).
    `split`: "interval" (every 8th name-sorted image is test: nerfstudio + I2-NeRF default) or
    "filename" (names start with train_/eval_: nerfstudio `--eval-mode filename`). With the right
    camera, use "filename": test views stay exactly the left frames of the left-only export, and
    the right frames taken at those instants are left out (7 cm away, they would leak the answer).
    """
    if cameras not in ("left", "left+right") or split not in ("interval", "filename"):
        raise ValueError(f"cameras={cameras!r} / split={split!r}")
    if cameras == "left+right" and split != "filename":
        raise ValueError("left+right needs split='filename' (keeps the test set = left frames only)")
    import cv2
    import open3d as o3d

    from reeftrack.io.video import FrameReader
    from reeftrack.recon.geometry import camera_center, project

    cams = np.load(run.recon_dir / "cameras.npz")
    frames, Km, E = cams["frames"].astype(int), cams["intrinsic"], cams["extrinsic"]
    reader = FrameReader(run.proxy)
    H0, W0 = reader[int(frames[0])].shape[:2]
    W = int(width)
    H = int(round(H0 * W / W0 / 2)) * 2
    S = np.diag([W / W0, H / H0, 1.0])
    K_i = np.stack([S @ full_res_intrinsics(k, cams["image_hw"], (H0, W0)) for k in Km])
    right_npz = run.recon_dir / "cameras_right.npz"
    camsR = np.load(right_npz) if right_npz.exists() and split == "filename" else None
    if cameras == "left+right" and camsR is None:
        raise FileNotFoundError(f"no right-camera poses in {run.recon_dir} (stereo reconstruction needed)")
    K_all = K_i
    if camsR is not None:  # shared camera from both, so left-only and left+right exports are identical on the left
        if not np.array_equal(camsR["frames"], frames):
            raise ValueError("left and right reconstructions have different keyframes")
        K_r = np.stack([S @ full_res_intrinsics(k, camsR["image_hw"], (H0, W0)) for k in camsR["intrinsic"]])
        K_all = np.concatenate([K_i, K_r])
    K = np.array([[K_all[:, 0, 0].max(), 0, K_all[:, 0, 2].mean()],
                  [0, K_all[:, 1, 1].max(), K_all[:, 1, 2].mean()], [0, 0, 1.0]])  # largest focal: zoom in only

    is_test = [i % LLFF_HOLD == 0 for i in range(len(frames))]  # frames are in time order = name order
    if split == "filename":
        names = [f"{'eval' if t else 'train'}_L_{i:04d}_f{int(f):05d}.jpg" for i, (f, t) in enumerate(zip(frames, is_test))]
    else:
        names = [f"{i:04d}_f{int(f):05d}.jpg" for i, f in enumerate(frames)]
    # (reader, intrinsics, extrinsics, name) for every exported image
    jobs = [(reader, K_i[i], E[i], names[i], int(f)) for i, f in enumerate(frames)]
    if cameras == "left+right":
        reader_r = FrameReader(run.proxy_right)
        jobs += [(reader_r, K_r[i], camsR["extrinsic"][i], f"train_R_{i:04d}_f{int(f):05d}.jpg", int(f))
                 for i, f in enumerate(frames) if not is_test[i]]

    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for j, (rd, Kj, Ej, name, f) in enumerate(jobs):
        img = cv2.resize(rd[f], (W, H), interpolation=cv2.INTER_AREA)
        Hw = K @ np.linalg.inv(Kj)
        img = cv2.warpPerspective(img, Hw, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(img_dir / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        images.append(ColmapImage(j + 1, name, Ej[:3, :3], Ej[:3, 3]))
        if j % 50 == 0:
            log(f"[export] {j + 1}/{len(jobs)} images")
    images.sort(key=lambda im: im.name)  # COLMAP ids in name order (nerfstudio's interval split follows ids)
    for k, im in enumerate(images, start=1):
        im.image_id = k

    pcd = o3d.io.read_point_cloud(str(run.recon_dir / "reef.ply"))
    P, C = np.asarray(pcd.points), np.asarray(pcd.colors)
    keep = np.random.default_rng(0).choice(len(P), min(len(P), max_points), replace=False)
    P, C = P[keep], (np.clip(C[keep], 0, 1) * 255).round().astype(np.uint8)
    write_colmap(out / "sparse" / "0", W, H, K, images, P, C)

    test = [n for n, t in zip(names, is_test) if t]
    if split == "interval":
        assert test == [n for i, n in enumerate(sorted(names)) if i % LLFF_HOLD == 0]
    (out / "test_images.txt").write_text("\n".join(test) + "\n")

    # depth range of the reef in view of each image: I2-NeRF's near/far (unbounded mode, in its
    # PCA-normalised units) and poses_bounds.npy (its forward-facing mode reads only the bounds)
    depths = []
    for i in range(len(frames)):
        uv, z = project(P, K, E[i])
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        depths.append(np.percentile(z[ok], [1, 99]) if ok.sum() > 100 else [np.nan, np.nan])
    d = np.array(depths)
    d = np.where(np.isnan(d), np.nanmedian(d, axis=0), d)
    pb = np.zeros((len(frames), 17))
    pb[:, -2:] = np.c_[0.9 * d[:, 0], 1.5 * d[:, 1]]
    np.save(out / "poses_bounds.npy", pb[np.argsort(names)])
    centres = np.stack([camera_center(e) for e in E])
    s = pca_scale(centres)
    near_m, far_m = float(0.5 * d[:, 0].min()), float(3.0 * d[:, 1].max())
    fm = run.frame_map()
    meta = {
        "run": str(run.root), "recon": run.recon_name, "cameras": cameras, "eval_mode": split,
        "width": W, "height": H, "n_images": len(images), "n_right_train": len(images) - len(names),
        "n_test": len(test), "n_points": int(len(P)), "llffhold": LLFF_HOLD,
        "shared_intrinsics": K.round(4).tolist(),
        "per_frame_fx_range": [round(float(K_i[:, 0, 0].min()), 2), round(float(K_i[:, 0, 0].max()), 2)],
        "units": "metres" if run.recon_name == "recon_stereo" else "arbitrary (mono)",
        "reef_depth_m": {"near_1pct": round(float(d[:, 0].min()), 3), "far_99pct": round(float(d[:, 1].max()), 3)},
        "i2nerf": {"pca_scale": round(s, 6), "near": round(near_m * s, 5), "far": round(far_m * s, 5)},
        "images": [{"name": n, "proxy_frame": int(f), "source_frame": int(fm.to_source(int(f))),
                    "time_s": round(float(fm.time_s(int(f))), 3), "test": n in test,
                    "camera": "right" if "_R_" in n else "left"} for (_, _, _, n, f) in jobs],
    }
    (out / "export.json").write_text(json.dumps(meta, indent=1))
    log(f"[export] {len(images)} images ({len(test)} held out, {len(images) - len(names)} right-camera), "
        f"{len(P):,} points, {W}x{H} -> {out}")
    return meta


def reprojection_check(out: Path, run, n: int = 4) -> list[Path]:
    """Draw the exported points onto a few exported images using only the exported COLMAP data."""
    import cv2

    from reeftrack.recon.geometry import project

    with open(out / "sparse/0/cameras.bin", "rb") as f:
        f.read(8)
        _, _, W, H = struct.unpack("<iiQQ", f.read(24))
        fx, fy, cx, cy = struct.unpack("<4d", f.read(32))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    P = np.loadtxt(out / "sparse/0/points3D.txt", comments="#", usecols=(1, 2, 3, 4, 5, 6))
    ims = read_images_bin(out / "sparse/0/images.bin")
    paths = []
    for _, name, R, t in ims[:: max(1, len(ims) // n)][:n]:
        img = cv2.imread(str(out / "images" / name))
        uv, z = project(P[:, :3], K, np.c_[R, t])
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        order = np.flatnonzero(ok)[np.argsort(-z[ok])]  # far first, near points overwrite
        splat = np.zeros_like(img)
        u, v = uv[order, 0].astype(int), uv[order, 1].astype(int)
        bgr = P[order][:, [5, 4, 3]].astype(np.uint8)
        for du in (0, 1):
            for dv in (0, 1):
                splat[v + dv, u + du] = bgr
        p = out / "checks" / f"reproject_{name}"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), np.hstack([img, splat, cv2.addWeighted(img, 0.5, splat, 0.5, 0)]))
        paths.append(p)
    return paths
