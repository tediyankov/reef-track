"""`reeftrack reconstruct`: keyframes -> chunked Wat3R/VGGT -> stitched -> levelled point cloud.

Outputs (recon/):
  keyframes.json  chosen keyframes + sharpness, chunk layout, alignment residuals
  cameras.npz     frames, intrinsic (N,3,3), extrinsic (N,3,4) camera-from-world, image size
  depth.npz       depth & confidence (float16) per keyframe, in world units
  reef.ply        cleaned, coloured reef point cloud (world frame: z up, seabed ~ z=0)
"""

from __future__ import annotations

import json
import logging
import time

import cv2
import numpy as np

from reeftrack.devices import resolve_device
from reeftrack.io.video import FrameReader
from reeftrack.recon.backends import FeedForwardRecon, mask_to_model, preprocess
from reeftrack.recon.geometry import transform_extrinsic
from reeftrack.recon.keyframes import make_chunks, select_keyframes
from reeftrack.recon.pointcloud import build_cloud, camera_centers, drop_above_cameras, levelling_transform
from reeftrack.recon.stitch import GlobalRecon, stitch
from reeftrack.tracking.masks import rle_decode

log = logging.getLogger(__name__)


class _FrameRange:
    """View of frames [lo, hi] of a reader, re-indexed from 0."""

    def __init__(self, reader, lo: int, hi: int):
        if not 0 <= lo <= hi < len(reader):
            raise ValueError(f"recon.frame_range [{lo}, {hi}] outside clip of {len(reader)} frames")
        self.reader, self.lo, self.n = reader, lo, hi - lo + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.reader[self.lo + i]


def _fish_masks(run, frames: np.ndarray, target: int, dilate_px: int = 3) -> dict[int, np.ndarray]:
    """Tracked fish masks at model resolution (slightly dilated), for keyframes that have one."""
    p = run.tracking_dir / "masks.npz"
    if not p.exists():
        return {}
    meta = json.loads((run.tracking_dir / "tracklet.json").read_text())
    shape = tuple(meta["shape"])
    out = {}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    with np.load(p) as z:
        for f in frames:
            key = f"m{int(f)}"
            if key in z.files:
                m = mask_to_model(rle_decode(z[key], shape), target).astype(np.uint8)
                out[int(f)] = cv2.dilate(m, k).astype(bool)
    return out


def apply_world_transform(recon: GlobalRecon, T: np.ndarray) -> None:
    for i in range(len(recon.frames)):
        recon.extrinsic[i], _ = transform_extrinsic(recon.extrinsic[i], T)


def save_recon(run, recon: GlobalRecon) -> None:
    d = run.recon_dir
    np.savez_compressed(d / "cameras.npz", frames=recon.frames, intrinsic=recon.intrinsic,
                        extrinsic=recon.extrinsic, image_hw=np.array(recon.depth.shape[1:]))
    np.savez_compressed(d / "depth.npz", frames=recon.frames, depth=recon.depth.astype(np.float16),
                        depth_conf=recon.depth_conf.astype(np.float16))


def load_recon(run, with_images: bool = False) -> GlobalRecon:
    d = run.recon_dir
    cams = np.load(d / "cameras.npz")
    dep = np.load(d / "depth.npz")
    images = None
    if with_images:
        with FrameReader(run.proxy) as r:
            images = preprocess([r[int(f)] for f in cams["frames"]], int(cams["image_hw"][1]))
    return GlobalRecon(cams["frames"], images, dep["depth"].astype(np.float32), dep["depth_conf"].astype(np.float32),
                       cams["extrinsic"], cams["intrinsic"], [])


def _preview(run) -> None:
    from reeftrack.fusion.viewer import write_scene

    write_scene(run, [], html_path=run.recon_dir / "reef.html", write_ply=False)


def rebuild_cloud(run, cfg) -> None:
    """Re-run only cleaning/merging from the saved depth maps (seconds instead of minutes)."""
    import open3d as o3d

    from reeftrack.recon.stereo import load_right

    recon = load_recon(run, with_images=True)
    right = load_right(run, with_images=True)  # stereo variant: include the right camera's depth too
    fish = _fish_masks(run, recon.frames, cfg.recon.image_size) if cfg.recon.mask_fish else {}
    pcd = build_cloud(recon, cfg.recon, fish, extra=[(right, None)] if right is not None else ())
    if cfg.recon.drop_above_camera is not None:  # saved cameras are already levelled
        pcd = drop_above_cameras(pcd, camera_centers(recon.extrinsic), cfg.recon.drop_above_camera,
                                 cfg.recon.camera_height_percentile)
    o3d.io.write_point_cloud(str(run.recon_dir / "reef.ply"), pcd)
    meta_p = run.recon_dir / "keyframes.json"
    meta = json.loads(meta_p.read_text())
    meta.update(n_points=len(pcd.points), fish_masked_keyframes=len(fish), cloud_config_digest=cfg.digest())
    meta_p.write_text(json.dumps(meta, indent=1))
    _preview(run)
    print(f"[recon] rebuilt cloud: {len(pcd.points):,} points ({len(fish)} keyframes with fish masked)")


def reconstruct(run, cfg) -> None:
    import open3d as o3d

    rc = cfg.recon
    device = resolve_device(rc.device)
    t0 = time.time()
    reader = FrameReader(run.proxy)
    fmap = run.frame_map()

    lo, hi = rc.frame_range or (0, len(reader) - 1)
    sub = _FrameRange(reader, lo, hi)
    kf, sharp = select_keyframes(sub, fmap.proxy_fps, rc.keyframe_fps, rc.sharpness_window, progress=True)
    kf, sharp = kf + lo, np.r_[np.full(lo, np.nan), sharp]
    chunks_idx = make_chunks(len(kf), rc.chunk_size, rc.chunk_overlap)
    print(f"[recon] {len(kf)} keyframes in {len(chunks_idx)} chunks of {rc.chunk_size} "
          f"(overlap {rc.chunk_overlap}) with {rc.backend} on {device}")

    model = FeedForwardRecon(rc.backend, rc.model_ids[rc.backend], device, rc.image_size)
    preds = []
    for ci, idx in enumerate(chunks_idx):
        tc = time.time()
        frames = [reader[int(kf[i])] for i in idx]
        preds.append((kf[idx], model(frames)))
        print(f"[recon] chunk {ci + 1}/{len(chunks_idx)}  ({time.time() - tc:.1f}s)")
        if device == "mps":
            import torch

            torch.mps.empty_cache()
    del model

    recon = stitch(preds, conf_percentile=50)
    fish = _fish_masks(run, recon.frames, rc.image_size) if rc.mask_fish else {}
    pcd = build_cloud(recon, rc, fish)
    T = levelling_transform(pcd, camera_centers(recon.extrinsic))
    pcd.transform(T)
    apply_world_transform(recon, T)
    if rc.drop_above_camera is not None:
        pcd = drop_above_cameras(pcd, camera_centers(recon.extrinsic), rc.drop_above_camera, rc.camera_height_percentile)

    save_recon(run, recon)
    o3d.io.write_point_cloud(str(run.recon_dir / "reef.ply"), pcd)
    (run.recon_dir / "keyframes.json").write_text(json.dumps({
        "backend": rc.backend, "model_id": rc.model_ids[rc.backend], "device": device,
        "config_digest": cfg.digest(), "keyframes": kf.tolist(),
        "sharpness": [None if np.isnan(s) else round(float(s), 2) for s in sharp[kf]],
        "chunks": [c.tolist() for c in chunks_idx], "align_residuals": recon.align_residual,
        "n_points": len(pcd.points), "fish_masked_keyframes": len(fish), "world_transform": T.tolist(),
        "runtime_s": round(time.time() - t0, 1),
    }, indent=1))
    _preview(run)
    print(f"[recon] {len(pcd.points):,} points -> {run.recon_dir / 'reef.ply'}  ({time.time() - t0:.0f}s)")
