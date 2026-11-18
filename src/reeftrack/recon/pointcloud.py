"""Depth maps -> one clean, coloured reef point cloud.

Per keyframe we keep a pixel only if
  * the model is confident about it (drop the lowest `conf_percentile` %),
  * it is not among the farthest pixels (`max_depth_percentile`): far pixels are mostly
    open water / back-scatter, where depth is meaningless,
  * it is not on the tracked fish (a moving object would smear through the static reef).
The merged cloud is voxel-downsampled (one point per small cube; removes the massive
redundancy between overlapping views) and statistical outliers ("floaters") are removed.
"""

from __future__ import annotations

import numpy as np

from reeftrack.recon.geometry import camera_center, rotation_to_z, unproject


def robust_extent(points: np.ndarray, q: float = 1.0) -> float:
    """Diagonal of the 1st-99th percentile box: the scene size, ignoring stray far points."""
    lo, hi = np.percentile(points, [q, 100 - q], axis=0)
    return float(np.linalg.norm(hi - lo))


def frame_points(depth, conf, image, K, E, conf_percentile, max_depth_percentile, exclude=None):
    valid = np.isfinite(depth) & (depth > 0)
    if exclude is not None:
        valid &= ~exclude
    if not valid.any():
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    keep = valid & (conf >= np.percentile(conf[valid], conf_percentile))
    keep &= depth <= np.percentile(depth[valid], max_depth_percentile)
    pts = unproject(depth, K, E)[keep]
    return pts, image[keep]


def collect_points(recon, rc, fish_masks: dict[int, np.ndarray] | None = None):
    """Filtered, coloured world points of every keyframe of one (global) reconstruction."""
    P, C = [], []
    for i, f in enumerate(recon.frames):
        excl = fish_masks.get(int(f)) if fish_masks else None
        p, c = frame_points(recon.depth[i], recon.depth_conf[i], recon.images[i], recon.intrinsic[i],
                            recon.extrinsic[i], rc.conf_percentile, rc.max_depth_percentile, excl)
        P.append(p)
        C.append(c)
    return np.concatenate(P), np.concatenate(C)


def clean_cloud(P: np.ndarray, C: np.ndarray, rc):
    """Merge -> voxel downsample (relative to robust scene size) -> remove floaters."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64) / 255.0)
    # Voxel size is relative to the (robust) scene extent, so mono (arbitrary units) and
    # stereo (metres) clouds get the same relative density.
    pcd = pcd.voxel_down_sample(rc.voxel_size * robust_extent(P))
    pcd, _ = pcd.remove_statistical_outlier(rc.outlier_nb_neighbors, rc.outlier_std_ratio)
    return pcd


def build_cloud(recon, rc, fish_masks: dict[int, np.ndarray] | None = None, extra=()):
    """Point cloud from one reconstruction (+ optional `extra` (recon, masks) pairs, e.g. right camera)."""
    parts = [collect_points(recon, rc, fish_masks)] + [collect_points(r, rc, m) for r, m in extra]
    return clean_cloud(np.concatenate([p for p, _ in parts]), np.concatenate([c for _, c in parts]), rc)


def levelling_transform(pcd, cam_centers: np.ndarray) -> np.ndarray:
    """4x4 rigid transform making the dominant plane (seabed) horizontal (+z = up).

    'Up' is chosen as the side of the plane where the cameras (the diver) are.
    """
    plane, _ = pcd.segment_plane(distance_threshold=0.02 * robust_extent(np.asarray(pcd.points)),
                                 ransac_n=3, num_iterations=2000)
    n = np.asarray(plane[:3])
    if (cam_centers.mean(0) @ n + plane[3]) < 0:
        n = -n
    R = rotation_to_z(n)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.asarray(pcd.points).mean(0)  # centre the scene at the origin
    return T


def drop_above_cameras(pcd, cam_centers: np.ndarray, margin_frac: float, cam_percentile: float = 50.0):
    """Remove points above the typical camera height (+ margin). Needs a levelled scene (z up).

    In shallow water the underside of the surface acts as a mirror and gets reconstructed
    as a 'ceiling' just above the diver. A snorkeller's camera regularly touches the
    surface, so we use a percentile of camera heights rather than the maximum.
    Not appropriate for walls/overhangs filmed from below (set recon.drop_above_camera: null).
    """
    P = np.asarray(pcd.points)
    z_max = np.percentile(cam_centers[:, 2], cam_percentile) + margin_frac * robust_extent(P)
    return pcd.select_by_index(np.flatnonzero(P[:, 2] <= z_max))


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    return np.stack([camera_center(E) for E in extrinsics])


def surface_thickness(P: np.ndarray, radius: float, n: int = 20_000, min_pts: int = 10, seed: int = 0) -> np.ndarray:
    """How 'thick' (smudged) the surface is: RMS distance of the points within `radius` of a
    sample point from their best-fit plane. A crisp surface gives ~ the true roughness of
    the reef; misaligned overlapping views (seams, doubled surfaces) make it larger.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(P)
    idx = np.random.default_rng(seed).choice(len(P), min(n, len(P)), replace=False)
    out = []
    for nb in tree.query_ball_point(P[idx], r=radius):
        if len(nb) < min_pts:
            continue
        Q = P[nb] - P[nb].mean(0)
        out.append(np.sqrt(max(0.0, np.linalg.eigvalsh(Q.T @ Q)[0]) / len(nb)))
    return np.asarray(out)
