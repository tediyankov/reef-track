"""Small, tested geometry kernel (numpy only).

Conventions (same as VGGT / Wat3R / OpenCV):
  * camera axes: x right, y down, z forward (into the scene)
  * ``extrinsic`` E = [R | t] is *camera-from-world*: X_cam = R @ X_world + t
  * ``intrinsic`` K is the 3x3 pinhole matrix in pixels of the image the model saw
  * depth = z-coordinate of the point in the camera frame (not the ray length)
"""

from __future__ import annotations

import numpy as np


def pixel_rays(K: np.ndarray, h: int, w: int) -> np.ndarray:
    """(h, w, 3) camera-frame directions with z = 1 for every pixel centre."""
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    pix = np.stack([u, v, np.ones_like(u)], -1)
    return pix @ np.linalg.inv(K).T


def unproject(depth: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Depth map (h, w) -> world points (h, w, 3)."""
    cam = pixel_rays(K, *depth.shape) * depth[..., None]
    R, t = E[:3, :3], E[:3, 3]
    return (cam - t) @ R  # == R.T @ (X_cam - t), row-vector form


def unproject_pixel(u: float, v: float, depth: float, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    cam = np.linalg.inv(K) @ np.array([u, v, 1.0]) * depth
    R, t = E[:3, :3], E[:3, 3]
    return R.T @ (cam - t)


def project(X: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (n, 3) -> pixels (n, 2) and depths (n,)."""
    cam = X @ E[:3, :3].T + E[:3, 3]
    z = cam[:, 2]
    uv = (cam @ K.T)[:, :2] / z[:, None]
    return uv, z


def camera_center(E: np.ndarray) -> np.ndarray:
    return -E[:3, :3].T @ E[:3, 3]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform: dst ~= s * R @ src + t  (Umeyama, 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_umeyama(src, dst, iters: int = 5, keep: float = 0.8, with_scale: bool = True):
    """Trimmed Umeyama: refit on the `keep` fraction of best-fitting correspondences."""
    s, R, t = umeyama(src, dst, with_scale)
    for _ in range(iters):
        res = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        idx = np.argsort(res)[: max(3, int(keep * len(src)))]
        s, R, t = umeyama(src[idx], dst[idx], with_scale)
    return s, R, t


def sim3_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def transform_points(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ M[:3, :3].T + M[:3, 3]


def transform_extrinsic(E: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, float]:
    """Re-express a camera-from-world pose after the world is mapped by Sim3 M (X' = M X).

    Returns the new 3x4 extrinsic and the scale factor that depths must be multiplied by.
    """
    sR = M[:3, :3]
    s = np.cbrt(np.linalg.det(sR))
    R_w = sR / s
    t_w = M[:3, 3]
    R, t = E[:3, :3], E[:3, 3]
    R_new = R @ R_w.T
    t_new = s * t - R_new @ t_w
    return np.hstack([R_new, t_new[:, None]]), float(s)


def rotation_to_z(normal: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector `normal` onto +z (Rodrigues)."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    c = float(n @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def slerp_extrinsics(E0: np.ndarray, E1: np.ndarray, a: float) -> np.ndarray:
    """Interpolate two camera poses (rotation SLERP, camera-centre LERP)."""
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([E0[:3, :3], E1[:3, :3]]))
    R = Slerp([0, 1], rots)(a).as_matrix()
    c = (1 - a) * camera_center(E0) + a * camera_center(E1)
    return np.hstack([R, (-R @ c)[:, None]])
