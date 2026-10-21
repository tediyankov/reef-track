"""Geometry kernel, keyframe/chunk logic and chunk stitching on synthetic scenes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reeftrack.recon.backends import ChunkPrediction, model_input_size, preprocess
from reeftrack.recon.geometry import (
    camera_center,
    project,
    robust_umeyama,
    rotation_to_z,
    sim3_matrix,
    slerp_extrinsics,
    transform_extrinsic,
    transform_points,
    umeyama,
    unproject,
    unproject_pixel,
)
from reeftrack.recon.keyframes import make_chunks, pick_sharpest
from reeftrack.recon.pointcloud import frame_points
from reeftrack.recon.stitch import stitch

K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])


def pose(rotvec=(0, 0, 0), center=(0, 0, 0)):
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return np.hstack([R, (-R @ np.asarray(center, float))[:, None]])


def test_unproject_project_roundtrip():
    E = pose((0.1, -0.2, 0.05), (1, 2, 3))
    depth = np.random.default_rng(0).uniform(2, 6, (48, 64))
    X = unproject(depth, K, E).reshape(-1, 3)
    uv, z = project(X, K, E)
    vv, uu = np.mgrid[:48, :64]
    assert np.allclose(uv, np.stack([uu.ravel(), vv.ravel()], 1), atol=1e-6)
    assert np.allclose(z, depth.ravel())
    assert np.allclose(unproject_pixel(10, 20, depth[20, 10], K, E), unproject(depth, K, E)[20, 10])
    assert np.allclose(camera_center(E), [1, 2, 3])


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(500, 3))
    R = Rotation.from_rotvec([0.3, -0.5, 1.0]).as_matrix()
    dst = 2.5 * src @ R.T + [1, -2, 3]
    s, R2, t = umeyama(src, dst)
    assert s == pytest.approx(2.5) and np.allclose(R2, R) and np.allclose(t, [1, -2, 3])
    dst_noisy = dst.copy()
    dst_noisy[:50] += 10  # 10 % gross outliers
    s3, R3, _ = robust_umeyama(src, dst_noisy)
    assert s3 == pytest.approx(2.5, rel=1e-3) and np.allclose(R3, R, atol=1e-3)


def test_transform_extrinsic_preserves_projection():
    E = pose((0.2, 0.1, -0.3), (0.5, 0, -1))
    M = sim3_matrix(3.0, Rotation.from_rotvec([0.4, 0.2, 0.1]).as_matrix(), np.array([2.0, 0, 1]))
    X = np.random.default_rng(2).normal(size=(20, 3)) + [0, 0, 5]
    uv, z = project(X, K, E)
    E2, s = transform_extrinsic(E, M)
    uv2, z2 = project(transform_points(M, X), K, E2)
    assert s == pytest.approx(3.0)
    assert np.allclose(uv, uv2) and np.allclose(z * s, z2)


def test_rotation_to_z():
    for n in [np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), np.array([1.0, 2, 3])]:
        R = rotation_to_z(n)
        assert np.allclose(R @ (n / np.linalg.norm(n)), [0, 0, 1])
        assert np.allclose(R @ R.T, np.eye(3))


def test_slerp_endpoints_and_midpoint():
    E0, E1 = pose((0, 0, 0), (0, 0, 0)), pose((0, 0.4, 0), (2, 0, 0))
    assert np.allclose(slerp_extrinsics(E0, E1, 0), E0)
    assert np.allclose(slerp_extrinsics(E0, E1, 1), E1)
    assert np.allclose(camera_center(slerp_extrinsics(E0, E1, 0.5)), [1, 0, 0])


def test_make_chunks_cover_and_overlap():
    chunks = make_chunks(100, 32, 8)
    assert chunks[0][0] == 0 and chunks[-1][-1] == 99
    assert all(len(c) == 32 for c in chunks)
    for a, b in zip(chunks[:-1], chunks[1:]):
        assert len(np.intersect1d(a, b)) >= 8
    assert len(make_chunks(10, 32, 8)) == 1
    with pytest.raises(ValueError):
        make_chunks(100, 8, 8)


def test_pick_sharpest():
    s = np.array([1, 5, 2, 9, 0, 0, 3])
    assert list(pick_sharpest(s, 3)) == [1, 3, 6]


def test_model_input_size_matches_vggt_rule():
    assert model_input_size(1080, 1920) == (294, 518)
    assert model_input_size(2000, 1000) == (518, 518)  # tall images are centre-cropped
    out = preprocess([np.zeros((1080, 1920, 3), np.uint8)] * 2)
    assert out.shape == (2, 294, 518, 3)


def test_frame_points_filters():
    depth = np.full((10, 10), 5.0)
    depth[0, 0] = 100  # far "water" pixel
    conf = np.ones((10, 10))
    conf[1, 1] = 0.01
    img = np.zeros((10, 10, 3), np.uint8)
    fish = np.zeros((10, 10), bool)
    fish[5, 5] = True
    p, c = frame_points(depth, conf, img, K, pose(), conf_percentile=5, max_depth_percentile=95, exclude=fish)
    assert len(p) == 100 - 3 and len(c) == len(p)


def _synthetic_chunks():
    """Flat 'seabed' at z=5 seen by a camera sliding along x; chunk 2 in a different Sim3 frame."""
    h, w = 48, 64
    Eg = [pose((0, 0, 0.02 * i), (0.3 * i, 0, 0)) for i in range(9)]
    depth_g = []
    for E in Eg:  # ray-cast plane z=5 (world)
        R, t = E[:3, :3], E[:3, 3]
        c = camera_center(E)
        rays_w = (np.dstack(np.meshgrid(np.arange(w), np.arange(h))) @ np.eye(2, 3) + [0, 0, 1]) @ np.linalg.inv(K).T @ R
        lam = (5 - c[2]) / rays_w[..., 2]
        Xw = c + lam[..., None] * rays_w
        depth_g.append((Xw @ R.T + t)[..., 2])
    img = np.zeros((h, w, 3), np.uint8)

    def chunk(ids, M):  # express frames `ids` in a chunk frame X_chunk = M^-1 X_global
        Minv = np.linalg.inv(M)
        Es, ds = zip(*[transform_extrinsic(Eg[i], Minv) for i in ids])
        return ChunkPrediction(np.stack([img] * len(ids)), np.stack([depth_g[i] * ds[k] for k, i in enumerate(ids)]),
                               np.ones((len(ids), h, w)), np.stack(Es), np.stack([K] * len(ids)))

    M2 = sim3_matrix(0.5, Rotation.from_rotvec([0.1, 0.3, -0.2]).as_matrix(), np.array([1.0, -1, 2]))
    return Eg, depth_g, [(np.arange(0, 6), chunk(range(0, 6), np.eye(4))), (np.arange(3, 9), chunk(range(3, 9), M2))]


def test_stitch_recovers_global_frame():
    Eg, depth_g, chunks = _synthetic_chunks()
    g = stitch(chunks, conf_percentile=0)
    assert list(g.frames) == list(range(9))
    for i in range(9):
        assert np.allclose(camera_center(g.extrinsic[i]), camera_center(Eg[i]), atol=1e-6)
        assert np.allclose(g.depth[i], depth_g[i], rtol=1e-5)
    assert max(g.align_residual) < 1e-6


def test_drop_above_cameras_removes_surface_layer():
    import open3d as o3d

    from reeftrack.recon.pointcloud import drop_above_cameras

    rng = np.random.default_rng(0)
    seabed = np.c_[rng.uniform(-1, 1, (500, 2)), rng.normal(0, 0.02, 500)]
    surface = np.c_[rng.uniform(-1, 1, (200, 2)), np.full(200, 1.0)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([seabed, surface])))
    cams = np.array([[0, 0, 0.4], [0.5, 0, 0.45]])
    out = np.asarray(drop_above_cameras(pcd, cams, 0.02).points)
    assert len(out) == 500 and out[:, 2].max() < 0.5


def test_surface_thickness_detects_doubled_surface():
    from reeftrack.recon.pointcloud import surface_thickness

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1, (40000, 2))
    crisp = np.c_[xy, rng.normal(0, 0.001, len(xy))]
    doubled = crisp.copy()
    doubled[::2, 2] += 0.01  # half the points from a view misaligned by 1 cm
    t1 = np.median(surface_thickness(crisp, 0.03, n=2000))
    t2 = np.median(surface_thickness(doubled, 0.03, n=2000))
    assert t1 == pytest.approx(0.001, rel=0.2) and t2 > 4 * t1
