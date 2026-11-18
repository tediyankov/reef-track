"""Stereo rig: pair splitting, metric rescaling via the baseline, drift-free stitching."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reeftrack.recon.backends import ChunkPrediction
from reeftrack.recon.geometry import camera_center, sim3_matrix, transform_extrinsic
from reeftrack.recon.stereo import assemble, metric_chunk, rig_stats, split_pairs
from reeftrack.recon.stitch import stitch

K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
H, W = 48, 64
BASELINE = 0.07  # metres


def pose(rotvec, center):
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return np.hstack([R, (-R @ np.asarray(center, float))[:, None]])


def rig_poses(n):
    """Left cameras slide along x looking at a seabed 1.5 m away; right lens 7 cm to the left camera's right."""
    left, right = [], []
    for i in range(n):
        rv = (0.0, 0.0, 0.03 * i)
        EL = pose(rv, (0.25 * i, 0.05 * np.sin(i), 0.0))
        R = EL[:3, :3]
        cR = camera_center(EL) + R.T @ np.array([BASELINE, 0, 0])  # +x in the left camera frame
        left.append(EL)
        right.append(np.hstack([R, (-R @ cR)[:, None]]))
    return left, right


def seabed_depth(E, z_world=1.5):
    R, t = E[:3, :3], E[:3, 3]
    c = camera_center(E)
    pix = np.dstack([*np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), np.ones((H, W))])
    rays_w = pix @ np.linalg.inv(K).T @ R
    lam = (z_world - c[2]) / rays_w[..., 2]
    return ((c + lam[..., None] * rays_w) @ R.T + t)[..., 2]


def chunk_from(Es_world, ids, M):
    """Interleaved L/R chunk expressed in an arbitrary chunk frame X_chunk = M^-1 X_world."""
    Minv = np.linalg.inv(M)
    Es, ds = [], []
    for E in Es_world:
        e, s = transform_extrinsic(E, Minv)
        Es.append(e)
        ds.append(seabed_depth(E) * s)
    n = len(Es)
    return ChunkPrediction(np.zeros((n, H, W, 3), np.uint8), np.stack(ds), np.ones((n, H, W)),
                           np.stack(Es), np.stack([K] * n))


def interleave(left, right, ids):
    out = []
    for i in ids:
        out += [left[i], right[i]]
    return out


def test_split_pairs_and_rig_stats():
    left, right = rig_poses(4)
    pred = chunk_from(interleave(left, right, range(4)), range(4), np.eye(4))
    L, R = split_pairs(pred)
    assert np.allclose(L.extrinsic, np.stack(left)) and np.allclose(R.extrinsic, np.stack(right))
    st = rig_stats(L, R)
    assert st["baseline_raw_median"] == pytest.approx(BASELINE)
    assert st["baseline_cv"] < 1e-9 and st["rel_rot_deg_median"] < 1e-6
    assert np.allclose(st["baseline_dir_left_cam"], [1, 0, 0], atol=1e-3)


def test_metric_chunk_recovers_metres():
    left, right = rig_poses(4)
    M = sim3_matrix(1 / 3.2, Rotation.from_rotvec([0.2, 0, 0.1]).as_matrix(), np.array([0.5, 0, 0]))
    pred = chunk_from(interleave(left, right, range(4)), range(4), M)  # model output at arbitrary scale
    L, R, st = metric_chunk(pred, BASELINE)
    assert st["scale_to_metres"] == pytest.approx(1 / 3.2)
    for EL, ER in zip(L.extrinsic, R.extrinsic):
        assert np.linalg.norm(camera_center(ER) - camera_center(EL)) == pytest.approx(BASELINE)
    assert np.allclose(L.depth[0], seabed_depth(left[0]))  # depths now in metres


def test_stereo_stitch_removes_scale_drift():
    """Two chunks returned at *different* arbitrary scales (= monocular scale drift)."""
    n = 9
    left, right = rig_poses(n)
    ids_a, ids_b = np.arange(0, 6), np.arange(3, 9)
    Ma = sim3_matrix(2.0, np.eye(3), np.zeros(3))
    Mb = sim3_matrix(0.4, Rotation.from_rotvec([0.1, 0.3, -0.2]).as_matrix(), np.array([1.0, -1, 2]))
    chunks_l, chunks_r = [], []
    for ids, M in [(ids_a, Ma), (ids_b, Mb)]:
        L, R, _ = metric_chunk(chunk_from(interleave(left, right, ids), ids, M), BASELINE)
        chunks_l.append((ids, L))
        chunks_r.append((ids, R))
    gl = stitch(chunks_l, conf_percentile=0, with_scale=False)
    gr = assemble(chunks_r, gl.chunk_transforms)
    # global frame = first chunk's first camera frame, in metres: compare relative geometry
    true_c = np.stack([camera_center(E) for E in left])
    est_c = np.stack([camera_center(E) for E in gl.extrinsic])
    true_d = np.linalg.norm(true_c - true_c[0], axis=1)
    est_d = np.linalg.norm(est_c - est_c[0], axis=1)
    assert np.allclose(est_d, true_d, atol=1e-6)  # metric distances along the path, no drift
    base = np.linalg.norm(np.stack([camera_center(E) for E in gr.extrinsic]) - est_c, axis=1)
    assert np.allclose(base, BASELINE, atol=1e-6)
    assert max(gl.align_residual) < 1e-6


def test_reconcile_scales_beats_independent_estimates():
    """Noisy per-chunk baselines + precise overlaps -> joint estimate close to the truth."""
    from reeftrack.recon.stereo import reconcile_scales

    rng = np.random.default_rng(0)
    n = 36
    true_lam = np.cumsum(rng.normal(0, 0.3, n))  # the model's raw scale wanders chunk to chunk
    base = true_lam + rng.normal(0, 0.25, n)  # each chunk's own baseline estimate: noisy
    joins = np.diff(true_lam) + rng.normal(0, 0.01, n - 1)  # overlaps: precise relative scale
    lam = reconcile_scales(base, np.full(n, 0.25**2), joins, np.full(n - 1, 0.01**2))
    err_joint = np.abs(lam - true_lam).max()
    err_indep = np.abs(base - true_lam).max()
    assert err_joint < 0.15 and err_joint < err_indep / 3
    assert np.abs(np.diff(lam) - np.diff(true_lam)).max() < 0.05  # neighbours stay consistent


def test_reconcile_scales_exact_without_noise():
    from reeftrack.recon.stereo import reconcile_scales

    lam_true = np.array([0.0, 0.5, -0.2, 0.3])
    lam = reconcile_scales(lam_true, np.ones(4), np.diff(lam_true), np.full(3, 1e-4))
    assert np.allclose(lam, lam_true)


def test_variance_estimation_recovers_from_wrong_noise_guesses():
    """Hand-set noise levels that over-trust the baselines (as in v2) are corrected from the data."""
    from reeftrack.recon.stereo import reconcile_scales

    rng = np.random.default_rng(1)
    n, sb, sj = 60, 0.25, 0.01
    true_lam = np.cumsum(rng.normal(0, 0.3, n))
    base = true_lam + rng.normal(0, sb, n)
    joins = np.diff(true_lam) + rng.normal(0, sj, n - 1)
    wrong_b, wrong_j = np.full(n, 0.04**2), np.full(n - 1, 0.1**2)  # baselines trusted 40x too much
    naive = reconcile_scales(base, wrong_b, joins, wrong_j)
    learned, f = reconcile_scales(base, wrong_b, joins, wrong_j, estimate_variances=True, return_factors=True)
    seam = lambda lam: np.abs(np.diff(lam) - np.diff(true_lam)).max()  # noqa: E731
    assert seam(learned) < seam(naive) / 3  # neighbours consistent again -> no seams
    assert np.sqrt(f[0]) * 0.04 == pytest.approx(sb, rel=0.35)  # learned baseline noise ~ true 0.25
    assert np.abs(learned - true_lam).mean() < np.abs(base - true_lam).mean()
