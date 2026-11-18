"""2D tracklet + known cameras/depth -> 3D trajectory with the expected positions."""

import numpy as np
import pytest

from reeftrack.config import load_config
from reeftrack.fusion.lift import interp_pose, lift_tracklet, measure_depth, summarise
from reeftrack.recon.backends import mask_to_model, to_model_pixels
from reeftrack.recon.geometry import camera_center
from reeftrack.tracking.masks import bbox, centroid
from reeftrack.tracking.tracklet import Observation, Segment, Tracklet

H, W = 294, 518  # chosen so proxy pixels == model pixels
K = np.array([[300.0, 0, 259], [0, 300.0, 147], [0, 0, 1]])
FISH_DEPTH, SEABED_DEPTH = 3.0, 5.0


def cam(x):
    return np.hstack([np.eye(3), np.array([[-x], [0], [0]])])  # camera at (x, 0, 0) looking +z


def fish_mask():
    m = np.zeros((H, W), bool)
    m[137:158, 239:280] = True  # ~ centred, 41 x 21 px
    return m


def make_scene(n_frames=41, kf_step=10, gap=None):
    kf = np.arange(0, n_frames, kf_step)
    depth = np.full((len(kf), H, W), SEABED_DEPTH, np.float32)
    depth[:, fish_mask()] = FISH_DEPTH
    E = np.stack([cam(0.1 * f) for f in kf])
    Ks = np.stack([K] * len(kf))
    tr = Tracklet(shape=(H, W))
    m = fish_mask()
    segs = [(0, n_frames - 1)] if gap is None else [(0, gap[0] - 1), (gap[1] + 1, n_frames - 1)]
    for sid, (a, b) in enumerate(segs):
        tr.segments.append(Segment(sid, a, (259, 147), a, b))
        for f in range(a, b + 1):
            tr.add_observation(Observation(f, sid, int(m.sum()), bbox(m), centroid(m), 5.0, 1.0), m)
    return tr, kf, depth, np.ones_like(depth), E, Ks


def test_model_pixel_mapping_1080p():
    u, v = to_model_pixels(1920, 1080, 1080, 1920)
    assert u == pytest.approx(518) and v == pytest.approx(294)
    assert mask_to_model(np.ones((1080, 1920), bool)).shape == (294, 518)


def test_measure_depth_ignores_background():
    depth = np.full((H, W), SEABED_DEPTH)
    depth[fish_mask()] = FISH_DEPTH
    d, n = measure_depth(depth, np.ones_like(depth), fish_mask())
    assert d == FISH_DEPTH and n == fish_mask().sum()
    assert np.isnan(measure_depth(depth, depth, np.zeros((H, W), bool))[0])


def test_interp_pose_between_keyframes():
    kf = np.array([0, 10])
    E = np.stack([cam(0), cam(1)])
    assert np.allclose(camera_center(interp_pose(kf, E, 5)), [0.5, 0, 0])
    assert interp_pose(kf, E, 11) is None


def test_lift_recovers_fish_positions():
    tr, kf, depth, conf, E, Ks = make_scene()
    fc = load_config({"fusion": {"mask_erode_px": 0}}).fusion
    rows = lift_tracklet(tr, kf, depth, conf, E, Ks, fc, image_size=518, fps=30)
    assert len(rows) == 41
    cx, cy = centroid(fish_mask())
    for r in rows:
        f = r["proxy_frame"]
        expected = np.array([0.1 * f + (cx - 259) / 300 * FISH_DEPTH, (cy - 147) / 300 * FISH_DEPTH, FISH_DEPTH])
        assert np.allclose([r["x"], r["y"], r["z"]], expected, atol=1e-6)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"measured", "depth_interp"}
    assert sum(r["kind"] == "measured" for r in rows) == len(kf)


def test_short_gap_is_bridged_long_gap_is_not():
    tr, kf, depth, conf, E, Ks = make_scene(gap=(15, 18))
    fc = load_config({"fusion": {"max_interp_gap_s": 1.0}}).fusion
    rows = lift_tracklet(tr, kf, depth, conf, E, Ks, fc, fps=30)
    assert sorted(r["proxy_frame"] for r in rows if r["kind"] == "gap_interp") == [15, 16, 17, 18]
    fc_strict = load_config({"fusion": {"max_interp_gap_s": 0.05}}).fusion
    rows = lift_tracklet(tr, kf, depth, conf, E, Ks, fc_strict, fps=30)
    assert not any(r["kind"] == "gap_interp" for r in rows)


def test_summary():
    tr, kf, depth, conf, E, Ks = make_scene()
    rows = lift_tracklet(tr, kf, depth, conf, E, Ks, load_config().fusion, fps=30)
    s = summarise(rows, fps=30, n_frames=41)
    assert s["coverage_frac"] == 1.0
    assert s["path_length"] == pytest.approx(4.0, rel=1e-3)  # camera (and fish) moves 0.1 / frame


def test_height_above_reef():
    from reeftrack.fusion.lift import height_above_reef

    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (5000, 2))
    reef = np.c_[xy, 0.2 * xy[:, 0]]  # sloping reef surface z = 0.2 x
    fish = np.array([[0.5, 0.0, 0.4], [-0.5, 0.0, 0.0], [5.0, 5.0, 1.0]])
    h = height_above_reef(fish, reef, radius=0.1)
    assert h[0] == pytest.approx(0.3, abs=0.01) and h[1] == pytest.approx(0.1, abs=0.01)
    assert np.isnan(h[2])  # no reef underneath
