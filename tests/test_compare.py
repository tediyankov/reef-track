"""Mono-vs-stereo comparison maths."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reeftrack.fusion.compare import align_paths, local_scale_profile


def test_align_paths_recovers_global_scale():
    rng = np.random.default_rng(0)
    Cs = np.cumsum(rng.normal(size=(50, 3)) * 0.1, axis=0)  # stereo path, metres
    R = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    Cm = (Cs - [1, 2, 3]) @ R / 0.25  # mono = arbitrary frame, 1 unit = 0.25 m
    frames = np.arange(50)
    M, common, ia, ib, res = align_paths(frames, Cm, frames, Cs)
    assert np.cbrt(np.linalg.det(M[:3, :3])) == pytest.approx(0.25)
    assert res.max() < 1e-9 and len(common) == 50


def test_local_scale_profile_detects_drift():
    t = np.linspace(0, 10, 100)
    Cs = np.c_[t, np.sin(t), 0.1 * t ** 1.5]
    drift = np.linspace(1.0, 2.0, 100)[:, None]  # mono scale wanders by x2 along the path
    Cm = np.cumsum(np.r_[Cs[:1], np.diff(Cs, axis=0) * drift[1:]], axis=0)
    _, ratios = local_scale_profile(Cm, Cs, s_global=1.0, window=10)
    assert ratios[0] > ratios[-1] * 1.5  # stereo/mono scale shrinks as mono over-scales later on
    _, flat = local_scale_profile(Cs, Cs, 1.0, 10)
    assert np.allclose(flat, 1.0)
