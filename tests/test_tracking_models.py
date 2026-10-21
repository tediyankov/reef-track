"""Motion (Kalman), appearance (colour signature) and loss-detection logic."""

import numpy as np
import pytest

from reeftrack.tracking.appearance import AppearanceModel, colour_signature, similarity
from reeftrack.tracking.motion import BoxKalman
from reeftrack.tracking.quality import FrameSignals, LossDetector


def test_kalman_learns_constant_velocity():
    kf = BoxKalman(np.array([0.0, 0.0, 20.0, 10.0]))
    for t in range(1, 30):
        kf.predict()
        kf.update(np.array([3.0 * t, 1.0 * t, 20.0, 10.0]))
    pred = kf.predict()
    assert pred[0] == pytest.approx(90.0, abs=1.0)
    assert pred[1] == pytest.approx(30.0, abs=1.0)


def test_kalman_gate_flags_jumps():
    kf = BoxKalman(np.array([100.0, 100.0, 20.0, 20.0]))
    for _ in range(10):
        kf.predict()
        kf.update(np.array([100.0, 100.0, 20.0, 20.0]))
    kf.predict()
    assert kf.mahalanobis(np.array([101.0, 100.0, 20, 20])) < 2
    assert kf.mahalanobis(np.array([300.0, 100.0, 20, 20])) > 20


def _patch(colour):
    img = np.zeros((20, 20, 3), np.uint8)
    img[:] = colour
    return img


def test_colour_signature_similarity():
    m = np.ones((20, 20), bool)
    a = colour_signature(_patch((200, 30, 30)), m)
    b = colour_signature(_patch((200, 32, 30)), m)
    c = colour_signature(_patch((30, 30, 200)), m)
    assert similarity(a, b) > 0.9
    assert similarity(a, c) < 0.2
    assert colour_signature(_patch((1, 2, 3)), np.zeros((20, 20), bool)) is None


def test_appearance_model_keeps_template():
    m = np.ones((20, 20), bool)
    a = colour_signature(_patch((200, 30, 30)), m)
    model = AppearanceModel(a)
    c = colour_signature(_patch((30, 30, 200)), m)
    for _ in range(50):  # running template drifts towards c...
        model.update(c)
    assert model.score(a) > 0.9  # ...but the click template still recognises the original
    assert model.score(None) == 0.0


def sig(area=100, x=50, presence=5.0, sim=0.9):
    return FrameSignals(area=area, bbox=(x, 50, x + 9, 59), presence_logit=presence, appearance_sim=sim,
                        frame_area=10_000)


def test_detector_patience():
    d = LossDetector(patience=3, min_area_frac=0)
    d.initialise(sig())
    assert d.assess(sig()).ok
    v1 = d.assess(sig(presence=-1))
    v2 = d.assess(sig(presence=-1))
    assert not v1.ok and not v1.broke and not v2.broke
    assert d.assess(sig()).ok  # recovered -> streak reset
    for _ in range(2):
        assert not d.assess(sig(presence=-1)).broke
    assert d.assess(sig(presence=-1)).broke


@pytest.mark.parametrize(
    "bad, flag",
    [
        (dict(area=0), "empty"),
        (dict(area=1000), "area"),
        (dict(sim=0.1), "appearance"),
        (dict(presence=-3), "presence"),
        (dict(x=90), "motion"),
    ],
)
def test_detector_flags(bad, flag):
    d = LossDetector(patience=5, min_area_frac=1e-4)
    d.initialise(sig())
    for _ in range(5):
        assert d.assess(sig()).ok
    s = sig(**bad)
    if bad.get("area") == 0:
        s.bbox = None
    assert flag in d.assess(s).flags
