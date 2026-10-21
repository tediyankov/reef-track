"""Audio-based camera synchronisation."""

import numpy as np
import pytest

from reeftrack.io import sync


def _clicks(n, sr, seed):
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.normal(size=n)
    for t in rng.integers(0, n - 200, 40):  # sharp transients (fin kicks, bubbles)
        x[t : t + 50] += rng.normal(size=50)
    return x.astype(np.float32)


def test_gcc_phat_recovers_subsample_delay():
    sr = 16000
    a = _clicks(sr * 5, sr, 0)
    b = np.roll(a, 1234) + 0.05 * np.random.default_rng(1).normal(size=a.size).astype(np.float32)
    d, sharp = sync.gcc_phat(a, b, sr, max_lag_s=1.0)
    assert d == pytest.approx(1234 / sr, abs=1 / sr)
    assert sharp > 10


def test_estimate_offset_sign_convention(monkeypatch):
    """A sound at left time t appears at right time t - offset."""
    sr, true_offset = 16000, 1.037  # right camera started 1.037 s after the left one
    total = _clicks(sr * 120, sr, 2)  # the 'world' soundtrack, world time = left time

    def fake_read(path, start_s, dur_s, sr_):
        shift = 0.0 if path == "L" else true_offset  # right time = world time - offset
        i0 = int(round((start_s + shift) * sr_))
        return total[i0 : i0 + int(dur_s * sr_)]

    monkeypatch.setattr(sync, "read_audio", fake_read)
    res = sync.estimate_offset("L", "R", fps=120, windows=[20, 50, 80], dur_s=10, max_lag_s=3, sr=sr)
    assert res.offset_s == pytest.approx(true_offset, abs=2 / sr)
    assert res.offset_frames == pytest.approx(true_offset * 120, abs=0.1)
    assert res.spread_ms < 1


def test_estimate_offset_rejects_false_peaks(monkeypatch):
    calls = iter([0.5, 0.5, -3.0, 0.5, 7.0])  # two windows lock onto wrong peaks

    def fake_gcc(a, b, sr, max_lag_s):
        return next(calls), 100.0

    monkeypatch.setattr(sync, "read_audio", lambda *a, **k: np.zeros(10, np.float32))
    monkeypatch.setattr(sync, "gcc_phat", fake_gcc)
    res = sync.estimate_offset("L", "R", fps=100, windows=[10, 20, 30, 40, 50], max_lag_s=2)
    assert res.offset_s == pytest.approx(2 - 0.5) and res.n_inliers == 3
