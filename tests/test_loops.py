"""Loop annotations: validation, CSV, 3D metrics from the trajectory."""

import csv

import numpy as np
import pytest

from reeftrack.annotation.loops import LoopLog, Trajectory, loop_metrics
from reeftrack.io.video import FrameMap

FM = FrameMap(start=12998, stride=4, n_proxy=6363, source_fps=120000 / 1001)


def circle_traj():
    """Fish swims a circle of radius 1 m over frames 100-200 and returns to its start."""
    f = np.arange(100, 201)
    a = 2 * np.pi * (f - 100) / 100
    xyz = np.c_[np.cos(a) - 1, np.sin(a), np.zeros_like(a)]  # starts and ends at (0, 0, 0)
    return Trajectory(f, xyz, "m", "fusion_stereo/trajectory.csv")


@pytest.fixture
def loops(tmp_path):
    return LoopLog(tmp_path / "annotations" / "loops.csv", FM, "/v/GX010703.MP4", "run1", circle_traj())


def test_loop_row_and_metrics(loops):
    lp = loops.add(100, 200, "around the bommie", "anna", confidence="probable", landmark="big bommie")
    r = list(csv.DictReader(open(loops.path)))[0]
    assert list(r)[:4] == ["loop_id", "confidence", "landmark", "start_frame"]
    assert r["loop_id"] == str(lp.event_id) and r["confidence"] == "probable" and r["landmark"] == "big bommie"
    assert float(r["return_distance"]) == pytest.approx(0, abs=1e-6)
    assert float(r["path_length"]) == pytest.approx(2 * np.pi, rel=1e-2)
    assert float(r["max_excursion"]) == pytest.approx(2.0, rel=1e-3)  # opposite side of the circle
    assert r["units"] == "m" and r["trajectory_source"].startswith("fusion_stereo")
    assert int(r["start_source_frame"]) == 12998 + 400


def test_partial_loop_and_untracked_frames(loops):
    lp = loops.add(100, 150)  # half circle: ends on the far side
    m = loops.row(lp)
    assert float(m["return_distance"]) == pytest.approx(2.0, rel=1e-3)
    far = loops.add(3000, 3100)  # no trajectory there -> 3D columns empty, row still saved
    assert loops.row(far)["return_distance"] == "" and loops.row(far)["units"] == "m"


def test_nearest_frame_tolerance():
    t = circle_traj()
    assert np.allclose(t.at(95), t.at(100))  # 5 frames before tracking starts: nearest is used
    assert t.at(50) is None  # too far
    assert loop_metrics(None, 1, 5) == {}


def test_validation_and_roundtrip(loops):
    with pytest.raises(ValueError):
        loops.add(10, 10)  # a loop needs at least 2 frames
    with pytest.raises(ValueError):
        loops.add(10, 20, confidence="maybe")
    lp = loops.add(60, 40, landmark="  sand patch ")  # swapped, trimmed, default confidence
    assert (lp.start_frame, lp.end_frame, lp.confidence, lp.landmark) == (40, 60, "certain", "sand patch")
    loops.update(lp.event_id, confidence="possible", comment="lost view briefly")
    again = LoopLog(loops.path, FM, "/v/GX010703.MP4", "run1")
    ev = again.get(lp.event_id)
    assert ev.confidence == "possible" and ev.comment == "lost view briefly" and again.next_id() == 2


def test_trajectory_for_run_prefers_stereo(tmp_path):
    from reeftrack.config import load_config
    from reeftrack.io.run import Run

    run = Run.create(tmp_path / "r", tmp_path / "v.mp4", FM, load_config())
    assert Trajectory.for_run(run) is None
    for d, x in (("fusion", 9.0), ("fusion_stereo", 1.0)):
        (run.root / d).mkdir(exist_ok=True)
        (run.root / d / "trajectory.csv").write_text(f"proxy_frame,x,y,z,kind\n5,{x},0,0,measured\n")
    t = Trajectory.for_run(run)
    assert t.units == "m" and t.at(5)[0] == 1.0
