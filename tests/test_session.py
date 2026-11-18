"""End-to-end tracking logic on a synthetic occlusion video with a fake segmenter."""

import numpy as np
import pytest

from reeftrack.config import load_config
from reeftrack.io.video import FrameMap
from reeftrack.tracking.backends import FakeBackend
from reeftrack.tracking.session import TrackingSession
from reeftrack.tracking.tracklet import Tracklet


@pytest.fixture
def session(occlusion_video):
    tc = load_config().tracking
    return TrackingSession(occlusion_video, tc, backend=FakeBackend())


def fish_xy(i):
    return (20 + 2.5 * i, 60)


def test_break_on_occlusion(session):
    seg = session.add_click(5, fish_xy(5))
    assert (seg.start, seg.end) == (0, 19)
    assert seg.end_reason == "lost"
    assert seg.start_reason == "video_start"
    assert session.tracklet.gaps(50) == [(20, 49)]


def test_reacquire_backfills_gap(session):
    session.add_click(5, fish_xy(5))
    seg = session.add_click(40, fish_xy(40))  # user notices the fish late...
    assert seg.start == 30  # ...backfill finds it re-emerging right after the coral
    assert seg.start_reason == "lost"
    assert seg.end == 49 and seg.end_reason == "video_end"
    assert session.tracklet.gaps(50) == [(20, 29)]


def test_backfill_stops_at_existing_segment(occlusion_video):
    tc = load_config({"tracking": {"patience": 100}}).tracking  # never break
    s = TrackingSession(occlusion_video, tc, backend=FakeBackend())
    s.add_click(0, fish_xy(0))  # forward runs to the end (patience huge), empty frames skipped
    assert s.tracklet.segments[0].end == 49


def test_correction_cuts_segment(session):
    session.add_click(5, fish_xy(5))
    seg = session.add_click(12, fish_xy(12))
    first = session.tracklet.segment(0)
    assert first.end == 11 and first.end_reason == "corrected"
    assert (seg.start, seg.end) == (12, 19)
    assert all(session.tracklet.owner(f) == 1 for f in range(12, 20))


def test_click_on_nothing_raises_and_leaves_no_segment(session, occlusion_video):
    with pytest.raises(ValueError):
        session.add_click(25, (10, 10))  # fish hidden
    assert session.tracklet.segments == []


def test_undo(session):
    session.add_click(5, fish_xy(5))
    session.add_click(40, fish_xy(40))
    session.undo()
    assert [s.segment_id for s in session.tracklet.segments] == [0]
    assert max(session.tracklet.frames()) == 19


def test_stop_flag(session):
    seen = []
    seg = session.add_click(5, fish_xy(5), callback=lambda f, m, v: seen.append(f), should_stop=lambda: len(seen) > 3)
    assert seg.end_reason == "user"


def test_save_load_roundtrip(session, tmp_path):
    session.add_click(5, fish_xy(5))
    session.add_click(40, fish_xy(40))
    fmap = FrameMap(start=1000, stride=4, n_proxy=50, source_fps=120.0)
    session.tracklet.save(tmp_path, fmap)
    tr = Tracklet.load(tmp_path)
    assert tr.frames() == session.tracklet.frames()
    assert np.array_equal(tr.mask(7), session.tracklet.mask(7))
    assert [s.segment_id for s in tr.segments] == [0, 1]
    rows = (tmp_path / "tracklet.csv").read_text().splitlines()
    assert rows[1].startswith("0,1000,")  # proxy frame 0 -> source frame 1000
    assert len(rows) == 1 + len(tr.frames())


def test_ownership_conflict(session):
    session.add_click(5, fish_xy(5))
    ob = session.tracklet.obs[6]
    ob2 = type(ob)(**{**ob.__dict__, "segment_id": 99})
    with pytest.raises(ValueError):
        session.tracklet.add_observation(ob2, session.tracklet.mask(6))
