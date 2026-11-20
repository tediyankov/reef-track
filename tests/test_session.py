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


def test_trim_end_of_segment_keeps_the_start(session):
    session.add_click(5, fish_xy(5))  # frames 0-19
    assert session.trim(15, keep="before") == (15, 19)  # SAM went wrong from frame 15
    seg = session.tracklet.segment(0)
    assert (seg.start, seg.end, seg.end_reason) == (0, 14, "trimmed")
    assert session.tracklet.frames() == list(range(15))


def test_trim_start_of_segment_keeps_the_end(session):
    session.add_click(5, fish_xy(5))
    assert session.trim(3, keep="after") == (0, 3)
    seg = session.tracklet.segment(0)
    assert (seg.start, seg.end, seg.start_reason) == (4, 19, "trimmed")


def test_trim_only_touches_the_segment_under_the_frame(session):
    session.add_click(5, fish_xy(5))  # 0-19
    session.add_click(40, fish_xy(40))  # 30-49
    session.trim(35, keep="before")
    assert session.tracklet.frames() == list(range(20)) + list(range(30, 35))


def test_trim_whole_segment_removes_it_and_untracked_frame_is_a_no_op(session):
    session.add_click(5, fish_xy(5))
    assert session.trim(25, keep="before") is None  # gap: nothing to cut
    session.trim(0, keep="before")
    assert session.tracklet.segments == [] and session.tracklet.obs == {}
    with pytest.raises(ValueError):
        session.trim(0, keep="sideways")


def test_retrack_after_trim_backfills_to_the_cut(session):
    session.add_click(5, fish_xy(5))
    session.trim(12, keep="before")  # keep 0-11
    seg = session.add_click(16, fish_xy(16))  # click the fish again a bit later
    assert seg.start == 12 and seg.start_reason == "met_segment"  # backfill closes the cut


def test_undo_steps_back_through_trims_deletes_and_corrections(session):
    session.add_click(5, fish_xy(5))  # 0-19
    session.add_click(12, fish_xy(12))  # correction: seg 0 cut to 0-11, seg 1 = 12-19
    session.trim(16, keep="before")
    session.delete_segment(0)
    assert session.undo() == "deleting segment 0"
    assert session.undo() == "trim of segment 1"
    assert session.tracklet.frames() == list(range(20))
    assert session.undo() == "tracking from frame 12"  # the correction's cut is restored too
    seg = session.tracklet.segment(0)
    assert (seg.end, seg.end_reason) == (19, "lost") and [s.segment_id for s in session.tracklet.segments] == [0]
    session.undo()
    assert session.tracklet.segments == [] and session.undo() is None


def test_failed_click_leaves_nothing_to_undo(session):
    session.add_click(5, fish_xy(5))
    with pytest.raises(ValueError):
        session.add_click(25, (10, 10))
    assert session.undo() == "tracking from frame 5"


class _CountingFake(FakeBackend):
    def __init__(self):
        super().__init__()
        self.steps = 0

    def step(self, frame):
        self.steps += 1
        return super().step(frame)


def _stepped(video, step=2, **tracking):
    tc = load_config({"tracking": {"frame_step": step, **tracking}}).tracking
    return TrackingSession(video, tc, backend=_CountingFake())


def test_frame_step_2_tracks_every_frame_with_half_the_sam_calls(occlusion_video):
    s1, s2 = _stepped(occlusion_video, 1), _stepped(occlusion_video, 2)
    for s in (s1, s2):
        s.add_click(5, fish_xy(5))
        s.add_click(40, fish_xy(40))
    assert s2.tracklet.frames() == s1.tracklet.frames()  # same coverage, no holes: 0-19, 30-49
    assert s2.backend.steps < 0.6 * s1.backend.steps
    assert s2.tracklet.meta["frame_step"] == {"0": 2, "1": 2}
    for f in (6, 8, 33, 47):  # in-between frames: the mask moved with the fish
        cx, cy = s2.tracklet.obs[f].centroid
        assert abs(cx - fish_xy(f)[0]) < 1.5 and abs(cy - 60) < 1


def test_frame_step_reaches_the_clip_ends_and_stops_exactly_at_another_segment(occlusion_video):
    s = _stepped(occlusion_video, 2)
    seg = s.add_click(4, fish_xy(4))  # even frame: steps 2, 0 backwards; forwards to 19
    assert (seg.start, seg.start_reason) == (0, "video_start")
    s.trim(15, keep="before")  # 0-14
    seg = s.add_click(18, fish_xy(18))  # backwards 16, then a 1-frame step to 15, then meets 14
    assert (seg.start, seg.start_reason) == (15, "met_segment")
    assert s.tracklet.frames() == list(range(20))
    seg = _stepped(occlusion_video, 2).add_click(45, fish_xy(45))  # odd distance to the last frame
    assert (seg.end, seg.end_reason) == (49, "video_end")


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


class _EmptyRestart(FakeBackend):
    """Finds the fish in the preview, but nothing when tracking restarts from the same click
    (seen with SAM 3 on a zoom window). A mask prompt still works."""

    def start(self, frame, points, labels=None, mask=None):
        if mask is None:
            self.start_mask = None
            return self._seg(np.zeros_like(frame))
        return super().start(frame, points, labels, mask)


@pytest.mark.parametrize("zoomed", [False, True])
def test_confirmed_preview_mask_used_when_restart_finds_nothing(occlusion_video, zoomed):
    tc = load_config().tracking
    s = TrackingSession(occlusion_video, tc, backend=_EmptyRestart())
    roi = (0, 20, 120, 80) if zoomed else None
    shown = s.preview(5, fish_xy(5), roi)
    assert shown.any()
    with pytest.raises(ValueError):  # without the confirmed preview: nothing to track
        s.add_click(5, fish_xy(5), roi=roi)
    seg = s.add_click(5, fish_xy(5), roi=roi, preview_mask=shown)
    assert (seg.start, seg.end) == (0, 19)  # tracked as if SAM had found it
    assert np.array_equal(s.tracklet.mask(5), shown)


def test_half_precision_submodule_keeps_the_models_dtype_outside():
    torch = pytest.importorskip("torch")
    from reeftrack.tracking.backends import _half_precision

    lin = torch.nn.Linear(8, 4)
    ref = lin(torch.ones(2, 8))
    _half_precision(lin, torch.float32)
    assert lin.weight.dtype == torch.float16
    out = lin(torch.ones(2, 8))  # float32 in, float16 inside, float32 out
    assert out.dtype == torch.float32 and torch.allclose(out, ref, atol=1e-2)
    _half_precision(None, torch.float32)  # backends without a memory_attention: no-op
