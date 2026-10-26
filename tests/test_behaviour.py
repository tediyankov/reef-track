"""Behaviour annotation log: validation, frame mapping, CSV round trip."""

import csv

import pytest

from reeftrack.annotation.behaviour import COLUMNS, BehaviourLog
from reeftrack.config import DEFAULT_BEHAVIOURS, load_config
from reeftrack.io.video import FrameMap

FM = FrameMap(start=12998, stride=4, n_proxy=6363, source_fps=120000 / 1001)


@pytest.fixture
def blog(tmp_path):
    return BehaviourLog(tmp_path / "annotations" / "behaviours.csv", FM, "/videos/GX010703.MP4", "run1",
                        DEFAULT_BEHAVIOURS)


def test_categories_match_ethogram():
    assert load_config().annotation.behaviours == DEFAULT_BEHAVIOURS
    assert DEFAULT_BEHAVIOURS == [
        "Foraging", "Near other Picasso triggerfish", "Chasing", "Escaping", "Fighting", "Cleaner fish interaction",
        "Scratching", "Housekeeping", "Sheltering", "Stationary", "Erratic swimming", "Calm swimming", "Other",
        "Out of view", "Calib"]
    defs = load_config().annotation.definitions
    assert set(defs) == set(DEFAULT_BEHAVIOURS) and all(defs.values())


def test_old_run_snapshot_gets_current_categories(tmp_path):
    import yaml

    from reeftrack.io.run import Run

    cfg = load_config()
    Run.create(tmp_path / "r", tmp_path / "v.mp4", FM, cfg)
    run = Run.open(tmp_path / "r")
    snap = yaml.safe_load(run.config_path.read_text())
    snap["annotation"]["behaviours"] = ["Calm Swimming", "Foraging"]  # list saved before the ethogram update
    run.config_path.write_text(yaml.safe_dump(snap))
    assert run.config().annotation.behaviours == DEFAULT_BEHAVIOURS


def test_add_writes_row_with_source_frames_and_times(blog):
    ev = blog.add("Foraging", 100, 129, "pecking at coral, near bommie", "anna")
    rows = list(csv.DictReader(open(blog.path)))
    assert list(rows[0]) == COLUMNS
    r = rows[0]
    assert r["behaviour"] == "Foraging" and r["event_id"] == str(ev.event_id)
    assert int(r["start_frame"]) == 100 and int(r["end_frame"]) == 129
    assert int(r["start_source_frame"]) == 12998 + 400 and int(r["end_source_frame"]) == 12998 + 516
    assert float(r["duration_s"]) == pytest.approx(30 * 4 * 1001 / 120000, abs=1e-3)  # 30 frames incl. end
    assert r["source_video"] == "/videos/GX010703.MP4" and r["run"] == "run1"
    assert r["comment"] == "pecking at coral, near bommie" and r["annotator"] == "anna"


def test_validation(blog):
    with pytest.raises(ValueError):
        blog.add("", 1, 2)  # every pair must have a behaviour
    with pytest.raises(ValueError):
        blog.add("Dancing", 1, 2)
    with pytest.raises(ValueError):
        blog.add("Foraging", 0, 10_000)
    ev = blog.add("Chasing", 50, 40)  # reversed marks are swapped, not rejected
    assert (ev.start_frame, ev.end_frame) == (40, 50)


def test_roundtrip_with_awkward_comments(blog):
    blog.add("Sheltering", 10, 20, 'under the "big" bommie,\nthen out', "a")
    blog.add("Foraging", 15, 30)
    again = BehaviourLog(blog.path, FM, "/videos/GX010703.MP4", "run1", DEFAULT_BEHAVIOURS)
    assert [(e.behaviour, e.start_frame, e.end_frame, e.comment) for e in again.sorted()] == [
        ("Sheltering", 10, 20, 'under the "big" bommie,\nthen out'), ("Foraging", 15, 30, "")]
    assert again.next_id() == 3


def test_overlaps_update_delete(blog):
    a = blog.add("Foraging", 10, 30)
    b = blog.add("Near other Picasso triggerfish", 20, 40)
    assert [e.event_id for e in blog.active_at(25)] == [a.event_id, b.event_id]
    assert blog.active_at(35) == [b]
    blog.update(a.event_id, behaviour="Chasing", end_frame=12, comment="short burst")
    assert blog.get(a.event_id).behaviour == "Chasing" and blog.get(a.event_id).end_frame == 12
    with pytest.raises(ValueError):
        blog.update(a.event_id, behaviour="Nope")
    blog.delete(b.event_id)
    rows = list(csv.DictReader(open(blog.path)))
    assert len(rows) == 1 and rows[0]["comment"] == "short burst"


def test_unknown_category_in_old_csv_is_kept(blog, caplog):
    blog.add("Foraging", 1, 2)
    text = blog.path.read_text().replace("Foraging", "Grazing")
    blog.path.write_text(text)
    again = BehaviourLog(blog.path, FM, "v", "run1", DEFAULT_BEHAVIOURS)
    assert again.events[0].behaviour == "Grazing" and "Grazing" in again.categories
