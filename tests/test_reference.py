"""Library videos: other annotators' labels shown read-only, and masks kept safe on the laptop
when the shared folder (RFS) is unreachable."""

import os
import shutil

import numpy as np
import pytest

from reeftrack.config import DEFAULT_BEHAVIOURS
from tests.test_library import REL, _write_video, lib  # noqa: F401  (lib is a fixture)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _library_run(lib):  # noqa: F811
    video = lib.video_root / REL
    _write_video(video)
    lib.compress(video)
    return lib.open_run(video)


@needs_ffmpeg
def test_other_annotators_labels_read_only(lib):  # noqa: F811
    from reeftrack.annotation.behaviour import BehaviourLog
    from reeftrack.annotation.loops import LoopLog
    from reeftrack.annotation.reference import load_reference, next_out_of_view, other_annotators

    run = _library_run(lib)
    cait = BehaviourLog.for_run(run, DEFAULT_BEHAVIOURS, "Cait B")
    cait.add("Foraging", 0, 9, annotator="Cait B")
    cait.add("Out of view", 10, 14, annotator="Cait B")
    cait.add("Out of view", 20, 25, annotator="Cait B")
    LoopLog.for_run(run, "Ben").add(3, 18, annotator="Ben")
    BehaviourLog.for_run(run, DEFAULT_BEHAVIOURS, "Rosie").add("Chasing", 1, 2, annotator="Rosie")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime) for p in run.root.glob("*.csv")}

    names = other_annotators(run, "Rosie")
    assert names == {"ben": "Ben", "cait_b": "Cait B"}  # everyone but me, by display name
    evs = load_reference(run, list(names), run.frame_map().n_proxy)
    assert [(e.who, e.label, e.start, e.end) for e in evs] == [
        ("Cait B", "Foraging", 0, 9), ("Ben", "Loop 1", 3, 18), ("Cait B", "Out of view", 10, 14),
        ("Cait B", "Out of view", 20, 25)]
    assert next_out_of_view(evs, 0) == 10 and next_out_of_view(evs, 10) == 20 and next_out_of_view(evs, 20) is None
    assert next_out_of_view(evs, 22, -1) == 20
    after = {p.name: (p.read_bytes(), p.stat().st_mtime) for p in run.root.glob("*.csv")}
    assert after == before  # nothing written

    (run.root / "behaviours_broken.csv").write_text("not,a,proper\nfile\n")
    assert len(load_reference(run, ["broken", "cait_b"], 30)) == 3  # bad files are skipped, not fatal


@needs_ffmpeg
def test_masks_backed_up_when_shared_folder_unreachable(lib, monkeypatch):  # noqa: F811
    from reeftrack.tracking import tracklet as tracklet_mod
    from reeftrack.tracking.tracklet import Observation, Segment, Tracklet, load_for_run, save_for_run

    def ob(f):
        return Observation(f, 1, 200, (10, 10, 30, 20), (20.0, 15.0), 5.0, 1.0)

    run = _library_run(lib)
    run.annotator = "Rosie"
    shared = run.tracking_dir
    tr = Tracklet(shape=(240, 320))
    m = np.zeros((240, 320), bool)
    m[10:20, 10:30] = True
    tr.segments.append(Segment(1, 0, (20.0, 15.0), 0, 0))
    tr.add_observation(ob(0), m)
    save_for_run(tr, run)
    backup = tracklet_mod._backup_dir(run)
    assert (shared / "masks.npz").exists() and (backup / "masks.npz").exists()
    assert load_for_run(run)[1] is False  # shared copy is current

    # RFS drops: the shared save fails, the laptop copy has the change
    real = Tracklet.save

    def flaky(self, folder, fmap=None):
        if folder == shared:
            raise OSError("Host is down")
        return real(self, folder, fmap)

    monkeypatch.setattr(Tracklet, "save", flaky)
    tr.segments[0].end = 1
    tr.add_observation(ob(1), m)
    with pytest.raises(OSError):
        save_for_run(tr, run)
    monkeypatch.setattr(Tracklet, "save", real)
    old = (shared / "tracklet.json").stat().st_mtime - 60
    os.utime(shared / "tracklet.json", (old, old))

    # next time the video is opened: the newer laptop copy wins and is pushed back to RFS
    got, restored = load_for_run(run)
    assert restored and 1 in got.obs
    assert 1 in Tracklet.load(shared).obs
    assert load_for_run(run)[1] is False  # in sync again
