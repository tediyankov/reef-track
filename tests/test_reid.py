"""Re-ID export: longest un-occluded stretch, spaced top-N selection, dataset layout."""

import csv
import json

import numpy as np
import pytest
import yaml

from reeftrack.library import Library, LibrarySettings
from reeftrack.reid.export import COLUMNS, Settings, export
from reeftrack.reid.intervals import Segment, consecutive_chapters, longest_clear_run, occlusion_mask
from reeftrack.reid.select import pick_spaced


def test_occlusion_mask_only_counts_occluding_behaviours():
    m = occlusion_mask(10, [("Out of view", 0, 1), ("Foraging", 2, 5), ("Sheltering", 8, 20), ("Calib", 6, 6)])
    assert m.tolist() == [True, True, False, False, False, False, False, False, True, True]


def test_longest_run_crosses_chapter_boundary():
    a = np.array([1, 0, 0, 0, 0, 0], bool)   # clear 1..5 runs into chapter 2
    b = np.array([0, 0, 1, 0, 0, 0], bool)   # clear 0..1, then 3..5
    assert longest_clear_run([("GX010001", a), ("GX020001", b)]) == [
        Segment("GX010001", 1, 5), Segment("GX020001", 0, 1)]
    assert longest_clear_run([("GX010001", np.ones(4, bool))]) == []


def test_consecutive_chapters_split_on_missing_chapter():
    assert consecutive_chapters(["GX030170", "GX010170", "GX040170", "GX010999"]) == [
        ["GX010170"], ["GX030170", "GX040170"], ["GX010999"]]


def test_pick_spaced_prefers_confidence_but_keeps_gap():
    t = [0.0, 0.1, 0.2, 1.5, 3.0]
    s = [0.9, 0.95, 0.8, 0.5, 0.0]
    assert pick_spaced(t, s, n=10, min_gap_s=1.0) == [1, 3]  # 0 and 2 too close to 1; 4 has no fish
    assert pick_spaced(t, s, n=1, min_gap_s=1.0) == [1]


REL = "Maldives/2024_02_14/AM/Cait/Left/GX0{c}0703"


@pytest.fixture
def lib(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setenv("REEFTRACK_HOME", str(tmp_path / "home"))
    rfs = tmp_path / "RFS" / "fish_in_the_wild"
    s = LibrarySettings(compressed_root=str(rfs / "compressed_videos"), annotations_root=str(rfs / "annotations"),
                        cache_dir=str(tmp_path / "cache"))
    lib = Library(s)
    events = {1: [("Out of view", 0, 9)], 2: [("Sheltering", 20, 29)]}  # clear: ch1 10-29 + ch2 0-19
    for c in (1, 2):
        rel = REL.format(c=c)
        mp4 = lib.compressed_root / f"{rel}.mp4"
        mp4.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for i in range(30):
            vw.write(np.full((48, 64, 3), 8 * i, np.uint8))
        vw.release()
        (lib.compressed_root / f"{rel}.reeftrack.json").write_text(json.dumps({"bytes": mp4.stat().st_size}))
        ann = lib.annotations_root / rel
        ann.mkdir(parents=True)
        (ann / "run.yaml").write_text(yaml.safe_dump({"frames": {
            "source_start": 0, "stride": 4, "n_proxy": 30, "source_fps": 120.0, "proxy_fps": 30.0}}))
        with (ann / "behaviours_rosie.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["behaviour", "start_frame", "end_frame"])
            w.writerows(events[c])
    return lib


def test_export_writes_frames_and_metadata(lib, tmp_path):
    tracker = tmp_path / "tracker.csv"
    head = ["Fish ID", "Site", "Date", "Observer", "Observation Period", "L/R view", "Video MP4 name",
            "Assigned to", "Out of view annotated"]
    with tracker.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerow([1, "Maldives", "2024-02-14", "Cait", "AM1", "Left", "GX010703", "Rosie", "Yes"])
        w.writerow([1, "Maldives", "2024-02-14", "Cait", "AM1", "Left", "GX020703", "Rosie", "Yes"])
        w.writerow([2, "Maldives", "2024-02-14", "Cait", "AM1", "Left", "GX010704", "Rosie", "No"])

    calls = []

    def fake_detect(frames):  # confidence rises with brightness; one box per frame
        calls.append(len(frames))
        return [(float(f.mean()) / 255 + 0.01, (1.0, 2.0, 3.0, 4.0), 1) for f in frames]

    out = tmp_path / "RFS" / "fish_in_the_wild" / "reid_dataset"
    s = Settings(weights=tmp_path / "w.pt", n_per_fish=3, min_gap_s=0.3, stride=1, batch=4)
    meta = export(tracker, out, s, lib, detect=fake_detect, log=lambda m: None)

    rows = list(csv.DictReader(meta.open()))
    assert list(rows[0]) == list(COLUMNS)
    assert sum(calls) == 40  # 20 + 20 clear frames, nothing outside the stretch
    assert {r["fish_num"] for r in rows} == {"wild_01"} and len(rows) == 3
    for r in rows:
        assert (out / r["filepath"]).is_file()
        assert r["filepath"] == f"frames/wild_01/{r['video_id']}/frame{r['frame_number']}.jpg"
        assert r["video_path"] == f"compressed_videos/{REL.format(c=r['video_id'][3])}.mp4"
        assert float(r["timestamp_sec"]) == pytest.approx(int(r["frame_number"]) / 30, abs=1e-3)
    # brightest frames are the last of each chapter: ch1 frames near 29 and ch2 frames near 19
    assert {r["video_id"] for r in rows} == {"GX010703", "GX020703"}
    interval = list(csv.DictReader((out / "intervals.csv").open()))[0]
    assert (interval["start_video"], interval["start_frame"], interval["end_video"], interval["end_frame"]) == \
        ("GX010703", "10", "GX020703", "19")
    assert "3 full-frame images of 1 Picasso triggerfish" in (out / "README.md").read_text()

    # second run resumes: nothing is re-detected
    calls.clear()
    export(tracker, out, s, lib, detect=fake_detect, log=lambda m: None)
    assert calls == []
    # redoing with a different N reuses the saved detector scores
    s.n_per_fish = 5
    meta = export(tracker, out, s, lib, overwrite=True, detect=fake_detect, log=lambda m: None)
    assert calls == [] and len(list(csv.DictReader(meta.open()))) == 5
