"""Shared video library: path mirroring, safe compression, local cache, per-annotator CSVs,
and not losing annotations when the network share is unreachable."""

import csv
import shutil

import numpy as np
import pytest

from reeftrack.config import DEFAULT_BEHAVIOURS
from reeftrack.library import CompressionBusy, Library, LibrarySettings

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
REL = "Maldives/2024_02_14/AM/Cait/Left/GX010703.MP4"


def _write_video(path, n=120, fps=120):
    """Synthetic 'GoPro' clip: frame i has brightness 2*i (so frame mapping can be checked)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(n):
        vw.write(np.full((240, 320, 3), 2 * i, np.uint8))
    vw.release()


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("REEFTRACK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("reeftrack.library._same_disk", lambda a, b: False)  # 'RFS' is a network share
    drive = tmp_path / "Seagate" / "originaldata"
    rfs = tmp_path / "RFS" / "fish_in_the_wild"
    for d in (drive, rfs / "compressed_videos", rfs / "annotations"):
        d.mkdir(parents=True)
    s = LibrarySettings(video_root=str(drive), compressed_root=str(rfs / "compressed_videos"),
                        annotations_root=str(rfs / "annotations"), cache_dir=str(tmp_path / "cache"),
                        height=240, bitrate="2M")
    s.save()
    return Library(LibrarySettings.load())


def test_settings_roundtrip_and_problems(lib, tmp_path):
    assert LibrarySettings.load().height == 240
    bad = LibrarySettings(video_root=str(tmp_path / "nope"))
    assert any("not found" in p for p in bad.problems()) and any("not set" in p for p in bad.problems())
    assert lib.s.problems() == []


def test_paths_mirror_source_structure(lib, tmp_path):
    video = lib.video_root / REL
    assert str(lib.rel(video)) == REL
    assert lib.compressed_path(lib.rel(video)).relative_to(lib.compressed_root).as_posix() == \
        "Maldives/2024_02_14/AM/Cait/Left/GX010703.mp4"
    assert lib.annotation_dir(lib.rel(video)).relative_to(lib.annotations_root).as_posix() == \
        "Maldives/2024_02_14/AM/Cait/Left/GX010703"
    # same drive mounted under another name -> still the same relative path
    other = tmp_path / "Seagate 1" / "originaldata" / REL
    assert str(lib.rel(other)) == REL
    with pytest.raises(ValueError, match="not inside"):
        lib.rel(tmp_path / "Downloads" / "x.MP4")


def test_empty_original_gives_clear_error(lib):
    bad = lib.video_root / "Maldives/2024_02_20/PM1/Theresa/Left/GX010068.MP4"
    bad.parent.mkdir(parents=True)
    bad.touch()  # many copies on the old drives are 0 bytes
    with pytest.raises(ValueError, match="0 bytes"):
        lib.frame_map_for(bad)


@needs_ffmpeg
def test_compress_open_and_annotate(lib):
    video = lib.video_root / REL
    _write_video(video)
    rel = lib.rel(video)
    with pytest.raises(FileNotFoundError):
        lib.open_run(video)  # nothing compressed yet and original not requested

    seen = []
    out = lib.compress(video, progress=lambda d, t: seen.append((d, t)))
    assert out == lib.compressed_path(rel) and out.exists() and seen
    info = lib.compressed_info(rel)
    assert info["n_frames"] == 30 and info["stride"] == 4 and info["source_rel"] == REL
    assert not lib._lock(rel).exists()  # lock released
    assert not list(lib.compressed_root.rglob("*.partial"))  # no half-written files on 'RFS'
    mtime = out.stat().st_mtime
    assert lib.compress(video) == out and out.stat().st_mtime == mtime  # never re-encoded

    run = lib.open_run(video)
    assert run.root == lib.annotation_dir(rel) and run.is_library
    assert not list(run.root.rglob("*.mp4"))  # no video in the annotations folder
    assert run.video == lib.cache_path(rel) and not run.video_is_raw
    assert run.meta()["source"]["rel"] == REL

    from reeftrack.annotation.behaviour import BehaviourLog
    from reeftrack.annotation.loops import LoopLog

    b = BehaviourLog.for_run(run, DEFAULT_BEHAVIOURS, "Anna Smith")
    b.add("Foraging", 3, 9, "test", "Anna Smith")
    LoopLog.for_run(run, "Anna Smith").add(1, 20, annotator="Anna Smith")
    BehaviourLog.for_run(run, DEFAULT_BEHAVIOURS, "Ben").add("Chasing", 0, 2, annotator="Ben")
    names = sorted(p.name for p in run.root.iterdir())
    assert names == ["behaviours_anna_smith.csv", "behaviours_ben.csv", "config.yaml", "loops_anna_smith.csv",
                     "run.yaml"]
    r = list(csv.DictReader(open(run.root / "behaviours_anna_smith.csv")))[0]
    assert r["source_video"] == REL and r["run"] == REL  # portable, not a machine-specific path
    assert int(r["start_source_frame"]) == 12 and int(r["end_source_frame"]) == 36


@needs_ffmpeg
@needs_ffmpeg
def test_copy_sent_without_info_file_is_recognised(lib, monkeypatch):
    """Only the .mp4 was sent: with the original at hand, the copy is checked and its info file rebuilt."""
    video = lib.video_root / REL
    _write_video(video)
    rel = lib.rel(video)
    lib.compress(video)
    sc, mp4 = lib.sidecar_path(rel), lib.compressed_path(rel)
    sc.unlink()
    assert lib.compressed_info(rel) is None and "info file" in lib.compressed_status(rel)
    assert lib.adopt_compressed(video)["n_frames"] == 30 and sc.exists()  # rebuilt, not re-encoded
    sc.unlink()
    assert lib.open_compressed(mp4).video == mp4 and sc.exists()  # File → Open compressed video, too

    # a copy with the wrong number of frames is not adopted
    sc.unlink()
    good = mp4.read_bytes()
    _write_video(mp4, n=40)
    assert lib.adopt_compressed(video) is None and not sc.exists()
    mp4.write_bytes(good)
    mp4.unlink()
    assert "No compressed copy found" in lib.compressed_status(rel)


@needs_ffmpeg
def test_compressed_folder_on_this_computer_plays_directly(lib, monkeypatch):
    video = lib.video_root / REL
    _write_video(video)
    rel = lib.rel(video)
    lib.compress(video)
    monkeypatch.setattr("reeftrack.library._same_disk", lambda a, b: True)
    shutil.rmtree(lib.cache_dir)
    assert lib.open_run(video).video == lib.compressed_path(rel)  # no second 1.5 GB copy in the cache
    assert not lib.cache_path(rel).exists()


def test_original_fallback_reads_same_frames(lib):
    from reeftrack.io.video import FrameReader, RawFrameReader

    video = lib.video_root / REL
    _write_video(video)
    raw_run = lib.open_run(video, use_original=True)
    assert raw_run.video_is_raw and raw_run.video == video
    raw = RawFrameReader(video, stride=4)
    assert len(raw) == raw_run.frame_map().n_proxy == 30
    lib.compress(video)
    prox = FrameReader(lib.open_run(video).video)
    for i in [0, 5, 29, 3, 17]:  # includes backward jumps
        assert abs(int(raw[i].mean()) - 2 * 4 * i) <= 3
        assert abs(int(prox[i].mean()) - int(raw[i].mean())) <= 4


@needs_ffmpeg
def test_compression_lock_blocks_second_user(lib):
    video = lib.video_root / REL
    _write_video(video)
    lock = lib._lock(lib.rel(video))
    lock.parent.mkdir(parents=True)
    lock.write_text("anna on laptop-1")
    with pytest.raises(CompressionBusy, match="anna"):
        lib.compress(video)


@needs_ffmpeg
def test_annotations_survive_network_outage(lib, monkeypatch):
    from reeftrack.annotation import events
    from reeftrack.annotation.behaviour import BehaviourLog

    video = lib.video_root / REL
    _write_video(video)
    run = lib.open_run(video, use_original=True)
    b = BehaviourLog.for_run(run, DEFAULT_BEHAVIOURS, "anna")
    b.add("Foraging", 1, 4, annotator="anna")
    shared = run.annotation_path("behaviours", "anna")

    real = events.IntervalLog._atomic_write

    def flaky(path, text):  # the shared file is unreachable, local disk is fine
        if path == shared:
            raise OSError("Host is down")
        real(path, text)

    monkeypatch.setattr(events.IntervalLog, "_atomic_write", staticmethod(flaky))
    b.add("Chasing", 10, 12, annotator="anna")
    assert b.last_save_error and "Host is down" in b.last_save_error
    assert len(list(csv.DictReader(open(shared)))) == 1  # shared copy is behind...
    assert len(list(csv.DictReader(open(b.backup)))) == 2  # ...but nothing is lost locally

    monkeypatch.setattr(events.IntervalLog, "_atomic_write", staticmethod(real))  # network back
    again = BehaviourLog.for_run(run, DEFAULT_BEHAVIOURS, "anna")
    assert again.restored_from_backup and len(again.events) == 2
    assert len(list(csv.DictReader(open(shared)))) == 2  # pushed back to the shared folder


def test_cache_pruned_least_recently_used(lib):
    import os
    import time

    lib.s.cache_limit_gb = 2.5e-9 * 3  # room for ~3 bytes x ... i.e. only the newest file below
    files = []
    for i in range(3):
        p = lib.cache_dir / f"v{i}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 4)
        os.utime(p, (time.time() + i, time.time() + i))
        files.append(p)
    lib.prune_cache(keep=files[0])
    assert files[0].exists() and not files[1].exists()  # 'keep' survives even though oldest


@needs_ffmpeg
def test_offline_annotation_then_sync_to_shared(lib, tmp_path, monkeypatch):
    """Colleague without RFS/hard drive: gets the compressed .mp4 + .json, annotates locally,
    later the annotations are merged into the shared folder."""
    import os
    import time

    from reeftrack.annotation.behaviour import BehaviourLog
    from reeftrack.library import sync_annotations

    video = lib.video_root / REL
    _write_video(video)
    lib.compress(video)
    rel = lib.rel(video)

    # what she receives (any folder, e.g. Downloads) and her settings: only a local annotations folder
    got = tmp_path / "Downloads"
    got.mkdir()
    shutil.copy(lib.compressed_path(rel), got / "GX010703.mp4")
    shutil.copy(lib.sidecar_path(rel), got / "GX010703.reeftrack.json")
    local_ann = tmp_path / "laptop" / "annotations"
    local_ann.mkdir(parents=True)
    her = Library(LibrarySettings(annotations_root=str(local_ann), cache_dir=str(tmp_path / "c2")))
    assert her.s.problems(required=("annotations_root",)) == [] and her.s.problems()  # drive/RFS optional

    run = her.open_compressed(got / "GX010703.mp4")
    assert run.root == local_ann / "Maldives/2024_02_14/AM/Cait/Left/GX010703"  # same layout as on RFS
    assert run.video == got / "GX010703.mp4" and run.frame_map().n_proxy == 30
    b = BehaviourLog.for_run(run, DEFAULT_BEHAVIOURS, "Anna")
    b.add("Foraging", 2, 9, "offline", "Anna")
    assert list(csv.DictReader(open(b.path)))[0]["source_video"] == REL

    # errors that matter in practice
    (got / "lonely.mp4").write_bytes((got / "GX010703.mp4").read_bytes())
    with pytest.raises(FileNotFoundError, match="reeftrack.json"):
        her.open_compressed(got / "lonely.mp4")
    shutil.copy(got / "GX010703.reeftrack.json", got / "lonely.reeftrack.json")
    with open(got / "lonely.mp4", "r+b") as fh:
        fh.truncate(1000)
    with pytest.raises(ValueError, match="incomplete"):
        her.open_compressed(got / "lonely.mp4")

    # someone already has a file for this video on the shared drive
    shared_run = lib.open_run(video)
    BehaviourLog.for_run(shared_run, DEFAULT_BEHAVIOURS, "Ben").add("Chasing", 0, 1, annotator="Ben")

    shared = lib.annotations_root
    preview = sync_annotations(local_ann, shared, dry_run=True, log=lambda *_: None)
    assert preview["copied"] == 1 and not (shared_run.root / "behaviours_anna.csv").exists()  # dry run copies nothing
    counts = sync_annotations(local_ann, shared, log=lambda *_: None)
    assert counts["copied"] == 1  # her CSV; the shared run.yaml/config.yaml are kept
    assert sorted(p.name for p in shared_run.root.glob("*.csv")) == ["behaviours_anna.csv", "behaviours_ben.csv"]
    assert sync_annotations(local_ann, shared, log=lambda *_: None)["same"] == 1  # idempotent

    # she edits offline later -> newer local file updates the shared one; an older one never overwrites
    time.sleep(0.01)
    b.add("Out of view", 10, 29, annotator="Anna")
    assert sync_annotations(local_ann, shared, log=lambda *_: None)["updated"] == 1
    old = time.time() - 3600
    os.utime(b.path, (old, old))
    (shared_run.root / "behaviours_anna.csv").write_text((shared_run.root / "behaviours_anna.csv").read_text() + "")
    b.path.write_text(b.path.read_text().replace("offline", "stale"))
    os.utime(b.path, (old, old))
    assert sync_annotations(local_ann, shared, log=lambda *_: None)["kept_newer_in_dest"] == 1

    # a folder whose frame layout disagrees is skipped, not merged
    import yaml

    y = yaml.safe_load((run.root / "run.yaml").read_text())
    y["frames"]["stride"] = 2
    (run.root / "run.yaml").write_text(yaml.safe_dump(y))
    assert sync_annotations(local_ann, shared, log=lambda *_: None)["skipped_mismatch"] == 1
