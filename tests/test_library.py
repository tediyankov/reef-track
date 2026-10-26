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
