"""Config loading, frame mapping and frame-exact proxy creation."""

import shutil

import numpy as np
import pydantic
import pytest

from reeftrack.config import load_config
from reeftrack.io.run import Run
from reeftrack.io.video import FrameMap, FrameReader, make_proxy, plan_frame_map, probe

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_config_defaults_and_override():
    cfg = load_config()
    assert cfg.tracking.backend in {"sam3", "sam2"}
    cfg2 = load_config({"recon": {"keyframe_fps": 1.5}})
    assert cfg2.recon.keyframe_fps == 1.5
    assert cfg2.recon.chunk_size == cfg.recon.chunk_size  # untouched keys survive the merge
    assert cfg.digest() != cfg2.digest()


def test_config_rejects_typos():
    with pytest.raises(pydantic.ValidationError):
        load_config({"tracking": {"patiense": 3}})


def test_frame_map():
    fm = FrameMap(start=12998, stride=4, n_proxy=6363, source_fps=120000 / 1001)
    assert fm.to_source(0) == 12998
    assert fm.to_source(10) == 13038
    assert fm.to_proxy(13038) == 10
    assert fm.proxy_fps == pytest.approx(29.97, abs=0.01)
    assert fm.time_s(0) == pytest.approx(12998 * 1001 / 120000)
    assert list(fm.to_source(np.array([0, 1]))) == [12998, 13002]


@needs_ffmpeg
def test_proxy_is_frame_exact(tmp_path):
    """Source frame i has brightness 4*i; proxy frame k must equal source frame start + k*stride."""
    import cv2

    src = tmp_path / "src.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 60, (64, 48))
    for i in range(60):
        vw.write(np.full((48, 64, 3), 4 * i, np.uint8))
    vw.release()

    info = probe(src)
    assert info.n_frames == 60
    fm = plan_frame_map(info, start=10, end=49, proxy_fps=15)
    assert fm.stride == 4 and fm.n_proxy == 10
    out = make_proxy(src, tmp_path / "proxy.mp4", fm, height=48, rotate=0, crf=0, gop=5)
    assert out.n_frames == 10
    with FrameReader(tmp_path / "proxy.mp4") as r:
        for k in [0, 3, 9, 1]:  # includes a backwards seek
            assert abs(int(r[k].mean()) - 4 * fm.to_source(k)) <= 3


def test_probe_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.mp4"
    p.touch()
    with pytest.raises(ValueError, match="0 bytes"):
        probe(p)


def test_plan_frame_map_bounds(tmp_path):
    from fractions import Fraction

    from reeftrack.io.video import VideoInfo

    info = VideoInfo(tmp_path, 10, 10, Fraction(120), 100, "h264")
    with pytest.raises(ValueError):
        plan_frame_map(info, 50, 200, 30)


def test_run_create_and_open(tmp_path):
    cfg = load_config()
    fm = FrameMap(start=5, stride=2, n_proxy=10, source_fps=60.0)
    Run.create(tmp_path / "r", tmp_path / "v.mp4", fm, cfg)
    run = Run.open(tmp_path / "r")
    assert run.frame_map() == fm
    assert run.config().digest() == cfg.digest()
    with pytest.raises(FileNotFoundError):
        Run.open(tmp_path / "nope")


def test_old_run_snapshot_with_obsolete_keys_still_loads(tmp_path):
    import yaml

    cfg = load_config()
    Run.create(tmp_path / "r", tmp_path / "v.mp4", FrameMap(0, 1, 5, 30.0), cfg)
    run = Run.open(tmp_path / "r")
    snap = yaml.safe_load(run.config_path.read_text())
    snap["tracking"]["window"] = 600  # a key removed in a later version
    snap["tracking"]["patience"] = 7
    run.config_path.write_text(yaml.safe_dump(snap))
    assert run.config().tracking.patience == 7
