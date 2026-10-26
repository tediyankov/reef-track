"""A *run* = one video clip + everything derived from it, in one self-describing folder.

    data/runs/<run_name>/
      run.yaml               provenance: source path, frame range, fps, config digest, git commit
      config.yaml            the exact merged config used
      proxy.mp4              1080p/30fps working copy (frame-exact)
      tracking/
        tracklet.json        segments, prompts, per-frame scores & status
        masks.npz            run-length-encoded masks for every tracked frame (one file)
      recon/
        keyframes.json       which proxy frames went into the 3D model and why
        cameras.npz          per-keyframe camera intrinsics + world-from-camera poses
        depth.npz            per-keyframe depth + confidence (float16)
        reef.ply             fused, cleaned reef point cloud (open in MeshLab/CloudCompare)
      fusion/
        trajectory.csv       fish 3D position per keyframe (+ interpolated rows, flagged)
        scene.html           interactive 3D viewer (reef + fish path + camera path)
        scene.ply            reef + trajectory points in one file

Few, large files on purpose: OneDrive / JASMIN group workspaces cope badly with 10^4 tiny files.
"""

from __future__ import annotations

import datetime as _dt
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reeftrack.config import REPO_ROOT, Config, drop_unknown_keys, load_config
from reeftrack.io.video import FrameMap


def slugify(name: str) -> str:
    """Safe, readable file-name part: 'Anna Smith' -> 'anna_smith'."""
    import re

    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "anon"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class Run:
    root: Path
    # Which reconstruction variant this handle reads/writes: "recon" (mono, default) or e.g.
    # "recon_stereo". Fusion outputs follow it ("fusion", "fusion_stereo"), so variants never
    # overwrite each other and can be compared side by side.
    recon_name: str = "recon"
    # Library runs (one folder per source video on shared storage) keep no video inside the
    # run folder: the GUI plays a compressed copy or the original, set here at open time.
    video: Path | None = None
    video_is_raw: bool = False  # True: `video` is the original (read with RawFrameReader)

    # ---- canonical paths -------------------------------------------------
    @property
    def meta_path(self) -> Path:
        return self.root / "run.yaml"

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def proxy(self) -> Path:
        return self.video if self.video is not None else self.root / "proxy.mp4"

    @property
    def is_library(self) -> bool:
        return self.meta().get("layout") == "library"

    def annotation_path(self, kind: str, annotator: str | None = None) -> Path:
        """Where an annotation CSV lives. Library runs: one file per annotator in the run
        folder (`behaviours_<name>.csv`), so annotators never overwrite each other."""
        if self.is_library:
            if not annotator:
                raise ValueError("library runs need an annotator name (files are per annotator)")
            return self.root / f"{kind}_{slugify(annotator)}.csv"
        return self.root / "annotations" / f"{kind}.csv"

    @property
    def tracking_dir(self) -> Path:
        return self.root / "tracking"

    @property
    def proxy_right(self) -> Path:
        return self.root / "proxy_right.mp4"

    @property
    def recon_dir(self) -> Path:
        return self.root / self.recon_name

    @property
    def fusion_dir(self) -> Path:
        suffix = self.recon_name.removeprefix("recon")
        return self.root / f"fusion{suffix}"

    def variant(self, recon_name: str) -> Run:
        return Run(self.root, recon_name, self.video, self.video_is_raw)

    # ---- metadata --------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        return yaml.safe_load(self.meta_path.read_text())

    def update_meta(self, **sections) -> None:
        meta = self.meta()
        meta.update(sections)
        self.meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    def stereo(self) -> dict | None:
        return self.meta().get("stereo")

    def frame_map_right(self) -> FrameMap:
        st = self.stereo()
        if not st:
            raise RuntimeError("run has no right camera; use `reeftrack add-right` first")
        fm = self.frame_map()
        return FrameMap(start=st["right_source_start"], stride=fm.stride, n_proxy=fm.n_proxy, source_fps=fm.source_fps)

    def frame_map(self) -> FrameMap:
        m = self.meta()["frames"]
        return FrameMap(start=m["source_start"], stride=m["stride"], n_proxy=m["n_proxy"], source_fps=m["source_fps"])

    def config(self, *overrides) -> Config:
        """Defaults <- this run's saved snapshot <- overrides (later wins)."""
        snapshot = None
        if self.config_path.exists():
            snapshot = yaml.safe_load(self.config_path.read_text()) or {}
            dropped = drop_unknown_keys(snapshot)
            if dropped:
                logging.getLogger(__name__).warning("ignoring obsolete config keys in %s: %s", self.config_path, dropped)
        return load_config(snapshot, *overrides)

    # ---- creation --------------------------------------------------------
    @classmethod
    def create(cls, root: Path, source: Path, fmap: FrameMap, cfg: Config, extra: dict | None = None,
               subdirs: bool = True) -> Run:
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("tracking", "recon", "fusion") if subdirs else ():
            (root / sub).mkdir(exist_ok=True)
        meta = {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "config_digest": cfg.digest(),
            "source": {"path": str(source)},
            "frames": {
                "source_start": fmap.start,
                "source_end": fmap.start + (fmap.n_proxy - 1) * fmap.stride,
                "stride": fmap.stride,
                "n_proxy": fmap.n_proxy,
                "source_fps": fmap.source_fps,
                "proxy_fps": fmap.proxy_fps,
            },
            **(extra or {}),
        }
        run = cls(root)
        run.meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
        run.config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        return run

    @classmethod
    def open(cls, root: str | Path, recon_name: str = "recon") -> Run:
        run = cls(Path(root), recon_name)
        if not run.meta_path.exists():
            raise FileNotFoundError(f"{root} is not a run directory (no run.yaml). Use `reeftrack prepare` first.")
        return run


def prepare_run(video: Path, start: int = 0, end: int | None = None, name: str | None = None,
                cfg: Config | None = None, force: bool = False, log=print) -> Run:
    """Probe `video`, create data/runs/<name>/ and the frame-exact proxy. Returns the Run."""
    from reeftrack.io.video import make_proxy, plan_frame_map, probe

    cfg = cfg or load_config()
    video = Path(video).resolve()
    info = probe(video)
    fmap = plan_frame_map(info, start, end, cfg.video.proxy_fps)
    run_name = name or f"{video.stem}_{fmap.start}-{fmap.start + (fmap.n_proxy - 1) * fmap.stride}"
    root = cfg.paths.data_root / run_name
    if not root.is_absolute():
        root = REPO_ROOT / root
    if (root / "run.yaml").exists() and not force:
        raise FileExistsError(f"{root} exists (use force=True / --force to overwrite)")
    run = Run.create(root, video, fmap, cfg, extra={"source": {
        "path": str(video), "width": info.width, "height": info.height,
        "fps": info.fps_float, "n_frames": info.n_frames, "codec": info.codec}})
    log(f"[prepare] {info.width}x{info.height}@{info.fps_float:.2f} -> {fmap.n_proxy} proxy frames "
        f"@ {fmap.proxy_fps:.2f} fps (every {fmap.stride}th frame), {cfg.video.proxy_height}p")
    make_proxy(video, run.proxy, fmap, cfg.video.proxy_height, cfg.video.rotate, cfg.video.crf, cfg.video.gop)
    log(f"[prepare] done: {root}")
    return run


def add_right_camera(run: Run, right_video: Path, baseline_m: float | None, cfg: Config | None = None,
                     n_windows: int = 8, log=print) -> dict:
    """Audio-sync a second (right) camera to the run's clip and build a frame-matched right proxy.

    Proxy frame i of `proxy_right.mp4` shows the same instant as proxy frame i of `proxy.mp4`
    (to within half a source frame, i.e. < 4.2 ms at 120 fps).
    """
    import numpy as np

    from reeftrack.io.sync import estimate_offset
    from reeftrack.io.video import make_proxy, probe

    cfg = cfg or run.config()
    right_video = Path(right_video).resolve()
    left_video = Path(run.meta()["source"]["path"])
    info_r = probe(right_video)
    fm = run.frame_map()
    t0, t1 = fm.time_s(0), fm.time_s(fm.n_proxy - 1)
    windows = list(np.linspace(t0, max(t0, t1 - 20), n_windows))
    sync = estimate_offset(left_video, right_video, fm.source_fps, windows)
    shift = -sync.offset_frames  # right source frame = left source frame + shift
    right_start = fm.start + int(round(shift))
    last = right_start + (fm.n_proxy - 1) * fm.stride
    if right_start < 0 or last >= info_r.n_frames:
        raise ValueError(f"right camera does not cover the clip (needs frames {right_start}-{last})")
    st = {
        "right_path": str(right_video),
        "baseline_m": baseline_m,
        "sync_offset_s": round(sync.offset_s, 6),
        "right_minus_left_frames": round(float(shift), 3),
        "right_source_start": right_start,
        "residual_frames": round(float(shift - round(shift)), 3),
        "sync_inliers": f"{sync.n_inliers}/{len(windows)}",
        "sync_spread_ms": round(sync.spread_ms, 2),
    }
    log(f"[stereo] right = left {shift:+.2f} source frames (inliers {st['sync_inliers']}, "
        f"spread {sync.spread_ms:.1f} ms) -> right proxy from source frame {right_start}")
    vc = cfg.video
    make_proxy(right_video, run.proxy_right, FrameMap(right_start, fm.stride, fm.n_proxy, fm.source_fps),
               vc.proxy_height, vc.rotate, vc.crf, vc.gop)
    run.update_meta(stereo=st)
    log(f"[stereo] done: {run.proxy_right}")
    return st
