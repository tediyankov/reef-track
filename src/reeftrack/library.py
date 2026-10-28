"""Shared video library: annotate many original videos with outputs on shared storage (RFS).

Three folders, set once per computer in `~/.reeftrack/library.yaml` (GUI: File → Library folders):

    video_root         originals, e.g. <hard drive>/originaldata          (read-only is fine)
    compressed_root    shared 1080p/30 fps copies, e.g. RFS/.../compressed_videos
    annotations_root   per-video annotation folders, e.g. RFS/.../annotations

A video's path below `video_root` is mirrored in both shared folders:

    originaldata/Maldives/2024_02_14/AM/Cait/Left/GX010703.MP4                   (original, 10-11 GB)
    compressed_videos/Maldives/2024_02_14/AM/Cait/Left/GX010703.mp4              (~1.4 GB, made once)
    compressed_videos/Maldives/2024_02_14/AM/Cait/Left/GX010703.reeftrack.json   (how it was made)
    annotations/Maldives/2024_02_14/AM/Cait/Left/GX010703/run.yaml               (video info, frame map)
    annotations/Maldives/2024_02_14/AM/Cait/Left/GX010703/behaviours_<name>.csv  (one per annotator)
    annotations/Maldives/2024_02_14/AM/Cait/Left/GX010703/loops_<name>.csv

No video is ever written to the annotations folder. The GUI plays a local copy of the
compressed video (`cache_dir`, pruned to `cache_limit_gb`), because jumping around a video
over the network takes ~0.3-1.7 s per jump; or, without a compressed copy, the original
(slower). Frame numbers are identical either way (frame i = original frame i x stride).

Compression is safe to share: it encodes on the local disk, checks the frame count against
the original, and only then copies the file to RFS under its final name, so a broken or
half-written copy never appears there. A lock file stops two people compressing the same
video at the same time. A copy whose .reeftrack.json went missing (e.g. only the .mp4 was
sent to someone) is recognised again once its frame count matches the original.

If `compressed_root` is on this computer (e.g. copies sent to an annotator without RFS),
videos play straight from it instead of being copied into the cache.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import yaml

from reeftrack.io.video import FrameMap, make_proxy, plan_frame_map, probe

ORIGINALS_DIRNAME = "originaldata"
SIDECAR_SUFFIX = ".reeftrack.json"
LOCK_STALE_H = 6.0


def settings_path() -> Path:
    return Path(os.environ.get("REEFTRACK_HOME", Path.home() / ".reeftrack")) / "library.yaml"


def _defaults() -> dict:
    if sys.platform == "darwin":
        rfs = "/Volumes/RFS/Triggerfish Navigation/fish_in_the_wild"
        return {"video_root": f"/Volumes/Seagate Backup Plus Drive/{ORIGINALS_DIRNAME}",
                "compressed_root": f"{rfs}/compressed_videos", "annotations_root": f"{rfs}/annotations"}
    return {"video_root": "", "compressed_root": "", "annotations_root": ""}  # Windows: picked in the GUI


@dataclass
class LibrarySettings:
    video_root: str = ""
    compressed_root: str = ""
    annotations_root: str = ""
    cache_dir: str = str(Path.home() / "reeftrack_cache")
    cache_limit_gb: float = 30.0
    height: int = 1080
    bitrate: str = "10M"
    proxy_fps: float = 30.0

    @classmethod
    def load(cls) -> LibrarySettings:
        p = settings_path()
        data = {**_defaults(), **(yaml.safe_load(p.read_text()) or {} if p.exists() else {})}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self) -> None:
        p = settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    LABELS = {"video_root": "Original videos (hard drive)", "compressed_root": "Compressed videos",
              "annotations_root": "Annotations"}

    def problems(self, required: tuple[str, ...] = ("video_root", "compressed_root", "annotations_root")) -> list[str]:
        """Human-readable reasons the library can't be used right now (empty = OK).

        Annotating from a compressed copy alone (no hard drive / RFS, e.g. annotations saved
        locally and synced later) only needs `annotations_root`.
        """
        out = []
        for key in required:
            label, v = self.LABELS[key], getattr(self, key)
            if not v:
                out.append(f"'{label}' folder is not set")
            elif not Path(v).is_dir():
                out.append(f"'{label}' folder not found: {v} (is the drive / RFS connected?)")
        return out


class CompressionBusy(RuntimeError):
    pass


class Library:
    def __init__(self, settings: LibrarySettings | None = None):
        self.s = settings or LibrarySettings.load()
        self.video_root = Path(self.s.video_root)
        self.compressed_root = Path(self.s.compressed_root)
        self.annotations_root = Path(self.s.annotations_root)
        self.cache_dir = Path(self.s.cache_dir).expanduser()

    # ---- paths -----------------------------------------------------------------------
    def rel(self, video: str | Path) -> PurePosixPath:
        """Path of `video` below the originals folder, e.g. Maldives/.../Left/GX010703.MP4.

        Falls back to the part after an `originaldata` folder, so a drive mounted under a
        different name (or drive letter) still maps to the same shared folders.
        """
        video = Path(video).resolve()
        try:
            return PurePosixPath(video.relative_to(self.video_root.resolve()).as_posix())
        except ValueError:
            parts = video.parts
            if ORIGINALS_DIRNAME in parts:
                i = len(parts) - 1 - parts[::-1].index(ORIGINALS_DIRNAME)
                return PurePosixPath(*parts[i + 1:])
            raise ValueError(f"{video} is not inside the original-videos folder ({self.video_root}). "
                             "Pick a video from that folder, or change it in File → Library folders.") from None

    def compressed_path(self, rel: PurePosixPath) -> Path:
        return self.compressed_root.joinpath(*rel.parent.parts, rel.stem + ".mp4")

    def sidecar_path(self, rel: PurePosixPath) -> Path:
        return self.compressed_root.joinpath(*rel.parent.parts, rel.stem + SIDECAR_SUFFIX)

    def annotation_dir(self, rel: PurePosixPath) -> Path:
        return self.annotations_root.joinpath(*rel.parent.parts, rel.stem)

    def cache_path(self, rel: PurePosixPath) -> Path:
        return self.cache_dir.joinpath(*rel.parent.parts, rel.stem + ".mp4")

    # ---- inspection ------------------------------------------------------------------
    def frame_map_for(self, video: Path) -> FrameMap:
        """Whole video at ~30 fps (every 4th frame of 120 fps GoPro footage). Raises a clear
        error for empty/corrupt files (many copies on the old drives are 0 bytes)."""
        return plan_frame_map(probe(video), 0, None, self.s.proxy_fps)

    def compressed_info(self, rel: PurePosixPath) -> dict | None:
        """Sidecar of a finished, verified compressed copy, or None."""
        sc, mp4 = self.sidecar_path(rel), self.compressed_path(rel)
        if not (sc.exists() and mp4.exists()):
            return None
        info = json.loads(sc.read_text())
        return info if mp4.stat().st_size == info.get("bytes") else None

    def compressed_status(self, rel: PurePosixPath) -> str:
        """Plain-language reason there is no usable compressed copy (for the 'not compressed' dialog)."""
        sc, mp4 = self.sidecar_path(rel), self.compressed_path(rel)
        if not mp4.exists():
            return f"No compressed copy found at {mp4}"
        if not sc.exists():
            return f"Found {mp4.name}, but its info file {sc.name} is missing next to it"
        return f"{mp4} doesn't match its info file {sc.name} (incomplete copy?)"

    def adopt_compressed(self, video: str | Path) -> dict | None:
        """Recognise a compressed copy whose .reeftrack.json is missing or stale (e.g. only the .mp4
        was sent): with the original at hand, check the copy has the expected number of frames and
        write a fresh info file. Returns the info, or None if there is no matching copy."""
        video = Path(video)
        rel = self.rel(video)
        mp4 = self.compressed_path(rel)
        if self.compressed_info(rel) or not mp4.exists():
            return self.compressed_info(rel)
        if mp4.with_suffix(".mp4.lock").exists():
            return None  # still being compressed
        fmap = self.frame_map_for(video)
        try:
            n = probe(mp4).n_frames
        except (OSError, ValueError):
            return None
        if n != fmap.n_proxy:
            return None
        self._write_sidecar(rel, video, fmap, n, mp4)
        return self.compressed_info(rel)

    def _write_sidecar(self, rel: PurePosixPath, video: Path, fmap: FrameMap, n_frames: int, mp4: Path) -> None:
        self.sidecar_path(rel).write_text(json.dumps({
            "source_rel": str(rel), "source_path": str(video), "source_frames": probe(video).n_frames,
            "source_fps": fmap.source_fps, "stride": fmap.stride, "n_frames": n_frames,
            "fps": fmap.proxy_fps, "height": self.s.height, "bitrate": self.s.bitrate,
            "bytes": mp4.stat().st_size, "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "created_by": f"{getpass.getuser()}@{platform.node()}",
        }, indent=1))

    def original_for(self, mp4: str | Path) -> Path | None:
        """The original video on the drive for a compressed copy inside `compressed_root`, if found."""
        try:
            rel = Path(mp4).resolve().relative_to(self.compressed_root.resolve())
        except ValueError:
            return None
        folder = self.video_root.joinpath(*rel.parent.parts)
        if not folder.is_dir():
            return None
        return next((p for p in folder.iterdir() if p.stem == rel.stem and p.suffix.lower() in (".mp4", ".mov")
                     and not p.name.startswith("._")), None)

    # ---- compression -----------------------------------------------------------------
    def _lock(self, rel) -> Path:
        return self.compressed_path(rel).with_suffix(".mp4.lock")

    def compress(self, video: str | Path, progress=None, should_stop=None, log=print) -> Path:
        """Make (once) the shared compressed copy of `video`. Returns its path on shared storage."""
        video = Path(video)
        rel = self.rel(video)
        if self.adopt_compressed(video):
            return self.compressed_path(rel)
        fmap = self.frame_map_for(video)
        lock = self._lock(rel)
        lock.parent.mkdir(parents=True, exist_ok=True)
        if lock.exists():
            age_h = (_dt.datetime.now().timestamp() - lock.stat().st_mtime) / 3600
            if age_h < LOCK_STALE_H:
                who = lock.read_text().strip()
                raise CompressionBusy(f"{rel} is already being compressed ({who}, started {age_h * 60:.0f} min ago). "
                                      "Try again later, or open the original video meanwhile.")
        lock.write_text(f"{getpass.getuser()} on {platform.node()} at {_dt.datetime.now():%Y-%m-%d %H:%M}")
        try:
            local = self.cache_path(rel)
            log(f"[compress] {rel}: {fmap.n_proxy} frames @ {fmap.proxy_fps:.2f} fps, {self.s.height}p -> {local}")
            out = make_proxy(video, local, fmap, self.s.height, 0, 23, 15, encoder="fast", bitrate=self.s.bitrate,
                             progress=progress, should_stop=should_stop)
            dst = self.compressed_path(rel)
            log(f"[compress] verified {out.n_frames} frames; copying to {dst}")
            _copy(local, dst, progress=None)
            self._write_sidecar(rel, video, fmap, out.n_frames, dst)
            self.prune_cache(keep=local)
            return dst
        finally:
            lock.unlink(missing_ok=True)

    # ---- local playback cache --------------------------------------------------------
    def ensure_local(self, rel: PurePosixPath, progress=None) -> Path:
        """Local copy of the compressed video (copied from shared storage if needed)."""
        info = self.compressed_info(rel)
        if info is None:
            raise FileNotFoundError(f"no compressed copy of {rel} yet")
        if _same_disk(self.compressed_path(rel), self.cache_dir):  # compressed folder is on this laptop
            return self.compressed_path(rel)
        local = self.cache_path(rel)
        if not (local.exists() and local.stat().st_size == info["bytes"]):
            _copy(self.compressed_path(rel), local, progress)
        os.utime(local)  # mark as recently used
        self.prune_cache(keep=local)
        return local

    def prune_cache(self, keep: Path | None = None) -> list[Path]:
        """Delete least-recently-used local copies above `cache_limit_gb`."""
        if not self.cache_dir.exists():
            return []
        files = sorted((p for p in self.cache_dir.rglob("*.mp4") if p.is_file()), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        removed = []
        for p in files:
            if total <= self.s.cache_limit_gb * 2**30:
                break
            if keep is not None and p.resolve() == Path(keep).resolve():
                continue
            total -= p.stat().st_size
            p.unlink()
            removed.append(p)
        return removed

    # ---- annotation runs -------------------------------------------------------------
    def open_run(self, video: str | Path, cfg=None, use_original: bool = False, progress=None):
        """Create/refresh the video's annotation folder and return a Run ready for the GUI.

        Plays the local copy of the compressed video, or the original if `use_original`
        (or no compressed copy exists and the caller chose so).
        """
        from reeftrack.config import load_config
        from reeftrack.io.run import Run

        video = Path(video)
        rel = self.rel(video)
        fmap = self.frame_map_for(video)
        root = self.annotation_dir(rel)
        cfg = cfg or load_config()
        source = {"path": str(video), "rel": str(rel), "bytes": video.stat().st_size}
        if not (root / "run.yaml").exists():
            run = Run.create(root, video, fmap, cfg, subdirs=False, extra={
                "layout": "library", "source": source,
                "compressed": str(PurePosixPath(*self.compressed_path(rel).relative_to(self.compressed_root).parts)),
            })
        else:
            run = Run.open(root)
            if run.frame_map() != fmap:
                raise RuntimeError(f"{root}/run.yaml describes a different video layout than {video}; "
                                   "refusing to mix annotations. Check the file / contact the maintainer.")
            run.update_meta(source=source)  # the drive may be mounted elsewhere on this computer
        if use_original or not self.adopt_compressed(video):
            if not use_original:
                raise FileNotFoundError(f"no compressed copy of {rel} yet")
            run.video, run.video_is_raw = video, True
        else:
            run.video, run.video_is_raw = self.ensure_local(rel, progress), False
        return run


    # ---- annotating from a compressed copy alone --------------------------------------
    @staticmethod
    def sidecar_for(mp4: str | Path) -> Path:
        mp4 = Path(mp4)
        return mp4.with_name(mp4.stem + SIDECAR_SUFFIX)

    def open_compressed(self, mp4: str | Path, cfg=None):
        """Open a compressed copy (+ its .reeftrack.json) without the original or shared storage.

        The annotation folder path comes from the info file (the original's path below
        `originaldata/`), so it is exactly the folder the video would have on RFS: annotations
        made offline can later be copied across with `reeftrack sync-annotations`.
        """
        from reeftrack.config import load_config
        from reeftrack.io.run import Run
        from reeftrack.io.video import probe

        mp4 = Path(mp4)
        sc = self.sidecar_for(mp4)
        original = None if sc.exists() else self.original_for(mp4)
        if original is not None:
            self.adopt_compressed(original)
        if not sc.exists():
            raise FileNotFoundError(f"{sc.name} is missing next to {mp4.name}. Copy both files (the .mp4 and the "
                                    ".reeftrack.json) from compressed_videos.")
        info = json.loads(sc.read_text())
        if mp4.stat().st_size != info.get("bytes"):
            raise ValueError(f"{mp4.name} is incomplete or not the file described by {sc.name} "
                             f"({mp4.stat().st_size} vs {info.get('bytes')} bytes); copy it again.")
        fmap = FrameMap(start=0, stride=int(info["stride"]), n_proxy=int(info["n_frames"]),
                        source_fps=float(info["source_fps"]))
        if probe(mp4).n_frames != fmap.n_proxy:
            raise ValueError(f"{mp4.name} doesn't have the expected {fmap.n_proxy} frames; copy it again.")
        rel = PurePosixPath(info["source_rel"])
        root = self.annotation_dir(rel)
        if not (root / "run.yaml").exists():
            run = Run.create(root, Path(info["source_path"]), fmap, cfg or load_config(), subdirs=False, extra={
                "layout": "library", "source": {"path": info["source_path"], "rel": str(rel),
                                                "frames": info.get("source_frames")},
                "compressed": str(rel.parent / (rel.stem + ".mp4"))})
        else:
            run = Run.open(root)
            if run.frame_map() != fmap:
                raise RuntimeError(f"{root}/run.yaml describes a different video layout than {mp4.name}; "
                                   "refusing to mix annotations.")
        run.video, run.video_is_raw = mp4, False
        return run


def sync_annotations(src: str | Path, dst: str | Path, dry_run: bool = False, log=print) -> dict:
    """Copy per-video annotation folders from `src` into `dst` (e.g. laptop -> RFS), safely.

    * annotation CSVs (per annotator): copied if missing in `dst`, or if the `src` file is
      newer and different; never replaced by an older copy.
    * run.yaml / config.yaml: copied only if the video folder is new in `dst`, and the frame
      layout must agree if it already exists there (otherwise the folder is skipped).
    Returns counts per action.
    """
    import filecmp

    src, dst = Path(src), Path(dst)
    counts = {"copied": 0, "updated": 0, "same": 0, "kept_newer_in_dest": 0, "skipped_mismatch": 0}
    for run_yaml in sorted(src.rglob("run.yaml")):
        d_src = run_yaml.parent
        rel = d_src.relative_to(src)
        d_dst = dst / rel
        if (d_dst / "run.yaml").exists():
            a = (yaml.safe_load(run_yaml.read_text()) or {}).get("frames")
            b = (yaml.safe_load((d_dst / "run.yaml").read_text()) or {}).get("frames")
            if a != b:
                log(f"SKIP   {rel}: frame layout differs from the destination's run.yaml")
                counts["skipped_mismatch"] += 1
                continue
        for f in sorted(p for p in d_src.iterdir() if p.is_file() and not p.name.startswith(".")):
            if f.suffix not in (".csv", ".yaml") or f.name.endswith(".tmp"):
                continue
            g = d_dst / f.name
            if f.suffix == ".yaml" and g.exists():
                continue  # keep the destination's video info
            if not g.exists():
                action = "copied"
            elif filecmp.cmp(f, g, shallow=False):
                counts["same"] += 1
                continue
            elif f.stat().st_mtime > g.stat().st_mtime:
                action = "updated"
            else:
                log(f"KEEP   {rel}/{f.name}: the destination copy is newer")
                counts["kept_newer_in_dest"] += 1
                continue
            log(f"{'WOULD ' if dry_run else ''}{action.upper():7s} {rel}/{f.name}")
            counts[action] += 1
            if not dry_run:
                _copy(f, g)
    return counts


def _same_disk(a: Path, b: Path) -> bool:
    """True if `a` is on the same (local) volume as `b`, so copying it into the cache gains nothing."""
    try:
        b.mkdir(parents=True, exist_ok=True)
        return a.stat().st_dev == b.stat().st_dev
    except OSError:
        return False


def _copy(src: Path, dst: Path, progress=None, chunk: int = 8 << 20) -> None:
    """Copy via a temporary name + rename, so `dst` is never a half-written file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    total, done = src.stat().st_size, 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while buf := fi.read(chunk):
            fo.write(buf)
            done += len(buf)
            if progress:
                progress(done, total)
    os.replace(tmp, dst)
    shutil.copystat(src, dst)
