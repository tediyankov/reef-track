"""Build a re-identification dataset from the annotated library.

For every individual in the annotations tracker whose videos are annotated for occlusions:

1. Read its behaviour annotations and find the longest stretch with no "Out of view" or
   "Sheltering" event (consecutive GoPro chapters count as one recording).
2. Run the triggerfish detector (RT-DETR) on that stretch of the compressed video.
3. Keep the N most confident frames, at least `min_gap_s` apart, as full-frame JPEGs.

Output (`out_dir`, e.g. RFS/.../fish_in_the_wild/reid_dataset):

    frames/<fish_num>/<video_id>/frame<frame_number>.jpg
    metadata.csv      one row per image (see COLUMNS)
    intervals.csv     the stretch used for each fish and how many frames came from it
    README.md         what the columns mean and how the dataset was made

Frame numbers are those of the compressed video (30 fps), the same numbers the annotation
CSVs use. Each fish is written separately (`.parts/`), so an interrupted export resumes.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

from reeftrack.library import Library
from reeftrack.reid.intervals import (
    OCCLUDED,
    Segment,
    consecutive_chapters,
    longest_clear_run,
    occlusion_mask,
)
from reeftrack.reid.select import pick_spaced

COLUMNS = {
    "filepath": "Image path, relative to the dataset folder (prepend the dataset's location, e.g. on JASMIN)",
    "fish_num_orig": "Fish ID in the annotations tracker",
    "fish_num": "Identity label for training (prefixed so it cannot clash with other datasets)",
    "source_folder": "Which export / sampling scheme the image comes from",
    "video_id": "GoPro file name of the source video",
    "frame_number": "Frame in the compressed video (30 fps), as used by the annotation CSVs",
    "timestamp_sec": "Time of the frame from the start of the original video file",
    "video_path": "Compressed video, relative to the fish_in_the_wild folder on RFS",
    "source_frame_number": "Frame in the original 120 fps GoPro file",
    "det_confidence": "Highest triggerfish confidence from the detector in this frame",
    "bbox_x1": "Detected box (pixels, full frame), left",
    "bbox_y1": "top",
    "bbox_x2": "right",
    "bbox_y2": "bottom",
    "n_detections": "Number of triggerfish boxes in the frame (>1: check the box is the focal fish)",
    "image_width": "Image width (pixels)",
    "image_height": "Image height (pixels)",
    "site": "Site",
    "date": "Date of the observation",
    "observer": "Diver who filmed the fish",
    "annotator": "Whose occlusion annotations defined the stretch",
}
INTERVAL_COLUMNS = ["fish_num_orig", "fish_num", "start_video", "start_frame", "end_video", "end_frame",
                    "duration_s", "frames_scored", "frames_with_fish", "frames_kept"]


@dataclass
class Individual:
    fish_id: int
    site: str
    date: str          # YYYY_MM_DD, as in the library folders
    period: str
    observer: str
    view: str
    annotator: str
    videos: list[str] = field(default_factory=list)


@dataclass
class Settings:
    weights: Path
    n_per_fish: int = 100
    min_gap_s: float = 1.0
    stride: int = 3            # run the detector on every 3rd frame (10 per second)
    conf: float = 0.25
    imgsz: int = 1024          # the checkpoint was trained at 1024
    batch: int = 8
    device: str = "auto"
    source_folder: str = "wild_longest_interval"
    label_prefix: str = "wild_"
    jpeg_quality: int = 95


# ---- inputs -------------------------------------------------------------------------------
def _yes(v) -> bool:
    return str(v).strip().lower() == "yes"


def read_tracker(path: str | Path) -> list[Individual]:
    """Individuals and their occlusion-annotated videos from the annotations tracker (.xlsx/.csv)."""
    import pandas as pd

    path = Path(path)
    t = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    t = t.dropna(subset=["Fish ID", "Video MP4 name"])
    t = t[t["Out of view annotated"].map(_yes)]
    fish: dict[int, Individual] = {}
    for _, r in t.iterrows():
        fid = int(r["Fish ID"])
        ind = fish.setdefault(fid, Individual(
            fid, str(r["Site"]).strip(), pd.Timestamp(r["Date"]).strftime("%Y_%m_%d"),
            str(r["Observation Period"]).strip(), str(r["Observer"]).strip(), str(r["L/R view"]).strip(),
            str(r["Assigned to"]).strip().lower()))
        ind.videos.append(str(r["Video MP4 name"]).strip())
    return [fish[k] for k in sorted(fish)]


def find_rel(lib: Library, ind: Individual, video: str) -> PurePosixPath | None:
    """Library path of a tracker row. The tracker says 'AM1' where some folders say 'AM'."""
    for period in dict.fromkeys([ind.period, re.sub(r"\d+$", "", ind.period)]):
        rel = PurePosixPath(ind.site, ind.date, period, ind.observer, ind.view, f"{video}.MP4")
        if (lib.annotation_dir(rel) / "run.yaml").exists():
            return rel
    return None


def behaviour_events(lib: Library, rel: PurePosixPath, annotator: str) -> list[tuple[str, int, int]]:
    """(behaviour, start, end) from the shared CSV, or from this computer's backup copy if newer
    (annotations saved while the shared drive was unreachable only exist there)."""
    name = f"behaviours_{annotator}.csv"
    home = Path(os.environ.get("REEFTRACK_HOME", Path.home() / ".reeftrack"))
    found = [p for p in (lib.annotation_dir(rel) / name, home.joinpath("backups", *rel.parent.parts, rel.stem, name))
             if p.exists()]
    if not found:
        raise FileNotFoundError(f"no {name} for {rel}")
    with max(found, key=lambda p: p.stat().st_mtime).open(newline="") as f:
        return [(r["behaviour"], int(r["start_frame"]), int(r["end_frame"])) for r in csv.DictReader(f)]


# ---- detector -----------------------------------------------------------------------------
class Detector:
    """RT-DETR triggerfish detector. Returns, per frame, (best confidence, best box, n boxes)."""

    def __init__(self, s: Settings):
        from ultralytics import RTDETR

        from reeftrack.devices import resolve_device

        self.model = RTDETR(str(s.weights))
        self.s = s
        self.device = resolve_device(s.device)

    def __call__(self, frames_bgr: list[np.ndarray]) -> list[tuple[float, tuple[float, ...], int]]:
        out = []
        for res in self.model.predict(frames_bgr, conf=self.s.conf, imgsz=self.s.imgsz, device=self.device,
                                      verbose=False):
            conf = res.boxes.conf.cpu().numpy()
            if len(conf) == 0:
                out.append((0.0, (np.nan,) * 4, 0))
                continue
            k = int(conf.argmax())
            out.append((float(conf[k]), tuple(float(x) for x in res.boxes.xyxy[k].cpu().numpy()), len(conf)))
        return out


def score_segment(video: Path, seg: Segment, detect: Callable, stride: int, batch: int,
                  progress: Callable[[int], None] | None = None) -> list[tuple[int, float, tuple, int]]:
    """(frame, confidence, box, n boxes) for every `stride`-th frame of the segment."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start)
    out, frames, idx = [], [], []

    def flush():
        for i, r in zip(idx, detect(frames)):
            out.append((i, *r))
        if progress:
            progress(len(frames))
        frames.clear()
        idx.clear()

    try:
        for i in range(seg.start, seg.end + 1):
            if (i - seg.start) % stride:
                if not cap.grab():
                    break
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
            idx.append(i)
            if len(frames) == batch:
                flush()
        if frames:
            flush()
    finally:
        cap.release()
    return out


def read_frame(video: Path, frame: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise OSError(f"failed to decode frame {frame} of {video}")
    return bgr


# ---- export -------------------------------------------------------------------------------
def export_individual(lib: Library, ind: Individual, s: Settings, out_dir: Path, detect: Callable,
                      log: Callable[[str], None] = print) -> tuple[list[dict], dict] | None:
    """Frames + metadata rows for one fish. None if it has no usable footage."""
    import cv2
    from tqdm import tqdm

    label = f"{s.label_prefix}{ind.fish_id:02d}"
    rels = {v: find_rel(lib, ind, v) for v in ind.videos}
    for v in [v for v, r in rels.items() if r is None]:
        log(f"  {label}: skipping {v} (no annotation folder)")
    meta = {v: yaml.safe_load((lib.annotation_dir(r) / "run.yaml").read_text()) for v, r in rels.items() if r}

    # longest clear stretch, over each run of consecutive chapters
    best: list[Segment] = []
    for group in consecutive_chapters(list(meta)):
        masks = []
        for v in group:
            n = int(meta[v]["frames"]["n_proxy"])
            masks.append((v, occlusion_mask(n, behaviour_events(lib, rels[v], ind.annotator), OCCLUDED)))
        cand = longest_clear_run(masks)
        if sum(x.n_frames for x in cand) > sum(x.n_frames for x in best):
            best = cand
    if not best:
        log(f"  {label}: no un-occluded footage")
        return None

    # score every stride-th frame, keeping one timeline across chapters for the spacing rule.
    # Scores are saved, so changing N / the gap later doesn't re-run the detector.
    cache = out_dir / ".parts" / f"{label}.scores.csv"
    stamp = f"{best[0].video}:{best[0].start}-{best[-1].video}:{best[-1].end}/{s.stride}"
    scored = _load_scores(cache, stamp)
    if scored is None:
        scored, offset = [], 0.0
        total = sum(x.n_frames for x in best) // s.stride
        with tqdm(total=total, desc=f"{label} detect", unit="frame", leave=False) as bar:
            for seg in best:
                fps = float(meta[seg.video]["frames"]["proxy_fps"])
                local = lib.ensure_local(rels[seg.video])
                for frame, conf, box, n in score_segment(local, seg, detect, s.stride, s.batch, bar.update):
                    scored.append({"video": seg.video, "frame": frame, "t": offset + (frame - seg.start) / fps,
                                   "conf": conf, "box": box, "n": n})
                offset += seg.n_frames / fps
        _save_scores(cache, stamp, scored)
    keep = pick_spaced([r["t"] for r in scored], [r["conf"] for r in scored], s.n_per_fish, s.min_gap_s)

    rows = []
    for k in tqdm(keep, desc=f"{label} write", unit="img", leave=False):
        r = scored[k]
        v, rel, fr = r["video"], rels[r["video"]], meta[r["video"]]["frames"]
        bgr = read_frame(lib.ensure_local(rel), r["frame"])
        relpath = PurePosixPath("frames", label, v, f"frame{r['frame']}.jpg")
        dst = out_dir.joinpath(*relpath.parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, s.jpeg_quality])
        if not ok:
            raise OSError(f"could not encode {relpath}")
        dst.write_bytes(buf.tobytes())
        src_frame = int(fr["source_start"]) + r["frame"] * int(fr["stride"])
        rows.append({
            "filepath": str(relpath), "fish_num_orig": ind.fish_id, "fish_num": label,
            "source_folder": s.source_folder, "video_id": v, "frame_number": r["frame"],
            "timestamp_sec": round(src_frame / float(fr["source_fps"]), 3),
            "video_path": str(PurePosixPath(lib.compressed_root.name, *rel.parent.parts, rel.stem + ".mp4")),
            "source_frame_number": src_frame, "det_confidence": round(r["conf"], 4),
            **{f"bbox_{c}": round(x, 1) for c, x in zip(("x1", "y1", "x2", "y2"), r["box"])},
            "n_detections": r["n"], "image_width": bgr.shape[1], "image_height": bgr.shape[0],
            "site": ind.site, "date": ind.date.replace("_", "-"), "observer": ind.observer,
            "annotator": ind.annotator,
        })
    fps0 = float(meta[best[0].video]["frames"]["proxy_fps"])
    interval = {"fish_num_orig": ind.fish_id, "fish_num": label, "start_video": best[0].video,
                "start_frame": best[0].start, "end_video": best[-1].video, "end_frame": best[-1].end,
                "duration_s": round(sum(x.n_frames for x in best) / fps0, 1), "frames_scored": len(scored),
                "frames_with_fish": sum(r["conf"] > 0 for r in scored), "frames_kept": len(rows)}
    log(f"  {label}: {interval['duration_s'] / 60:.1f} min stretch "
        f"({best[0].video} f{best[0].start} -> {best[-1].video} f{best[-1].end}), {len(rows)} frames")
    return rows, interval


_SCORE_COLS = ["stamp", "video", "frame", "t", "conf", "x1", "y1", "x2", "y2", "n"]


def _save_scores(path: Path, stamp: str, scored: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"stamp": stamp, "video": r["video"], "frame": r["frame"], "t": round(r["t"], 4),
             "conf": round(r["conf"], 5), **dict(zip(("x1", "y1", "x2", "y2"), r["box"])), "n": r["n"]}
            for r in scored]
    _write_csv(path, rows, _SCORE_COLS)


def _load_scores(path: Path, stamp: str) -> list[dict] | None:
    """Saved detector scores for this exact stretch and stride, or None."""
    if not path.exists():
        return None
    rows = _read_csv(path)
    if not rows or rows[0]["stamp"] != stamp:
        return None
    return [{"video": r["video"], "frame": int(r["frame"]), "t": float(r["t"]), "conf": float(r["conf"]),
             "box": tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2")), "n": int(r["n"])} for r in rows]


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def export(tracker: str | Path, out_dir: str | Path, s: Settings, lib: Library | None = None,
           fish: list[int] | None = None, overwrite: bool = False, detect: Callable | None = None,
           log: Callable[[str], None] = print) -> Path:
    """Export every individual (or just `fish`) and write metadata.csv. Returns its path."""
    lib = lib or Library()
    out_dir = Path(out_dir)
    parts = out_dir / ".parts"
    parts.mkdir(parents=True, exist_ok=True)
    individuals = [i for i in read_tracker(tracker) if fish is None or i.fish_id in fish]
    detect = detect or Detector(s)
    for ind in individuals:
        label = f"{s.label_prefix}{ind.fish_id:02d}"
        done = parts / f"{label}.csv"
        if done.exists() and not overwrite:
            log(f"  {label}: already exported (use --overwrite to redo)")
            continue
        res = export_individual(lib, ind, s, out_dir, detect, log)
        if res is not None:
            rows, interval = res
            _write_csv(parts / f"{label}.interval.csv", [interval], INTERVAL_COLUMNS)
            _write_csv(done, rows, list(COLUMNS))

    rows = [r for p in sorted(parts.glob(f"{s.label_prefix}*[0-9].csv")) for r in _read_csv(p)]
    intervals = [r for p in sorted(parts.glob("*.interval.csv")) for r in _read_csv(p)]
    _write_csv(out_dir / "metadata.csv", rows, list(COLUMNS))
    _write_csv(out_dir / "intervals.csv", intervals, INTERVAL_COLUMNS)
    (out_dir / "README.md").write_text(_readme(s, len(rows), len(intervals)))
    log(f"{len(rows)} images of {len(intervals)} fish -> {out_dir}")
    return out_dir / "metadata.csv"


def _readme(s: Settings, n_images: int, n_fish: int) -> str:
    cols = "\n".join(f"| `{k}` | {v} |" for k, v in COLUMNS.items())
    return f"""# Re-identification frames: fish in the wild

{n_images} full-frame images of {n_fish} Picasso triggerfish, exported
{_dt.date.today().isoformat()} with `reeftrack reid-export` (code: `src/reeftrack/reid/` in the
reef-track repository).

How the frames were chosen, per individual:

1. The longest stretch of its recording with no "Out of view" or "Sheltering" annotation
   (behaviour CSVs in `../annotations`). Consecutive GoPro chapters count as one recording.
2. The triggerfish detector (`{Path(s.weights).name}`, RT-DETR, {s.imgsz} px, confidence >= {s.conf})
   was run on one frame in every {s.stride} of that stretch in the compressed video (1080p, 30 fps).
3. The {s.n_per_fish} frames with the highest detection confidence were kept, at least
   {s.min_gap_s:g} s apart so they are not near-duplicates.

The detector finds triggerfish, not this individual: where `n_detections` > 1, check that the
box is the focal fish. `intervals.csv` lists the stretch used for each fish.

`metadata.csv` columns:

| Column | Meaning |
| --- | --- |
{cols}
"""
