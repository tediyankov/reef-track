"""Re-ID images from the aquarium background-variation experiment (one fish per video, many backdrops).

    reeftrack reid-aquarium background_var_experiments.csv --out <dataset folder>

Input: a CSV with `video_path, video_name, fish_ID` (rows without a fish_ID, and macOS `._`
files, are skipped). For every video:

1. Read only its **keyframes** (one per second in GoPro files), at full resolution. Reading the
   card is the bottleneck, and keyframes need no inter-frame decoding.
2. Run the RT-DETR triggerfish detector. A keyframe is a **candidate** when it has exactly one
   detection (two = the fish plus its mirror image in the tank's glass end panel, or the fish cut
   in two by the panel edge: rejected) with confidence >= `conf_keep`.
3. Save a square crop around the fish (box's long side + `pad`, full resolution), like the existing
   aquarium images (`fish_<id>_<video>_<frame>_bbox.jpg`).

Then, per individual, sample **the same number** of images (`n_per_fish`): spread as evenly as
possible over that fish's videos (= backdrops), random within a video, never two closer than
`min_gap_s` in time. If a fish cannot supply `n_per_fish`, every fish gets the smallest count, so
classes stay balanced.

Output folder:
    images/                     the crops
    metadata.csv                one row per image: fish, video, frame, time, box, crop, confidence
    fish_id_data.csv            the pipeline's all_fish_id_data.csv format, JASMIN paths
    fish_id_data_rfs.csv        same, paths in this folder (e.g. on RFS)
    detections.csv              every keyframe scored (provenance; lets the sampling be redone)
    summary.csv                 images per fish and video
    README.md
Candidate crops and per-video detections are cached (`cache_dir`), so an interrupted run resumes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from reeftrack.config import REPO_ROOT
from reeftrack.reid.select import pick_spaced

JASMIN_ROOT = "/gws/ssde/j25b/iecdt/tyankov/triggerfish-id/data"
META_COLUMNS = ["filepath", "fish_num", "source", "video_id", "frame_number", "timestamp_sec", "video_path",
                "det_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "crop_x", "crop_y", "crop_size",
                "image_width", "image_height", "md5"]
DET_COLUMNS = ["video_id", "fish_num", "frame_number", "timestamp_sec", "det_confidence", "n_detections",
               "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "candidate"]


@dataclass
class AquariumSettings:
    weights: Path = REPO_ROOT / "models" / "RTDETR_aq_and_wild.pt"
    n_per_fish: int = 120
    min_gap_s: float = 2.0
    conf: float = 0.25        # detections counted from this confidence (reflections included)
    conf_keep: float = 0.7    # a candidate's single fish must be at least this confident (below 0.7:
                              # backdrop prints, objects outside the tank, fish+reflection boxes)
    imgsz: int = 1024         # detector input size (the model was trained at 1024)
    device: str = "auto"
    pad: float = 0.1          # crop side = box long side * (1 + pad)
    jpeg_quality: int = 95
    seed: int = 0
    source: str = "aquarium_background_experiment"
    exclude_fish: tuple[int, ...] = ()  # individuals left out (e.g. too little footage to reach n_per_fish)
    jasmin_root: str = JASMIN_ROOT
    rfs_dir: str | None = None  # where the folder will live on RFS (fish_id_data_rfs.csv); default: `out`
    layout: str = "flat"        # "flat": images/<crop> | "per_individual": images/fish_<id>/<crop> + labels.csv


# ------------------------------------------------------------------------------------------ input
def read_video_list(path: str | Path) -> list[tuple[Path, int]]:
    """(video, fish id) for rows with a fish_ID; macOS resource forks (._*) are skipped."""
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("video_name") or Path(row["video_path"]).name).strip()
            fid = (row.get("fish_ID") or "").strip()
            if name.startswith("._") or not fid:
                continue
            out.append((Path(row["video_path"]), int(float(fid))))
    return out


# ------------------------------------------------------------------------------------------ video
def probe(video: Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def keyframes(video: Path, w: int, h: int):
    """Yield (time_s, BGR frame) for every keyframe, decoding nothing else."""
    # showinfo reports each frame's timestamp at log level "info" (on stderr, read by a thread)
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-skip_frame", "nokey", "-i", str(video),
           "-fps_mode", "passthrough",
           "-vf", "showinfo", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times: list[float] = []

    def read_info():  # showinfo prints one line per output frame on stderr
        for line in p.stderr:
            m = re.search(rb"pts_time:\s*([0-9.]+)", line)
            if m:
                times.append(float(m.group(1)))

    t = threading.Thread(target=read_info, daemon=True)
    t.start()
    size = w * h * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            for _ in range(500):  # the info line can lag the frame slightly (stderr is read concurrently)
                if len(times) > i or not t.is_alive():
                    break
                t.join(0.01)
            if len(times) <= i:
                raise RuntimeError(f"no timestamp for keyframe {i} of {video.name}")
            ts = times[i]
            yield ts, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()


def square_crop(box, w: int, h: int, pad: float) -> tuple[int, int, int]:
    """(x0, y0, side): square around the box, slid inside the frame."""
    x1, y1, x2, y2 = box
    side = int(round(min(max(x2 - x1, y2 - y1) * (1 + pad), w, h)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = int(np.clip(round(cx - side / 2), 0, w - side))
    y0 = int(np.clip(round(cy - side / 2), 0, h - side))
    return x0, y0, side


def crop_name(fish: int, video_id: str, frame: int) -> str:
    return f"fish_{fish}_{video_id}_{frame:06d}_bbox.jpg"


# ------------------------------------------------------------------------------------------ scoring
def score_video(video: Path, fish: int, detect, s: AquariumSettings, cache: Path, log=print) -> list[dict]:
    """All keyframes of one video, scored; candidate crops saved in cache/<video_id>/. Cached."""
    import cv2

    vid = video.stem
    done = cache / f"{vid}.csv"
    if done.exists():
        return _read_csv(done)
    w, h, fps = probe(video)
    crops = cache / vid
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    batch: list[tuple[float, np.ndarray]] = []

    def flush():
        for (ts, frame), (conf, box, n) in zip(batch, detect([f for _, f in batch])):
            fr = int(round(ts * fps))
            cand = n == 1 and conf >= s.conf_keep
            row = {"video_id": vid, "fish_num": fish, "frame_number": fr, "timestamp_sec": round(ts, 3),
                   "det_confidence": round(conf, 4), "n_detections": n, "candidate": int(cand),
                   **{k: (round(v, 1) if n else "") for k, v in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), box)}}
            if cand:
                x0, y0, side = square_crop(box, w, h, s.pad)
                cv2.imwrite(str(crops / crop_name(fish, vid, fr)), frame[y0:y0 + side, x0:x0 + side],
                            [cv2.IMWRITE_JPEG_QUALITY, s.jpeg_quality])
                row.update(crop_x=x0, crop_y=y0, crop_size=side, image_width=w, image_height=h)
            rows.append(row)
        batch.clear()

    for ts, frame in keyframes(video, w, h):
        batch.append((ts, frame))
        if len(batch) == 8:
            flush()
    if batch:
        flush()
    _write_csv(done, rows, DET_COLUMNS + ["crop_x", "crop_y", "crop_size", "image_width", "image_height"])
    n_c = sum(r["candidate"] for r in rows)
    log(f"[aquarium] {vid} (fish {fish}): {len(rows)} keyframes, {n_c} candidates")
    return rows


# ------------------------------------------------------------------------------------------ sampling
def allocate(capacity: list[int], n: int) -> list[int]:
    """Spread n picks over videos as evenly as their capacities allow (round robin)."""
    take = [0] * len(capacity)
    while sum(take) < n:
        open_ = [i for i, c in enumerate(capacity) if take[i] < c]
        if not open_:
            break
        for i in open_:
            if sum(take) >= n:
                break
            take[i] += 1
    return take


def spaced_random_order(rows: list[dict], min_gap_s: float, rng) -> list[dict]:
    """Candidates of one video in a random order that keeps every pair >= min_gap_s apart.

    The list is as long as the video allows; taking its first k gives k random, well-spaced frames.
    """
    if not rows:
        return []
    t = [float(r["timestamp_sec"]) for r in rows]
    idx = pick_spaced(t, rng.random(len(rows)) + 1e-9, len(rows), min_gap_s)  # random priorities
    order = sorted(idx, key=lambda i: rng.random())
    return [rows[i] for i in order]


def sample(det_rows: list[dict], n_per_fish: int, min_gap_s: float, seed: int) -> tuple[dict[int, list[dict]], int]:
    """fish -> chosen candidate rows (equal count per fish), and that count."""
    rng = np.random.default_rng(seed)
    by_fish: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in det_rows:
        if int(r["candidate"]):
            by_fish[int(r["fish_num"])][r["video_id"]].append(r)
    orders = {f: {v: spaced_random_order(sorted(rs, key=lambda r: float(r["timestamp_sec"])), min_gap_s, rng)
                  for v, rs in sorted(vids.items())} for f, vids in sorted(by_fish.items())}
    n = min([n_per_fish] + [sum(len(o) for o in vids.values()) for vids in orders.values()])
    chosen = {}
    for f, vids in orders.items():
        names = list(vids)
        take = allocate([len(vids[v]) for v in names], n)
        chosen[f] = [r for v, k in zip(names, take) for r in vids[v][:k]]
    return chosen, n


# ------------------------------------------------------------------------------------------ export
def export(video_csv: str | Path, out: str | Path, s: AquariumSettings, cache: str | Path | None = None,
           detect=None, log=print) -> dict:
    from reeftrack.reid.export import Detector

    out = Path(out)
    cache = Path(cache or REPO_ROOT / "data" / "reid_aquarium_cache")
    videos = read_video_list(video_csv)
    if not videos:
        raise ValueError(f"no videos with a fish_ID in {video_csv}")
    missing = [v for v, _ in videos if not v.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos not found, e.g. {missing[0]} (is the card mounted?)")
    videos = [(v, f) for v, f in videos if f not in set(s.exclude_fish)]
    if s.exclude_fish:
        log(f"[aquarium] excluding fish {sorted(s.exclude_fish)}")
    log(f"[aquarium] {len(videos)} videos, {len({f for _, f in videos})} individuals")
    detect = detect or Detector(s)
    det_rows = []
    for k, (v, f) in enumerate(videos, 1):
        log(f"[aquarium] video {k}/{len(videos)}: {v.name}")
        det_rows += score_video(v, f, detect, s, cache, log)
    for r in det_rows:  # cached scores may come from a run with a lower threshold: re-apply this one
        r["candidate"] = int(int(r["n_detections"]) == 1 and float(r["det_confidence"]) >= s.conf_keep)
    chosen, n = sample(det_rows, s.n_per_fish, s.min_gap_s, s.seed)
    if n < s.n_per_fish:
        log(f"[aquarium] NOTE: the scarcest fish has only {n} usable frames -> {n} per fish for all")

    img_dir = out / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    vpath = {v.stem: str(v) for v, _ in videos}
    meta, pipe_j, pipe_r, summary = [], [], [], defaultdict(int)
    for f, rows in chosen.items():
        for r in sorted(rows, key=lambda r: (r["video_id"], int(r["frame_number"]))):
            name = crop_name(f, r["video_id"], int(r["frame_number"]))
            rel = f"fish_{f}/{name}" if s.layout == "per_individual" else name  # path inside images/
            (img_dir / rel).parent.mkdir(exist_ok=True)
            shutil.copyfile(cache / r["video_id"] / name, img_dir / rel)
            md5 = hashlib.md5((img_dir / rel).read_bytes()).hexdigest()
            meta.append({"filepath": f"images/{rel}", "fish_num": f, "source": s.source, "video_id": r["video_id"],
                         "frame_number": r["frame_number"], "timestamp_sec": r["timestamp_sec"],
                         "video_path": vpath[r["video_id"]], "md5": md5,
                         **{k: r[k] for k in ("det_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "crop_x",
                                              "crop_y", "crop_size", "image_width", "image_height")}})
            pipe_j.append({"filepath": f"{s.jasmin_root.rstrip('/')}/{out.name}/images/{rel}", "fish_num": f,
                           "file_exists": True, "img_hash": md5})
            rfs = Path(s.rfs_dir) if s.rfs_dir else out.resolve()
            pipe_r.append({"filepath": str(rfs / "images" / rel), "fish_num": f, "file_exists": True, "img_hash": md5})
            summary[(f, r["video_id"])] += 1
    _write_csv(out / "metadata.csv", meta, META_COLUMNS)
    _write_csv(out / "fish_id_data.csv", pipe_j, ["filepath", "fish_num", "file_exists", "img_hash"])
    _write_csv(out / "fish_id_data_rfs.csv", pipe_r, ["filepath", "fish_num", "file_exists", "img_hash"])
    if s.layout == "per_individual":  # the JASMIN loader's minimal format: absolute path + identity
        _write_csv(out / "labels.csv", pipe_j, ["filepath", "fish_num"])
    _write_csv(out / "detections.csv", det_rows,
               DET_COLUMNS + ["crop_x", "crop_y", "crop_size", "image_width", "image_height"])
    vids_of = defaultdict(list)
    for v, f in videos:
        vids_of[f].append(v.stem)
    cand = defaultdict(int)
    for r in det_rows:
        cand[(int(r["fish_num"]), r["video_id"])] += int(r["candidate"])
    _write_csv(out / "summary.csv", [{"fish_num": f, "video_id": v, "candidates": cand[(f, v)], "images": summary[(f, v)]}
                                     for f in sorted(vids_of) for v in vids_of[f]],
               ["fish_num", "video_id", "candidates", "images"])
    (out / "README.md").write_text(_readme(s, n, len(chosen), len(meta), len(videos), out.name))
    log(f"[aquarium] {len(meta)} images ({n} per fish x {len(chosen)} fish) -> {out}")
    return {"n_per_fish": n, "n_fish": len(chosen), "n_images": len(meta)}


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _readme(s: AquariumSettings, n: int, n_fish: int, n_img: int, n_videos: int, name: str) -> str:
    cfg = {k: str(v) for k, v in asdict(s).items()}
    return f"""# Aquarium background-variation re-ID images

{n_img} images: **{n} per individual x {n_fish} individuals**, from {n_videos} videos (one fish per
video, a different tank backdrop per video).{f" Fish {', '.join(map(str, sorted(s.exclude_fish)))} left out "
"(too little footage for this many images per fish)." if s.exclude_fish else ""} Made by `reeftrack reid-aquarium` (reef-track repo,
`src/reeftrack/reid/aquarium.py`).

## How images were chosen
* Only keyframes were read (one per second, full 4K resolution).
* RT-DETR triggerfish detector (`{Path(s.weights).name}`, input {s.imgsz}). A frame was usable when it
  had exactly one detection at confidence >= {s.conf} and that detection had confidence >= {s.conf_keep}.
  Two detections = the fish plus its mirror image in the glass end panel (or split by the panel edge),
  so those frames were dropped.
* Square crop around the box (long side + {int(s.pad * 100)}%), at full resolution, named like the
  existing aquarium images: `fish_<id>_<video>_<frame>_bbox.jpg` (frame = frame number in the
  original video).
* Per individual, the same number of images, spread as evenly as possible over its videos
  (backdrops), random within a video (seed {s.seed}), no two closer than {s.min_gap_s} s.

## Files
* `images/` - the crops{" (one folder per individual, `fish_<id>/`)" if s.layout == "per_individual" else ""}.
* `labels.csv` (per-individual layout) - `filepath, fish_num`: absolute JASMIN paths, one row per image.
* `fish_id_data.csv` - pipeline format (`filepath, fish_num, file_exists, img_hash`) with JASMIN
  paths under `{s.jasmin_root}/{name}/`; `fish_id_data_rfs.csv` - same with RFS paths.
* `metadata.csv` - per image: video, frame, time, detector box and confidence, crop, source path.
* `detections.csv` - every keyframe of every video, scored (to re-sample without re-reading videos).
* `summary.csv` - usable frames and chosen images per fish and video.

## Using it for re-ID evaluation
`fish_num` is the original aquarium ID, the same individuals as in the other aquarium data. Frames
from one video share a backdrop and are seconds apart: for a test split that is not optimistic,
keep each video (`video_id`) entirely in train *or* test for known individuals, rather than
splitting images at random.

## Settings
```
{json.dumps(cfg, indent=1)}
```
"""
