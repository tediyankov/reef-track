# Re-ID frame export

`reeftrack reid-export` builds a still-image dataset for re-identification from the annotated
library. The code is in `src/reeftrack/reid/`; it only reads the library and doesn't touch the
tracking / 3D pipeline.

```bash
conda activate ~/envs/reef-track
caffeinate -i reeftrack reid-export "Fish in the wild_ annotations tracker.xlsx"
```

Needs RFS connected (compressed videos + annotations, as set in File → Library folders) and
the detector checkpoint at `models/RTDETR_aq_and_wild.pt` (or `--weights`).

## What it does, per individual

1. Takes every video the tracker marks **Out of view annotated = Yes**, and reads its
   `behaviours_<assigned annotator>.csv` (or this computer's backup copy, if newer).
2. Finds the longest stretch with no **Out of view** or **Sheltering** event. GoPro chapters
   (GX01…, GX02…) of the same recording are joined, so a stretch can cross from one into the next.
3. Runs the RT-DETR triggerfish detector on every 3rd frame (10 per second) of that stretch.
4. Keeps the 100 most confident frames that are at least 1 s apart, as full 1080p JPEGs.

## Output

Default: `fish_in_the_wild/reid_dataset/` on RFS (next to `compressed_videos/`).

```
reid_dataset/
  frames/wild_01/GX010703/frame12345.jpg
  metadata.csv     one row per image: filepath, fish_num_orig, fish_num, source_folder,
                   video_id, frame_number, timestamp_sec, video_path, + detector box/confidence
  intervals.csv    the stretch used per fish, frames scored / with a fish / kept
  README.md        column descriptions
  .parts/          per-fish results + saved detector scores (lets a run resume)
```

`filepath` is relative to `reid_dataset/`; `video_path` is relative to `fish_in_the_wild/`.
`frame_number` is the compressed-video frame (the one the annotation CSVs use) and
`source_frame_number` the frame in the original 120 fps file.

## Options

| Option | Default | |
| --- | --- | --- |
| `--n` | 100 | frames per fish |
| `--min-gap` | 1.0 | minimum seconds between kept frames |
| `--stride` | 3 | run the detector on every Nth frame |
| `--fish 4 --fish 7` | all | only these Fish IDs |
| `--out` | `…/fish_in_the_wild/reid_dataset` | dataset folder |
| `--overwrite` | off | redo fish already exported (saved detector scores are reused, so changing `--n` or `--min-gap` is fast) |

The detector finds *a* triggerfish, not *this* one: check rows with `n_detections` > 1.

## Aquarium background-variation experiment (`reeftrack reid-aquarium`)

One fish per video, a different tank backdrop per video; per-video fish IDs in a CSV
(`video_path, video_name, fish_ID`; rows without an ID are skipped). Code:
`src/reeftrack/reid/aquarium.py`, tests: `tests/test_reid_aquarium.py`.

* Only keyframes are read (one per second, full 4K): reading the SD card is the bottleneck.
* RT-DETR (`models/RTDETR_aq_and_wild.pt`): a frame is usable with exactly one detection
  (>= 0.25) of confidence >= 0.7. Two detections are the fish and its reflection in the glass end
  panel; below 0.7 the detector also fires on fish printed on the backdrops, objects outside the
  tank and fish+reflection boxes.
* Square crop (box long side + 10%) at full resolution, named like the existing aquarium images:
  `fish_<id>_<video>_<frame>_bbox.jpg`.
* The same number of images per individual, spread evenly over its videos (backdrops), random
  within a video, at least 2 s apart. Detections and candidate crops are cached
  (`data/reid_aquarium_cache/`), so re-sampling with other settings takes seconds.

Built in Nov 2026 as 11 fish x 120 images (fish 54 left out: only three short videos, ~80 usable
images at 2 s spacing):

```bash
reeftrack reid-aquarium background_var_experiments.csv --out data/reid_datasets/background_experiment \
  --rfs-dir /Volumes/RFS/triggerfish_individual_id/Datasets/background_experiment \
  --n 120 --min-gap 2 --exclude-fish 54 --conf-keep 0.7
rsync -a data/reid_datasets/background_experiment/ /Volumes/RFS/triggerfish_individual_id/Datasets/background_experiment/
bash scripts/jasmin/push_dataset.sh data/reid_datasets/background_experiment   # -> JASMIN .../triggerfish-id/data/
```

Build locally, then copy: writing ~1,300 files straight to RFS failed once when the share's login
expired mid-run.
