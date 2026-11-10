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
