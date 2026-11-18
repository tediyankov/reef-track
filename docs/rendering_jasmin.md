# Rendering the reef: WaterSplatting + I²-NeRF on JASMIN

Goal: for held-out frames of each video, the **real frame** next to what each method renders
from **the same camera** (position, direction, field of view), plus PSNR/SSIM.

| | WaterSplatting (3DV 2025) | I²-NeRF (NeurIPS 2025) |
|---|---|---|
| what | 3D Gaussians for the reef + a volumetric water model | NeRF (ZipNeRF) with a physical absorption/scattering model |
| code | github.com/water-splatting/water-splatting (Apache-2.0) | github.com/ShuhongLL/I2-NeRF (Apache-2.0) |
| training | ~15k steps, expect tens of minutes per scene on an A100 | 20k steps, expect ~1 h per scene |
| extra output | the same view **with the water removed** | – |

Neither needs pretrained weights: both are fitted to each scene. What they need from us — camera
poses and a starting point cloud — comes from reef-track's stereo Wat3R reconstruction, which is
also why COLMAP (which fails in turbid water) is not needed.

```
laptop                                   JASMIN
------                                   ------
1 reeftrack export-colmap  ──rsync──►  2 setup_envs.sh (once)
                                         3 submit_all.sh  →  GPU jobs (ORCHID)
5 reeftrack render-compare ◄──rsync──  4 renders of the held-out frames
```

## 1. Laptop: export each run (done for the three stereo runs)

```bash
reeftrack export-colmap data/runs/valerio_GX020083_loop
reeftrack export-colmap data/runs/cait_GX010176_loop
reeftrack export-colmap data/runs/cait_GX010703_12998-38448
```

Each makes `data/render_exports/<run>/` (60–130 MB): 960×540 keyframes warped to one shared
camera, COLMAP poses + 300k reef points, `poses_bounds.npy`, `test_images.txt` (every 8th image,
held out from training by both methods) and `checks/` (the points drawn through the exported
cameras next to the real frame: they must line up).

## 2. JASMIN: one-off setup

Prerequisites: a JASMIN account with the `jasmin-login` role **and the `orchid` role** (GPU
cluster; request it at accounts.jasmin.ac.uk → Services → Additional services → orchid), and your
JASMIN SSH key loaded on the laptop (`ssh-add ~/.ssh/<your jasmin key>`).

Copy the exports and the scripts (from the reef-track folder on the laptop; replace `USER`):

```bash
rsync -av --exclude checks --exclude compare data/render_exports/ USER@xfer-vm-01.jasmin.ac.uk:reefrender/scenes/
rsync -av scripts/jasmin/ USER@xfer-vm-01.jasmin.ac.uk:reefrender/jasmin/
```

Log in and build the two environments (~45–60 min, once). `gpuhost001` is JASMIN's interactive
GPU machine; any sci server works too (the GPU type is set explicitly, so building needs no GPU):

```bash
ssh -A USER@login-01.jasmin.ac.uk
ssh -A gpuhost001.jc.rl.ac.uk
nohup bash ~/reefrender/jasmin/setup_envs.sh > ~/reefrender/setup.log 2>&1 &
tail -f ~/reefrender/setup.log          # Ctrl+C stops watching, not the setup
```

It ends with `watersplatting OK …`, `i2nerf OK …` and `setup finished`. Everything goes inside the
reefrender folder (set `REEFRENDER` to put that folder in a group workspace): its own Miniforge
(`miniforge3/`, with two environments, ~15 GB), download caches (`cache/`), temporary files
(`tmp/`) and the two code repositories (`code/`). Nothing large is written to your home
directory. Safe to re-run if anything fails half-way.

## 3. JASMIN: submit the jobs

From a sci server (Slurm commands live there):

```bash
ssh -A sci-vm-01.jasmin.ac.uk
bash ~/reefrender/jasmin/submit_all.sh          # 3 scenes x 2 methods = 6 GPU jobs
squeue -u $USER                                 # PD = waiting, R = running
tail -f ~/reefrender/logs/ws_<jobid>.log        # or i2nerf_<jobid>.log
```

Each job trains, then renders the held-out views:
* WaterSplatting → `~/reefrender/outputs/watersplatting/<scene>/renders/test/{rgb,rgb_clear}/`
* I²-NeRF        → `~/reefrender/outputs/i2nerf_unbounded/<scene>/test_preds/color_NNN.png`

To try one scene first: `bash ~/reefrender/jasmin/submit_all.sh valerio_GX020083_loop`.

## 4. Laptop: fetch the renders

```bash
rsync -av --exclude 'nerfstudio_models' --exclude '*.ckpt' USER@xfer-vm-01.jasmin.ac.uk:reefrender/outputs/ data/render_results/
```

## 5. Laptop: side-by-sides

```bash
for r in valerio_GX020083_loop cait_GX010176_loop cait_GX010703_12998-38448; do
  reeftrack render-compare data/render_exports/$r \
    --watersplatting data/render_results/watersplatting/$r/renders \
    --i2nerf data/render_results/i2nerf_unbounded/$r
done
```

Per run, in `data/render_exports/<run>/compare/`: `compare_<image>.png` for every held-out frame
(real | WaterSplatting | WaterSplatting without water | I²-NeRF), `contact_sheet.jpg` (6 frames
spread over the video) and `metrics.csv` (PSNR / SSIM against the real frame).

## If something goes wrong

* **Setup fails building `tiny-cuda-nn` or an extension**: send me the last ~40 lines of
  `~/reefrender/setup.log`.
* **A job fails**: send me the end of its log in `~/reefrender/logs/`. `sacct -j <jobid>` shows
  the exit state.
* **Out of GPU memory (I²-NeRF)**: lower the batch size, e.g.
  `--gin_bindings="Config.batch_size = 4096"` in `train_i2nerf.sbatch`.
* **I²-NeRF renders are empty/garbage**: its underwater configs were only tested by the authors on
  forward-facing scenes; our default runs it in its unbounded (any camera path) mode. The fallback
  is the authors' exact forward-facing setup:
  `I2_MODE=ff sbatch ~/reefrender/jasmin/train_i2nerf.sbatch <scene>` (outputs in
  `outputs/i2nerf_ff/`). Best for transects where the camera keeps facing one way.
* **Pending for long (`QOSGrpCpuLimit` / `Resources`)**: the ORCHID queue is busy; jobs start when
  GPUs free up (max 8 jobs per user, 24 h each).

## Notes

* **Why images are slightly zoomed**: Wat3R estimates the focal length per frame (±10%) but
  I²-NeRF needs one shared camera. Changing only a pinhole camera's intrinsics is an exact image
  warp, so every frame is resampled to the run's largest focal length (crop, never black borders).
  The real frames in the comparison are these same warped images, so the pairs match exactly.
* **Resolution**: 960×540 keeps training quick. For sharper renders re-export with
  `--width 1920` (4× the pixels; expect ~2–4× longer training).
* **The fish** moves, so it is not part of the static reef: expect it to be blurred or missing in
  renders. Fine for judging the reef.
