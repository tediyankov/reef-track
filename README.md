# reef-track

Label-free pipeline that takes **one swim-through video** and produces

1. a **tracklet of a focal fish** (e.g. a Picasso triggerfish), started and patched by clicks in a GUI,
2. a **3D point cloud of the reef** (underwater feed-forward reconstruction), and
3. the **fish trajectory placed inside that 3D reef**.

Everything runs out of the box, with no training or labelling. The aim is to see how far
current foundation models (SAM 2/3 for tracking, Wat3R/VGGT for 3D) get on their own.

```
video ──prepare──► proxy.mp4 ──GUI clicks──► tracklet (2D masks, 30 fps)
                       │                              │
                       └──reconstruct──► reef.ply + cameras + depth
                                                      │
                                     fuse ◄───────────┘──► trajectory.csv + scene.html
```

Method details, in plain language: [docs/methods.md](docs/methods.md).

## Install

```bash
mamba env create --prefix ~/envs/reef-track -f environment.yml   # or: conda env create ...
eval "$(mamba shell hook --shell zsh)"                         # zsh (macOS default); use --shell bash on Linux/JASMIN
mamba activate ~/envs/reef-track                                # or, without mamba: conda activate ~/envs/reef-track
pip install -e . --no-deps
bash scripts/fetch_wat3r.sh           # vendors Wat3R model code into third_party/ (pinned)
pytest                                # ~90 tests, ~1 min, no weights or GPU needed
```

Model weights download automatically into `~/.cache/huggingface` on first use (set `HF_HOME`
to move them): Wat3R ≈ 4.7 GB, SAM 2.1-small ≈ 0.2 GB, SAM 3 ≈ 3.4 GB.
**SAM 3 is gated:** request access at <https://huggingface.co/facebook/sam3>, then run
`hf auth login`. Until access is granted the tracker automatically falls back to SAM 2.1.

## Use

```bash
# 1. make a run (any video, any directory); frames are *source* frame numbers
reeftrack prepare "/Volumes/.../GX010703.MP4" --start 12998 --end 38448 --name cait_GX010703_12998-38448

# 2. track the fish interactively
reeftrack gui data/runs/cait_GX010703_12998-38448

# 3. reconstruct the reef (GPU-heavy) and 4. place the fish in it
reeftrack reconstruct data/runs/cait_GX010703_12998-38448 -c configs/runs/cait_GX010703.yaml
reeftrack fuse        data/runs/cait_GX010703_12998-38448 -c configs/runs/cait_GX010703.yaml
reeftrack view        data/runs/cait_GX010703_12998-38448          # opens scene.html (or recon/reef.html)

# after tracking (or changing cleaning params) rebuild the cloud in seconds, fish masked out:
reeftrack reconstruct data/runs/cait_GX010703_12998-38448 -c configs/runs/cait_GX010703.yaml --reuse
```

### Stereo rig (Left + Right GoPros) → metric scale

```bash
R=data/runs/cait_GX010703_12998-38448
reeftrack add-right   $R "/Volumes/.../Cait/Right/GX010029.MP4" --baseline-cm 7   # audio-sync + right proxy
reeftrack reconstruct $R -c configs/runs/cait_GX010703.yaml --stereo             # -> recon_stereo/ (metres)
reeftrack fuse        $R -c configs/runs/cait_GX010703.yaml --recon recon_stereo # -> fusion_stereo/
reeftrack compare     $R                                                         # -> compare/ (mono vs stereo)
reeftrack view        $R --recon recon_stereo
```

Mono (`recon/`, `fusion/`) and stereo (`recon_stereo/`, `fusion_stereo/`) results live side by
side; nothing overwrites the other.

`reeftrack --help` lists everything. Other commands: `track` (headless single click) and
`replay` (re-run tracking from the clicks saved in `tracklet.json`, e.g. with SAM 3 on a GPU).

### GUI cheat-sheet

| action | how |
|---|---|
| start a tracklet | left-click the fish → white preview → **Enter** |
| fish lost (red tick on timeline) | **N** jumps to the next gap → scrub until it's back → click it → Enter (also back-fills the gap) |
| tracker drifted | click the fish on the bad frame → that segment is cut there and re-tracked |
| undo / save | **Ctrl+Z** / autosaves after every change (**Ctrl+S** to force) |
| playback | **Space**, **←/→** (±1 frame), **Shift+←/→** (±1 s), timeline click/drag |
| behaviour event (Behaviour tab) | **I** at first frame → **O** at last frame → pick behaviour (number key) + optional comment → OK |
| edit / delete behaviour | click a row to jump, double-click to edit, *Delete* button |
| loop (Loops tab) | **I** where the loop starts → **O** when the fish is back at that spot (start frame shown picture-in-picture) → confidence, landmark, comment → OK |

**Behaviour annotation.** The GUI also records frame-level behaviours (start frame, end frame,
behaviour, comment, annotator) to `annotations/behaviours.csv` in the run, saved immediately
after every change and independent of the mask tracking. For annotators who only do
behaviours there's a fast mode that doesn't load the tracking model:

```bash
reeftrack gui data/runs/<run> --behaviour-only
```

Categories are set in `configs/default.yaml` (`annotation.behaviours`). Annotator
instructions: [docs/annotating_behaviour.md](docs/annotating_behaviour.md).

## Outputs (per run, in `data/runs/<run>/`)

| file | what |
|---|---|
| `run.yaml`, `config.yaml` | provenance: source path, frame range/fps mapping, git commit, config snapshot |
| `proxy.mp4` | frame-exact 1080p/30 fps working copy |
| `tracking/tracklet.json`, `masks.npz`, `tracklet.csv` | segments, clicks, per-frame boxes/scores; RLE masks; CSV with **source** frame numbers and timestamps |
| `recon/reef.ply`, `reef.html` | coloured reef point cloud + browser preview (open in [CloudCompare](https://cloudcompare.org) / MeshLab / Open3D) |
| `recon/cameras.npz`, `depth.npz`, `keyframes.json` | per-keyframe camera poses/intrinsics, depth+confidence, diagnostics |
| `fusion/trajectory.csv` | fish x, y, z per frame (+ `kind`: measured / depth_interp / gap_interp) |
| `fusion/scene.html`, `scene.ply`, `summary.json` | interactive 3D viewer (offline, shareable), combined cloud, path stats |
| `proxy_right.mp4`, `run.yaml: stereo` | frame-matched right-camera proxy; audio-sync offset + baseline |
| `recon_stereo/…` (+ `cameras_right.npz`, `depth_right.npz`) | stereo reconstruction **in metres**; `keyframes.json` holds rig diagnostics |
| `fusion_stereo/…` | fish trajectory in metres (incl. `height_above_reef`) |
| `annotations/behaviours.csv` | behaviour events: start/end frame (run + source), times, behaviour, comment, annotator |
| `annotations/loops.csv` | loops (fish returns to a known place): frames/times, confidence, landmark, comment, annotator + 3D return distance, path length, max excursion from the trajectory |
| `compare/report.json`, `compare.html`, `overlay.ply` | mono vs stereo: scale, scale drift, cloud-to-cloud distances, fish paths (`reeftrack compare --mono A --stereo B` compares any two variants) |

## Mac or JASMIN?

Measured on an M3 Pro (36 GB) for the 3.5-min example clip (6,363 frames at 30 fps):

| step | Mac (MPS) | comment |
|---|---|---|
| prepare (4K HEVC → proxy) | ~5 min | I/O-bound (USB drive) |
| SAM 2.1 tracking | ~0.6 s / frame | fine for interactive segments; a full-clip pass ≈ 1 h |
| Wat3R reconstruction | ~40 s / 32-frame chunk, ~11 GB | 13 min for 427 keyframes (18 chunks) |

**Recommendation:** for one test video the Mac is sufficient. Do the clicking in the GUI
locally, and run full-clip SAM 3 replays and reconstructions of many videos on JASMIN
ORCHID GPUs (`scripts/jasmin/`, which also covers env setup with the HF cache on a group workspace).
The code picks CUDA → MPS → CPU automatically (`device: auto`).

```bash
bash scripts/jasmin/setup_env.sh /gws/nopw/j04/<gws>/reef-track-cache   # once
mkdir -p logs && sbatch scripts/jasmin/reconstruct.slurm data/runs/<run> configs/runs/<run>.yaml
```

## Repository layout

```
configs/            default.yaml (all parameters, commented) + runs/<run>.yaml overrides
src/reeftrack/
  io/               video probing, frame-exact proxy, run folders & provenance
  tracking/         SAM backends, loss detection (presence/area/motion/appearance), tracklet model, session logic
  recon/            keyframes, Wat3R/VGGT wrapper, chunk stitching (Sim3), point-cloud cleaning
  fusion/           2D tracklet → 3D trajectory, HTML/PLY scene export
  gui/              PySide6 app (video canvas, timeline, live tracking, behaviour annotation)
  annotation/       interval annotations: behaviours + loops (one CSV each per run)
  cli.py            `reeftrack` command
scripts/            fetch_wat3r.sh, jasmin/ SLURM jobs
tests/              unit + synthetic end-to-end tests (no weights needed)
data/, third_party/ git-ignored (outputs, vendored code)
```
