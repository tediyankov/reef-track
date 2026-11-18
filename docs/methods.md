# Methods, in plain language

This document explains what each stage does, why it was chosen, and what its limits are.
Parameters referenced here live in `configs/default.yaml`.

---

## 0. Preparing the video

Raw GoPro files here are 4K HEVC at 119.88 fps, about 11 GB per 20 minutes. Nothing downstream
needs that, so `prepare` decodes the chosen frame range **once** into a 1080p, 29.97 fps
H.264 *proxy* (every 4th frame). The proxy has a keyframe every 15 frames, so the GUI can
jump to any frame exactly and quickly.

The mapping `source_frame = start + 4 × proxy_frame` is stored in `run.yaml`, and every output
(CSV files, viewer labels) is reported in **original source frame numbers and seconds**, so
results can always be traced back to the raw footage. The trim is frame-exact: a unit test
checks it on a synthetic video whose frame numbers are encoded as brightness.

## 1. Tracking the focal fish

### Why SAM (and not YOLO-seg)?
YOLO11-seg is trained on COCO, which has **no fish class**, so it cannot segment a
triggerfish out of the box. SAM-family models are *promptable*: given one click they segment
"the object under this point", whatever it is, which is exactly the label-free starting point
we need. Their video versions then carry that mask from frame to frame using a *memory* of
recent frames.

* **SAM 3** (Meta, Nov 2025) is the current state of the art and the default. Its tracker
  keeps the SAM 2 interface and adds a stronger backbone and a "presence" head that predicts
  whether the object is visible at all. Its weights are gated, so access must be requested.
* **SAM 2.1** is used automatically until access is granted. In tests on this video it already
  gives clean masks of a small striped fish 100+ frames after the click.

### When does the tracklet break? (`tracking.*`)
SAM trackers rarely "give up": when the fish hides behind coral they keep returning
something. We therefore score every frame with five cheap, label-free signals:

| signal | idea |
|---|---|
| presence | SAM's own "object visible?" score is negative |
| empty | the mask vanished (or is a few pixels) |
| area | the mask suddenly grows or shrinks by more than ×4 versus the recent median (it has swallowed coral or kept only a fin) |
| motion | the mask's centre is far from where a **Kalman filter** (a constant-velocity motion model) predicted it. This is the SAMURAI idea, and it catches jumps onto a neighbouring fish |
| appearance | the colour histogram (hue/saturation/value) inside the mask no longer resembles the fish you clicked |

If any signal fires the frame is *suspect*, and it is **not stored** in SAM's memory, so the
tracker doesn't learn the wrong object (SAMURAI-style memory gating). After 3 consecutive
suspect frames (`patience`) the segment ends at the last good frame. Suspect frames are never
written to the tracklet.

### How patching works
A tracklet is one fish identity made of **segments** separated by gaps. Each click creates a segment:

1. SAM segments the clicked fish.
2. The tracker runs **forwards** until the fish is lost or the clip ends.
3. It also runs **backwards** from the click until it is lost again or it meets the previous
   segment. You usually notice the fish a little after it re-emerges from the coral; the
   backward pass recovers those frames automatically, so gaps shrink to the true occlusion.

A frame is owned by at most one segment (no double counting). Clicking a frame that is
already tracked is a **correction**: the segment is cut there and re-tracked from your click.
Every click is saved, so `reeftrack replay` reproduces the whole tracklet, for example with
SAM 3 on JASMIN.

## 2. Reconstructing the reef in 3D

### Feed-forward reconstruction
Classic photogrammetry (Structure-from-Motion + Multi-View Stereo, e.g. COLMAP/Agisoft) matches
keypoints between photos and optimises camera positions for hours, and it often fails in
turbid, low-texture or repetitive underwater scenes. **VGGT** (CVPR 2025) is a large
transformer that, given a set of frames, predicts in a single forward pass:

* a **depth map** per frame (distance of every pixel from the camera),
* a **confidence** per pixel,
* the **camera pose** (position and orientation) and **intrinsics** (focal length) of every frame.

**Wat3R** (Ren et al., 2026) is VGGT adapted to underwater footage **without labels**: a teacher
network generates pseudo-targets on unlabelled underwater videos, and a student learns from
them with a cross-view consistency loss. This makes it more robust to the colour loss and haze
of water. We use it by default; plain VGGT is selectable (`recon.backend: vggt`) for comparison.

### From depth maps to a point cloud
A depth map plus a camera pose is enough to place every pixel in 3D (*unprojection*). Doing
that for all keyframes and merging them gives the reef **point cloud**: a list of (x, y, z)
coordinates with an RGB colour each, stored as a standard `.ply` file.

Steps and their parameters:

1. **Keyframes** (`keyframe_fps: 3`): consecutive frames at 30 fps are nearly identical, so
   3D needs viewpoint change (*parallax*). We take about 3 frames per second, picking the
   sharpest frame in each slot to avoid motion blur.
2. **Chunks** (`chunk_size: 32`, `chunk_overlap: 8`): GPU memory limits how many frames the
   model sees at once. Each chunk is reconstructed in its own coordinate frame and scale.
3. **Stitching:** neighbouring chunks share 8 frames. For those frames both chunks predict a
   3D point for every pixel, which gives thousands of matching point pairs. From them we solve
   for the *similarity transform* (rotation, translation and one scale factor) that maps one
   chunk onto the other (Umeyama's method, with outlier trimming), and chain these
   transforms. The median alignment residual per chunk is stored in `keyframes.json` as a
   quality check.
4. **Cleaning:**
   - drop the least-confident 40 % of pixels per frame;
   - drop the farthest 5 % (open water and back-scatter, where depth is meaningless);
   - remove the tracked fish, since moving objects smear;
   - *voxel downsample*: keep one point per small cube, which removes redundancy between views;
   - remove statistical outliers ("floaters");
   - after levelling, remove points above the *typical* (median) camera height. In shallow
     water the underside of the surface acts as a mirror and the model reconstructs it as a
     "ceiling" just above a snorkeller. On the example clip this layer sat clearly apart from
     the reef (reef z ≤ 0.03, surface z ≈ 0.22–0.29, camera median 0.07). Switch it off
     (`drop_above_camera: null`) for walls or overhangs filmed from below.
5. **Levelling:** a plane is fitted to the cloud with RANSAC (the seabed) and the scene is
   rotated so the seabed is horizontal, with +z pointing towards the diver's side. The
   z-coordinate then reads as height relative to the seabed plane. The fish trajectory also
   gets a `height_above_reef` column: the fish's z minus the median z of reef points directly
   beneath it, which follows coral heads rather than a flat plane.

### Units and scale (important)
A single moving camera cannot tell a small reef filmed close-up from a big reef filmed from
afar: monocular reconstruction is correct **up to an unknown scale**. Shapes, relative
distances, angles and the fish's path *relative to the reef* are meaningful. Metres are not.
Ways to get metric scale, roughly in order of effort:

* **the stereo rig (Left + Right GoPros):** the known baseline between the cameras fixes the
  scale exactly. Implemented: see "Stereo reconstruction" below.
* a known-size object in view (scale bar, quadrat, a diver's fin), measured once;
* the GoPro's IMU telemetry (GPMF), which gives gravity direction and, more weakly, motion scale.

### Stereo reconstruction (`--stereo`)
The rig carries two GoPros 7 cm apart (lens centre to centre: the *baseline*).

1. **Synchronising the cameras.** They are started by hand, so their clocks differ. Both
   record the same sounds (breathing, fin kicks, bubbles), so we cross-correlate the two
   audio tracks (GCC-PHAT, a standard time-delay estimator that emphasises sharp sounds) in 8
   windows across the clip. For the example clip all 8 agree to within 0.8 ms: the right
   video lags by 10.42 source frames. The right proxy is cut 10 frames later, leaving a
   residual of 0.42 source frames (3.5 ms), which is negligible at 30 fps.
2. **Pairs, not singles.** Each keyframe is fed to Wat3R as a Left+Right pair taken at the same
   instant. The second viewpoint helps the model with depth and lets the right camera's depth
   maps add coverage to the cloud.
3. **Metric scale, estimated jointly.** In each chunk the model predicts both camera positions of
   every pair. Their distance *should* be 7 cm, so each chunk gives its own (noisy) estimate of
   the factor that turns it into metres. Neighbouring chunks also share frames, and aligning
   them measures their *relative* scale. All 36 chunk scales are solved **together** by weighted
   least squares. How much to trust each kind of evidence is **learned from the data**
   (variance-component estimation: each group's noise level is adjusted until its residuals are
   consistent with it), rather than set by hand.

   How we got here (all versions are kept in the run folder; numbers from `compare*/report.json`):

   | version | per-chunk scale | seam misfit (median / max) | surface thickness | mono drift seen |
   |---|---|---|---|---|
   | mono `recon` | free per join | 0.035 / 0.087 | 7.6 mm | (reference) |
   | `recon_stereo_v1` | own baselines only | 0.105 / 0.748 | 9.9 mm | ×8.8 (inflated by v1's own noise) |
   | `recon_stereo_v2` | joint, hand-set weights | 0.053 / 0.244 | 9.6 mm | ×3.2 |
   | `recon_stereo` (final) | joint, learned weights | 0.052 / 0.239 | 9.6 mm | ×3.2 |
   | `recon_stereo_global` | overlaps + one global scale | 0.035 / 0.106 | 10.0 mm | ×2.3 (keeps drift) |

   The learned weights showed that overlap joins and per-chunk baselines are *about equally*
   noisy (~10 % in scale each), so the final result is essentially v2. Wat3R's neighbouring
   chunks genuinely disagree by ~10 % in scale: mono absorbs that as drift, stereo as small seams.

   **Why the stereo cloud looks slightly more smudged than mono.** The surface is ~2 mm thicker
   (≈9.6 vs 7.6 mm at a 3 cm scale), but *not* because of depth quality or chunk seams:
   - single keyframes are equally crisp (2.1 mm stereo vs 2.0 mm mono);
   - the `global` variant has mono-quality seams yet is just as thick;
   - Left-only and Right-only stereo clouds are each ~9 mm.

   The extra thickness appears when keyframes are merged: in pair mode, consecutive views are
   registered to each other a couple of mm less precisely. Plausibly this is because each
   chunk then spans half the time (16 instants instead of 32), so the model sees less parallax
   per chunk. Adding the Right camera's depth adds ~0.7 mm more (`stereo_right_depth: false`
   removes it, at the cost of ~25 % fewer points). An earlier version of the comparison page
   also exaggerated the difference by drawing a fixed number of points per cloud; both are now
   shown on the same metric grid.
4. **No scale drift.** After rescaling, chunks are joined with rotation and translation only.
   The mono pipeline must also estimate a free scale at every join, and small errors there
   accumulate as *scale drift* along the swim.
5. **Checks** (in `recon_stereo/keyframes.json`): within each chunk, how constant the predicted
   baseline is (coefficient of variation) and the predicted Left-Right rotation. A rigid rig
   should give near-constant values; large spreads mean the model was unsure.

Caveat: a 7 cm baseline is short (similar to human eyes). It pins down *scale* very well
across many frames, but on its own it would triangulate poorly beyond a couple of metres.
That is why it is combined with Wat3R's multi-view geometry rather than used for classical
stereo matching.

### Comparing mono and stereo (`reeftrack compare`)
* **surface thickness** (mm): around 20,000 sample points, how far the points within 3 cm
  scatter from their best-fit local plane. A crisp reconstruction scores close to the reef's
  real small-scale roughness; misaligned overlapping views ("smudging") score higher. This
  objectively measures what you see by eye;
* both clouds are displayed on the **same metric voxel grid**, because a fixed point budget per
  cloud made the denser stereo cloud look grainier than it is;
Both reconstructions use the same keyframes, so their camera paths can be matched frame for
frame:
* one similarity transform maps the mono path onto the stereo path; its scale converts mono
  units to metres, and the residual says how similar the two path *shapes* are;
* the same fit on ~10 s sliding windows shows the **mono scale drift** (a ratio of 1.0 means
  no drift);
* after alignment, nearest-neighbour distances between the clouds (in cm) show where the two
  geometries disagree; `compare/overlay.ply` shows them together (mono blue, stereo orange);
* if both fusions exist, the fish paths are compared too (length, speed, point-wise distance).

### Storage formats
* `.ply`: the standard point-cloud format; opens in CloudCompare, MeshLab, Blender and Open3D.
* `.npz`: compressed numpy arrays (cameras, depth maps) for further analysis in Python.
* Masks are run-length encoded, so a whole tracklet fits in one small file.

## 3. Placing the fish in the reef

For each tracked frame:

1. Take the fish's 2D centre from the tracklet (30 fps).
2. Get the camera pose. At keyframes it comes from the reconstruction; in between, the
   rotation is interpolated smoothly (SLERP) and the position linearly.
3. Get the fish's distance from the camera. At keyframes it is the **median depth inside the
   (slightly eroded) fish mask** (`measured`); between keyframes it is interpolated in time
   (`depth_interp`). A short median filter suppresses one-off depth errors.
4. Shoot a ray from the camera through the fish centre to that distance: that is the fish's 3D
   position in reef coordinates.
5. Gaps shorter than `max_interp_gap_s` (2 s) between segments are bridged with straight
   lines (`gap_interp`); longer gaps stay empty.

`trajectory.csv` keeps the `kind` column so analyses can use measured points only.

## Known limitations and next steps

* **Small fish depth:** at the model's 518-px input width a distant fish covers only tens
  of pixels, so its depth is noisier than the reef's. Possible improvements: crops around the
  fish at higher resolution, or a monocular metric-depth model as a second opinion.
* **Look-alike fish:** motion and colour cues do not separate two Picasso triggerfish that
  cross closely. That is when a correction click is needed. A learned re-ID embedding (e.g.
  DINOv2 features) is a natural next step.
* **Surface / above-water footage** confuses the 3D model; restrict reconstruction with
  `recon.frame_range`.
* **Speed on a Mac:** SAM tracking runs at about 0.6 s per frame on an M3 Pro, so long
  passes belong on a GPU.
