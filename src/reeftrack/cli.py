"""Command-line entry point: `reeftrack <command> ...`.

Typical workflow (any video, any directory):

    reeftrack prepare /path/to/GX010703.MP4 --start 12998 --end 38448 --name cait_0703
    reeftrack gui data/runs/cait_0703              # click the fish, patch breaks, save
    reeftrack reconstruct data/runs/cait_0703      # 3D reef (GPU-heavy; JASMIN-friendly)
    reeftrack fuse data/runs/cait_0703             # fish trajectory in reef coordinates
    reeftrack view data/runs/cait_0703             # open the interactive 3D scene

Heavy imports happen inside commands so `reeftrack --help` stays instant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__.split("\n\n")[0])

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML overriding configs/default.yaml")]


@app.command()
def prepare(
    video: Annotated[Path, typer.Argument(help="Source video (any directory).")],
    start: Annotated[int, typer.Option(help="First source frame (inclusive).")] = 0,
    end: Annotated[int | None, typer.Option(help="Last source frame (inclusive); default = end of video.")] = None,
    name: Annotated[str | None, typer.Option(help="Run name; default = <video stem>_<start>_<end>.")] = None,
    config: ConfigOpt = None,
    force: bool = typer.Option(False, help="Overwrite an existing run."),
) -> None:
    """Create a run folder and a frame-exact 1080p working proxy of the clip."""
    from reeftrack.config import load_config
    from reeftrack.io.run import prepare_run

    try:
        prepare_run(video, start, end, name, load_config(config), force, log=typer.echo)
    except FileExistsError as e:
        raise typer.BadParameter(str(e)) from e


@app.command("add-right")
def add_right(
    run_dir: Path,
    right_video: Annotated[Path, typer.Argument(help="The right-camera video of the same dive.")],
    baseline_cm: Annotated[float | None, typer.Option(help="Lens centre-to-centre distance in cm.")] = None,
    left_original: Annotated[Path | None, typer.Option(
        help="Left camera's original video, for its audio (needed when the run uses a compressed copy).")] = None,
    config: ConfigOpt = None,
) -> None:
    """Add the second camera of a stereo rig: audio-sync it and build a frame-matched right proxy."""
    from reeftrack.io.run import Run, add_right_camera

    run = Run.open(run_dir)
    try:
        add_right_camera(run, right_video, None if baseline_cm is None else baseline_cm / 100, run.config(config),
                         log=typer.echo, left_audio=left_original)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@app.command()
def annotate(
    video: Annotated[Path | None, typer.Argument(help="Original video (on the hard drive). Omit to pick one.")] = None,
    config: ConfigOpt = None,
    behaviour_only: bool = typer.Option(False, "--behaviour-only",
                                        help="No mask tracking, even if PyTorch is installed (faster start)."),
) -> None:
    """Annotate behaviours/loops on library videos (hard drive + RFS). The main command for annotators.

    With the full environment installed, fish can be mask-tracked here too; masks are saved per
    annotator in the video's annotation folder (tracking_<name>/), next to behaviours_<name>.csv."""
    from reeftrack.gui.app import main

    main(None, config, behaviour_only=behaviour_only, library=True, video=video)


@app.command("check-install")
def check_install(
    tracking: bool = typer.Option(True, help="Also require the mask-tracking packages (PyTorch, transformers)."),
) -> None:
    """Check that every package the GUI needs is importable; exits with an error naming what's missing."""
    import importlib

    mods = ["numpy", "cv2", "PySide6.QtWidgets", "yaml", "pydantic"]
    if tracking:
        mods += ["torch", "torchvision", "transformers", "huggingface_hub"]
    missing = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # ImportError, or a broken binary package
            missing.append(f"{m} ({str(e).splitlines()[0][:120]})")
    if missing:
        typer.echo("MISSING / BROKEN:\n  - " + "\n  - ".join(missing))
        raise typer.Exit(1)
    typer.echo("all packages OK")


@app.command("fetch-models")
def fetch_models(config: ConfigOpt = None) -> None:
    """Download the tracking model(s) now, so the first click in the GUI isn't a multi-minute wait.

    Gets the configured tracker (SAM 3 by default: gated, needs `hf auth login` with access granted)
    and always the small open SAM 2.1 fallback. Safe to re-run: cached files are not downloaded again."""
    from huggingface_hub import snapshot_download

    from reeftrack.config import load_config

    tc = load_config(config).tracking
    wanted = [tc.backend] + (["sam2"] if tc.backend != "sam2" else [])
    ok = []
    for name in wanted:
        repo = tc.model_ids[name]
        typer.echo(f"{name}: {repo} …")
        try:
            snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            ok.append(name)
            typer.echo(f"{name}: ready")
        except Exception as e:  # gated repo without access, offline, ...
            typer.echo(f"{name}: not downloaded ({str(e).splitlines()[0][:160]})")
            if name == "sam3":
                typer.echo("  (SAM 3 needs access: request it at https://huggingface.co/facebook/sam3, then run "
                           "`hf auth login`. Until then the GUI uses SAM 2.1.)")
    if not ok:
        raise typer.Exit(1)


@app.command()
def library() -> None:
    """Show the library folders (set them in the GUI: File → Library folders) and check they're reachable."""
    from reeftrack.library import LibrarySettings, settings_path

    s = LibrarySettings.load()
    typer.echo(f"settings file: {settings_path()}")
    for k in ("video_root", "compressed_root", "annotations_root", "cache_dir"):
        typer.echo(f"  {k:17s} {getattr(s, k) or '(not set)'}")
    probs = s.problems()
    typer.echo("OK" if not probs else "Problems:\n  - " + "\n  - ".join(probs))


@app.command()
def compress(videos: Annotated[list[Path], typer.Argument(help="Original video file(s).")]) -> None:
    """Make the shared compressed copy of one or more original videos (skips ones already done)."""
    _compress_many(videos)


@app.command("compress-batch")
def compress_batch(
    folder: Annotated[Path, typer.Argument(help="Folder of originals, e.g. <drive>/originaldata/Maldives/2024_02_14")],
    pattern: str = typer.Option("*.MP4", help="File pattern (case-insensitive)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only list what would be compressed, with an estimate."),
) -> None:
    """Compress every original video below FOLDER that has no shared compressed copy yet (e.g. overnight)."""
    import fnmatch

    vids = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("._")
                  and fnmatch.fnmatch(p.name.lower(), pattern.lower()))
    _compress_many(vids, dry_run)


def _compress_many(videos: list[Path], dry_run: bool = False) -> None:
    from reeftrack.library import CompressionBusy, Library

    lib = Library()
    probs = lib.s.problems()
    if probs:
        raise typer.BadParameter("; ".join(probs) + " (set folders with `reeftrack annotate` → File → Library folders)")
    todo, minutes = [], 0.0
    for v in videos:
        try:
            rel = lib.rel(v)
            fm = lib.frame_map_for(v)
        except (ValueError, FileNotFoundError) as e:
            typer.echo(f"SKIP  {v}: {e}")
            continue
        if lib.adopt_compressed(v):
            typer.echo(f"done  {rel}")
            continue
        todo.append(v)
        minutes += fm.n_proxy / fm.proxy_fps / 60 * 1.1
        typer.echo(f"todo  {rel}  ({v.stat().st_size / 2**30:.1f} GB original)")
    typer.echo(f"{len(todo)} to compress, roughly {minutes / 60:.1f} h on a recent Mac (~{1.5 * len(todo):.1f} GB on RFS)")
    if dry_run:
        return
    from tqdm import tqdm

    failed = []
    for i, v in enumerate(todo, 1):
        bar = tqdm(total=100, desc=f"[{i}/{len(todo)}] {v.name}", unit="%")
        last = [0]

        def prog(d, t, bar=bar, last=last):
            pct = int(100 * d / max(t, 1))
            bar.update(pct - last[0])
            last[0] = pct

        try:
            lib.compress(v, progress=prog, log=lambda *_: None)
        except (CompressionBusy, RuntimeError, OSError, ValueError) as e:
            failed.append((v, e))
        finally:
            bar.close()
    for v, e in failed:
        typer.echo(f"FAILED {v}: {e}")
    typer.echo(f"finished: {len(todo) - len(failed)} compressed, {len(failed)} failed")


@app.command("sync-annotations")
def sync_annotations_cmd(
    src: Annotated[Path, typer.Argument(help="Annotations folder to copy from, e.g. a laptop's local folder.")],
    dst: Annotated[Path, typer.Argument(help="Annotations folder to copy into, e.g. RFS/.../fish_in_the_wild/annotations")],
    dry_run: bool = typer.Option(False, "--dry-run", help="Only show what would be copied."),
) -> None:
    """Merge offline annotations (CSVs + tracking_<name>/ masks) into the shared folder (never overwrites a newer file)."""
    from reeftrack.library import sync_annotations

    if not src.is_dir() or not dst.is_dir():
        raise typer.BadParameter("both folders must exist (is RFS connected?)")
    counts = sync_annotations(src, dst, dry_run, log=typer.echo)
    typer.echo(("[dry run] " if dry_run else "") + ", ".join(f"{k}: {v}" for k, v in counts.items()))


@app.command("reid-export")
def reid_export(
    tracker: Annotated[Path, typer.Argument(help="Annotations tracker spreadsheet (.xlsx or .csv).")],
    out: Annotated[Path | None, typer.Option(help="Dataset folder (default: reid_dataset next to the "
                                                  "library's compressed_videos folder).")] = None,
    weights: Annotated[Path, typer.Option(help="RT-DETR checkpoint.")] = Path("models/RTDETR_aq_and_wild.pt"),
    n: Annotated[int, typer.Option("--n", help="Frames per fish.")] = 100,
    min_gap: Annotated[float, typer.Option(help="Minimum seconds between kept frames.")] = 1.0,
    stride: Annotated[int, typer.Option(help="Run the detector on every Nth frame.")] = 3,
    fish: Annotated[list[int] | None, typer.Option("--fish", help="Only these Fish IDs (repeatable).")] = None,
    device: str = typer.Option("auto", help="cuda / mps / cpu"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Redo fish that were already exported."),
) -> None:
    """Re-ID dataset: top-N detector frames from each fish's longest un-occluded stretch."""
    from reeftrack.library import Library
    from reeftrack.reid.export import Settings, export

    lib = Library()
    if problems := lib.s.problems(required=("compressed_root", "annotations_root")):
        raise typer.BadParameter("; ".join(problems))
    if not weights.exists():
        raise typer.BadParameter(f"detector checkpoint not found: {weights}")
    s = Settings(weights=weights, n_per_fish=n, min_gap_s=min_gap, stride=stride, device=device)
    export(tracker, out or lib.compressed_root.parent / "reid_dataset", s, lib, fish, overwrite, log=typer.echo)


@app.command("reid-aquarium")
def reid_aquarium(
    videos_csv: Annotated[Path, typer.Argument(help="CSV with video_path, video_name, fish_ID (one fish per video).")],
    out: Annotated[Path, typer.Option(help="Dataset folder to write (e.g. on RFS).")],
    n: int = typer.Option(120, help="Images per individual (equal for all; lowered to the scarcest fish if needed)."),
    min_gap: float = typer.Option(2.0, help="Minimum seconds between two images of the same video."),
    cache: Annotated[Path | None, typer.Option(help="Cache for per-video detections and candidate crops.")] = None,
    jasmin_root: str = typer.Option("/gws/ssde/j25b/iecdt/tyankov/triggerfish-id/data",
                                    help="Folder the dataset will live in on JASMIN (for fish_id_data.csv paths)."),
    exclude_fish: Annotated[list[int] | None, typer.Option(help="Leave this individual out (repeatable).")] = None,
    rfs_dir: Annotated[str | None, typer.Option(help="Final RFS location, when building locally first.")] = None,
    layout: str = typer.Option("flat", help="flat (images/<crop>) | per_individual (images/fish_<id>/<crop> + labels.csv)"),
    conf_keep: float = typer.Option(0.7, help="Minimum detector confidence for a usable frame."),
    device: str = typer.Option("auto", help="auto | mps | cuda | cpu"),
    seed: int = typer.Option(0),
) -> None:
    """Balanced re-ID crops from aquarium videos (one fish per video), e.g. the background experiment."""
    from reeftrack.reid.aquarium import AquariumSettings, export

    s = AquariumSettings(n_per_fish=n, min_gap_s=min_gap, device=device, seed=seed, jasmin_root=jasmin_root,
                         exclude_fish=tuple(exclude_fish or ()), conf_keep=conf_keep, rfs_dir=rfs_dir, layout=layout)
    try:
        export(videos_csv, out, s, cache, log=typer.echo)
    except (ValueError, FileNotFoundError) as e:
        raise typer.BadParameter(str(e)) from e


@app.command()
def gui(run_dir: Annotated[Path | None, typer.Argument()] = None, config: ConfigOpt = None,
        behaviour_only: bool = typer.Option(False, "--behaviour-only",
                                            help="Behaviour annotation only: no SAM model is loaded (fast start).")) -> None:
    """Open the GUI: mask tracking (click the fish) and behaviour annotation (I/O to mark start/end)."""
    from reeftrack.gui.app import main

    main(run_dir, config, behaviour_only)


@app.command()
def track(
    run_dir: Path,
    frame: Annotated[int, typer.Option(help="Proxy frame of the click.")],
    x: Annotated[float, typer.Option(help="Click x in proxy pixels.")],
    y: Annotated[float, typer.Option(help="Click y in proxy pixels.")],
    config: ConfigOpt = None,
) -> None:
    """Headless tracking from a single click (e.g. on JASMIN). Appends a segment to the tracklet."""
    from reeftrack.io.run import Run
    from reeftrack.tracking.session import TrackingSession

    run = Run.open(run_dir)
    session = TrackingSession.from_run(run, run.config(config))
    seg = session.add_click(frame, (x, y), progress=True)
    session.save()
    typer.echo(f"[track] segment {seg.segment_id}: frames {seg.start}-{seg.end} ({seg.end_reason})")


@app.command()
def replay(run_dir: Path, config: ConfigOpt = None) -> None:
    """Re-run tracking from the prompts saved in tracklet.json (reproducibility / new backend)."""
    from reeftrack.io.run import Run
    from reeftrack.tracking.session import TrackingSession

    run = Run.open(run_dir)
    prompts = json.loads((run.tracking_dir / "tracklet.json").read_text())["prompts"]
    session = TrackingSession.from_run(run, run.config(config), fresh=True)
    from reeftrack.tracking.roi import rect_at

    for p in prompts:
        roi = rect_at(p["xy"], p["roi"], session.shape) if p.get("roi") else None
        if p.get("polygon"):  # hand-corrected mask prompt
            session.add_mask(p["frame"], p["polygon"], progress=True, roi=roi, enhance=p.get("enhance"))
        else:
            session.add_click(p["frame"], tuple(p["xy"]), progress=True, roi=roi, enhance=p.get("enhance"))
    session.save()


@app.command()
def reconstruct(
    run_dir: Path,
    config: ConfigOpt = None,
    stereo: bool = typer.Option(False, help="Use both rig cameras (needs `add-right`); writes recon_stereo/ "
                                            "in metres, leaving the mono recon/ untouched."),
    out: str = typer.Option("recon_stereo", help="Output folder name for --stereo (to keep several variants)."),
    reuse: bool = typer.Option(False, help="Reuse saved depth/cameras; only rebuild the point cloud "
                                           "(e.g. after tracking, to mask the fish, or after changing cleaning params)."),
) -> None:
    """3D-reconstruct the reef from keyframes (Wat3R/VGGT) into recon/reef.ply."""
    from reeftrack.io.run import Run
    from reeftrack.recon.pipeline import rebuild_cloud
    from reeftrack.recon.pipeline import reconstruct as _reconstruct

    run = Run.open(run_dir, out if stereo else "recon")
    if stereo and not reuse:
        from reeftrack.recon.stereo import reconstruct_stereo

        reconstruct_stereo(run, run.config(config), out)
    else:
        (rebuild_cloud if reuse else _reconstruct)(run, run.config(config))


@app.command()
def fuse(run_dir: Path, config: ConfigOpt = None,
         recon: str = typer.Option("recon", help="Reconstruction to use: recon (mono) | recon_stereo")) -> None:
    """Place the fish tracklet into the reconstruction -> fusion*/trajectory.csv + scene.html."""
    from reeftrack.fusion.lift import fuse as _fuse
    from reeftrack.io.run import Run

    run = Run.open(run_dir, recon)
    _fuse(run, run.config(config))


@app.command()
def compare(run_dir: Path, mono: str = "recon", stereo: str = "recon_stereo",
            window_s: float = typer.Option(10.0, help="Window length for the mono scale-drift profile."),
            out: str = typer.Option("compare", help="Output folder name inside the run.")) -> None:
    """Compare two reconstructions of a run (default mono vs stereo) -> compare/report.json + compare.html."""
    from reeftrack.fusion.compare import compare as _compare
    from reeftrack.io.run import Run

    _compare(Run.open(run_dir), mono, stereo, window_s, out)


@app.command("export-colmap")
def export_colmap_cmd(
    run_dir: Path,
    out: Annotated[Path | None, typer.Option(help="Output folder; default data/render_exports/<run name>.")] = None,
    recon: str = typer.Option("recon_stereo", help="Reconstruction to export: recon_stereo (metric) | recon (mono)"),
    width: int = typer.Option(960, help="Image width for training (height keeps the aspect)."),
    max_points: int = typer.Option(300_000, help="Reef points kept to initialise the Gaussians."),
    cameras: str = typer.Option("left", help="left | left+right (right frames as extra training views)."),
    split: str = typer.Option("interval", help="interval (every 8th = test; both methods) | filename "
                              "(train_/eval_ names; WaterSplatting only; required for left+right)"),
) -> None:
    """Export a run as a COLMAP scene (images, one shared camera, poses, points) for WaterSplatting / I2-NeRF."""
    from reeftrack.config import REPO_ROOT
    from reeftrack.io.run import Run
    from reeftrack.render.colmap_export import export_colmap, reprojection_check

    run = Run.open(run_dir, recon)
    out = out or REPO_ROOT / "data" / "render_exports" / run.root.name
    try:
        export_colmap(run, out, width=width, max_points=max_points, cameras=cameras, split=split, log=typer.echo)
    except (ValueError, FileNotFoundError) as e:
        raise typer.BadParameter(str(e)) from e
    for p in reprojection_check(out, run):
        typer.echo(f"[export] check image: {p}")


@app.command("render-compare")
def render_compare_cmd(
    export: Annotated[Path, typer.Argument(help="Scene folder made by `reeftrack export-colmap`.")],
    watersplatting: Annotated[list[str] | None, typer.Option(
        help="Folder with WaterSplatting's `ns-render dataset` output; repeat as LABEL=PATH to compare runs.")] = None,
    i2nerf: Annotated[Path | None, typer.Option(help="I2-NeRF experiment folder (contains test_preds/).")] = None,
    out: Annotated[Path | None, typer.Option(help="Output folder; default <export>/compare.")] = None,
    n: int = typer.Option(6, help="Frames on the contact sheet."),
) -> None:
    """Real held-out frames next to each method's render from the same camera, plus PSNR/SSIM."""
    from reeftrack.render.compare import compare

    try:
        compare(export, out or export / "compare", watersplatting, i2nerf, n, log=typer.echo)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e)) from e


@app.command()
def view(run_dir: Path, native: bool = typer.Option(False, help="Open3D window instead of the browser."),
         recon: str = typer.Option("recon", help="recon (mono) | recon_stereo")) -> None:
    """Open the 3D scene (reef + fish trajectory)."""
    from reeftrack.fusion.viewer import open_scene
    from reeftrack.io.run import Run

    open_scene(Run.open(run_dir, recon), native=native)


if __name__ == "__main__":
    app()
