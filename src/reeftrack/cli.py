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
    config: ConfigOpt = None,
) -> None:
    """Add the second camera of a stereo rig: audio-sync it and build a frame-matched right proxy."""
    from reeftrack.io.run import Run, add_right_camera

    run = Run.open(run_dir)
    add_right_camera(run, right_video, None if baseline_cm is None else baseline_cm / 100, run.config(config),
                     log=typer.echo)


@app.command()
def annotate(
    video: Annotated[Path | None, typer.Argument(help="Original video (on the hard drive). Omit to pick one.")] = None,
    config: ConfigOpt = None,
) -> None:
    """Annotate behaviours/loops on library videos (hard drive + RFS). The main command for annotators."""
    from reeftrack.gui.app import main

    main(None, config, behaviour_only=True, library=True, video=video)


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
        if lib.compressed_info(rel):
            typer.echo(f"done  {rel}")
            continue
        todo.append(v)
        minutes += fm.n_proxy / fm.proxy_fps / 60 * 1.1
        typer.echo(f"todo  {rel}  ({v.stat().st_size / 2**30:.1f} GB original)")
    typer.echo(f"{len(todo)} to compress, roughly {minutes / 60:.1f} h on a recent Mac (~{1.4 * len(todo):.0f} GB on RFS)")
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
    for p in prompts:
        session.add_click(p["frame"], tuple(p["xy"]), progress=True)
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


@app.command()
def view(run_dir: Path, native: bool = typer.Option(False, help="Open3D window instead of the browser."),
         recon: str = typer.Option("recon", help="recon (mono) | recon_stereo")) -> None:
    """Open the 3D scene (reef + fish trajectory)."""
    from reeftrack.fusion.viewer import open_scene
    from reeftrack.io.run import Run

    open_scene(Run.open(run_dir, recon), native=native)


if __name__ == "__main__":
    app()
