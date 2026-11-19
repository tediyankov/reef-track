"""
reef-track mask-tracking diagnostic. gets useful info about env and crashes that u can then send back to me.
"""

import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("REEFTRACK_DIAG_OUT", Path.home() / "Desktop" / "reeftrack_diagnostic.txt"))
lines: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def sh(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(failed: {e})"


SIGNALS = {-10: "BUS ERROR (SIGBUS)", -11: "SEGMENTATION FAULT (SIGSEGV)", -6: "ABORT (SIGABRT)", -9: "KILLED (SIGKILL, often out of memory)"}

# tests (each runs as `python -X faulthandler -c <code> <args>`) 
MPS_BASIC = r"""
import torch
x = torch.randn(512, 512, device="mps"); y = (x @ x).sum().item()
c = torch.nn.Conv2d(3, 16, 3).to("mps"); c(torch.randn(1, 3, 256, 256, device="mps")).sum().item()
print("MPS maths OK")
"""

TRACK = r"""
import sys, threading, time, numpy as np
device, video, in_thread = sys.argv[1], sys.argv[2], sys.argv[3] == "thread"
from reeftrack.tracking.backends import HFVideoTracker
if video != "-":
    from reeftrack.io.video import FrameReader
    r = FrameReader(video)
    i0 = min(len(r) // 3, 12000)
    frames = [r[i0 + k] for k in range(8)]
    print(f"frames from the real video: {video} (frame {i0}, {frames[0].shape[1]}x{frames[0].shape[0]})", flush=True)
else:  # synthetic: a bright ellipse moving over a grey background, 1080p like the GUI
    import cv2
    frames = []
    for k in range(8):
        f = np.full((1080, 1920, 3), 90, np.uint8)
        cv2.ellipse(f, (900 + 12 * k, 540), (90, 45), 0, 0, 360, (240, 200, 60), -1)
        frames.append(f)
    print("synthetic frames 1920x1080", flush=True)

def run():
    t0 = time.time()
    tr = HFVideoTracker("sam2", "facebook/sam2.1-hiera-small", device)
    print(f"model loaded on {device} in {time.time() - t0:.1f} s", flush=True)
    h, w = frames[0].shape[:2]
    res = tr.start(frames[0], [(w / 2, h / 2)])
    print(f"start OK (mask {int(res.mask.sum())} px)", flush=True)
    for k in range(1, len(frames)):
        t = time.time(); tr.step(frames[k]); print(f"step {k} OK ({time.time() - t:.2f} s)", flush=True)
    print("TRACKING OK", flush=True)

if in_thread:  # like the GUI: tracking runs off the main thread
    err = []
    def wrapped():
        try:
            run()
        except Exception as e:
            err.append(e)
    th = threading.Thread(target=wrapped); th.start(); th.join()
    if err:
        raise err[0]
else:
    run()
"""


def run_test(name: str, code: str, *args: str, timeout: int = 600) -> None:
    log(f"\n--- TEST: {name}")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code, *args], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"RESULT: TIMED OUT after {timeout} s")
        return
    out = "\n".join(ln for ln in (r.stdout + r.stderr).splitlines()
                    if not any(k in ln for k in ("httpx", "Loading weights", "rope_theta", "Redirects are")))
    log(out[-6000:])
    rc = r.returncode
    log(f"RESULT: {'PASSED' if rc == 0 else 'CRASHED: ' + SIGNALS.get(rc, f'exit code {rc}')}  ({time.time() - t0:.0f} s)")


def newest_crash_reports(n: int = 2, since: float = 0.0) -> None:
    """The faulting thread of the newest macOS crash reports for python (native stack)."""
    files = sorted((f for f in glob.glob(str(Path.home() / "Library/Logs/DiagnosticReports/*python*.ips"))
                    if os.path.getmtime(f) > since), key=os.path.getmtime, reverse=True)[:n]
    if not files:
        log("no python crash reports" + (" since the tests started" if since else " in ~/Library/Logs/DiagnosticReports"))
    for f in files:
        log(f"\n--- crash report {Path(f).name} ({time.ctime(os.path.getmtime(f))})")
        try:
            text = Path(f).read_text(errors="replace")
            body = json.loads(text[text.index("\n") + 1:])
            exc = body.get("exception", {})
            log(f"exception: {exc.get('type')} {exc.get('signal')} {exc.get('subtype', '')}")
            images = body.get("usedImages", [])
            ti = body.get("faultingThread", 0)
            th = body.get("threads", [])[ti]
            log(f"faulting thread {ti} ({th.get('name') or th.get('queue') or 'unnamed'}):")
            for fr in th.get("frames", [])[:30]:
                img = images[fr["imageIndex"]].get("name", "?") if fr.get("imageIndex", -1) < len(images) else "?"
                log(f"   {img:32s} {fr.get('symbol', '')}")
        except Exception as e:  # noqa: BLE001
            log(f"(could not read: {e})")


def main() -> None:
    log("reef-track tracking diagnostic", time.strftime("%Y-%m-%d %H:%M"))
    log("\n=== computer")
    log(sh("sw_vers"))
    log("model:", sh("sysctl -n hw.model"), "| chip:", sh("sysctl -n machdep.cpu.brand_string"))
    log(f"memory: {int(sh('sysctl -n hw.memsize') or 0) / 2**30:.0f} GB | free disk (home):", sh("df -h ~ | tail -1"))
    log(sh("memory_pressure | tail -1"))

    log("\n=== python + packages")
    log("python:", sys.executable, platform.python_version(), platform.machine())
    from importlib import metadata
    for p in ("torch", "torchvision", "transformers", "huggingface_hub", "safetensors", "accelerate", "numpy",
              "opencv-python-headless", "opencv-python", "PySide6", "av", "reeftrack"):
        try:
            log(f"  {p:24s} {metadata.version(p)}")
        except metadata.PackageNotFoundError:
            log(f"  {p:24s} -")
    log("pip check:", sh(f'"{sys.executable}" -m pip check'))
    log("env vars:", {k: v for k, v in os.environ.items() if k.startswith(("PYTORCH", "HF_", "OMP", "KMP", "MKL", "CONDA_PREFIX"))})
    try:
        import reeftrack
        from reeftrack.config import load_config
        from reeftrack.devices import resolve_device
        tc = load_config().tracking
        log("reeftrack code:", Path(reeftrack.__file__).parent)
        log(f"tracker from configs/default.yaml: {tc.backend} on {tc.device} -> device {resolve_device(tc.device)}")
    except Exception as e:  # noqa: BLE001
        log("reeftrack import failed:", e)
    log("torch MPS available:", sh(f'"{sys.executable}" -c "import torch;print(torch.backends.mps.is_available())"'))

    # newest compressed copy in the local playback cache: tracking on Rosie's real frames
    cache = sorted(glob.glob(str(Path.home() / "reeftrack_cache" / "**" / "*.mp4"), recursive=True), key=os.path.getmtime)
    video = cache[-1] if cache else "-"
    log("video for the tests:", video)

    log("\n=== crash reports from before this diagnostic")
    newest_crash_reports()

    log("\n=== tests")
    t_tests = time.time()
    run_test("MPS (Mac GPU) basic maths", MPS_BASIC)
    run_test("SAM2 on MPS, synthetic frames", TRACK, "mps", "-", "main")
    run_test("SAM2 on MPS, background thread (like the GUI)", TRACK, "mps", "-", "thread")
    if video != "-":
        run_test("SAM2 on MPS, frames of the real video", TRACK, "mps", video, "thread")
    run_test("SAM2 on CPU, synthetic frames", TRACK, "cpu", "-", "main")

    log("\n=== crash reports written by the tests above (if any crashed)")
    time.sleep(3)  # macOS writes the report a moment after the crash
    newest_crash_reports(2, since=t_tests)

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nDone. Send this file back: {OUT}")


if __name__ == "__main__":
    main()
