"""Ground truth vs rendered views, for the held-out frames of an exported scene.

Inputs (whichever exist):
* WaterSplatting: `ns-render dataset --split test` output -> <dir>/test/rgb/<image name>.jpg
  (and rgb_clear/ = the same view with the water removed)
* I2-NeRF: `eval.py` output -> <exp>/test_preds/color_NNN.png (and J_NNN.png = water removed),
  NNN = index in test_images.txt

Output: one side-by-side PNG per held-out frame, a contact sheet of `n` evenly spaced frames,
and metrics.csv (PSNR / SSIM of each method's underwater render against the real frame).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

LABEL_H = 34


def _read(path: Path, size) -> np.ndarray:
    import cv2

    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-10)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Standard SSIM (Wang et al. 2004; 11x11 Gaussian, sigma 1.5), averaged over RGB channels."""
    import cv2

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for ch in range(3):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(m[5:-5, 5:-5].mean())
    return float(np.mean(vals))


def _labelled(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    band = np.full((LABEL_H, img.shape[1], 3), 15, np.uint8)
    cv2.putText(band, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return np.vstack([band, img])


def _runs(arg) -> list[tuple[str, Path]]:
    """None | path | [path or "label=path", ...] -> [(label suffix, path)]."""
    if not arg:
        return []
    items = [arg] if isinstance(arg, (str, Path)) else list(arg)
    out = []
    for it in items:
        label, _, path = str(it).rpartition("=") if "=" in str(it) else ("", "", str(it))
        out.append((f" ({label})" if label else "", Path(path)))
    return out


def find_renders(export: Path, watersplatting, i2nerf: Path | None) -> dict[str, dict[str, Path]]:
    """method label -> {test image name -> rendered file}. `watersplatting` may list several runs
    as "label=path" (e.g. left-only vs left+right) to compare them on the same held-out frames."""
    tests = (export / "test_images.txt").read_text().split()
    out: dict[str, dict[str, Path]] = {}
    for suffix, ws in _runs(watersplatting):
        for sub, label in (("rgb", "WaterSplatting"), ("rgb_clear", "WaterSplatting, water removed")):
            found = {}
            for n in tests:
                hits = sorted(ws.glob(f"**/{sub}/{Path(n).stem}.*"))
                if hits:
                    found[n] = hits[0]
            if found:
                out[label + suffix] = found
    if i2nerf:
        d = Path(i2nerf)
        d = d / "test_preds" if (d / "test_preds").is_dir() else d
        # color_ = rendered underwater view; J_ = scene radiance with the medium removed
        for prefix, label in (("color", "I2-NeRF"), ("J", "I2-NeRF, water removed")):
            found = {n: d / f"{prefix}_{i:03d}.png" for i, n in enumerate(tests) if (d / f"{prefix}_{i:03d}.png").exists()}
            if found:
                out[label] = found
    # underwater renders next to the real frame, then the water-removed views together
    return dict(sorted(out.items(), key=lambda kv: "removed" in kv[0]))


def compare(export: Path, out: Path, watersplatting=None, i2nerf: Path | None = None,
            n: int = 6, log=print) -> Path:
    import cv2

    meta = json.loads((export / "export.json").read_text())
    size = (meta["width"], meta["height"])
    info = {im["name"]: im for im in meta["images"]}
    tests = (export / "test_images.txt").read_text().split()
    renders = find_renders(export, watersplatting, i2nerf)
    if not renders:
        raise FileNotFoundError("no rendered test views found (check the --watersplatting / --i2nerf folders)")
    for label, found in renders.items():
        log(f"[compare] {label}: {len(found)}/{len(tests)} test views")
    out.mkdir(parents=True, exist_ok=True)
    rows, panels = [], {}
    for name in tests:
        gt = _read(export / "images" / name, size)
        im = info[name]
        cols = [_labelled(gt, f"Real frame {im['source_frame']} ({im['time_s']:.1f} s)")]
        row = {"image": name, "source_frame": im["source_frame"], "time_s": im["time_s"]}
        for label, found in renders.items():
            if name not in found:
                cols.append(_labelled(np.zeros_like(gt), f"{label}: missing"))
                continue
            r = _read(found[name], size)
            text = label
            if "removed" not in label:  # restored view is not meant to match the underwater frame
                row[f"{label} PSNR"], row[f"{label} SSIM"] = round(psnr(r, gt), 2), round(ssim(r, gt), 4)
                text += f"  (PSNR {row[f'{label} PSNR']:.1f} dB)"
            cols.append(_labelled(r, text))
        panel = np.hstack([np.pad(c, ((0, 0), (0, 6), (0, 0))) for c in cols])[:, :-6]
        cv2.imwrite(str(out / f"compare_{Path(name).stem}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        panels[name] = panel
        rows.append(row)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("image", "source_frame", "time_s"), k))
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    pick = [tests[int(i)] for i in np.linspace(0, len(tests) - 1, min(n, len(tests))).round()]
    sheet = np.vstack([np.pad(panels[p], ((0, 8), (0, 0), (0, 0))) for p in pick])
    w = 2400
    if sheet.shape[1] > w:
        sheet = cv2.resize(sheet, (w, int(sheet.shape[0] * w / sheet.shape[1])), interpolation=cv2.INTER_AREA)
    sheet_path = out / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    for label in renders:
        vals = [r[f"{label} PSNR"] for r in rows if f"{label} PSNR" in r]
        if vals:
            log(f"[compare] {label}: mean PSNR {np.mean(vals):.2f} dB over {len(vals)} held-out frames")
    log(f"[compare] {len(rows)} side-by-sides + {sheet_path}")
    return sheet_path
