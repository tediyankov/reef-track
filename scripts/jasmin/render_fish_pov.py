"""Render a trained WaterSplatting scene along a fish-POV camera path (from
`python -m reeftrack.render.fish_pov` on the laptop). Run inside the watersplatting env, on a GPU:

    python render_fish_pov.py <config.yml> <scene export dir> <fish_pov_path.json> <out dir>

The path's poses are in the reconstruction's world frame (OpenCV camera axes). nerfstudio trains
in its own frame (OpenGL axes, re-oriented, re-centred, rescaled), so the mapping is fitted from
the training cameras themselves: each training image's pose in the export (sparse/0/images.txt)
against the same image's pose in the trained model. It is checked twice before anything is
rendered: the fit's residual, and a held-out real frame rendered through the same mapping and
scored against the photo (it should match the ~23 dB of the evaluation).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_images_txt(path):
    """{name: 4x4 camera-to-world, OpenCV axes} from a COLMAP images.txt."""
    out = {}
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if len(tok) == 10 and not line.startswith("#") and tok[9].lower().endswith((".jpg", ".png")):
            R = qvec_to_rotmat([float(v) for v in tok[1:5]])
            t = np.array([float(v) for v in tok[5:8]])
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R.T, -R.T @ t
            out[tok[9]] = M
    return out


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1
    R = U @ d @ Vt
    s = np.trace(np.diag(sig) @ d) / S.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> OpenGL (nerfstudio)


def main(cfg, scene, path_json, out):
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.utils.eval_utils import eval_setup
    from PIL import Image

    scene, out = Path(scene), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _, pipeline, _, step = eval_setup(Path(cfg), test_mode="inference")
    model, dev = pipeline.model, pipeline.device
    model.eval()
    dpo = pipeline.datamanager.train_dataparser_outputs
    ns_c2w = dpo.cameras.camera_to_worlds.cpu().numpy()  # (N, 3, 4), OpenGL, nerfstudio frame
    ours = read_images_txt(scene / "sparse" / "0" / "images.txt")
    names = [Path(p).name for p in dpo.image_filenames]
    src = np.stack([ours[n][:3, 3] for n in names])
    s, R, t = umeyama(src, ns_c2w[:, :, 3])

    def to_ns(M):
        return np.concatenate([R @ M[:3, :3] @ FLIP, (s * R @ M[:3, 3] + t)[:, None]], 1)

    pos_err = np.linalg.norm(np.stack([to_ns(ours[n])[:, 3] for n in names]) - ns_c2w[:, :, 3], axis=1)
    rot_err = [np.degrees(np.arccos(np.clip((np.trace(to_ns(ours[n])[:, :3].T @ c[:, :3]) - 1) / 2, -1, 1)))
               for n, c in zip(names, ns_c2w)]
    print(f"model step {step}; frame fit over {len(names)} training cameras: scale {s:.4f}, "
          f"position error max {pos_err.max() / s * 100:.2f} cm, rotation error max {max(rot_err):.3f} deg")
    if pos_err.max() / s > 0.01 or max(rot_err) > 0.5:
        sys.exit("frame fit is off: not rendering (send me this log)")

    c0 = dpo.cameras[0]
    fx, fy, cx, cy = (float(getattr(c0, k).item()) for k in ("fx", "fy", "cx", "cy"))
    W, H = int(c0.width.item()), int(c0.height.item())

    @torch.no_grad()
    def render(M):
        cam = Cameras(camera_to_worlds=torch.tensor(to_ns(M), dtype=torch.float32)[None],
                      fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H, camera_type=CameraType.PERSPECTIVE)
        o = model.get_outputs_for_camera(cam.to(dev))
        return {k: (o[k].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for k in ("rgb", "rgb_clear") if k in o}

    # check: a held-out real frame through the same mapping
    held = sorted(set(ours) - set(names))
    if held:
        gt = np.asarray(Image.open(scene / "images" / held[0]).convert("RGB")).astype(float)
        r = render(ours[held[0]])["rgb"].astype(float)
        if r.shape == gt.shape:
            psnr = 10 * np.log10(255 ** 2 / np.mean((r - gt) ** 2))
            print(f"check: held-out {held[0]} rendered through the mapping: PSNR {psnr:.2f} dB (expect ~20-24)")
            if psnr < 15:
                sys.exit("held-out check failed: the mapping is wrong, not rendering (send me this log)")

    path = json.loads(Path(path_json).read_text())
    frames = {"rgb": [], "rgb_clear": []}
    for i, f in enumerate(path["frames"]):
        for k, img in render(np.array(f["c2w"])).items():
            frames[k].append(img)
            d = out / k
            d.mkdir(exist_ok=True)
            Image.fromarray(img).save(d / f"{i:04d}.png")
    fps = path["fps"]
    for k, imgs in frames.items():
        if not imgs:
            continue
        try:
            import mediapy
            mediapy.write_video(out / f"fish_pov_{k}.mp4", imgs, fps=fps)
            print(f"wrote {out / f'fish_pov_{k}.mp4'}")
        except Exception as e:  # no ffmpeg on the node: the PNG frames are enough
            print(f"video encoding failed ({e}); frames are in {out / k}/")
    print(f"done: {len(path['frames'])} frames at {W}x{H} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
