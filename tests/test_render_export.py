"""COLMAP export for GS/NeRF training: pose maths and file round-trips."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reeftrack.recon.backends import to_model_pixels
from reeftrack.render.colmap_export import (
    ColmapImage,
    full_res_intrinsics,
    pca_scale,
    qvec_to_rotmat,
    read_images_bin,
    rotmat_to_qvec,
    write_colmap,
)


def test_quaternion_matches_colmap_convention():
    R = Rotation.random(100, random_state=1)
    for r, xyzw in zip(R.as_matrix(), R.as_quat()):
        q = rotmat_to_qvec(r)
        ref = xyzw[[3, 0, 1, 2]] * np.sign(xyzw[3])
        assert np.allclose(q, ref, atol=1e-9) and np.allclose(qvec_to_rotmat(q), r, atol=1e-9)


def test_full_res_intrinsics_inverts_model_resize():
    K_model = np.array([[330.0, 0, 259], [0, 331, 147], [0, 0, 1]])
    K = full_res_intrinsics(K_model, (294, 518), (1080, 1920))
    X = np.array([0.3, -0.2, 2.0])  # a camera-frame point
    u, v = (K @ X)[:2] / X[2]
    um, vm = to_model_pixels(u, v, 1080, 1920)
    assert np.allclose([um, vm], (K_model @ X)[:2] / X[2], atol=1e-6)


def test_colmap_binary_roundtrip(tmp_path):
    R = Rotation.random(3, random_state=2).as_matrix()
    ims = [ColmapImage(i + 1, f"{i:04d}_f{i:05d}.jpg", R[i], np.array([i, -1.0, 2.5])) for i in range(3)]
    K = np.array([[700.0, 0, 480], [0, 690, 270], [0, 0, 1]])
    write_colmap(tmp_path, 960, 540, K, ims, np.random.default_rng(3).random((5, 3)), np.full((5, 3), 200, np.uint8))
    back = read_images_bin(tmp_path / "images.bin")
    assert [b[1] for b in back] == [im.name for im in ims]
    for (iid, _, Rb, tb), im in zip(back, ims):
        assert iid == im.image_id and np.allclose(Rb, im.R) and np.allclose(tb, im.t)
    assert (tmp_path / "points3D.bin").stat().st_size == 8 + 5 * (8 + 24 + 3 + 8 + 8)
    assert "1 PINHOLE 960 540 700.000000 690.000000" in (tmp_path / "cameras.txt").read_text()


def test_pca_scale_unit_extent():
    c = np.array([[0, 0, 0], [4, 0, 0], [2, 1, 0], [2, -1, 0.0]])
    assert pca_scale(c) == pytest.approx(0.5)


def test_compare_pairs_renders_with_held_out_frames(tmp_path):
    """Fake method outputs laid out exactly as ns-render / I2-NeRF eval write them."""
    import json
    import shutil

    import cv2

    from reeftrack.render.compare import compare

    exp = tmp_path / "scene"
    (exp / "images").mkdir(parents=True)
    names = [f"{i:04d}_f{10 * i:05d}.jpg" for i in range(17)]
    rng = np.random.default_rng(0)
    for nm in names:
        cv2.imwrite(str(exp / "images" / nm), rng.integers(0, 255, (54, 96, 3), dtype=np.uint8))
    tests = [nm for i, nm in enumerate(sorted(names)) if i % 8 == 0]
    (exp / "test_images.txt").write_text("\n".join(tests) + "\n")
    (exp / "export.json").write_text(json.dumps({"width": 96, "height": 54, "images": [
        {"name": nm, "source_frame": 100 + i, "time_s": i / 3, "test": nm in tests} for i, nm in enumerate(names)]}))
    ws = tmp_path / "ws_renders" / "test"
    for sub in ("rgb", "rgb_clear"):
        (ws / sub).mkdir(parents=True)
        for nm in tests:  # perfect render = the frame itself
            shutil.copy(exp / "images" / nm, ws / sub / nm)
    i2 = tmp_path / "i2" / "test_preds"
    i2.mkdir(parents=True)
    for k in range(len(tests)):
        cv2.imwrite(str(i2 / f"color_{k:03d}.png"), np.zeros((54, 96, 3), np.uint8))
    sheet = compare(exp, tmp_path / "out", tmp_path / "ws_renders", tmp_path / "i2", n=2, log=lambda *_: None)
    assert sheet.exists() and len(list((tmp_path / "out").glob("compare_*.png"))) == len(tests) == 3
    rows = (tmp_path / "out" / "metrics.csv").read_text().splitlines()
    head = rows[0].split(",")
    first = dict(zip(head, rows[1].split(",")))
    assert float(first["WaterSplatting PSNR"]) > 40 > float(first["I2-NeRF PSNR"])  # pairing is right
