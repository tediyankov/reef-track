"""Aquarium (background-experiment) re-ID export: input parsing, crops, balanced sampling."""

import numpy as np

from reeftrack.reid.aquarium import allocate, read_video_list, sample, square_crop


def test_read_video_list_skips_resource_forks_and_unlabelled(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("video_path,video_name,fish_ID\n/x/._GX1.MP4,._GX1.MP4,\n/x/GX1.MP4,GX1.MP4,46\n"
                 "/x/GX2.MP4,GX2.MP4,\n/x/GX3.MP4,GX3.MP4,59\n")
    assert [(v.name, f) for v, f in read_video_list(p)] == [("GX1.MP4", 46), ("GX3.MP4", 59)]


def test_square_crop_inside_frame():
    assert square_crop((100, 100, 300, 200), 3840, 2160, 0.1) == (90, 40, 220)
    x0, y0, side = square_crop((3800, 2100, 3840, 2160), 3840, 2160, 0.1)  # at the corner: slid inside
    assert x0 + side <= 3840 and y0 + side <= 2160


def test_allocate_spreads_evenly_and_respects_capacity():
    assert allocate([50, 50, 50], 9) == [3, 3, 3]
    assert allocate([2, 50, 50], 10) == [2, 4, 4]
    assert sum(allocate([1, 1], 5)) == 2


def _rows(fish, video, n, step=1.0, cand=1):
    return [{"fish_num": fish, "video_id": video, "timestamp_sec": i * step, "candidate": cand,
             "frame_number": int(i * step * 120)} for i in range(n)]


def test_sample_balanced_spaced_and_equal_per_fish():
    rows = _rows(1, "A", 60) + _rows(1, "B", 60) + _rows(2, "C", 30) + _rows(2, "D", 100, cand=0)
    chosen, n = sample(rows, n_per_fish=20, min_gap_s=2.0, seed=0)
    # fish 2 has 30 candidate keyframes 1 s apart: at most 15 at >= 2 s; random packing gets a bit fewer
    assert 10 <= n <= 15
    assert {f: len(c) for f, c in chosen.items()} == {1: n, 2: n}
    a = [r for r in chosen[1] if r["video_id"] == "A"]
    assert abs(len(a) - n / 2) <= 0.5  # split evenly across fish 1's two videos
    for c in chosen.values():
        for v in {r["video_id"] for r in c}:
            t = np.sort([r["timestamp_sec"] for r in c if r["video_id"] == v])
            assert (np.diff(t) >= 2.0).all()
    again, _ = sample(rows, 20, 2.0, seed=0)
    assert again == chosen  # reproducible
