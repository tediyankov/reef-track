import numpy as np
import pytest

from reeftrack.tracking.masks import (
    area,
    bbox,
    centroid,
    erode,
    iou,
    largest_component,
    rle_decode,
    rle_encode,
)


@pytest.mark.parametrize("seed", range(5))
def test_rle_roundtrip_random(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((37, 53)) > 0.7
    assert np.array_equal(rle_decode(rle_encode(m), m.shape), m)


@pytest.mark.parametrize("fill", [True, False])
def test_rle_roundtrip_uniform(fill):
    m = np.full((10, 10), fill)
    runs = rle_encode(m)
    assert runs[0] == (0 if fill else 100)
    assert np.array_equal(rle_decode(runs, m.shape), m)


def test_rle_is_compact_for_blobs():
    m = np.zeros((1080, 1920), bool)
    m[500:560, 900:1000] = True
    assert rle_encode(m).nbytes < 1000  # vs 2 MB raw


def test_geometry():
    m = np.zeros((20, 30), bool)
    m[5:10, 10:20] = True
    assert area(m) == 50
    assert bbox(m) == (10, 5, 19, 9)
    assert centroid(m) == pytest.approx((14.5, 7.0))
    assert bbox(np.zeros_like(m)) is None and centroid(np.zeros_like(m)) is None
    assert iou(m, m) == 1.0 and iou(m, ~m) == 0.0


def test_largest_component_drops_fragments():
    m = np.zeros((50, 50), bool)
    m[10:30, 10:30] = True
    m[45:47, 45:47] = True
    out = largest_component(m)
    assert out.sum() == 400 and not out[46, 46]


def test_erode_never_deletes_tiny_mask():
    m = np.zeros((10, 10), bool)
    m[5, 5] = True
    assert erode(m, 2).sum() == 1
