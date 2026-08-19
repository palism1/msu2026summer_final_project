# FILE MAP | Unit tests for the pure per-image probability cache (src/ensemble/cache.py).
#   No torch, no GPU. Covers the write/read round trip, bit-packed GT round trip on a
#   non-byte-multiple shape, the fingerprint, alignment checks, path scheme, and the
#   environment record's torch-free fallback.
"""GPU-free tests for the ensemble probability cache."""

import numpy as np
import pytest

from src.ensemble.cache import (
    SplitCache,
    assert_aligned,
    cache_path,
    environment_record,
    gt_fingerprint,
    pack_gt,
    read_split_cache,
    unpack_gt,
    write_split_cache,
)


def test_write_read_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    probs = rng.random((4, 6, 6)).astype(np.float32)
    gt = (rng.random((4, 6, 6)) > 0.5).astype(np.float32)
    image_paths = [f"img_{i}.png" for i in range(4)]
    mask_paths = [f"mask_{i}.png" for i in range(4)]

    path = write_split_cache(tmp_path / "seed42" / "seen_kvasir.npz", probs, gt,
                             image_paths, mask_paths)
    cache = read_split_cache(path)

    assert np.allclose(cache.probs, probs, atol=1e-3)
    assert np.array_equal(cache.gt, gt)
    assert cache.image_paths == image_paths
    assert cache.mask_paths == mask_paths


def test_pack_unpack_gt_round_trips_non_byte_multiple_shape():
    rng = np.random.default_rng(1)
    gt = (rng.random((3, 5, 7)) > 0.5).astype(np.float32)   # 105 bits, not a multiple of 8
    packed = pack_gt(gt)
    restored = unpack_gt(packed, gt.shape)
    assert np.array_equal(restored, gt)


def test_gt_fingerprint_changes_on_one_pixel_flip():
    gt = np.zeros((2, 4, 4), dtype=np.float32)
    packed_before = pack_gt(gt)
    fp_before = gt_fingerprint(packed_before)

    gt[0, 0, 0] = 1.0
    packed_after = pack_gt(gt)
    fp_after = gt_fingerprint(packed_after)

    assert fp_before != fp_after


def _cache(image_paths, gt_sha256):
    return SplitCache(
        probs=np.zeros((1, 2, 2), np.float32),
        gt=np.zeros((1, 2, 2), np.float32),
        image_paths=image_paths,
        mask_paths=["m.png"],
        gt_sha256=gt_sha256,
    )


def test_assert_aligned_raises_on_path_mismatch():
    a = _cache(["a.png"], "same")
    b = _cache(["b.png"], "same")
    with pytest.raises(ValueError):
        assert_aligned([a, b])


def test_assert_aligned_raises_on_fingerprint_mismatch():
    a = _cache(["a.png"], "sha1")
    b = _cache(["a.png"], "sha2")
    with pytest.raises(ValueError):
        assert_aligned([a, b])


def test_assert_aligned_passes_when_matched():
    a = _cache(["a.png"], "same")
    b = _cache(["a.png"], "same")
    assert_aligned([a, b])   # no raise


def test_cache_path_scheme():
    p = cache_path("cache", "sam_vit_h", 42, "seen_kvasir")
    assert str(p) == str(p.__class__("cache/sam_vit_h/seed42/seen_kvasir.npz"))


def test_environment_record_all_string_values_no_exception():
    record = environment_record()
    assert isinstance(record, dict)
    assert all(isinstance(v, str) for v in record.values())
    assert "python" in record
    assert "torch" in record
