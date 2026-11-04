from pathlib import Path

import numpy as np
import pytest

from src.feature_cache import cached_features


def test_cache_reuse_and_identity(tmp_path: Path):
    images = np.arange(4 * 28 * 28, dtype=np.int32).reshape(4, 28, 28).astype(np.uint8)
    indices = np.arange(4, dtype=np.int64)
    calls = []

    def extract(pixels):
        calls.append(1)
        return pixels.reshape(len(pixels), -1).astype(np.float32)

    kwargs = dict(representation="pixels", subset="cluster_train", images=images,
                  indices=indices, config={"version": 1}, extractor=extract)
    first, info = cached_features(tmp_path, **kwargs)
    second, info2 = cached_features(tmp_path, **kwargs)
    assert calls == [1]
    assert info["cache_hit"] is False and info2["cache_hit"] is True
    np.testing.assert_array_equal(first, second)
    kwargs["images"] = images.copy()
    kwargs["images"][0, 0, 0] ^= 1
    _, info3 = cached_features(tmp_path, **kwargs)
    assert info3["cache_hit"] is False
    assert calls == [1, 1]


def test_wrong_size_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        cached_features(tmp_path, representation="x", subset="train",
                        images=np.zeros((3, 28, 28), dtype=np.uint8),
                        indices=np.array([0, 1]), config={"version": 1},
                        extractor=lambda arr: arr)
