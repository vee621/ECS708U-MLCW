from pathlib import Path

import numpy as np
import pytest

from src.splits import make_splits, save_splits, load_splits, validate_splits


def dummy_labels():
    # 10 classes, 6000 each train / 1000 each test, deliberately shuffled.
    rng = np.random.default_rng(42)
    train = np.repeat(np.arange(10, dtype=np.uint8), 6000)
    test = np.repeat(np.arange(10, dtype=np.uint8), 1000)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def test_splits_reproducible_disjoint_and_stratified(tmp_path: Path):
    y, t = dummy_labels()
    one = make_splits(y, t, seed=708)
    two = make_splits(y, t, seed=708)
    assert one.keys() == two.keys()
    for key in one:
        np.testing.assert_array_equal(one[key], two[key])
    assert len(one["train"]) == 50_000
    assert len(one["validation"]) == 10_000
    assert len(one["fit_train"]) == 10_000
    assert len(one["cluster_train"]) == 3_000
    assert len(one["cluster_validation"]) == 1_000
    assert len(one["cluster_test"]) == 1_000
    assert np.bincount(y[one["cluster_train"]], minlength=10).tolist() == [300] * 10
    assert np.bincount(t[one["cluster_test"]], minlength=10).tolist() == [100] * 10
    manifest = save_splits(tmp_path, one, y, t, seed=708)
    assert manifest["seed"] == 708
    save_splits(tmp_path, two, y, t, seed=708)  # idempotent
    saved, _ = load_splits(tmp_path, y, t)
    for key in one:
        np.testing.assert_array_equal(saved[key], one[key])
    with pytest.raises(FileExistsError):
        save_splits(tmp_path, make_splits(y, t, seed=709), y, t, seed=709)


def test_tampering_and_cross_subset_leakage_detected(tmp_path: Path):
    y, t = dummy_labels()
    good = make_splits(y, t)
    bad = {key: value.copy() for key, value in good.items()}
    bad["cluster_train"][0] = bad["validation"][0]
    with pytest.raises(ValueError):
        validate_splits(bad, y, t)
    save_splits(tmp_path, good, y, t)
    with pytest.raises(ValueError, match="labels changed"):
        altered = y.copy()
        altered[0] = (int(altered[0]) + 1) % 10
        load_splits(tmp_path, altered, t)
