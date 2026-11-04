"""Immutable, seeded, stratified train/validation and feature-extraction indices."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_SEED = 708
EXPECTED_SIZES = {
    "train": 50_000,
    "validation": 10_000,
    "fit_train": 10_000,
    "cluster_train": 3_000,
    "cluster_validation": 1_000,
    "cluster_test": 1_000,
}


def _stratified_subset(indices: np.ndarray, labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Draw from the given index domain; labels affect selection ONLY."""
    indices = np.asarray(indices, dtype=np.int64)
    if count >= len(indices):
        raise ValueError("Subset must be smaller than its parent")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=seed)
    positions, _ = next(splitter.split(np.zeros(len(indices)), labels[indices]))
    return np.sort(indices[positions])


def make_splits(train_labels: np.ndarray, test_labels: np.ndarray, seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """Prepare all indices before experimentation; no test pixels are read."""
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    if train_labels.shape != (60_000,) or test_labels.shape != (10_000,):
        raise ValueError("Expected official 60,000 training and 10,000 test labels")
    if not np.all((train_labels >= 0) & (train_labels <= 9)) or not np.all((test_labels >= 0) & (test_labels <= 9)):
        raise ValueError("Expected integer class IDs 0 to 9")
    if not np.issubdtype(train_labels.dtype, np.integer) or not np.issubdtype(test_labels.dtype, np.integer):
        raise ValueError("Labels must be integer class IDs")

    official_train = np.arange(len(train_labels), dtype=np.int64)
    # Keep the complete official test partition reserved. Select its fixed 1k
    # evaluation subset now using labels only; read no test pixels here.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=10_000, random_state=seed)
    train_positions, validation_positions = next(splitter.split(np.zeros(len(official_train)), train_labels))
    train_idx = np.sort(train_positions.astype(np.int64))
    validation_idx = np.sort(validation_positions.astype(np.int64))
    fit_train_idx = _stratified_subset(train_idx, train_labels, 10_000, seed + 1)
    cluster_train_idx = _stratified_subset(fit_train_idx, train_labels, 3_000, seed + 2)
    cluster_val_idx = _stratified_subset(validation_idx, train_labels, 1_000, seed + 3)
    cluster_test_idx = _stratified_subset(np.arange(10_000), test_labels, 1_000, seed + 4)
    splits = {
        "train": train_idx,
        "validation": validation_idx,
        "fit_train": fit_train_idx,
        "cluster_train": cluster_train_idx,
        "cluster_validation": cluster_val_idx,
        "cluster_test": cluster_test_idx,
    }
    validate_splits(splits, train_labels, test_labels)
    return splits


def validate_splits(splits: dict[str, np.ndarray], train_labels: np.ndarray, test_labels: np.ndarray) -> None:
    if set(splits) != set(EXPECTED_SIZES):
        raise ValueError(f"Unexpected index keys: {list(splits)}")
    for name, expected in EXPECTED_SIZES.items():
        arr = np.asarray(splits[name])
        upper = 10_000 if name == "cluster_test" else 60_000
        if arr.shape != (expected,) or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Wrong index shape/type for {name}: {arr.shape}/{arr.dtype}")
        if np.any(arr < 0) or np.any(arr >= upper) or len(np.unique(arr)) != expected:
            raise ValueError(f"Invalid, repeated or out-of-range indices: {name}")
        # Allow at most 1 sample/class discrepancy from the population's
        # stratified expectation. Class proportions need not be balanced.
        parent_labels = test_labels if name == "cluster_test" else train_labels
        parent = {
            "train": np.arange(60_000),
            "validation": np.arange(60_000),
            "fit_train": splits["train"],
            "cluster_train": splits["fit_train"],
            "cluster_validation": splits["validation"],
            "cluster_test": np.arange(10_000),
        }[name]
        expected_counts = np.bincount(parent_labels[parent].astype(int), minlength=10) * (expected / len(parent))
        observed_counts = np.bincount(parent_labels[arr].astype(int), minlength=10)
        if np.max(np.abs(observed_counts - expected_counts)) > 2:
            raise ValueError(f"Class proportions drift in {name}: {observed_counts}")
    train = set(map(int, splits["train"]))
    val = set(map(int, splits["validation"]))
    if train & val or len(train | val) != 60_000:
        raise ValueError("Train/validation must partition official training set exactly")
    if not set(map(int, splits["fit_train"])) <= train:
        raise ValueError("fit_train must be within train")
    if not set(map(int, splits["cluster_train"])) <= set(map(int, splits["fit_train"])):
        raise ValueError("cluster_train must be within fit_train")
    if not set(map(int, splits["cluster_validation"])) <= val:
        raise ValueError("cluster_validation must be within validation")
    # Test indices live in a SEPARATE official namespace; no overlap check
    # between integer IDs in test and IDs in train is meaningful.


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(splits: dict[str, np.ndarray], train_labels: np.ndarray, test_labels: np.ndarray, seed: int) -> dict:
    return {
        "dataset": "Fashion-MNIST",
        "input_shape": [28, 28, 1],
        "classes": 10,
        "official_train_size": 60_000,
        "official_test_size": 10_000,
        "seed": seed,
        "sampling_seeds": {"partition": seed, "fit_train": seed + 1,
                           "cluster_train": seed + 2, "cluster_validation": seed + 3,
                           "cluster_test": seed + 4},
        "index_file": "splits/indices.npz",
        "namespace": {"train": "official train IDX", "validation": "official train IDX",
                      "fit_train": "official train IDX", "cluster_train": "official train IDX",
                      "cluster_validation": "official train IDX", "cluster_test": "official test IDX"},
        "label_sha256": {"official_train": _sha256_bytes(np.asarray(train_labels, dtype=np.uint8).tobytes()),
                          "official_test": _sha256_bytes(np.asarray(test_labels, dtype=np.uint8).tobytes())},
        "indices": {
            k: {"count": int(len(v)),
                "sha256": _sha256_bytes(np.asarray(v, dtype="<i8").tobytes()),
                "class_counts": np.bincount(
                    (test_labels if k == "cluster_test" else train_labels)[v].astype(int),
                    minlength=10,
                ).astype(int).tolist()}
            for k, v in splits.items()
        },
        "test_policy": "No test pixels or test metrics before final evaluation; test labels used solely for fixed 1k stratified index selection.",
    }


def save_splits(folder: str | Path, splits: dict[str, np.ndarray], train_labels: np.ndarray,
                test_labels: np.ndarray, seed: int = DEFAULT_SEED, *, force: bool = False) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / "indices.npz"
    manifest_path = folder / "manifest.json"
    manifest = build_manifest(splits, train_labels, test_labels, seed)
    if index_path.exists() or manifest_path.exists():
        if not (index_path.exists() and manifest_path.exists()):
            raise FileExistsError("Incomplete existing splits; inspect and use --force explicitly")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(index_path, allow_pickle=False) as saved:
            identical = set(saved.files) == set(splits) and all(
                np.array_equal(saved[k], splits[k]) for k in splits)
        if identical and existing == manifest:
            return manifest
        if not force:
            raise FileExistsError("Existing split differs; supply --force to replace explicitly")
    # Pre-write all data and replace atomically to avoid partially-written npz.
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as temp:
        tmp_npz = Path(temp.name)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".json", mode="w", encoding="utf-8", delete=False) as temp:
        tmp_json = Path(temp.name)
    try:
        np.savez_compressed(tmp_npz, **splits)
        tmp_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_npz, index_path)
        os.replace(tmp_json, manifest_path)
    finally:
        tmp_npz.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
    return manifest


def load_splits(folder: str | Path, train_labels: np.ndarray | None = None,
                test_labels: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with np.load(folder / "indices.npz", allow_pickle=False) as f:
        splits = {k: f[k] for k in f.files}
    if set(splits) != set(EXPECTED_SIZES):
        raise ValueError("Missing or extra saved splits")
    for name, arr in splits.items():
        checksum = _sha256_bytes(np.asarray(arr, dtype="<i8").tobytes())
        if checksum != manifest["indices"][name]["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")
    if train_labels is not None and test_labels is not None:
        if _sha256_bytes(np.asarray(train_labels, dtype=np.uint8).tobytes()) != manifest["label_sha256"]["official_train"]:
            raise ValueError("Official training labels changed since split creation")
        if _sha256_bytes(np.asarray(test_labels, dtype=np.uint8).tobytes()) != manifest["label_sha256"]["official_test"]:
            raise ValueError("Official test labels changed since split creation")
        validate_splits(splits, train_labels, test_labels)
    return splits, manifest
