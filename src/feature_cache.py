"""Versioned feature caching; extractors receive pixels, never ground-truth labels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Callable

import numpy as np


def cached_features(cache_root: str | Path, *, representation: str, subset: str,
                    images: np.ndarray, indices: np.ndarray, config: dict,
                    extractor: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, dict]:
    """Cache representation of a fixed index subset; validate cache identity.

    `images` must be the *already indexed* N x H x W image array. No labels are
    accepted, to avoid accidental label-informed feature extraction.
    `config` must fully describe the extractor and preprocessing version.
    """
    indices = np.asarray(indices, dtype="<i8")
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[0] != len(indices) or images.shape[1:] != (28, 28):
        raise ValueError("Pass selected greyscale 28x28 images and matching indices")
    if not representation or not subset:
        raise ValueError("Representation and subset identifiers required")
    if not config:
        raise ValueError("Cache config must describe extractor version/parameters")
    # Hash source pixel values as well as indices: never serve stale features if
    # image preprocessing or the underlying data changes under the same IDs.
    identity = {"representation": representation, "subset": subset, "config": config,
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "image_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(cache_root)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{representation}-{subset}-{signature}.npy"
    started = perf_counter()
    if file.exists():
        features = np.load(file, allow_pickle=False)
        hit = True
    else:
        features = np.asarray(extractor(images))
        if features.ndim != 2 or features.shape[0] != len(indices) or not np.issubdtype(features.dtype, np.number):
            raise ValueError("Extractor must return numeric N x D features")
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as temp:
            tmp_file = Path(temp.name)
        try:
            np.save(tmp_file, features, allow_pickle=False)
            os.replace(tmp_file, file)
        finally:
            tmp_file.unlink(missing_ok=True)
        hit = False
    if features.ndim != 2 or features.shape[0] != len(indices):
        raise ValueError("Cached feature array shape mismatch")
    return features, {"path": str(file), "cache_hit": hit,
                      "elapsed_seconds": round(perf_counter() - started, 6),
                      "feature_shape": list(features.shape), "signature": signature}
