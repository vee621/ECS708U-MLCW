"""Minimal CPU benchmark: use 10k fit only, 10k validation, no test pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashion_mnist import load_images, load_labels  # noqa: E402
from src.feature_cache import cached_features  # noqa: E402
from src.splits import load_splits  # noqa: E402
from src.timing import hardware_info  # noqa: E402


def avg_pool_2x2(images: np.ndarray) -> np.ndarray:
    # 28x28 grayscale -> 14x14 = 196 features, no labels involved.
    pixels = images.astype(np.float32) / 255.0
    return pixels.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cpu-smoke.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = ROOT / "data" / "raw"
    labels_train = load_labels(data_dir, "train")
    labels_test = load_labels(data_dir, "test")  # label provenance check only
    split, manifest = load_splits(ROOT / "splits", labels_train, labels_test)
    if cfg["split_seed"] != manifest["seed"]:
        raise ValueError("Experiment seed disagrees with immutable saved split")
    fit_ids, val_ids = split["fit_train"], split["validation"]
    # ONLY official train pixels are loaded; test images never accessed.
    images = load_images(data_dir, "train", download=True)
    cache_dir = ROOT / "data" / "features"
    x_train, info_train = cached_features(
        cache_dir, representation="avgpool2x2", subset="fit_train",
        images=images[fit_ids], indices=fit_ids,
        config={"version": 1, "normalization": "uint8/255", "pooling": "mean2x2"},
        extractor=avg_pool_2x2)
    x_val, info_val = cached_features(
        cache_dir, representation="avgpool2x2", subset="validation",
        images=images[val_ids], indices=val_ids,
        config={"version": 1, "normalization": "uint8/255", "pooling": "mean2x2"},
        extractor=avg_pool_2x2)
    model = SGDClassifier(loss="log_loss", max_iter=cfg["max_iter"],
                          tol=cfg["tol"], random_state=cfg["model_seed"])
    start = perf_counter()
    model.fit(x_train, labels_train[fit_ids])
    train_time = perf_counter() - start
    start = perf_counter()
    predictions = model.predict(x_val)
    inference_time = perf_counter() - start
    report = {
        "experiment_id": cfg["experiment_id"], "config": cfg,
        "hardware": hardware_info(), "seed": manifest["seed"],
        "partition": {"fit_train": len(fit_ids), "validation": len(val_ids)},
        "runtime_seconds": {
            "extraction_fit_train": info_train["elapsed_seconds"],
            "extraction_validation": info_val["elapsed_seconds"],
            "training": round(train_time, 6), "inference_validation": round(inference_time, 6)},
        "feature_cache": {"fit_train": info_train, "validation": info_val},
        "validation_accuracy": round(float(accuracy_score(labels_train[val_ids], predictions)), 6),
        "test_evaluated": False,
        "note": "Extraction durations include cache read when cache_hit=true; no test images used."
    }
    output = ROOT / "results" / "cpu-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
