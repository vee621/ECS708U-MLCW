"""Fetch and verify all four original Fashion-MNIST IDX gzip archives.

This downloads the official 60k/10k image and label archives. It only reads
image *headers* and verifies file checksums; it never trains on or evaluates
test pixels. Output JSON records the exact downloaded bytes for provenance.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashion_mnist import FILES, CLASS_NAMES, ensure_file, load_labels  # noqa: E402

SOURCE = "https://github.com/zalandoresearch/fashion-mnist#get-the-data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_header(path: Path, count: int) -> dict:
    # No image array or pixel payload is decoded here (including test pixels).
    with gzip.open(path, "rb") as f:
        header = f.read(16)
    if len(header) != 16:
        raise ValueError(f"Truncated image IDX header: {path}")
    magic, n, rows, cols = struct.unpack(">IIII", header)
    if (magic, n, rows, cols) != (2051, count, 28, 28):
        raise ValueError(f"Invalid image IDX header in {path}: {(magic, n, rows, cols)}")
    return {"count": n, "height": rows, "width": cols}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "source_manifest.json")
    parser.add_argument("--verify-only", action="store_true", help="Do not download missing files")
    args = parser.parse_args()

    records = {}
    for name, expected_md5 in FILES.items():
        path = ensure_file(args.data_dir, name, download=not args.verify_only)
        records[name] = {
            "size_bytes": path.stat().st_size,
            "md5": expected_md5,
            "sha256": sha256(path),
            "source_path": "data/fashion/" + name,
        }
        if "images" in name:
            expected_count = 60_000 if name.startswith("train-") else 10_000
            records[name]["idx_header"] = image_header(path, expected_count)

    train_labels = load_labels(args.data_dir, "train")
    test_labels = load_labels(args.data_dir, "test")
    counts = {
        "official_train": np.bincount(train_labels, minlength=10).tolist(),
        "official_test": np.bincount(test_labels, minlength=10).tolist(),
    }
    if counts != {"official_train": [6000] * 10, "official_test": [1000] * 10}:
        raise ValueError(f"Unexpected Fashion-MNIST class distribution: {counts}")
    manifest = {
        "dataset": "Fashion-MNIST",
        "upstream": SOURCE,
        "upstream_repository": "https://github.com/zalandoresearch/fashion-mnist",
        "shape": [28, 28, 1],
        "class_names": list(CLASS_NAMES),
        "official_sizes": {"train": 60_000, "test": 10_000},
        "label_class_counts": counts,
        "archives": records,
        "test_policy": "Test images are downloaded and checksum-verified, but never decoded, trained on, or evaluated during data preparation.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source": SOURCE, "files": records, "class_counts": counts}, indent=2))
    print(f"Verified official archives and wrote {args.manifest}")


if __name__ == "__main__":
    main()
