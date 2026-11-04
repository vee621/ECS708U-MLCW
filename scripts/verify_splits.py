"""Check fixed indices, label provenance, class proportions, and disjointness."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashion_mnist import load_labels  # noqa: E402
from src.splits import load_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    args = parser.parse_args()
    train = load_labels(args.data_dir, "train")
    test = load_labels(args.data_dir, "test")
    splits, manifest = load_splits(args.split_dir, train, test)
    print("Verified seed:", manifest["seed"])
    for key, index in splits.items():
        print(f"{key}: {len(index):,} indices, class counts {manifest['indices'][key]['class_counts']}")
    print("All split hashes, class proportions and subset relations verified. No images accessed.")


if __name__ == "__main__":
    main()
