"""Download canonical label files and save all fixed indices BEFORE experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashion_mnist import load_labels  # noqa: E402
from src.splits import DEFAULT_SEED, make_splits, save_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--force", action="store_true", help="Explicitly replace different saved indices")
    args = parser.parse_args()
    labels_train = load_labels(args.data_dir, "train", download=True)
    labels_test = load_labels(args.data_dir, "test", download=True)
    partitions = make_splits(labels_train, labels_test, seed=args.seed)
    manifest = save_splits(args.split_dir, partitions, labels_train, labels_test,
                           seed=args.seed, force=args.force)
    print(json.dumps({"seed": manifest["seed"], "indices": manifest["indices"]}, indent=2))
    print(f"Saved splits to {args.split_dir} (do not change after inspecting results).")


if __name__ == "__main__":
    main()
