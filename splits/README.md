# Frozen Fashion-MNIST indices

Run `python scripts/prepare_data.py` **once, before any experiments**. It uses
only labels to stratify; it never reads test images. The command writes:

- `indices.npz` — six named arrays of original official-dataset row indices.
- `manifest.json` — seed(s), class counts, source-label SHA-256 hashes, index
  SHA-256 hashes, partition sizes, and index-namespace definitions.

Both generated files are **intended to be committed** to version control so
all experiments share the exact indices. Subsequent calls refuse to overwrite
changed files unless `--force` is explicitly provided; do not use `--force`
after looking at any experimental results. Verify with
`python scripts/verify_splits.py` (reads labels only).

The official test set contains 10,000 held-out images. `cluster_test` lists
1,000 frozen test indices in the *official test namespace*, not the training
namespace. No test images or test metrics are accessed by setup/smoke code;
open those pixels only at the final evaluation stage.
