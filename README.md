# ECS708U-MLCW — Fashion-MNIST reproducible setup

Setup for sections **0.1–0.3**: fixed stratified Fashion-MNIST partitions,
CPU-friendly fitting subset, shared clustering/feature-extraction subsets,
versioned feature caching, and a small runnable CPU validation-only smoke test.

## Dataset and frozen indices

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 28×28 greyscale
images, 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
Sneaker, Bag, Ankle boot. The official data has 60,000 train and 10,000 test
images. Downloaded IDX archives are checksum-verified and stored under
`data/raw/` (not committed). Only labels are read when constructing splits;
class IDs are **never passed** to distance calculations, centroids, cluster
assignments, or the feature extractor.

| Named index | Count | Original index namespace | Purpose |
| --- | ---: | --- | --- |
| `train` | 50,000 | Official train | Primary fitting partition |
| `validation` | 10,000 | Official train | Model selection and reporting |
| `fit_train` | 10,000 | Official train | Optional fixed CPU-friendly subset of `train` |
| `cluster_train` | 3,000 | Official train | Fixed subset of `fit_train` for all representations |
| `cluster_validation` | 1,000 | Official train | Fixed subset of `validation` for all representations |
| `cluster_test` | 1,000 | Official test | Preselected test indices for **final evaluation only** |

All subsets use deterministic `StratifiedShuffleSplit` with base seed **708**.
Offsets +1 to +4 generate the four nested subsets. The original official test
set remains held out; the setup reads *test labels* only to choose the fixed
stratified 1,000 test indices, not test pixels. The smoke run accesses **no**
test pixels or test metrics. Full official test evaluation must happen only
once, after model selection. Never use `cluster_test` to tune models.

Generate `splits/indices.npz` and `splits/manifest.json` once **before any
experiments**, then commit those two files. The manifest includes exact
seeds, source label hashes, index hashes, sizes, and class counts. Rerunning
the generator does not resample the data. Changed splits are rejected unless
explicitly forced. `python scripts/verify_splits.py` checks for leakage and
class-proportion drift without reading any images.

## Install and run (CPU, no paid service)

Use Python 3.10+ with a virtual environment. Pinned dependencies are in
`requirements.txt`. Run on your own CPU, Google Colab, or **QMUL Teaching
Hub → School of EECS → Deep Learning and Computer Vision** via
https://hub.comp-teach.qmul.ac.uk/hub/user/ . No GPU is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python scripts/prepare_data.py    # download label archives + freeze indices
python scripts/verify_splits.py   # validate seed, disjointness, proportions
python scripts/smoke_run.py       # downloads official train images; NO test images
python -m pytest -q
```

Or run `bash scripts/run_cpu_smoke.sh` on Linux/macOS/Teaching Hub. The first
run needs internet access to download **official labels** for index setup and
**official train images** for the smoke run. Later runs reuse the downloaded
archives. Never modify the indices after inspecting validation results.

## Hardware, timings, feature extraction

The `cpu-smoke` experiment (`configs/cpu-smoke.json`) uses a CPU-only
`SGDClassifier` on the fixed 10,000-image fit partition, evaluates on the
same 10,000-image validation partition, and **never evaluates test**. This is
only an infrastructure check, not a claimed coursework result.

It extracts label-free 14×14 average-pooled pixels and caches arrays in
`data/features/`. Cache identity includes the representation, parameters,
source indices and **pixel bytes**. Cache hits avoid recomputing features.
It writes `results/cpu-smoke.json` with measured training, feature-extraction,
and validation inference times, cache status, Python/package versions and
hardware/CPU information. Timings depend on the actual machine; **no
measurements are claimed until that command runs.** For future experiments,
record timings separately for each experiment ID and distinguish extraction
on cache hit from first-time extraction. Record whether hardware was QMUL
Teaching Hub, Google Colab CPU, or personal computer in the results note.

## Repo layout

```text
.
├── README.md                 # these instructions
├── requirements.txt          # pinned dependencies
├── configs/                  # config per experiment ID
├── splits/                   # tracked frozen indices + seed manifest
├── src/                      # validated data, split, feature cache, hardware code
├── scripts/                  # prepare, verify, smoke benchmark, shell runner
├── tests/                    # offline reproducibility/leakage/cache tests
├── results/                  # JSON experiment register and timing logs
├── figures/                  # tracked notebook figures plus provenance
└── data/                     # downloaded IDX and cached features (gitignored)
```

The clustering comparisons must use `cluster_train` / `cluster_validation` /
`cluster_test` **identically for every representation**. Labels are allowed
for fixed stratification and *post hoc evaluation* only, never in feature
extraction, distances, centroid fitting, or cluster assignment. Test subset
images are reserved until the final evaluation stage.
