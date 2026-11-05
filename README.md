# ECS708U-MLCW

Fashion-MNIST work for ECS708U. The images are 28×28 greyscale and have 10 clothing classes.

The data is from [Zalando's Fashion-MNIST repo](https://github.com/zalandoresearch/fashion-mnist#get-the-data). It has 60,000 training images and 10,000 test images. The four original `.gz` files are in `data/raw/`. Checksums are saved in `data/source_manifest.json`.

## Run

Use Python 3.11+ (CPU is fine).

```bash
python -m pip install -r requirements.txt
python scripts/download_official_data.py --verify-only
python scripts/verify_splits.py
python scripts/smoke_run.py
python -m pytest -q
```

Open `notebooks/part0_part1.ipynb` in Jupyter and run all cells.

The split files are already included, so you don't need to make them again. If they're missing, run `python scripts/prepare_data.py` before doing any experiments. Don't change the splits after looking at results.

## Data splits

The split seed is **708**. Indices are saved in `splits/indices.npz` and details (seeds, checksums and class counts) are in `splits/manifest.json`.

| Split | Images | Where they come from |
| --- | ---: | --- |
| train | 50,000 | original training set |
| validation | 10,000 | original training set |
| fit_train | 10,000 | smaller training subset for CPU runs |
| cluster_train | 3,000 | from fit_train |
| cluster_validation | 1,000 | from validation |
| cluster_test | 1,000 | original test set; final evaluation only |

Sampling is stratified. Labels are used to select the indices, not for clustering distances or assignments. Use the same cluster subsets when comparing features. Test images should only be used for the final evaluation.

## Files

- `src/` — dataset loading, splits and feature caching
- `scripts/` — data checks, split generation and a small CPU run
- `configs/cpu-smoke.json` — settings for the example run
- `results/cpu-smoke.json` — validation accuracy, hardware and timing from the GitHub Actions CPU run
- `tests/` — basic split and cache checks

`scripts/smoke_run.py` trains a small SGD model using `fit_train` and checks it on validation only. It stores extracted features under `data/features/` so they can be reused. The recorded times are from the GitHub runner, not from a personal laptop.
