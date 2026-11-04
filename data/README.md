# Official Fashion-MNIST archives

Source: [Zalando Research — Get the Data](https://github.com/zalandoresearch/fashion-mnist#get-the-data).

`.github/workflows/pin-fashion-mnist.yml` downloads and commits all four **unaltered** official compressed IDX archives under `data/raw/`, after checking their published MD5 digests. `data/source_manifest.json` records the actual archive SHA-256 digests, byte counts, image headers and source.

To independently verify the committed data without downloading anything:

```bash
python scripts/download_official_data.py --verify-only
python scripts/verify_splits.py
```

The test image contents remain held out until final evaluation. Derived data in `data/features/` is disposable and ignored by Git.
