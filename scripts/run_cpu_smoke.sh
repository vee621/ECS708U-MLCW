#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_data.py
python scripts/verify_splits.py
python scripts/smoke_run.py
python -m pytest -q
