"""Download and read the canonical Fashion-MNIST IDX files without torch/GPU."""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import struct
import tempfile
from urllib.request import urlopen

import numpy as np

# MD5s published with torchvision's FashionMNIST dataset implementation.
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
MIRRORS = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
)
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_file(root: str | Path, name: str, *, download: bool = True) -> Path:
    """Return a verified official compressed file; never accept damaged bytes."""
    if name not in FILES:
        raise ValueError(f"Unknown Fashion-MNIST file: {name}")
    root = Path(root)
    path = root / name
    if path.exists() and _md5(path) == FILES[name]:
        return path
    if path.exists() and not download:
        raise ValueError(f"Checksum mismatch: {path}")
    if not download:
        raise FileNotFoundError(f"Missing {path}. Run scripts/prepare_data.py")
    root.mkdir(parents=True, exist_ok=True)
    errors = []
    for mirror in MIRRORS:
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".download-", delete=False) as tmp:
                tmp_name = tmp.name
                with urlopen(mirror + name, timeout=90) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        tmp.write(chunk)
            if _md5(Path(tmp_name)) != FILES[name]:
                raise ValueError("MD5 mismatch")
            os.replace(tmp_name, path)
            return path
        except (OSError, ValueError) as exc:
            errors.append(f"{mirror}: {exc}")
        finally:
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
    raise RuntimeError(f"Unable to obtain {name}: {'; '.join(errors)}")


def _read_idx(path: Path, *, expected_magic: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated IDX header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != expected_magic:
            raise ValueError(f"Wrong IDX magic in {path}: {magic}")
        shape = (count,)
        if expected_magic == 2051:
            dims = f.read(8)
            if len(dims) != 8:
                raise ValueError(f"Truncated image dimensions: {path}")
            rows, cols = struct.unpack(">II", dims)
            if (rows, cols) != (28, 28):
                raise ValueError(f"Unexpected dimensions in {path}: {rows}x{cols}")
            shape = (count, rows, cols)
        payload = f.read()
    if len(payload) != int(np.prod(shape)):
        raise ValueError(f"Wrong IDX byte count in {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


def load_labels(root: str | Path, partition: str, *, download: bool = False) -> np.ndarray:
    """Load labels ONLY for deterministic stratification/metrics (never clustering)."""
    if partition not in ("train", "test"):
        raise ValueError("partition must be 'train' or 'test'")
    filename = "train-labels-idx1-ubyte.gz" if partition == "train" else "t10k-labels-idx1-ubyte.gz"
    labels = _read_idx(ensure_file(root, filename, download=download), expected_magic=2049)
    expected = 60000 if partition == "train" else 10000
    if labels.shape != (expected,) or not np.all(labels < 10):
        raise ValueError(f"Unexpected {partition} label data: {labels.shape}")
    return labels


def load_images(root: str | Path, partition: str, *, download: bool = False) -> np.ndarray:
    """Load image pixels only when used for fitting or evaluation; uint8 N x 28 x 28."""
    if partition not in ("train", "test"):
        raise ValueError("partition must be 'train' or 'test'")
    filename = "train-images-idx3-ubyte.gz" if partition == "train" else "t10k-images-idx3-ubyte.gz"
    images = _read_idx(ensure_file(root, filename, download=download), expected_magic=2051)
    expected = 60000 if partition == "train" else 10000
    if images.shape != (expected, 28, 28):
        raise ValueError(f"Unexpected {partition} image data: {images.shape}")
    return images
