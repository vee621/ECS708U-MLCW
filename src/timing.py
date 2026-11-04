"""Hardware provenance for any CPU/GPU experiment."""
from __future__ import annotations

import os
import platform
import sys

import numpy
import sklearn


def hardware_info() -> dict:
    """Record what we can measure; don't guess CPU model or thread count."""
    name = platform.processor() or "unknown"
    if name == "unknown" and sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {"host": platform.node(), "system": platform.platform(),
            "cpu": name, "logical_cpu_count": os.cpu_count(),
            "python": sys.version.split()[0], "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "accelerator": "CPU only (this smoke experiment)"}
