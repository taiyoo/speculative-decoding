"""
Utility helpers: deterministic seeding, GPU timing, CSV logging.
"""

import csv
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class GPUTimer:
    """
    Context manager for GPU-synchronised timing.

    Usage:
        timer = GPUTimer()
        with timer:
            model.generate(...)
        print(timer.elapsed_ms)
    """

    def __init__(self):
        self.elapsed_s: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self.elapsed_s = time.perf_counter() - self._start
        self.elapsed_ms = self.elapsed_s * 1000.0
        return False


def append_csv(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    """
    Append a single row (dict) to a CSV file.  Creates the file with a header
    if it doesn't exist yet.
    """
    exists = path.exists()
    if fieldnames is None:
        fieldnames = list(row.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as a complete CSV file (overwrites)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_env_info() -> dict:
    """Gather environment metadata for reproducibility."""
    info = {
        "python": os.popen("python3 --version").read().strip(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "not installed"
    return info
