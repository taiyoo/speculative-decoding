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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # No additional deterministic backend flags are currently exposed for MPS.
            pass
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
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except ImportError:
            pass
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
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
            info["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["mps_available"] = True
        else:
            info["device"] = "cpu"
    except ImportError:
        info["torch"] = "not installed"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "not installed"
    return info


def maybe_compile_model(model, label: str = "model"):
    """Optionally wrap an inference model with torch.compile when configured."""
    try:
        import torch
    except ImportError:
        return model

    from config import GPU_TRY_TORCH_COMPILE

    if not GPU_TRY_TORCH_COMPILE:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        print(f"[compile] Skipping {label}: torch.compile is unavailable.")
        return model

    if hasattr(model, "_orig_mod"):
        return model

    if bool(getattr(model, "is_loaded_in_8bit", False)) or bool(getattr(model, "is_loaded_in_4bit", False)):
        print(f"[compile] Skipping {label}: quantized models are left in eager mode.")
        return model

    quant_method = getattr(model, "quantization_method", None)
    if quant_method is not None and str(quant_method).lower() not in {"none", ""}:
        print(f"[compile] Skipping {label}: quantization backend is not compile-safe.")
        return model

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        devices = {str(v) for v in hf_device_map.values()}
        if len(devices) > 1:
            print(f"[compile] Skipping {label}: multi-device placement is not supported.")
            return model

    try:
        device = next(model.parameters()).device
    except StopIteration:
        print(f"[compile] Skipping {label}: model has no parameters.")
        return model

    if device.type != "cuda":
        print(f"[compile] Skipping {label}: reduce-overhead compile is only enabled on CUDA.")
        return model

    try:
        compiled = compile_fn(model, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        print(f"[compile] Failed to compile {label}; using eager mode ({exc}).")
        return model

    print(f"[compile] Enabled torch.compile for {label}.")
    return compiled
