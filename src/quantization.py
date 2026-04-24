"""
Helpers for selecting a supported model-loading quantization mode.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch

from config import QUANT_MODE


_warned_messages: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned_messages:
        print(f"[quantization] {message}")
        _warned_messages.add(message)


def _fp16_kwargs() -> dict:
    return {"torch_dtype": torch.float16, "device_map": "auto"}


def get_quant_kwargs(mode: str | None = None) -> tuple[dict, str]:
    """
    Return supported loading kwargs and the resolved quantization mode.

    Parameters
    ----------
    mode:
        Explicit quantization mode override. When ``None`` (default), falls
        back to the global ``QUANT_MODE`` from :mod:`config`. Use this to
        load the target and draft models with different precisions, e.g.
        target=int8, draft=fp16 on a 24 GB GPU.

    Falls back to fp16 when optional quantization dependencies are unavailable.
    """
    resolved = (mode if mode is not None else QUANT_MODE) or "fp16"
    resolved = resolved.lower()

    if resolved == "int8":
        if find_spec("bitsandbytes") is None:
            _warn_once(
                "QUANT_MODE=int8 requested but bitsandbytes is not installed; "
                "falling back to fp16."
            )
            return _fp16_kwargs(), "fp16"

        from transformers import BitsAndBytesConfig

        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": "auto",
        }, "int8"

    if resolved == "fp8":
        if find_spec("optimum") is None:
            _warn_once(
                "QUANT_MODE=fp8 requested but optimum-quanto is not installed; "
                "falling back to fp16."
            )
            return _fp16_kwargs(), "fp16"

        from transformers import QuantoConfig

        return {
            "quantization_config": QuantoConfig(weights="float8"),
            "device_map": "auto",
        }, "fp8"

    return _fp16_kwargs(), "fp16"
