"""
Hugging Face offline-first helpers.

Default behavior prefers local cache and avoids hub connectivity checks.
Set SPECDEC_HF_OFFLINE_FIRST=0 to allow normal online resolution.
"""

from __future__ import annotations

import os


def _as_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def hf_offline_first() -> bool:
    return _as_bool("SPECDEC_HF_OFFLINE_FIRST", True)


def apply_hf_mode_env() -> bool:
    """
    Apply environment flags for offline-first execution.

    Returns True when offline mode is active.
    """
    offline = hf_offline_first()
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return offline


def hf_model_kwargs() -> dict:
    """Common kwargs for tokenizer/model loading."""
    kwargs = {"trust_remote_code": True}
    if hf_offline_first():
        kwargs["local_files_only"] = True
    return kwargs
