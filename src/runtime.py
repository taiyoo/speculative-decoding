"""
Notebook/runtime helpers so each phase can run independently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import MANIFESTS_DIR, RESULTS_DIR
from data_loader import load_all_datasets, freeze_manifests, save_full_data, load_from_manifests
from evaluate import evaluate_results
from metrics import compute_latency_metrics


def bootstrap_notebook():
    """
    Ensure src/ is importable no matter where the notebook cell is executed from.

    Works locally (notebook next to src/) and on Colab (notebook in /content
    while the repo lives elsewhere on disk).
    """
    cwd = Path.cwd().resolve()

    def _has_src(p: Path) -> bool:
        return (p / "src" / "baseline.py").exists()

    project_root = None
    # 1. Walk upward from cwd.
    for candidate in [cwd, *cwd.parents]:
        if _has_src(candidate):
            project_root = candidate
            break
    # 2. Common Colab locations.
    if project_root is None and Path("/content").exists():
        colab_candidates = [Path("/content")]
        for child in Path("/content").iterdir():
            if child.is_dir():
                colab_candidates.append(child)
        for candidate in colab_candidates:
            if _has_src(candidate):
                project_root = candidate
                break
    if project_root is None:
        project_root = cwd

    src_dir = str(project_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return project_root


def ensure_data(ns: dict) -> dict:
    if "data" in ns and ns["data"]:
        return ns["data"]

    manifests_exist = all(
        (MANIFESTS_DIR / f"{task}_data.json").exists()
        for task in ["gsm8k", "mmlu", "cnndm"]
    )
    if manifests_exist:
        data = load_from_manifests()
    else:
        data = load_all_datasets()
        freeze_manifests(data)
        save_full_data(data)
    ns["data"] = data
    return data


def ensure_target_model(ns: dict):
    if "target_model" in ns and "target_tokenizer" in ns:
        return ns["target_model"], ns["target_tokenizer"]

    from baseline import load_target_model

    target_model, target_tokenizer = load_target_model()
    ns["target_model"] = target_model
    ns["target_tokenizer"] = target_tokenizer
    return target_model, target_tokenizer


def ensure_draft_model(ns: dict, draft_label: str):
    suffix = "05" if draft_label == "0.5B" else "15"
    model_key = f"draft_{suffix}_model"
    tokenizer_key = f"draft_{suffix}_tokenizer"
    if model_key in ns and tokenizer_key in ns:
        return ns[model_key], ns[tokenizer_key]

    from speculative import load_draft_model

    draft_model, draft_tokenizer = load_draft_model(draft_label)
    ns[model_key] = draft_model
    ns[tokenizer_key] = draft_tokenizer
    return draft_model, draft_tokenizer


def _read_results_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def ensure_baseline_results(ns: dict):
    data = ensure_data(ns)
    ensure_target_model(ns)

    from baseline import run_baseline

    for regime_name, key_suffix in [
        ("deterministic", "det"),
        ("stochastic", "stoch"),
    ]:
        result_key = f"baseline_{key_suffix}"
        quality_key = f"base_quality_{key_suffix}"
        latency_key = f"base_lat_{key_suffix}"
        csv_path = RESULTS_DIR / f"baseline_{regime_name}.csv"

        if result_key not in ns:
            results = _read_results_csv(csv_path)
            if not results:
                results = run_baseline(
                    data,
                    regime_name,
                    ns["target_model"],
                    ns["target_tokenizer"],
                )
            ns[result_key] = results

        if quality_key not in ns:
            ns[quality_key] = evaluate_results(ns[result_key], data)

        if latency_key not in ns:
            ns[latency_key] = compute_latency_metrics(ns[result_key])


def ensure_spec_results(ns: dict, draft_label: str) -> dict[str, list[dict]]:
    suffix = "05" if draft_label == "0.5B" else "15"
    result_key = f"spec_results_{suffix}"
    if result_key in ns and ns[result_key]:
        return ns[result_key]

    results = {}
    regime_suffixes = {
        "deterministic": "det",
        "stochastic": "stoch",
    }
    for k_val in [4, 8, 16]:
        for regime_name, regime_short in regime_suffixes.items():
            csv_path = RESULTS_DIR / f"spec_{draft_label}_k{k_val}_{regime_short}.csv"
            rows = _read_results_csv(csv_path)
            if rows:
                results[f"{draft_label}_k{k_val}_{regime_name}"] = rows

    ns[result_key] = results
    return results


def ensure_df_all(ns: dict):
    if "df_all" in ns and not ns["df_all"].empty:
        return ns["df_all"]

    summary_path = RESULTS_DIR / "all_configs_summary.csv"
    combined_path = RESULTS_DIR / "all_configs_combined.csv"
    for path in [summary_path, combined_path]:
        if path.exists():
            ns["df_all"] = pd.read_csv(path)
            return ns["df_all"]

    return pd.DataFrame()
