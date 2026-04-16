"""
Metric computation for speculative decoding experiments.

Computes the 12 metrics defined in sec/03_metrics.tex:
  T        — wall-clock latency (s)
  R_tok    — tokens/second throughput
  TTFT     — time to first token (ms)
  TPOT     — time per output token (ms)
  α        — token acceptance rate
  B_eff    — effective block size
  S        — speedup vs. baseline
  Q_gsm    — GSM8K exact-match accuracy (%)
  Q_mmlu   — MMLU letter-match accuracy (%)
  Q_sum    — CNN/DM ROUGE-L F1 (%)
  ΔQ       — quality delta (spec − baseline)
  σ_S      — speedup std across seeds
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_latency_metrics(results: list[dict]) -> dict:
    """Compute aggregate latency metrics from per-sample results."""
    df = pd.DataFrame(results)
    return {
        "T_mean_s": round(df["latency_s"].mean(), 4),
        "T_median_s": round(df["latency_s"].median(), 4),
        "R_tok_mean": round(df["tokens_per_sec"].mean(), 2),
        "TTFT_mean_ms": round(df["ttft_ms"].mean(), 2),
        "TPOT_mean_ms": round(df["tpot_ms"].mean(), 2),
        "total_tokens": int(df["num_tokens"].sum()),
        "n_samples": len(df),
    }


def compute_acceptance_metrics(spec_results: list[dict]) -> dict:
    """
    Compute acceptance rate α and effective block size B_eff
    from speculative decoding results.

    Each result row should have 'total_proposed' and 'total_accepted' fields.
    """
    total_proposed = sum(r.get("total_proposed", 0) for r in spec_results)
    total_accepted = sum(r.get("total_accepted", 0) for r in spec_results)

    alpha = total_accepted / total_proposed if total_proposed > 0 else 0.0

    # B_eff = mean accepted tokens per verify step
    n_verify = sum(r.get("n_verify_steps", 0) for r in spec_results)
    b_eff = total_accepted / n_verify if n_verify > 0 else 0.0

    return {
        "alpha": round(alpha, 4),
        "B_eff": round(b_eff, 2),
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "n_verify_steps": n_verify,
    }


def compute_speedup(
    baseline_results: list[dict],
    spec_results: list[dict],
) -> float:
    """
    Compute speedup S = T_baseline / T_speculative.
    Matches on sample_id for fair comparison.
    """
    base_latency = {r["sample_id"]: r["latency_s"] for r in baseline_results}
    spec_latency = {r["sample_id"]: r["latency_s"] for r in spec_results}

    common_ids = set(base_latency) & set(spec_latency)
    if not common_ids:
        return 0.0

    total_base = sum(base_latency[sid] for sid in common_ids)
    total_spec = sum(spec_latency[sid] for sid in common_ids)

    return round(total_base / total_spec, 4) if total_spec > 0 else 0.0


def compute_quality_delta(
    baseline_quality: dict[str, float],
    spec_quality: dict[str, float],
) -> dict[str, float]:
    """
    Compute ΔQ = Q_spec − Q_base for each task.
    Values are in percentage points.
    """
    delta = {}
    for task in baseline_quality:
        if task in spec_quality:
            delta[f"delta_{task}"] = round(
                spec_quality[task] - baseline_quality[task], 2
            )
    return delta


def compute_speedup_stability(speedups: list[float]) -> float:
    """Compute σ_S — standard deviation of speedup across seeds."""
    if len(speedups) < 2:
        return 0.0
    return round(float(np.std(speedups, ddof=1)), 4)


def build_config_summary(
    config_name: str,
    draft_label: str,
    k: int,
    regime: str,
    latency_metrics: dict,
    acceptance_metrics: dict | None,
    quality: dict[str, float],
    speedup: float,
    quality_delta: dict[str, float] | None = None,
    sigma_s: float | None = None,
) -> dict:
    """
    Build a single-row summary dict for one experiment configuration.
    """
    row = {
        "config": config_name,
        "draft": draft_label,
        "k": k,
        "regime": regime,
        "T_mean_s": latency_metrics["T_mean_s"],
        "R_tok": latency_metrics["R_tok_mean"],
        "TTFT_ms": latency_metrics["TTFT_mean_ms"],
        "TPOT_ms": latency_metrics["TPOT_mean_ms"],
        "S": speedup,
    }

    if acceptance_metrics:
        row["alpha"] = acceptance_metrics["alpha"]
        row["B_eff"] = acceptance_metrics["B_eff"]
    else:
        row["alpha"] = None
        row["B_eff"] = None

    row["Q_gsm"] = quality.get("gsm8k", None)
    row["Q_mmlu"] = quality.get("mmlu", None)
    row["Q_sum"] = quality.get("cnndm", None)

    if quality_delta:
        row["dQ_gsm"] = quality_delta.get("delta_gsm8k", None)
        row["dQ_mmlu"] = quality_delta.get("delta_mmlu", None)
        row["dQ_sum"] = quality_delta.get("delta_cnndm", None)

    if sigma_s is not None:
        row["sigma_S"] = sigma_s

    return row
