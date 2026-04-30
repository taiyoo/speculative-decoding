"""
Unified visualization-evaluation backend for speculative decoding results.

This module reads CSV outputs directly from results/, builds robust paired
comparisons against baseline outputs, and provides publication-ready summary
plots and tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASELINE_FILE_RE = re.compile(r"^baseline_(deterministic|stochastic)\.csv$")
SPEC_FILE_RE = re.compile(r"^spec_([0-9.]+B)_k(\d+)_(det|stoch)\.csv$")


@dataclass
class EvalBundle:
    results_dir: Path
    discovered_files: list[Path]
    baseline_df: pd.DataFrame
    spec_df: pd.DataFrame
    merged_df: pd.DataFrame
    run_summary: pd.DataFrame
    task_summary: pd.DataFrame


def _regime_long(short_name: str) -> str:
    return "deterministic" if short_name == "det" else "stochastic"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path}") from exc


def discover_result_files(results_dir: str | Path = "results") -> dict[str, list[Path]]:
    """Discover baseline/spec result files in a results directory."""
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    baseline_files: list[Path] = []
    spec_files: list[Path] = []

    for path in sorted(root.glob("*.csv")):
        name = path.name
        if BASELINE_FILE_RE.match(name):
            baseline_files.append(path)
        elif SPEC_FILE_RE.match(name):
            spec_files.append(path)

    return {
        "baseline": baseline_files,
        "spec": spec_files,
        "all": sorted(baseline_files + spec_files),
    }


def _load_baseline_df(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        df = _safe_read_csv(path)
        m = BASELINE_FILE_RE.match(path.name)
        if m is None:
            continue
        regime = m.group(1)
        if "regime" not in df.columns:
            df["regime"] = regime
        df["source_file"] = path.name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    numeric_cols = ["latency_s", "ttft_ms", "tpot_ms", "num_tokens", "tokens_per_sec"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_spec_df(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        df = _safe_read_csv(path)
        m = SPEC_FILE_RE.match(path.name)
        if m is None:
            continue

        draft, k_str, regime_short = m.groups()
        regime = _regime_long(regime_short)
        k_val = int(k_str)

        df["draft"] = df.get("draft", draft)
        df["k"] = pd.to_numeric(df.get("k", k_val), errors="coerce").fillna(k_val).astype(int)
        df["regime"] = df.get("regime", regime)
        df["source_file"] = path.name
        df["config"] = df.apply(
            lambda r: f"{r['draft']}_k{int(r['k'])}_{r['regime']}",
            axis=1,
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "seed",
        "latency_s",
        "ttft_ms",
        "tpot_ms",
        "num_tokens",
        "tokens_per_sec",
        "total_proposed",
        "total_accepted",
        "n_verify_steps",
        "alpha",
        "B_eff",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_pairwise_df(baseline_df: pd.DataFrame, spec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair each speculative row with baseline by (sample_id, regime).

    Produces per-sample comparative metrics for visualization and diagnostics.
    """
    if baseline_df.empty or spec_df.empty:
        return pd.DataFrame()

    base_cols = [
        "sample_id",
        "regime",
        "latency_s",
        "ttft_ms",
        "tpot_ms",
        "num_tokens",
        "tokens_per_sec",
        "output_text",
    ]
    existing = [c for c in base_cols if c in baseline_df.columns]
    base = baseline_df[existing].copy().rename(
        columns={
            "latency_s": "baseline_latency_s",
            "ttft_ms": "baseline_ttft_ms",
            "tpot_ms": "baseline_tpot_ms",
            "num_tokens": "baseline_num_tokens",
            "tokens_per_sec": "baseline_tokens_per_sec",
            "output_text": "baseline_output_text",
        }
    )

    merged = spec_df.merge(base, on=["sample_id", "regime"], how="left", validate="many_to_one")

    merged["speedup_sample"] = np.where(
        merged["latency_s"] > 0,
        merged["baseline_latency_s"] / merged["latency_s"],
        np.nan,
    )

    spec_text = merged.get("output_text", "").fillna("").astype(str).str.strip()
    base_text = merged.get("baseline_output_text", "").fillna("").astype(str).str.strip()
    merged["disagree_with_baseline"] = spec_text != base_text

    merged["num_tokens_ratio"] = np.where(
        merged["baseline_num_tokens"] > 0,
        merged["num_tokens"] / merged["baseline_num_tokens"],
        np.nan,
    )
    merged["latency_delta_s"] = merged["latency_s"] - merged["baseline_latency_s"]
    merged["tokens_per_sec_delta"] = merged["tokens_per_sec"] - merged["baseline_tokens_per_sec"]

    return merged


def summarize_runs(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create run-level summary for each speculative config."""
    if merged_df.empty:
        return pd.DataFrame()

    grouped = merged_df.groupby(["config", "draft", "k", "regime"], as_index=False)
    summary = grouped.agg(
        n_samples=("sample_id", "count"),
        n_tasks=("task", "nunique"),
        latency_mean_s=("latency_s", "mean"),
        baseline_latency_mean_s=("baseline_latency_s", "mean"),
        speedup_mean=("speedup_sample", "mean"),
        speedup_median=("speedup_sample", "median"),
        speedup_p10=("speedup_sample", lambda s: np.nanquantile(s, 0.10)),
        speedup_p90=("speedup_sample", lambda s: np.nanquantile(s, 0.90)),
        ttft_mean_ms=("ttft_ms", "mean"),
        tpot_mean_ms=("tpot_ms", "mean"),
        tokens_per_sec_mean=("tokens_per_sec", "mean"),
        baseline_tokens_per_sec_mean=("baseline_tokens_per_sec", "mean"),
        alpha_mean=("alpha", "mean"),
        B_eff_mean=("B_eff", "mean"),
        disagreement_rate=("disagree_with_baseline", "mean"),
        output_length_ratio_mean=("num_tokens_ratio", "mean"),
    )

    for col in [
        "latency_mean_s",
        "baseline_latency_mean_s",
        "speedup_mean",
        "speedup_median",
        "speedup_p10",
        "speedup_p90",
        "ttft_mean_ms",
        "tpot_mean_ms",
        "tokens_per_sec_mean",
        "baseline_tokens_per_sec_mean",
        "alpha_mean",
        "B_eff_mean",
        "disagreement_rate",
        "output_length_ratio_mean",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    summary["disagreement_rate_pct"] = summary["disagreement_rate"] * 100.0
    summary = summary.sort_values(["speedup_mean", "disagreement_rate"], ascending=[False, True]).reset_index(drop=True)
    return summary


def summarize_by_task(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create config x task summary."""
    if merged_df.empty:
        return pd.DataFrame()

    grouped = merged_df.groupby(["config", "draft", "k", "regime", "task"], as_index=False)
    task_summary = grouped.agg(
        n_samples=("sample_id", "count"),
        speedup_mean=("speedup_sample", "mean"),
        speedup_median=("speedup_sample", "median"),
        latency_mean_s=("latency_s", "mean"),
        baseline_latency_mean_s=("baseline_latency_s", "mean"),
        alpha_mean=("alpha", "mean"),
        B_eff_mean=("B_eff", "mean"),
        disagreement_rate=("disagree_with_baseline", "mean"),
        output_length_ratio_mean=("num_tokens_ratio", "mean"),
    )
    task_summary["disagreement_rate_pct"] = task_summary["disagreement_rate"] * 100.0
    return task_summary.sort_values(["task", "speedup_mean"], ascending=[True, False]).reset_index(drop=True)


def load_visual_evaluation(results_dir: str | Path = "results") -> EvalBundle:
    """Load all available result CSVs and build evaluation-ready data bundles."""
    discovered = discover_result_files(results_dir)
    baseline_df = _load_baseline_df(discovered["baseline"])
    spec_df = _load_spec_df(discovered["spec"])
    merged_df = build_pairwise_df(baseline_df, spec_df)
    run_summary = summarize_runs(merged_df)
    task_summary = summarize_by_task(merged_df)
    return EvalBundle(
        results_dir=Path(results_dir),
        discovered_files=discovered["all"],
        baseline_df=baseline_df,
        spec_df=spec_df,
        merged_df=merged_df,
        run_summary=run_summary,
        task_summary=task_summary,
    )


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120


def plot_speedup_by_config(run_summary: pd.DataFrame, top_n: int | None = None, ax=None):
    if run_summary.empty:
        raise ValueError("run_summary is empty")
    set_plot_style()
    df = run_summary.copy()
    if top_n is not None:
        df = df.head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    sns.barplot(data=df, x="config", y="speedup_mean", hue="regime", ax=ax)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Mean Speedup by Configuration")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Speedup (baseline/spec)")
    ax.tick_params(axis="x", rotation=35)
    return ax


def plot_pareto_speedup_vs_disagreement(run_summary: pd.DataFrame, ax=None):
    if run_summary.empty:
        raise ValueError("run_summary is empty")
    set_plot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=run_summary,
        x="speedup_mean",
        y="disagreement_rate_pct",
        hue="draft",
        style="regime",
        s=100,
        ax=ax,
    )
    for _, row in run_summary.iterrows():
        ax.text(row["speedup_mean"], row["disagreement_rate_pct"], row["config"], fontsize=8)

    ax.set_title("Pareto: Speedup vs Output Disagreement")
    ax.set_xlabel("Mean Speedup")
    ax.set_ylabel("Disagreement with baseline (%)")
    return ax


def plot_acceptance_heatmap(run_summary: pd.DataFrame, regime: str, metric: str = "alpha_mean", ax=None):
    if run_summary.empty:
        raise ValueError("run_summary is empty")
    set_plot_style()

    subset = run_summary[run_summary["regime"] == regime].copy()
    if subset.empty:
        raise ValueError(f"No rows for regime={regime}")

    pivot = subset.pivot_table(index="k", columns="draft", values=metric, aggfunc="mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{metric} Heatmap ({regime})")
    ax.set_xlabel("Draft")
    ax.set_ylabel("k")
    return ax


def plot_task_speedup_heatmap(task_summary: pd.DataFrame, ax=None):
    if task_summary.empty:
        raise ValueError("task_summary is empty")
    set_plot_style()

    pivot = task_summary.pivot_table(index="task", columns="config", values="speedup_mean", aggfunc="mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0, ax=ax)
    ax.set_title("Task-level Mean Speedup")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Task")
    return ax


def plot_speedup_distribution(merged_df: pd.DataFrame, top_configs: int = 6, ax=None):
    if merged_df.empty:
        raise ValueError("merged_df is empty")
    set_plot_style()

    cfg_rank = (
        merged_df.groupby("config", as_index=False)["speedup_sample"]
        .mean()
        .sort_values("speedup_sample", ascending=False)
    )
    keep = cfg_rank.head(top_configs)["config"]
    subset = merged_df[merged_df["config"].isin(keep)]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    sns.boxplot(data=subset, x="config", y="speedup_sample", ax=ax)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Per-sample Speedup Distribution (Top {top_configs} configs)")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Sample speedup")
    ax.tick_params(axis="x", rotation=35)
    return ax


def plot_latency_gain_vs_acceptance(run_summary: pd.DataFrame, ax=None):
    if run_summary.empty:
        raise ValueError("run_summary is empty")
    set_plot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    run_summary = run_summary.copy()
    run_summary["latency_gain_pct"] = (
        (run_summary["baseline_latency_mean_s"] - run_summary["latency_mean_s"])
        / run_summary["baseline_latency_mean_s"]
        * 100.0
    )

    sns.scatterplot(
        data=run_summary,
        x="alpha_mean",
        y="latency_gain_pct",
        hue="draft",
        style="regime",
        s=100,
        ax=ax,
    )
    ax.set_title("Acceptance vs Latency Gain")
    ax.set_xlabel("Mean acceptance rate (alpha)")
    ax.set_ylabel("Latency reduction vs baseline (%)")
    return ax


def export_summary_tables(bundle: EvalBundle, output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary_path = out_dir / "run_summary.csv"
    task_summary_path = out_dir / "task_summary.csv"
    pairwise_path = out_dir / "pairwise_sample_metrics.csv"

    bundle.run_summary.to_csv(run_summary_path, index=False)
    bundle.task_summary.to_csv(task_summary_path, index=False)
    bundle.merged_df.to_csv(pairwise_path, index=False)

    return {
        "run_summary": run_summary_path,
        "task_summary": task_summary_path,
        "pairwise_sample_metrics": pairwise_path,
    }


def export_all_figures(bundle: EvalBundle, output_dir: str | Path, dpi: int = 180) -> dict[str, Path]:
    if bundle.run_summary.empty or bundle.merged_df.empty:
        raise ValueError("No data available for figure export")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_speedup_by_config(bundle.run_summary, ax=ax)
    fig.tight_layout()
    paths["speedup_by_config"] = out_dir / "speedup_by_config.png"
    fig.savefig(paths["speedup_by_config"], dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_pareto_speedup_vs_disagreement(bundle.run_summary, ax=ax)
    fig.tight_layout()
    paths["pareto_speedup_vs_disagreement"] = out_dir / "pareto_speedup_vs_disagreement.png"
    fig.savefig(paths["pareto_speedup_vs_disagreement"], dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, regime in enumerate(["deterministic", "stochastic"]):
        if regime in bundle.run_summary["regime"].unique():
            plot_acceptance_heatmap(bundle.run_summary, regime=regime, metric="alpha_mean", ax=axes[i])
        else:
            axes[i].set_axis_off()
            axes[i].set_title(f"No {regime} data")
    fig.tight_layout()
    paths["acceptance_heatmaps"] = out_dir / "acceptance_heatmaps.png"
    fig.savefig(paths["acceptance_heatmaps"], dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_task_speedup_heatmap(bundle.task_summary, ax=ax)
    fig.tight_layout()
    paths["task_speedup_heatmap"] = out_dir / "task_speedup_heatmap.png"
    fig.savefig(paths["task_speedup_heatmap"], dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_speedup_distribution(bundle.merged_df, ax=ax)
    fig.tight_layout()
    paths["speedup_distribution"] = out_dir / "speedup_distribution.png"
    fig.savefig(paths["speedup_distribution"], dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_latency_gain_vs_acceptance(bundle.run_summary, ax=ax)
    fig.tight_layout()
    paths["latency_gain_vs_acceptance"] = out_dir / "latency_gain_vs_acceptance.png"
    fig.savefig(paths["latency_gain_vs_acceptance"], dpi=dpi)
    plt.close(fig)

    return paths


__all__ = [
    "EvalBundle",
    "discover_result_files",
    "load_visual_evaluation",
    "summarize_runs",
    "summarize_by_task",
    "plot_speedup_by_config",
    "plot_pareto_speedup_vs_disagreement",
    "plot_acceptance_heatmap",
    "plot_task_speedup_heatmap",
    "plot_speedup_distribution",
    "plot_latency_gain_vs_acceptance",
    "export_summary_tables",
    "export_all_figures",
]
