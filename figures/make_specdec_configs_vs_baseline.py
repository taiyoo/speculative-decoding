from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RESULTS_DIR / "RTX4090_Cuda_Graph" / "all_configs_summary.csv")
    baseline_det = pd.read_csv(RESULTS_DIR / "baseline_deterministic.csv")
    baseline_stoch = pd.read_csv(RESULTS_DIR / "baseline_stochastic.csv")
    return summary, baseline_det, baseline_stoch


def _build_metrics_df(summary: pd.DataFrame, baseline_det: pd.DataFrame, baseline_stoch: pd.DataFrame) -> pd.DataFrame:
    base = {
        "deterministic": {
            "latency_s": float(baseline_det["latency_s"].mean()),
            "ttft_ms": float(baseline_det["ttft_ms"].mean()),
            "tpot_ms": float(baseline_det["tpot_ms"].mean()),
            "tokens_per_sec": float(baseline_det["tokens_per_sec"].mean()),
        },
        "stochastic": {
            "latency_s": float(baseline_stoch["latency_s"].mean()),
            "ttft_ms": float(baseline_stoch["ttft_ms"].mean()),
            "tpot_ms": float(baseline_stoch["tpot_ms"].mean()),
            "tokens_per_sec": float(baseline_stoch["tokens_per_sec"].mean()),
        },
    }

    df = summary.copy()
    df["regime_short"] = df["regime"].map({"deterministic": "D", "stochastic": "S"})
    df["label"] = df.apply(lambda r: f"{r['draft']}-k{int(r['k'])}-{r['regime_short']}", axis=1)

    # Keep deterministic and stochastic groups visually separated and ordered by draft,k.
    df = df.sort_values(["regime", "draft", "k"], key=lambda col: col.map({"deterministic": 0, "stochastic": 1}) if col.name == "regime" else col)

    base_latency = df["regime"].map(lambda r: base[r]["latency_s"]).astype(float)
    base_ttft = df["regime"].map(lambda r: base[r]["ttft_ms"]).astype(float)
    base_tpot = df["regime"].map(lambda r: base[r]["tpot_ms"]).astype(float)
    base_tps = df["regime"].map(lambda r: base[r]["tokens_per_sec"]).astype(float)

    df["latency_reduction_pct"] = (1.0 - (df["T_mean_s"] / base_latency)) * 100.0
    df["throughput_gain_pct"] = ((df["R_tok_mean"] / base_tps) - 1.0) * 100.0
    df["ttft_delta_pct"] = ((df["TTFT_mean_ms"] / base_ttft) - 1.0) * 100.0
    df["tpot_delta_pct"] = ((df["TPOT_mean_ms"] / base_tpot) - 1.0) * 100.0

    keep_cols = [
        "config",
        "draft",
        "k",
        "regime",
        "label",
        "S",
        "T_mean_s",
        "R_tok_mean",
        "TTFT_mean_ms",
        "TPOT_mean_ms",
        "alpha",
        "B_eff",
        "latency_reduction_pct",
        "throughput_gain_pct",
        "ttft_delta_pct",
        "tpot_delta_pct",
    ]
    return df[keep_cols]


def _plot(df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    x = np.arange(len(df))
    bar_w = 0.38
    regime_colors = {"deterministic": "#1f77b4", "stochastic": "#ff7f0e"}
    colors = [regime_colors[r] for r in df["regime"]]

    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=200)
    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.bar(x, df["S"], color=colors, alpha=0.9)
    ax1.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    ax1.set_title("Speedup vs Regime Baseline")
    ax1.set_ylabel("S (x)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=35, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    for xi, yi in zip(x, df["S"]):
        ax1.text(xi, yi + 0.03, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x - bar_w / 2, df["latency_reduction_pct"], width=bar_w, color="#2ca02c", label="Latency reduction (%)")
    ax2.bar(x + bar_w / 2, df["throughput_gain_pct"], width=bar_w, color="#9467bd", label="Throughput gain (%)")
    ax2.axhline(0.0, color="#333333", linewidth=1.0)
    ax2.set_title("Core Baseline Deltas")
    ax2.set_ylabel("Delta vs baseline (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["label"], rotation=35, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    ax2.legend(frameon=False, fontsize=9)

    ax3.bar(x - bar_w / 2, df["ttft_delta_pct"], width=bar_w, color="#d62728", label="TTFT delta (%)")
    ax3.bar(x + bar_w / 2, df["tpot_delta_pct"], width=bar_w, color="#8c564b", label="TPOT delta (%)")
    ax3.axhline(0.0, color="#333333", linewidth=1.0)
    ax3.set_title("Token Timing Deltas")
    ax3.set_ylabel("Delta vs baseline (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["label"], rotation=35, ha="right")
    ax3.grid(axis="y", linestyle="--", alpha=0.25)
    ax3.legend(frameon=False, fontsize=9)

    ax4.plot(x, df["alpha"], marker="o", color="#17becf", linewidth=2.0, label="alpha")
    ax4b = ax4.twinx()
    ax4b.plot(x, df["B_eff"], marker="s", color="#bcbd22", linewidth=2.0, label="B_eff")
    ax4.set_title("Acceptance Dynamics")
    ax4.set_ylabel("alpha")
    ax4b.set_ylabel("B_eff")
    ax4.set_xticks(x)
    ax4.set_xticklabels(df["label"], rotation=35, ha="right")
    ax4.grid(axis="y", linestyle="--", alpha=0.25)

    lines_a, labels_a = ax4.get_legend_handles_labels()
    lines_b, labels_b = ax4b.get_legend_handles_labels()
    ax4.legend(lines_a + lines_b, labels_a + labels_b, frameon=False, loc="upper right", fontsize=9)

    # Explicitly explain label coding once, to keep ticks compact.
    fig.suptitle(
        "SpecDec configurations vs baseline (RTX4090, k=4/8/16)\n"
        "Label format: <draft>-k<k>-<regime>, where D=deterministic and S=stochastic",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary, baseline_det, baseline_stoch = _load_inputs()
    metrics = _build_metrics_df(summary, baseline_det, baseline_stoch)

    csv_path = RESULTS_DIR / "specdec_configs_vs_baseline_metrics.csv"
    png_path = FIG_DIR / "specdec_configs_vs_baseline_metrics.png"
    pdf_path = FIG_DIR / "specdec_configs_vs_baseline_metrics.pdf"

    metrics.to_csv(csv_path, index=False)
    _plot(metrics, png_path, pdf_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()