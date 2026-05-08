from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent


def _pick_results_dir() -> Path:
    preferred = ROOT / "results" / "RTX5090"
    if (preferred / "gpu_opt_benchmark_delta.csv").exists():
        return preferred
    return ROOT / "results"


def _load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    delta = pd.read_csv(results_dir / "gpu_opt_benchmark_delta.csv")
    ci = pd.read_csv(results_dir / "gpu_opt_benchmark_bootstrap_ci.csv")
    task = pd.read_csv(results_dir / "gpu_opt_benchmark_task_delta.csv")
    return delta, ci, task


def _build_overall_df(delta: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    metric_order = ["latency_s", "draft_elapsed_s", "verify_elapsed_s", "ttft_ms"]

    base_map = delta.set_index("metric")["baseline_mean"].to_dict()
    pct_map = delta.set_index("metric")["pct_delta_compare_minus_base"].to_dict()

    ci_map = ci.set_index("metric")[["ci95_low", "ci95_high"]].to_dict("index")

    rows = []
    for metric in metric_order:
        baseline = float(base_map[metric])
        pct = float(pct_map[metric])
        ci_low_abs = float(ci_map[metric]["ci95_low"])
        ci_high_abs = float(ci_map[metric]["ci95_high"])
        ci_low_pct = (ci_low_abs / baseline) * 100.0
        ci_high_pct = (ci_high_abs / baseline) * 100.0

        rows.append(
            {
                "metric": metric,
                "pct_delta": pct,
                "ci_low_pct": ci_low_pct,
                "ci_high_pct": ci_high_pct,
                "err_low": max(0.0, pct - ci_low_pct),
                "err_high": max(0.0, ci_high_pct - pct),
            }
        )

    out = pd.DataFrame(rows)
    label_map = {
        "latency_s": "End-to-end latency",
        "draft_elapsed_s": "Draft stage time",
        "verify_elapsed_s": "Verify stage time",
        "ttft_ms": "Time to first token",
    }
    out["label"] = out["metric"].map(label_map)
    return out


def render() -> tuple[Path, Path]:
    results_dir = _pick_results_dir()
    delta, ci, task = _load_inputs(results_dir)
    overall = _build_overall_df(delta, ci)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    y = np.arange(len(overall))
    colors = ["#b22222" if x > 0 else "#1f7a1f" for x in overall["pct_delta"]]
    ax0.barh(y, overall["pct_delta"], color=colors, alpha=0.88)
    ax0.errorbar(
        overall["pct_delta"],
        y,
        xerr=np.vstack([overall["err_low"], overall["err_high"]]),
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=3,
    )
    ax0.axvline(0.0, color="#444444", linewidth=1.0)
    ax0.set_yticks(y)
    ax0.set_yticklabels(overall["label"])
    ax0.invert_yaxis()
    ax0.set_xlabel("Delta vs gpu_opt_off (%)")
    ax0.set_title("Overall effect with 95% bootstrap CI")
    ax0.grid(axis="x", linestyle="--", alpha=0.25)

    for yi, x in enumerate(overall["pct_delta"]):
        ax0.text(x + (0.08 if x >= 0 else -0.08), yi, f"{x:+.2f}%", va="center", ha="left" if x >= 0 else "right", fontsize=9)

    task_df = task.loc[:, ["task", "latency_s_delta_pct"]].copy()
    task_df = task_df.sort_values("latency_s_delta_pct", ascending=True)
    task_colors = ["#b22222" if x > 0 else "#1f7a1f" for x in task_df["latency_s_delta_pct"]]

    ax1.barh(task_df["task"], task_df["latency_s_delta_pct"], color=task_colors, alpha=0.88)
    ax1.axvline(0.0, color="#444444", linewidth=1.0)
    ax1.set_xlabel("Latency delta (%)")
    ax1.set_title("Task-level latency delta")
    ax1.grid(axis="x", linestyle="--", alpha=0.25)

    for i, x in enumerate(task_df["latency_s_delta_pct"]):
        ax1.text(x + (0.08 if x >= 0 else -0.08), i, f"{x:+.2f}%", va="center", ha="left" if x >= 0 else "right", fontsize=9)

    match_row = (results_dir / "gpu_opt_benchmark_equivalence.csv")
    subtitle = "Output-equivalent benchmark"
    if match_row.exists():
        eq = pd.read_csv(match_row)
        if not eq.empty:
            subtitle = (
                f"Output match: {float(eq.loc[0, 'output_match_rate_pct']):.1f}% "
                f"(n={int(eq.loc[0, 'n_pairs'])} paired samples)"
            )

    fig.suptitle(
        "CUDA optimisation toggle benchmark on RTX 5090 Laptop\n"
        f"{subtitle}",
        fontsize=11,
        y=1.03,
    )

    fig.tight_layout()

    png_path = FIG_DIR / "cuda_optimisation_comparison.png"
    pdf_path = FIG_DIR / "cuda_optimisation_comparison.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = render()
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
