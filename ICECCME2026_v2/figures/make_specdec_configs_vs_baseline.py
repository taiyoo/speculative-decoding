from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parent
PAPER_DIR = FIGURE_DIR.parent
ARTIFACT_DIR = PAPER_DIR / "camera_ready_exact_artifacts"
OUTPUT_PATH = FIGURE_DIR / "specdec_configs_vs_baseline_metrics.pdf"

REGIME_STYLE = {
    "deterministic": {"label": "Deterministic", "linestyle": "-", "marker": "o"},
    "stochastic": {"label": "Stochastic", "linestyle": "--", "marker": "s"},
}
FAMILY_COLOR = {"Qwen2.5": "#0072B2", "Qwen3": "#D55E00"}
HARDWARE_COLOR = {"RTX4090": "#0072B2", "RTX5090-laptop": "#009E73"}


def _load_summary(relative_path: str, family: str, hardware: str, draft: str) -> pd.DataFrame:
    frame = pd.read_csv(ARTIFACT_DIR / relative_path)
    frame = frame.loc[frame["k"].isin([4, 8, 16]) & (frame["draft"] == draft)].copy()
    frame["family"] = family
    frame["hardware"] = hardware
    return frame


def _load_qwen3_rtx5090() -> pd.DataFrame:
    source_dir = ARTIFACT_DIR / "qwen3_rtx5090_laptop"
    rows: list[dict[str, float | int | str]] = []

    for regime, suffix in (("deterministic", "det"), ("stochastic", "stoch")):
        baseline = pd.read_csv(source_dir / f"baseline_{regime}.csv")
        baseline_latency = float(baseline["latency_s"].mean())

        for k in (4, 8, 16):
            run = pd.read_csv(source_dir / f"spec_0.6B_k{k}_{suffix}.csv")
            rows.append(
                {
                    "k": k,
                    "regime": regime,
                    "S": baseline_latency / float(run["latency_s"].mean()),
                    "alpha": float(run["total_accepted"].sum() / run["total_proposed"].sum()),
                    "B_eff": float(run["total_accepted"].sum() / run["n_verify_steps"].sum()),
                    "family": "Qwen3",
                    "hardware": "RTX5090-laptop",
                }
            )

    return pd.DataFrame(rows)


def _load_stability() -> pd.DataFrame:
    run_families = (
        ("Qwen2.5", "RTX4090", "qwen25_rtx4090", "0.5B"),
        ("Qwen3", "RTX4090", "qwen3_rtx4090", "0.6B"),
        ("Qwen2.5", "RTX5090-laptop", "qwen25_rtx5090_laptop", "0.5B"),
        ("Qwen3", "RTX5090-laptop", "qwen3_rtx5090_laptop", "0.6B"),
    )
    rows: list[dict[str, float | int | str]] = []

    for family, hardware, directory, draft in run_families:
        for regime, suffix in (("deterministic", "det"), ("stochastic", "stoch")):
            seed_latency: dict[int, float] = {}
            for seed in (123, 999):
                path = ARTIFACT_DIR / directory / "stability" / f"spec_{draft}_k4_{suffix}_seed{seed}.csv"
                run = pd.read_csv(path)
                if set(run["seed"].unique()) != {seed}:
                    raise ValueError(f"Unexpected seed values in {path}")
                seed_latency[seed] = float(run["latency_s"].mean())

            rows.append(
                {
                    "family": family,
                    "hardware": hardware,
                    "regime": regime,
                    "seed_123_latency_s": seed_latency[123],
                    "seed_999_latency_s": seed_latency[999],
                    "latency_change_pct": (seed_latency[999] / seed_latency[123] - 1.0) * 100.0,
                }
            )

    return pd.DataFrame(rows)


def _plot_lines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    group_column: str,
    colors: dict[str, str],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for group, color in colors.items():
        for regime, style in REGIME_STYLE.items():
            subset = frame.loc[(frame[group_column] == group) & (frame["regime"] == regime)].sort_values("k")
            axis.plot(
                subset["k"],
                subset[metric],
                color=color,
                label=f"{group}, {style['label'].lower()}",
                linewidth=1.6,
                markersize=4.5,
                **{key: style[key] for key in ("linestyle", "marker")},
            )

    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Proposal length, $k$")
    axis.set_ylabel(ylabel)
    axis.set_xticks([4, 8, 16])
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.6)
    axis.legend(frameon=False, fontsize=6.6, ncol=2, columnspacing=0.8, handlelength=2.2)


def _plot_stability(axis: plt.Axes, stability: pd.DataFrame) -> None:
    categories = (
        ("Qwen2.5", "RTX4090", "Qwen2.5\nRTX4090"),
        ("Qwen3", "RTX4090", "Qwen3\nRTX4090"),
        ("Qwen2.5", "RTX5090-laptop", "Qwen2.5\nRTX5090-L"),
        ("Qwen3", "RTX5090-laptop", "Qwen3\nRTX5090-L"),
    )
    x = np.arange(len(categories))
    width = 0.34

    for offset, (regime, style) in zip((-width / 2, width / 2), REGIME_STYLE.items()):
        values = []
        for family, hardware, _ in categories:
            row = stability.loc[
                (stability["family"] == family)
                & (stability["hardware"] == hardware)
                & (stability["regime"] == regime),
                "latency_change_pct",
            ]
            values.append(float(row.iloc[0]))

        bars = axis.bar(
            x + offset,
            values,
            width,
            color="#0072B2" if regime == "deterministic" else "#D55E00",
            label=style["label"],
        )
        for bar, value in zip(bars, values):
            vertical_alignment = "bottom" if value >= 0 else "top"
            label_offset = 0.12 if value >= 0 else -0.12
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset,
                f"{value:+.1f}",
                ha="center",
                va=vertical_alignment,
                fontsize=6.3,
            )

    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_title("(d) Repeat-seed latency stability at $k=4$", loc="left", fontweight="bold")
    axis.set_ylabel("Seed 999 vs 123 change (%)")
    axis.set_xticks(x, [label for _, _, label in categories])
    axis.set_ylim(-6.2, 1.2)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.6)
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")


def main() -> None:
    qwen25_rtx4090 = _load_summary(
        "qwen25_rtx4090/all_configs_summary.csv", "Qwen2.5", "RTX4090", "0.5B"
    )
    qwen3_rtx4090 = _load_summary(
        "qwen3_rtx4090/all_configs_summary.csv", "Qwen3", "RTX4090", "0.6B"
    )
    qwen3_rtx5090 = _load_qwen3_rtx5090()
    stability = _load_stability()

    if not all(len(frame) == 6 for frame in (qwen25_rtx4090, qwen3_rtx4090, qwen3_rtx5090)):
        raise ValueError("Each fixed-policy comparison must contain six k/regime rows")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 5,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.35), constrained_layout=True)

    rtx4090 = pd.concat([qwen25_rtx4090, qwen3_rtx4090], ignore_index=True)
    _plot_lines(
        axes[0, 0],
        rtx4090,
        "family",
        FAMILY_COLOR,
        "S",
        "Speedup, $S$ ($\\times$)",
        "(a) Model-family comparison on RTX4090",
    )
    axes[0, 0].axhline(1.0, color="#555555", linestyle=":", linewidth=0.9)
    axes[0, 0].set_ylim(0.75, 3.25)

    qwen3_hardware = pd.concat([qwen3_rtx4090, qwen3_rtx5090], ignore_index=True)
    _plot_lines(
        axes[0, 1],
        qwen3_hardware,
        "hardware",
        HARDWARE_COLOR,
        "S",
        "Speedup, $S$ ($\\times$)",
        "(b) Qwen3 hardware portability",
    )
    axes[0, 1].axhline(1.0, color="#555555", linestyle=":", linewidth=0.9)
    axes[0, 1].set_ylim(0.0, 3.05)

    _plot_lines(
        axes[1, 0],
        qwen3_hardware,
        "hardware",
        HARDWARE_COLOR,
        "alpha",
        "Acceptance rate, $\\alpha$",
        "(c) Qwen3 acceptance portability",
    )
    axes[1, 0].set_ylim(0.05, 0.38)

    _plot_stability(axes[1, 1], stability)

    figure.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()