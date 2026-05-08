"""Render DriftDiffuse architecture and pipeline diagrams to PNG.

Run from the assignment root:
    python figures/make_diagrams.py

Editable .drawio sources for the same diagrams sit alongside this script
(driftdiffuse_architecture.drawio, driftdiffuse_pipeline.drawio). Open them in
VSCode with the drawio extension and re-export to PNG for higher-fidelity
versions if desired.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent

# --- Colour palette (consistent across diagrams) ---
CTX = "#cfe3ff"
MASK = "#ffd7d7"
TOK = "#dff0d8"
LAYER = "#ececec"
ACCENT = "#666666"
ARROW = "#444444"


def _box(ax, xy, w, h, text, fc=LAYER, ec=ACCENT, fontsize=9, weight="normal"):
    p = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight)


def _arrow(ax, src, dst, color=ARROW, style="-|>", lw=1.1):
    a = FancyArrowPatch(src, dst, arrowstyle=style, mutation_scale=10,
                        color=color, linewidth=lw)
    ax.add_patch(a)


def render_architecture(path: Path) -> None:
    """DriftDiffuser model internals (left = inputs, right = output)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    # ---- Inputs row ----
    # context tokens
    for i in range(6):
        _box(ax, (0.2 + i * 0.55, 4.6), 0.5, 0.4, f"c{i}", fc=CTX, fontsize=8)
    # block: alternating revealed / masked
    block_labels = ["x0", "[M]", "[M]", "[M]"]
    block_fcs = [TOK, MASK, MASK, MASK]
    for i, (lab, fc) in enumerate(zip(block_labels, block_fcs)):
        _box(ax, (3.7 + i * 0.55, 4.6), 0.5, 0.4, lab, fc=fc, fontsize=8)
    ax.text(0.45 + 6 * 0.275, 5.15, "context (frozen)", fontsize=8, color="#555")
    ax.text(3.95 + 4 * 0.275, 5.15, "block (k positions)", fontsize=8, color="#555")

    # ---- Embedding lookup ----
    _box(ax, (0.2, 3.7), 5.95, 0.5, "Token Embedding   |   MASK Embedding (sentinel for [M])", fc="#fff7e0", fontsize=9)

    # Pos + time
    _box(ax, (6.4, 3.7), 1.5, 0.5, "+ Positional Emb", fc="#eef", fontsize=9)
    _box(ax, (8.1, 3.7), 1.5, 0.5, "+ Time Emb (t_i)", fc="#eef", fontsize=9)
    ax.text(8.85, 4.3, "(masked positions only)", fontsize=7, color="#666", ha="center")

    # ---- Transformer stack (6 bidirectional blocks) ----
    stack_x, stack_y = 1.5, 1.4
    stack_w, stack_h = 8.0, 1.9
    _box(ax, (stack_x, stack_y), stack_w, stack_h, "", fc="#f6f6fb", ec=ACCENT)
    ax.text(stack_x + stack_w / 2, stack_y + stack_h - 0.18,
            "Bidirectional Transformer  ×  N layers (RMSNorm  +  SDPA  +  SiLU FFN)",
            ha="center", fontsize=10, weight="bold")
    # individual mini-layer slabs
    n_layers = 6
    slab_w = (stack_w - 0.6) / n_layers
    for i in range(n_layers):
        _box(ax, (stack_x + 0.3 + i * slab_w, stack_y + 0.2),
             slab_w * 0.92, 1.05, f"L{i+1}", fc=LAYER, fontsize=8)

    # ---- Output head ----
    _box(ax, (3.0, 0.4), 5.0, 0.55,
         "Tied Output Head  →  logits over target vocab (V = 152{,}064)",
         fc="#e7f0ff", fontsize=10, weight="bold")

    # ---- Arrows ----
    _arrow(ax, (3.1, 4.55), (3.1, 4.25))                          # inputs to embed
    _arrow(ax, (3.1, 3.65), (3.1, 3.35))                          # embed to stack-top
    _arrow(ax, (5.5, 1.35), (5.5, 1.0))                           # stack to head
    _arrow(ax, (7.15, 3.65), (5.7, 3.35), style="->", lw=0.9)
    _arrow(ax, (8.85, 3.65), (5.9, 3.35), style="->", lw=0.9)

    # ---- Legend ----
    legend_handles = [
        mpatches.Patch(color=CTX, label="context token"),
        mpatches.Patch(color=TOK, label="revealed block token"),
        mpatches.Patch(color=MASK, label="[MASK] sentinel"),
        mpatches.Patch(color=LAYER, label="transformer layer"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True,
              bbox_to_anchor=(0.0, -0.02))

    ax.set_title("DriftDiffuser architecture: parallel k-token proposer with masked diffusion",
                 fontsize=11, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_pipeline(path: Path) -> None:
    """End-to-end DriftDiffuse speculative decoding loop."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.set_axis_off()

    # ---- Stage 1: Drift schedule sampling ----
    _box(ax, (0.2, 4.7), 2.7, 1.3,
         "1. Drift Schedule\n\nt_i = clip(t + λ·i/k, 0, T)\nlater positions stay\nmasked longer",
         fc="#fff5e6", fontsize=9, weight="normal")

    # ---- Stage 2: Iterative unmask (T denoising steps) ----
    _box(ax, (3.3, 4.7), 3.3, 1.3, "", fc="#f6f6fb")
    ax.text(3.3 + 1.65, 5.88, "2. Iterative Unmasking",
            ha="center", fontsize=9.5, weight="bold")
    ax.text(3.3 + 1.65, 5.68, "(n_denoise steps)",
            ha="center", fontsize=8, color="#555")
    # mini step diagrams
    step_y = 5.0
    cells = [
        ("[M][M][M][M]", MASK),
        ("x0[M][M][M]", TOK),
        ("x0 x1[M]x3", TOK),
        ("x0 x1 x2 x3", TOK),
    ]
    for i, (lbl, fc) in enumerate(cells):
        _box(ax, (3.38 + i * 0.78, step_y), 0.72, 0.4, lbl, fc=fc, fontsize=6.5)
    ax.text(3.3 + 1.65, 4.83, "freeze top-confidence positions per step",
            ha="center", fontsize=7, color="#555")

    # ---- Stage 3: Target verify ----
    _box(ax, (7.0, 4.7), 2.6, 1.3,
         "3. Target Verify\n\nsingle forward pass\nover prefix + draft\n→ k+1 logit rows",
         fc="#e7f0ff", fontsize=9)

    # ---- Stage 4: Accept/reject (two paths) ----
    _box(ax, (10.0, 4.7), 1.8, 1.3,
         "4. Accept/Reject\n\nblock-mode\nor token-mode",
         fc="#dff0d8", fontsize=9, weight="bold")

    # ---- Bottom row: details ----
    _box(ax, (0.5, 2.2), 5.0, 1.7,
         "BLOCK MODE\n\nBernoulli on  Π_i p(x_i) / q(x_i)\nover the common vocabulary\n\n"
         "On reject → token-level fallback\n(unbiased w.r.t. target distribution)",
         fc="#eafff0", fontsize=9)

    _box(ax, (6.5, 2.2), 5.0, 1.7,
         "TOKEN MODE  (Leviathan-style)\n\nq(x_i | x_<i, ctx) via one batched\ndrifter forward of size k\n(row i reveals positions [0..i))\n\n"
         "Per-token rejection sampling",
         fc="#fff0f0", fontsize=9)

    # ---- Bottom legend bar ----
    _box(ax, (0.5, 0.5), 11.0, 1.2,
         "Latency:  T_drift = n_denoise · t_drifter   (independent of k)\n"
         "vs AR draft:  T_draft = k · t_draft_step\n"
         "→ break-even threshold for k=8 lowered when n_denoise ≪ k",
         fc="#fafafa", fontsize=10, weight="bold")

    # ---- Arrows between stages ----
    _arrow(ax, (2.9, 5.35), (3.3, 5.35))
    _arrow(ax, (6.6, 5.35), (7.0, 5.35))
    _arrow(ax, (9.6, 5.35), (10.0, 5.35))
    # loop-back from accept stage to drift schedule
    _arrow(ax, (10.9, 4.7), (10.9, 4.4), style="->", lw=0.9)
    _arrow(ax, (10.9, 4.4), (1.5, 4.4), style="->", lw=0.9)
    _arrow(ax, (1.5, 4.4), (1.5, 4.7), style="->", lw=0.9)
    ax.text(6.0, 4.32, "loop until max_new_tokens or EOS",
            ha="center", fontsize=7.5, color="#666")

    ax.set_title("DriftDiffuse: parallel speculative decoding with diffusion drafter",
                 fontsize=11, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arch_png = OUT_DIR / "driftdiffuse_architecture.png"
    pipe_png = OUT_DIR / "driftdiffuse_pipeline.png"
    render_architecture(arch_png)
    render_pipeline(pipe_png)
    print(f"Wrote {arch_png}")
    print(f"Wrote {pipe_png}")


if __name__ == "__main__":
    main()
