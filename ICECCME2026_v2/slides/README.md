# ICECCME 2026 conference talk

Slides and speaker notes for *Characterizing Speculative Decoding Dynamics for
Large Language Models on Consumer-Class GPUs* (`../conference_101719.tex`).

## Files

| File | Purpose |
|---|---|
| `slides.tex` | The whole deck. Every frame is followed by its `\note{...}` block. |
| `preamble.tex` | Theme, colours, and the `\panelA`–`\panelD` macros that clip the paper's 2×2 figure into single panels. |
| `slides-notes.tex` | Three-line wrapper that builds the notes-only PDF. |
| `Makefile` | Build targets. |

## Build

```bash
make          # slides.pdf   -- 16:9 projector deck (24 pages)
make notes    # slides-notes.pdf -- speaker notes, one page per note (21 pages)
make all      # both
```

Both PDFs come from `slides.tex`; `slides-notes.tex` only defines `\NOTESMODE`
before inputting it, so the notes can never drift from the slides.

## Presenter display

For a dual-screen setup (notes on the laptop, slides on the projector), edit the
`\else` branch of the notes switch near the top of `slides.tex`:

```latex
% \setbeameroption{hide notes}
\setbeameroption{show notes on second screen=right}
```

Then open the resulting PDF in `pympress`, `dspdfviewer`, or Impressive.

## Timing

Planned for **~10:40**, inside the 8–12 minute slot. Each `\note` block opens
with its target time window, e.g. `[6:30--7:50 | RQ4 portability]`.

Slides marked as compressible in their notes, if you are running long:

1. **RQ3 (deterministic vs. stochastic)** — state the one-line takeaway and move on (~35 s).
2. **The three gaps** on "What is missing" — go straight to the RQ block (~20 s).

## Structure

13 content frames plus 6 backup frames after `\appendix` (backup frames are
excluded from the page numbering). Backups cover the full Qwen3 and Qwen2.5
grids, acceptance portability, metric definitions, and the hardware profiles —
the last two are the ones most likely to be needed in Q&A.

## Figure panels

`figures/specdec_configs_vs_baseline_metrics.pdf` is a single 2×2 matplotlib
grid. The `\panelA`–`\panelD` macros in `preamble.tex` crop it with
`trim`/`clip` so each panel can be shown on its own. **If that figure is
regenerated with a different layout, the trim values must be re-measured.**
