# Exact Artifact Citation Map (Tables/Figures)

## Table sources in paper text

- sec/06_results.tex Table `tab:headline-4090` (Qwen3 RTX4090 fixed-policy matrix)
  - qwen3_rtx4090/all_configs_summary.csv

- sec/06_results.tex Table `tab:headline-5090` (Qwen3 RTX5090-laptop matrix)
  - qwen3_rtx5090_laptop/baseline_deterministic.csv
  - qwen3_rtx5090_laptop/baseline_stochastic.csv
  - qwen3_rtx5090_laptop/spec_0.6B_k4_det.csv
  - qwen3_rtx5090_laptop/spec_0.6B_k8_det.csv
  - qwen3_rtx5090_laptop/spec_0.6B_k16_det.csv
  - qwen3_rtx5090_laptop/spec_0.6B_k4_stoch.csv
  - qwen3_rtx5090_laptop/spec_0.6B_k8_stoch.csv
  - qwen3_rtx5090_laptop/spec_0.6B_k16_stoch.csv

- sec/06_results.tex Cross-family paragraph (Qwen2.5 values cited in text)
  - qwen25_rtx4090/all_configs_summary.csv

- sec/07_acsd.tex ACSD quantitative evidence (Qwen2.5 RTX4090)
  - qwen25_rtx4090_acsd/acsd_0.5B_to_1.5B_det.csv
  - qwen25_rtx4090_acsd/acsd_0.5B_to_1.5B_stoch.csv
  - qwen25_rtx4090_acsd/acsd_summary.csv

## Figure sources

- sec/04_execution.tex Figure `fig:specdec-impl-flow` is a TikZ diagram embedded in LaTeX; no external result artifact file is required.

## Notes

- This package intentionally excludes non-cited artifacts.
- Original source files remain unchanged in project directories.
