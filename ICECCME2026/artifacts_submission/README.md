# ICECCME 2026 Submission Artifacts

This package organizes experiment artifacts by model family and hardware.

## Folder Map

- `qwen25_rtx4090`: Qwen2.5 runs on RTX4090 (`results/RTX4090_Cuda_Graph` source)
- `qwen25_rtx5090`: Qwen2.5 runs on RTX5090 (`results/RTX5090` source)
- `qwen25_rtx5090_laptop`: Qwen2.5 runs on RTX5090-laptop (`results/RTX5090_Laptop_Cuda_Graph` source)
- `qwen3_rtx4090`: Qwen3 review runs on RTX4090 (`Review_RTX4090/results` source)
- `qwen3_rtx5090_laptop`: Qwen3 review runs on RTX5090-laptop (`Review_RTX5090_Laptop/results` source)

Each group contains:

- `results_csv/`: canonical CSV artifacts (timestamped and debug CSVs removed)
- Optional `verify_logs/` and `visual_eval/` when present in source

## Reproducibility Metadata

- `MANIFEST.csv`: one row per file with group, relative path, size, and SHA256
- `SUMMARY_COUNTS.csv`: per-group counts of CSV and non-CSV files

## Curation Rules

- Included only top-level CSV files from each source result folder.
- Excluded timestamped duplicates (`*_YYYYMM...csv`) to avoid run-version ambiguity.
- Excluded debug CSVs (`*debug*`).
- Raw source folders remain unchanged outside this submission package.
