# Speculative Decoding for Efficient LLM Inference

COMP5329 Assignment 2 project for measuring latency/quality trade-offs of speculative decoding on the Qwen 2.5 family.

## Overview

This project implements and evaluates:
- Baseline autoregressive decoding (target-only)
- Speculative decoding with two draft sizes
- Deterministic and stochastic decoding regimes
- Task-level quality and system-level latency metrics

Target model and drafts:
- Target: `Qwen/Qwen2.5-7B-Instruct`
- Draft A: `Qwen/Qwen2.5-0.5B-Instruct`
- Draft B: `Qwen/Qwen2.5-1.5B-Instruct`

Benchmarks:
- GSM8K (300 samples)
- MMLU (500 samples, 5 subjects x 100)
- CNN/DailyMail (200 samples)

Total experiment configurations:
- 2 baseline runs + 12 speculative runs = 14 configurations

## Repository Layout

- `src/config.py`: central configuration (models, datasets, regimes, success criteria)
- `src/data_loader.py`: dataset loading/sampling + manifest freezing
- `src/baseline.py`: target-only baseline runner
- `src/speculative.py`: speculative decoding runner (draft-then-verify)
- `src/evaluate.py`: quality evaluation (GSM8K EM, MMLU letter match, CNN/DM ROUGE-L)
- `src/metrics.py`: latency/acceptance/speedup/quality delta aggregation
- `src/visualize.py`: visualization placeholder
- `experiment.ipynb`: primary end-to-end workflow
- `sec/*.tex`: paper sections
- `main.tex`, `main.bib`: paper entrypoint and bibliography

## Environment

Validated environment (see `env_spec.txt`):
  GPU version (tested environment):
  - RTX5090 Laptop (Windows 11) / RTX4090 (Windows 10)
  - Python 3.12
  - torch 2.11.0+cu128
  - torchvision 0.26.0+cu128

  No GPU version:
  - Apple MPS backend (`cuda_available: False`)
  - Python 3.14
  - torch 2.11.0

  Dependencies are pinned in `requirements.txt`.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the experiment from `experiment.ipynb`.

The notebook is the recommended execution path because it orchestrates:
- Phase 1: data preparation
- Phase 2: baseline benchmarking
- Phase 3: speculative grid (0.5B draft)
- Phase 4: speculative grid (1.5B draft)
- Stability analysis and final summary tables

## Run on Teammate GPU (Linux/Windows + NVIDIA)

Use this path for local CUDA machines.

1. Clone the repository and enter the project directory.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

4. Sanity-check runtime:

```bash
python -c "from src.utils import get_env_info; print(get_env_info())"
```

Expected: `cuda_available: True` and `device: cuda`.

5. Run `experiment.ipynb` from the project root.

Notes:
- Keep notebook working directory at repository root so `src/` is discoverable.
- First run downloads models and datasets; subsequent runs reuse cache/manifests.

## Run in Google Colab

Use this path if teammates do not have a local GPU.

1. In Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU.
2. In a first code cell, run:

```python
!git clone <YOUR_REPO_URL>
%cd speculative-decoding/
!pip install -U pip
!pip install -r requirements-colab.txt
!python -c "from src.utils import get_env_info; print(get_env_info())"
```

3. Open and run `experiment.ipynb` from top to bottom.

Colab tips:
- If authentication is required for Qwen models, run `huggingface-cli login` in a cell.
- For long runs, prefer Colab Pro/A100 if available.
- Ensure the repo path remains at `.../Assignment 2` before running notebook cells.

## Optional Scripted Data Prep (Phase 1)

You can run data preparation directly:

```bash
python src/data_loader.py --verify-tokenizer --smoke-test 20
```

If running in Colab, use:

```bash
python src/data_loader.py --smoke-test 5
```

Outputs created in `manifests/`:
- `gsm8k.json`, `mmlu.json`, `cnndm.json`
- `gsm8k_data.json`, `mmlu_data.json`, `cnndm_data.json`

## Results and Outputs

Generated artifacts are stored under `results/`:
- `baseline_deterministic.csv`
- `baseline_stochastic.csv`
- `spec_0.5B_k{4,8,16}_{det,stoch}.csv`
- `spec_1.5B_k{4,8,16}_{det,stoch}.csv`
- `stability/` per-seed reruns for top configs
- `all_configs_combined.csv` (from notebook summary cell)

## Metrics Tracked

Implemented in `src/metrics.py` and aligned with the paper:
- Latency: `T`, `TTFT`, `TPOT`
- Throughput: `R_tok`
- Acceptance: `alpha`, `B_eff`
- Speedup: `S`
- Quality: GSM8K exact match, MMLU letter match, CNN/DM ROUGE-L
- Deltas and stability: `dQ`, `sigma_S`

## Paper Build

Build the LaTeX paper from project root:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Main file includes:
- `sec/00_abstract.tex`
- `sec/01_introduction.tex`
- `sec/02_experiment.tex`
- `sec/03_metrics.tex`
- `sec/04_execution.tex`
- `sec/05_discussion.tex`

## Notes

- `ignore/` contains exploratory material and is excluded from version control.
- Current scope is Qwen 2.5 autoregressive speculative decoding (no production DFlash integration in this branch).
- `requirements-colab.txt` is the recommended dependency set for Colab/GPU portability.
