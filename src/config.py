"""
Speculative Decoding Experiment — Central Configuration

All model IDs, dataset parameters, prompt templates, decoding hyperparameters,
and output paths are defined here. No magic constants elsewhere.
"""

from pathlib import Path
from dataclasses import dataclass, field

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
RESULTS_DIR = PROJECT_ROOT / "results"
STABILITY_DIR = RESULTS_DIR / "stability"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Models ─────────────────────────────────────────────────────────────────────
TARGET_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DRAFT_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}

# ── Datasets ───────────────────────────────────────────────────────────────────
SEED = 42

DATASETS = {
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "hf_config": "main",
        "split": "test",
        "n_samples": 300,
        "max_new_tokens": 256,
    },
    "mmlu": {
        "hf_name": "cais/mmlu",
        "hf_config": "all",
        "split": "test",
        "n_samples": 500,         # 5 subjects × 100
        "subjects": [
            "abstract_algebra",
            "college_chemistry",
            "computer_security",
            "econometrics",
            "us_foreign_policy",
        ],
        "samples_per_subject": 100,
        "max_new_tokens": 32,
    },
    "cnndm": {
        "hf_name": "abisee/cnn_dailymail",
        "hf_config": "3.0.0",
        "split": "test",
        "n_samples": 200,
        "max_new_tokens": 160,
    },
}

# ── Prompt Templates ───────────────────────────────────────────────────────────
PROMPTS = {
    "gsm8k": (
        "You are a careful math assistant. Solve the problem step by step "
        "and end with Final Answer: <number>.\n"
        "Question: {question}"
    ),
    "mmlu": (
        "You are a knowledgeable assistant. Choose exactly one option.\n"
        "Question: {question}\n"
        "Options: A. {A}  B. {B}  C. {C}  D. {D}\n"
        "Answer:"
    ),
    "cnndm": (
        "Summarize the following article in 3-4 sentences, "
        "preserving key facts and entities.\n"
        "Article: {article}\n"
        "Summary:"
    ),
}

# ── Decoding Regimes ───────────────────────────────────────────────────────────
@dataclass
class DecodingRegime:
    name: str
    temperature: float
    top_p: float

REGIMES = {
    "deterministic": DecodingRegime(name="deterministic", temperature=0.0, top_p=1.0),
    "stochastic": DecodingRegime(name="stochastic", temperature=0.7, top_p=0.9),
}

# ── Speculative Decoding ──────────────────────────────────────────────────────
DRAFT_LENGTHS = [4, 8, 16]
# DRAFT_LENGTHS = [8, 16]

# ── Stability Seeds ───────────────────────────────────────────────────────────
STABILITY_SEEDS = [42, 123, 999]

# ── Quantization ──────────────────────────────────────────────────────────────
# Options: "fp16", "int8" (bitsandbytes), "fp8" (quanto float8_e4m3)
#
# QUANT_MODE is the legacy single-knob default (kept for backward compat: any
# code path that does not explicitly request a target/draft mode falls back to
# this value). Prefer TARGET_QUANT / DRAFT_QUANT below.
QUANT_MODE = "int8"  # was "fp8" — switched because optimum-quanto+torchao crashes on Colab ("Cannot copy out of meta tensor")

# Per-model overrides. Set either to None to inherit QUANT_MODE.
#
# Recommended on a 24 GB consumer GPU (e.g. RTX 5090 laptop):
#     TARGET_QUANT = "int8"   # 7B model, dominates wall-clock cost
#     DRAFT_QUANT  = "fp16"   # small model on the critical path; avoid bnb dequant tax
TARGET_QUANT: str | None = "fp16"   # 3B target 
DRAFT_QUANT: str | None = "fp16"    # 0.5B draft — full precision, no dequant tax

# ── GPU Decode Optimization ───────────────────────────────────────────────────
# Applied in src/speculative.py. All switches are safe defaults for eager mode.
GPU_USE_SEPARATE_STREAMS = True
GPU_PREALLOCATE_STEP_BUFFERS = True
GPU_USE_STABLE_STEP_SHAPES = True
GPU_TRY_CUDA_GRAPHS = False  # Experimental: auto-falls back to eager when unsupported.

# ── DriftDiffuse (Phase 7/8) ──────────────────────────────────────────────────
DRIFTER_CHECKPOINT_DIR = RESULTS_DIR / "drifter_ckpt"
DRIFTER_CONFIG = {
    "hidden": 512,
    "n_layers": 6,
    "n_heads": 8,
    "ffn_mult": 4,
    "max_ctx_len": 768,
    "k_max": 16,
    "n_steps": 8,
    "dropout": 0.0,
    "tie_embeddings": True,
}
DRIFTER_TRAIN = {
    "regimes": ("deterministic",),
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "n_epochs": 1,
    "max_steps": 2000,           # smoke run; raise for paper run
    "log_every": 50,
    "val_every": 500,
    "val_split": 0.05,
    "drift_lambda": 2.0,
}
DRIFTER_EVAL = {
    "k_values": (8, 16),
    "n_denoise_steps": (3,),
    "accept_modes": ("block", "token"),
    "regimes": ("deterministic", "stochastic"),
}

ACSD_EVAL = {
    "primary_draft": "0.5B",
    "rescue_draft": "1.5B",
    "base_k": 8,
    "k_choices": (4, 8, 16),
    "regimes": ("deterministic", "stochastic"),
    "accept_window": 6,
    "k_low_threshold": 0.20,
    "k_high_threshold": 0.40,
    "rescue_trigger_alpha": 0.18,
    "rescue_trigger_consecutive": 5,
    "rescue_hold_steps": 4,
    "rescue_cooldown_steps": 10,
    "ar_fallback_alpha": 0.08,
    "ar_fallback_min_tokens": 96,
    "ar_fallback_consecutive": 3,
    "checkpoint_every": 50,
}

# ── Success Criteria ──────────────────────────────────────────────────────────
MIN_SPEEDUP = 1.3          # S >= 1.3x
MAX_QUALITY_DROP = 1.0     # |ΔQ| <= 1.0 point
