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
TARGET_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
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

# ── Stability Seeds ───────────────────────────────────────────────────────────
STABILITY_SEEDS = [42, 123, 999]

# ── Quantization ──────────────────────────────────────────────────────────────
# Options: "fp16", "int8" (bitsandbytes), "fp8" (quanto float8_e4m3)
QUANT_MODE = "fp8"

# ── Success Criteria ──────────────────────────────────────────────────────────
MIN_SPEEDUP = 1.3          # S >= 1.3x
MAX_QUALITY_DROP = 1.0     # |ΔQ| <= 1.0 point
