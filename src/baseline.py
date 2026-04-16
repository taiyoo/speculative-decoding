"""
Baseline runner: target-only autoregressive decoding (Phase 2).

Loads Qwen2.5-7B-Instruct, runs all 1,000 samples in both decoding regimes,
records per-sample latency/throughput/output.
"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    TARGET_MODEL_ID, DATASETS, REGIMES, RESULTS_DIR, SEED,
)
from utils import set_seed, GPUTimer, write_csv


def load_target_model():
    """Load the target model and tokenizer once."""
    print(f"Loading target model: {TARGET_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_ID, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_baseline_sample(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    """
    Generate autoregressively for a single sample.
    Returns dict with timing and output info.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    n_input = input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    # --- TTFT: time to first token (single-token generate) ---
    ttft_timer = GPUTimer()
    with ttft_timer:
        first_out = model.generate(input_ids, max_new_tokens=1, **{
            k: v for k, v in gen_kwargs.items() if k != "max_new_tokens"
        })
    ttft_ms = ttft_timer.elapsed_ms

    # --- Full generation ---
    full_timer = GPUTimer()
    with full_timer:
        output_ids = model.generate(input_ids, **gen_kwargs)

    new_ids = output_ids[0, n_input:]
    n_new = len(new_ids)
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # TPOT: time per output token (excluding first token)
    tpot_ms = ((full_timer.elapsed_ms - ttft_ms) / max(n_new - 1, 1)) if n_new > 1 else 0.0
    tokens_per_sec = n_new / full_timer.elapsed_s if full_timer.elapsed_s > 0 else 0.0

    return {
        "latency_s": round(full_timer.elapsed_s, 4),
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "num_tokens": n_new,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "output_text": output_text,
    }


def run_baseline(
    data: dict[str, list[dict]],
    regime_name: str,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """
    Run baseline autoregressive decoding for all samples under a given regime.
    Returns list of per-sample result dicts.
    """
    regime = REGIMES[regime_name]
    if model is None or tokenizer is None:
        model, tokenizer = load_target_model()

    results = []
    total = sum(len(v) for v in data.values())
    done = 0

    for task_name, samples in data.items():
        max_new_tokens = DATASETS[task_name]["max_new_tokens"]
        for sample in samples:
            done += 1
            set_seed(SEED)

            out = run_baseline_sample(
                model, tokenizer,
                sample["prompt"],
                max_new_tokens,
                regime.temperature,
                regime.top_p,
            )

            row = {
                "sample_id": sample["sample_id"],
                "task": task_name,
                "regime": regime_name,
                **out,
            }
            results.append(row)

            if done % 50 == 0 or done == total:
                print(f"  [{regime_name}] {done}/{total} samples done")

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"baseline_{regime_name}.csv"
    write_csv(csv_path, results)
    print(f"  Saved → {csv_path}")
    return results
