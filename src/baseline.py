"""
Baseline runner: target-only autoregressive decoding (Phase 2).

Loads Qwen2.5-7B-Instruct, runs all 1,000 samples in both decoding regimes,
records per-sample latency/throughput/output.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    TARGET_MODEL_ID, DATASETS, REGIMES, RESULTS_DIR, SEED,
    QUANT_MODE, TARGET_QUANT,
)
from quantization import get_quant_kwargs
from sampling import sample_next_token
from hf_utils import apply_hf_mode_env, hf_model_kwargs
from utils import set_seed, GPUTimer, write_csv


def _get_quant_kwargs():
    """Return model loading kwargs based on TARGET_QUANT (fallback QUANT_MODE)."""
    kwargs, _ = get_quant_kwargs(TARGET_QUANT)
    return kwargs


def load_target_model():
    """Load the target model and tokenizer once."""
    offline = apply_hf_mode_env()
    requested = TARGET_QUANT if TARGET_QUANT is not None else QUANT_MODE
    quant_kwargs, resolved_quant_mode = get_quant_kwargs(TARGET_QUANT)
    print(
        f"Loading target model: {TARGET_MODEL_ID} "
        f"(quant={requested} -> {resolved_quant_mode}, offline_first={offline})"
    )
    hf_kwargs = hf_model_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, **hf_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        **quant_kwargs,
        **hf_kwargs,
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
    if max_new_tokens <= 0:
        return {
            "latency_s": 0.0,
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "num_tokens": 0,
            "tokens_per_sec": 0.0,
            "output_text": "",
        }

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    n_input = input_ids.shape[1]

    gen_kwargs = {"do_sample": temperature != 0.0}
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    generated_tokens: list[torch.Tensor] = []
    eos_token_id = tokenizer.eos_token_id

    ttft_timer = GPUTimer()
    full_timer = GPUTimer()

    with full_timer:
        with torch.inference_mode():
            # Prefix forward + first token (TTFT).
            with ttft_timer:
                outputs = model(input_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                next_token, _ = sample_next_token(logits, gen_kwargs)

            if eos_token_id is None or int(next_token.item()) != int(eos_token_id):
                generated_tokens.append(next_token)
                prev_token = next_token

                for _ in range(max_new_tokens - 1):
                    outputs = model(
                        prev_token,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    next_token, _ = sample_next_token(logits, gen_kwargs)

                    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                        break

                    generated_tokens.append(next_token)
                    prev_token = next_token

    if generated_tokens:
        new_ids = torch.cat(generated_tokens, dim=-1).squeeze(0)
    else:
        new_ids = torch.empty((0,), dtype=input_ids.dtype, device=input_ids.device)

    n_new = int(new_ids.numel())
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # TPOT: time per output token (excluding first token)
    ttft_ms = ttft_timer.elapsed_ms
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
