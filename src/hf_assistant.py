"""
HF assistant_model runner: speculative decoding using transformers' built-in
``model.generate(assistant_model=...)`` path.

This is the same Leviathan-style algorithm as our hand-rolled
``src/speculative.py``, but executed through HuggingFace's optimised C++/CUDA
implementation. It serves as a reference point that isolates the wall-clock
cost of our Python-level verify loop from the underlying algorithm.

Outputs CSVs with the same schema as ``baseline.py`` so they slot directly into
the same paired-speedup analysis. The α / B_eff columns are not produced by
``generate()`` and are left blank in the CSV.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DATASETS, REGIMES, RESULTS_DIR, SEED,
    DRAFT_MODELS, TARGET_MODEL_ID,
    TARGET_QUANT, DRAFT_QUANT,
)
from quantization import get_quant_kwargs
from hf_utils import apply_hf_mode_env, hf_model_kwargs
from utils import set_seed, GPUTimer, write_csv


def _load_model(model_id: str, quant_mode: str | None = None):
    apply_hf_mode_env()
    quant_kwargs, _ = get_quant_kwargs(quant_mode)
    hf_kwargs = hf_model_kwargs()
    tok = AutoTokenizer.from_pretrained(model_id, **hf_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, **quant_kwargs, **hf_kwargs
    )
    mdl.eval()
    return mdl, tok


def run_hf_assistant_sample(
    target_model,
    target_tokenizer,
    assistant_model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_assistant_tokens: int,
) -> dict:
    """Run a single sample through HF's built-in assistant_model path."""
    if max_new_tokens <= 0:
        return {
            "latency_s": 0.0, "ttft_ms": 0.0, "tpot_ms": 0.0,
            "num_tokens": 0, "tokens_per_sec": 0.0, "output_text": "",
        }

    inputs = target_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    input_ids = inputs["input_ids"].to(target_model.device)
    n_input = input_ids.shape[1]

    do_sample = temperature != 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        assistant_model=assistant_model,
        num_assistant_tokens=num_assistant_tokens,
        num_assistant_tokens_schedule="constant",
        pad_token_id=target_tokenizer.pad_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    timer = GPUTimer()
    with timer, torch.inference_mode():
        out_ids = target_model.generate(input_ids, **gen_kwargs)

    new_ids = out_ids[0, n_input:]
    n_new = int(new_ids.numel())
    output_text = target_tokenizer.decode(new_ids, skip_special_tokens=True)

    tokens_per_sec = n_new / timer.elapsed_s if timer.elapsed_s > 0 else 0.0
    # generate() does not expose per-token timing; we leave ttft/tpot blank.
    return {
        "latency_s": round(timer.elapsed_s, 4),
        "ttft_ms": 0.0,
        "tpot_ms": 0.0,
        "num_tokens": n_new,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "output_text": output_text,
    }


def run_hf_assistant(
    data: dict[str, list[dict]],
    regime_name: str,
    draft_label: str,
    num_assistant_tokens: int,
    target_model=None,
    target_tokenizer=None,
    assistant_model=None,
) -> list[dict]:
    """Run HF assistant_model speculative decoding over all samples."""
    regime = REGIMES[regime_name]

    if target_model is None or target_tokenizer is None:
        target_model, target_tokenizer = _load_model(
            TARGET_MODEL_ID, TARGET_QUANT
        )
    if assistant_model is None:
        assistant_model, _ = _load_model(
            DRAFT_MODELS[draft_label], DRAFT_QUANT
        )

    # transformers requires the assistant to live on the same device as the target.
    if assistant_model.device != target_model.device:
        try:
            assistant_model.to(target_model.device)
        except (NotImplementedError, ValueError):
            # bitsandbytes-quantised models cannot be moved post-hoc; assume the
            # caller already loaded both with the same device_map.
            pass

    results = []
    total = sum(len(v) for v in data.values())
    done = 0
    short = "det" if regime_name == "deterministic" else "stoch"

    for task_name, samples in data.items():
        max_new_tokens = DATASETS[task_name]["max_new_tokens"]
        for sample in samples:
            done += 1
            set_seed(SEED)

            out = run_hf_assistant_sample(
                target_model, target_tokenizer, assistant_model,
                sample["prompt"], max_new_tokens,
                regime.temperature, regime.top_p,
                num_assistant_tokens,
            )

            results.append({
                "sample_id": sample["sample_id"],
                "task": task_name,
                "regime": regime_name,
                "draft": draft_label,
                "k": num_assistant_tokens,
                **out,
            })

            if done % 50 == 0 or done == total:
                print(f"  [hf_assist {draft_label}/k={num_assistant_tokens}/{short}] {done}/{total}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = (
        RESULTS_DIR
        / f"hfassist_{draft_label}_k{num_assistant_tokens}_{short}.csv"
    )
    write_csv(csv_path, results)
    print(f"  Saved → {csv_path}")
    return results
