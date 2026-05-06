"""
Adaptive Cascaded Speculative Decoding (ACSD).

ACSD keeps vanilla speculative verification but adapts both draft length (k)
and draft model online to avoid long-tail latency blowups:
- Primary draft: fast model (0.5B)
- Rescue draft: stronger model (1.5B) when acceptance collapses
- Optional fallback to target-only AR for pathological samples
"""

from __future__ import annotations

import json
from collections import deque

import torch

from config import DATASETS, REGIMES, RESULTS_DIR, SEED
from sampling import sample_next_token_and_prob
from speculative import (
    VERIFY_LOG_DIR,
    _crop_cache,
    _draft_generate,
    _verify_block,
    load_draft_model,
)
from utils import GPUTimer, set_seed, write_csv


def _rolling_alpha(window: deque[tuple[int, int]]) -> float:
    if not window:
        return 1.0
    acc = sum(a for a, _ in window)
    prop = sum(p for _, p in window)
    return (acc / prop) if prop > 0 else 0.0


def _select_k(
    rolling_alpha: float,
    k_choices: tuple[int, ...],
    low_thresh: float,
    high_thresh: float,
    base_k: int,
) -> int:
    ordered = sorted(set(int(x) for x in k_choices))
    if len(ordered) == 1:
        return ordered[0]

    low_k = ordered[0]
    high_k = ordered[-1]

    # Prefer a middle value nearest base_k if available.
    mid_candidates = [k for k in ordered if k not in {low_k, high_k}]
    if mid_candidates:
        mid_k = min(mid_candidates, key=lambda x: abs(x - base_k))
    else:
        mid_k = ordered[len(ordered) // 2]

    if rolling_alpha < low_thresh:
        return low_k
    if rolling_alpha < high_thresh:
        return mid_k
    return high_k


def _continue_with_target_ar(
    target_model,
    target_pending_input: torch.Tensor,
    target_past,
    gen_kwargs: dict,
    accepted_tokens: list[int],
    max_new_tokens: int,
    eos_id: int | None,
) -> tuple[list[int], object, torch.Tensor, bool]:
    stopped_eos = False
    one_token_buf = torch.empty((1, 1), device=target_pending_input.device, dtype=target_pending_input.dtype)

    while len(accepted_tokens) < max_new_tokens:
        with torch.inference_mode():
            out = target_model(
                target_pending_input,
                past_key_values=target_past,
                use_cache=True,
            )
        target_past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_token, _, _ = sample_next_token_and_prob(logits, gen_kwargs, return_probs=False)
        emitted = int(next_token.item())
        accepted_tokens.append(emitted)

        if eos_id is not None and emitted == int(eos_id):
            accepted_tokens.pop()
            stopped_eos = True
            break

        one_token_buf.fill_(emitted)
        target_pending_input = one_token_buf

    return accepted_tokens, target_past, target_pending_input, stopped_eos


def acsd_decode_sample(
    target_model,
    draft_models: dict[str, object],
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    primary_label: str = "0.5B",
    rescue_label: str = "1.5B",
    base_k: int = 8,
    k_choices: tuple[int, ...] = (4, 8, 16),
    accept_window: int = 4,
    k_low_threshold: float = 0.20,
    k_high_threshold: float = 0.40,
    rescue_trigger_alpha: float = 0.18,
    rescue_trigger_consecutive: int = 3,
    rescue_hold_steps: int = 6,
    rescue_cooldown_steps: int = 8,
    ar_fallback_alpha: float = 0.12,
    ar_fallback_min_tokens: int = 48,
    ar_fallback_consecutive: int = 3,
) -> dict:
    """Run ACSD for one sample."""
    if primary_label not in draft_models or rescue_label not in draft_models:
        raise ValueError("Both primary and rescue draft models must be loaded")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(target_model.device)
    n_input = input_ids.shape[1]

    if temperature == 0.0:
        gen_kwargs = {"do_sample": False}
    else:
        gen_kwargs = {"do_sample": True, "temperature": temperature, "top_p": top_p}

    keep_step_logits = bool(gen_kwargs.get("do_sample", False))
    target_device = input_ids.device
    eos_id = tokenizer.eos_token_id

    target_past = None
    target_pending_input = input_ids

    # Keep cache for active draft only; on switch we rebuild from prefix.
    draft_cache = {primary_label: None, rescue_label: None}
    active_label = primary_label
    last_active_label = active_label

    accepted_tokens: list[int] = []
    total_proposed = 0
    total_accepted = 0
    n_verify_steps = 0
    verify_log: list[dict] = []

    recent = deque(maxlen=max(1, int(accept_window)))
    low_alpha_consecutive = 0
    rescue_remaining = 0
    rescue_cooldown_remaining = 0
    fallback_low_consecutive = 0
    used_ar_fallback = False
    used_rescue_steps = 0
    k_usage = {int(k): 0 for k in k_choices}
    stopped_eos = False

    ttft_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and target_device.type == "cuda" else None
    ttft_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and target_device.type == "cuda" else None

    full_timer = GPUTimer()
    with full_timer:
        while len(accepted_tokens) < max_new_tokens:
            remaining = max_new_tokens - len(accepted_tokens)
            if remaining <= 0:
                break

            if recent:
                r_alpha = _rolling_alpha(recent)
                k_now = min(_select_k(r_alpha, k_choices, k_low_threshold, k_high_threshold, base_k), remaining)
            else:
                # Avoid a cold-start jump to high-k before we observe acceptance.
                k_now = min(int(base_k), remaining)
            k_usage[int(k_now)] = k_usage.get(int(k_now), 0) + 1

            # Rescue state machine.
            if rescue_remaining > 0:
                active_label = rescue_label
                rescue_remaining -= 1
                if rescue_remaining == 0:
                    rescue_cooldown_remaining = max(0, int(rescue_cooldown_steps))
            else:
                active_label = primary_label
                if rescue_cooldown_remaining > 0:
                    rescue_cooldown_remaining -= 1

            if active_label != last_active_label:
                # Invalidate cache for newly active draft; rebuild from prefix.
                draft_cache[active_label] = None
                last_active_label = active_label

            draft_model = draft_models[active_label]
            draft_device = next(draft_model.parameters()).device

            if draft_cache[active_label] is None:
                prefix = torch.cat(
                    [input_ids[0], torch.tensor(accepted_tokens, device=target_device, dtype=input_ids.dtype)],
                    dim=0,
                ).unsqueeze(0)
                draft_pending_input = prefix.to(draft_device)
            else:
                if draft_device == target_device:
                    draft_pending_input = target_pending_input
                else:
                    draft_pending_input = target_pending_input.to(draft_device, non_blocking=True)

            draft_tokens, q_token_probs, q_step_logits, draft_past = _draft_generate(
                draft_model,
                draft_pending_input,
                draft_cache[active_label],
                k_now,
                gen_kwargs,
                keep_step_logits,
            )

            draft_tokens_t = draft_tokens if draft_tokens.device == target_device else draft_tokens.to(target_device, non_blocking=True)
            target_input = torch.cat([target_pending_input, draft_tokens_t], dim=-1)

            if n_verify_steps == 0 and ttft_start is not None:
                ttft_start.record()

            with torch.inference_mode():
                target_out = target_model(
                    target_input,
                    past_key_values=target_past,
                    use_cache=True,
                )
            target_past = target_out.past_key_values
            verify_logits = target_out.logits[0, -(k_now + 1):, :]

            if n_verify_steps == 0 and ttft_end is not None:
                ttft_end.record()

            emitted, n_acc = _verify_block(
                verify_logits,
                draft_tokens.squeeze(0).tolist(),
                q_token_probs,
                q_step_logits,
                gen_kwargs,
            )

            total_proposed += k_now
            total_accepted += n_acc
            n_verify_steps += 1
            verify_log.append(
                {
                    "step": n_verify_steps,
                    "proposed": k_now,
                    "accepted": n_acc,
                    "draft": active_label,
                    "abs_position": n_input + len(accepted_tokens),
                }
            )

            step_alpha = (n_acc / k_now) if k_now > 0 else 0.0
            recent.append((n_acc, k_now))

            if active_label == rescue_label:
                used_rescue_steps += 1
            else:
                # Trigger rescue only while on primary draft, and require cooldown
                # between rescue episodes to prevent oscillation.
                if step_alpha < rescue_trigger_alpha:
                    low_alpha_consecutive += 1
                else:
                    low_alpha_consecutive = 0

                if (
                    low_alpha_consecutive >= rescue_trigger_consecutive
                    and rescue_remaining == 0
                    and rescue_cooldown_remaining == 0
                ):
                    rescue_remaining = max(1, int(rescue_hold_steps))
                    low_alpha_consecutive = 0

            keep_len = n_input + len(accepted_tokens) + n_acc
            target_past = _crop_cache(target_past, keep_len)
            draft_cache[active_label] = _crop_cache(draft_past, keep_len)

            accepted_draft = draft_tokens.squeeze(0)[:n_acc].tolist()
            accepted_tokens.extend(int(t) for t in accepted_draft)
            accepted_tokens.append(int(emitted))

            if len(accepted_tokens) > max_new_tokens:
                accepted_tokens = accepted_tokens[:max_new_tokens]
                break

            one_tok = torch.empty((1, 1), device=target_device, dtype=input_ids.dtype)
            one_tok.fill_(int(emitted))
            target_pending_input = one_tok

            if eos_id is not None and (int(emitted) == int(eos_id) or int(eos_id) in accepted_draft):
                if int(eos_id) in accepted_tokens:
                    eos_pos = accepted_tokens.index(int(eos_id))
                    accepted_tokens = accepted_tokens[:eos_pos]
                stopped_eos = True
                break

            # Guardrail: if acceptance collapses after enough generated tokens,
            # finish with plain AR to avoid runaway latency tails.
            if len(accepted_tokens) >= int(ar_fallback_min_tokens):
                if _rolling_alpha(recent) < ar_fallback_alpha:
                    fallback_low_consecutive += 1
                else:
                    fallback_low_consecutive = 0

                if fallback_low_consecutive >= max(1, int(ar_fallback_consecutive)):
                    used_ar_fallback = True
                    break
            else:
                fallback_low_consecutive = 0

        if used_ar_fallback and len(accepted_tokens) < max_new_tokens:
            accepted_tokens, target_past, target_pending_input, stopped_eos = _continue_with_target_ar(
                target_model,
                target_pending_input,
                target_past,
                gen_kwargs,
                accepted_tokens,
                max_new_tokens,
                eos_id,
            )

    n_new = len(accepted_tokens)
    output_text = tokenizer.decode(accepted_tokens, skip_special_tokens=True)
    ttft_ms = 0.0
    if ttft_start is not None and ttft_end is not None:
        torch.cuda.synchronize(target_device)
        ttft_ms = float(ttft_start.elapsed_time(ttft_end))

    tpot_ms = ((full_timer.elapsed_ms - ttft_ms) / max(n_new - 1, 1)) if n_new > 1 else 0.0
    tokens_per_sec = n_new / full_timer.elapsed_s if full_timer.elapsed_s > 0 else 0.0
    alpha = total_accepted / total_proposed if total_proposed > 0 else 0.0
    b_eff = total_accepted / n_verify_steps if n_verify_steps > 0 else 0.0

    return {
        "latency_s": round(full_timer.elapsed_s, 4),
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "num_tokens": n_new,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "n_verify_steps": n_verify_steps,
        "alpha": round(alpha, 4),
        "B_eff": round(b_eff, 2),
        "stopped_eos": stopped_eos,
        "output_text": output_text,
        "verify_log": verify_log,
        "acsd_primary_draft": primary_label,
        "acsd_rescue_draft": rescue_label,
        "acsd_rescue_steps": used_rescue_steps,
        "acsd_used_ar_fallback": used_ar_fallback,
        "acsd_mean_recent_alpha": round(_rolling_alpha(recent), 4),
        "acsd_k_usage": json.dumps(k_usage, sort_keys=True),
    }


def run_acsd_grid(
    data: dict[str, list[dict]],
    regime_name: str,
    target_model,
    target_tokenizer,
    draft_models: dict[str, object] | None = None,
    primary_label: str = "0.5B",
    rescue_label: str = "1.5B",
    base_k: int = 8,
    k_choices: tuple[int, ...] = (4, 8, 16),
    seed: int = SEED,
    label: str = "acsd",
    checkpoint_every: int | None = 50,
    progress_callback=None,
    **acsd_kwargs,
) -> list[dict]:
    """Run ACSD for one regime over all configured datasets."""
    regime = REGIMES[regime_name]
    regime_short = "det" if regime_name == "deterministic" else "stoch"

    if draft_models is None:
        draft_models = {}
    if primary_label not in draft_models:
        draft_models[primary_label], _ = load_draft_model(primary_label)
    if rescue_label not in draft_models:
        draft_models[rescue_label], _ = load_draft_model(rescue_label)

    total_samples = sum(len(v) for v in data.values())
    done = 0
    results = []
    verify_logs: dict[str, list[dict]] = {}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{label}_{primary_label}_to_{rescue_label}_{regime_short}.csv"

    log_path = VERIFY_LOG_DIR / f"{label}_{primary_label}_to_{rescue_label}_{regime_short}_seed{seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush_checkpoint(partial: bool):
        write_csv(csv_path, results)
        with open(log_path, "w") as f:
            json.dump(verify_logs, f)
        if partial:
            print(f"Checkpoint -> {csv_path} ({len(results)}/{total_samples})")

    try:
        for task_name, samples in data.items():
            max_new = DATASETS[task_name]["max_new_tokens"]
            for sample in samples:
                set_seed(seed)
                out = acsd_decode_sample(
                    target_model=target_model,
                    draft_models=draft_models,
                    tokenizer=target_tokenizer,
                    prompt=sample["prompt"],
                    max_new_tokens=max_new,
                    temperature=regime.temperature,
                    top_p=regime.top_p,
                    primary_label=primary_label,
                    rescue_label=rescue_label,
                    base_k=base_k,
                    k_choices=k_choices,
                    **acsd_kwargs,
                )

                row = {
                    "sample_id": sample["sample_id"],
                    "task": task_name,
                    "draft": label,
                    "k": base_k,
                    "regime": regime_name,
                    "seed": seed,
                    "primary_draft": primary_label,
                    "rescue_draft": rescue_label,
                    **{k: v for k, v in out.items() if k != "verify_log"},
                }
                results.append(row)
                verify_logs[sample["sample_id"]] = out.get("verify_log", [])

                done += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "done": done,
                            "total": total_samples,
                            "task": task_name,
                            "sample_id": sample.get("sample_id"),
                        }
                    )
                elif done % 50 == 0 or done == total_samples:
                    print(f"  [acsd {regime_short}] {done}/{total_samples}")

                if checkpoint_every and done % int(checkpoint_every) == 0:
                    _flush_checkpoint(partial=True)
    finally:
        _flush_checkpoint(partial=(done < total_samples))

    if done < total_samples:
        print(f"Saved partial -> {csv_path} ({done}/{total_samples})")
    else:
        print(f"Saved -> {csv_path}")
    return results
