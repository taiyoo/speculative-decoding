"""
Speculative decoding runner (Phases 3-4).

Implements the draft-then-verify loop with **KV-cache reuse for both target
and draft models**, matching Leviathan et al. (2023) Algorithm 1:
  1. Draft model generates k tokens autoregressively (with cached prefix).
  2. Target model verifies all k tokens in a single forward pass that only
     consumes the new tokens (cache covers everything before).
  3. Accept the longest matching prefix; resample from the divergence point.
  4. Crop both target and draft KV caches back to the accepted prefix length
     so the next iteration starts from the correct state.

Supports both deterministic and stochastic regimes.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    TARGET_MODEL_ID, DRAFT_MODELS, DATASETS, REGIMES,
    DRAFT_LENGTHS, RESULTS_DIR, STABILITY_DIR, SEED, STABILITY_SEEDS,
    QUANT_MODE,
)
from quantization import get_quant_kwargs
from sampling import probs_from_logits, sample_next_token
from utils import set_seed, GPUTimer, write_csv


VERIFY_LOG_DIR = RESULTS_DIR / "verify_logs"


def _get_quant_kwargs():
    kwargs, _ = get_quant_kwargs()
    return kwargs


def load_draft_model(draft_label: str):
    model_id = DRAFT_MODELS[draft_label]
    quant_kwargs, resolved_quant_mode = get_quant_kwargs()
    print(
        f"Loading draft model: {model_id} "
        f"(quant={QUANT_MODE} -> {resolved_quant_mode})"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **quant_kwargs,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _crop_cache(past_key_values, length: int):
    """Crop a KV cache to `length` tokens. Supports DynamicCache and tuple-of-tuples."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(length)
        return past_key_values
    new_layers = []
    for layer in past_key_values:
        new_layers.append(tuple(t[..., :length, :] for t in layer))
    return tuple(new_layers)


def _draft_generate(
    draft_model,
    new_input_ids: torch.Tensor,
    past_key_values,
    k: int,
    gen_kwargs: dict,
    keep_step_logits: bool,
):
    """
    Generate k draft tokens with KV-cache reuse.

    Args:
        new_input_ids: tokens not yet in `past_key_values` (full prompt on the
            very first call; a single newly-emitted token on later calls).
        past_key_values: existing draft cache (or None on the first call).
        keep_step_logits: only required for stochastic acceptance correction.

    Returns:
        proposed: (1, k) tensor of draft tokens (on draft device)
        q_token_probs: list[float] of length k (q_i(y_i) under draft policy)
        q_step_logits: (k, vocab) tensor on the draft device, or None
        past_key_values: updated cache covering prefix + k draft tokens
    """
    q_token_probs: list[float] = []
    q_step_logits: list[torch.Tensor] = []
    proposed_tokens: list[torch.Tensor] = []

    with torch.inference_mode():
        outputs = draft_model(
            new_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        for step in range(k):
            if keep_step_logits:
                q_step_logits.append(logits.squeeze(0).detach())

            next_token, probs = sample_next_token(logits, gen_kwargs)
            q_prob = probs.gather(1, next_token).squeeze(1)
            q_token_probs.append(float(q_prob.item()))
            proposed_tokens.append(next_token)

            if step < k - 1:
                outputs = draft_model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

    proposed = torch.cat(proposed_tokens, dim=-1)
    q_logits_tensor = torch.stack(q_step_logits, dim=0) if keep_step_logits else None
    return proposed, q_token_probs, q_logits_tensor, past_key_values


def _verify_block(
    verify_logits: torch.Tensor,
    draft_tokens: list[int],
    q_token_probs: list[float],
    q_step_logits,
    gen_kwargs: dict,
) -> tuple[int, int]:
    """
    Verify k draft tokens against k+1 target distributions.

    Args:
        verify_logits: (k+1, vocab) target logits where row i is the target
            distribution for predicting the i-th token (i in 0..k).
        draft_tokens: list[int] of length k.
    Returns:
        emitted_token: int — the new token to append (corrected or bonus).
        n_accepted:    int — number of draft tokens accepted (0..k).
    """
    k = len(draft_tokens)
    sample_mode = gen_kwargs.get("do_sample", False)

    for i in range(k):
        logits_i = verify_logits[i].unsqueeze(0)
        d_tok = int(draft_tokens[i])

        if sample_mode:
            p_probs = probs_from_logits(logits_i, gen_kwargs)
            p_tok = float(p_probs[0, d_tok].item())
            q_tok = max(float(q_token_probs[i]), 1e-12)
            accept_prob = min(1.0, p_tok / q_tok)

            if float(torch.rand(1).item()) < accept_prob:
                continue
            # Reject: sample from residual max(p - q, 0).
            q_logits_i = q_step_logits[i].unsqueeze(0).to(p_probs.device)
            q_probs = probs_from_logits(q_logits_i, gen_kwargs)
            residual = torch.clamp(p_probs - q_probs, min=0.0)
            if float(residual.sum().item()) <= 1e-12:
                corrected = p_probs
            else:
                corrected = residual / residual.sum(dim=-1, keepdim=True)
            emitted = int(torch.multinomial(corrected, 1).item())
            return emitted, i
        else:
            target_tok = int(logits_i.argmax(dim=-1).item())
            if target_tok == d_tok:
                continue
            return target_tok, i

    # All k draft tokens accepted -> sample bonus from p_{k+1}.
    bonus_logits = verify_logits[k].unsqueeze(0)
    if sample_mode:
        bonus_probs = probs_from_logits(bonus_logits, gen_kwargs)
        emitted = int(torch.multinomial(bonus_probs, 1).item())
    else:
        emitted = int(bonus_logits.argmax(dim=-1).item())
    return emitted, k


def speculative_decode_sample(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    k: int,
    temperature: float,
    top_p: float,
) -> dict:
    """Run speculative decoding for a single sample with full KV-cache reuse."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(target_model.device)
    n_input = input_ids.shape[1]

    gen_kwargs = {}
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    target_device = input_ids.device
    draft_device = next(draft_model.parameters()).device

    eos_id = tokenizer.eos_token_id
    keep_step_logits = bool(gen_kwargs.get("do_sample", False))

    target_past = None
    draft_past = None
    target_pending_input = input_ids
    draft_pending_input = input_ids if draft_device == target_device else input_ids.to(draft_device)

    accepted_tokens: list[int] = []
    total_proposed = 0
    total_accepted = 0
    n_verify_steps = 0
    verify_log: list[dict] = []
    stopped_eos = False

    ttft_timer = GPUTimer()
    full_timer = GPUTimer()

    with full_timer:
        while len(accepted_tokens) < max_new_tokens:
            remaining = max_new_tokens - len(accepted_tokens)
            effective_k = min(k, remaining)
            if effective_k <= 0:
                break

            # ---- Draft phase ----
            draft_tokens, q_token_probs, q_step_logits, draft_past = _draft_generate(
                draft_model,
                draft_pending_input,
                draft_past,
                effective_k,
                gen_kwargs,
                keep_step_logits,
            )

            # ---- Verify phase ----
            if draft_device != target_device:
                draft_tokens_t = draft_tokens.to(target_device, non_blocking=True)
            else:
                draft_tokens_t = draft_tokens

            target_input = torch.cat([target_pending_input, draft_tokens_t], dim=-1)

            if n_verify_steps == 0:
                ttft_timer.__enter__()

            with torch.inference_mode():
                target_outputs = target_model(
                    target_input,
                    past_key_values=target_past,
                    use_cache=True,
                )
            target_past = target_outputs.past_key_values

            # Last (k+1) logits = predictions for the next k+1 positions
            # (p_1 conditioned on prefix; p_{k+1} = bonus).
            verify_logits = target_outputs.logits[0, -(effective_k + 1):, :]

            emitted, n_acc = _verify_block(
                verify_logits,
                draft_tokens.squeeze(0).tolist(),
                q_token_probs,
                q_step_logits,
                gen_kwargs,
            )

            if n_verify_steps == 0:
                ttft_timer.__exit__(None, None, None)

            total_proposed += effective_k
            total_accepted += n_acc
            n_verify_steps += 1
            verify_log.append({
                "step": n_verify_steps,
                "proposed": effective_k,
                "accepted": n_acc,
                "abs_position": n_input + len(accepted_tokens),
            })

            # ---- Crop both caches back to the accepted prefix length ----
            # After the forward, both caches cover `n_input + len(accepted) + effective_k`
            # positions. Accept n_acc draft tokens, so caches must shrink to
            # `n_input + len(accepted) + n_acc`. The newly emitted token is
            # *not* in either cache; we feed it as input on the next iteration.
            keep_len = n_input + len(accepted_tokens) + n_acc
            target_past = _crop_cache(target_past, keep_len)
            draft_past = _crop_cache(draft_past, keep_len)

            accepted_draft = draft_tokens.squeeze(0)[:n_acc].tolist()
            accepted_tokens.extend(int(t) for t in accepted_draft)
            accepted_tokens.append(int(emitted))

            # An iteration can emit up to (n_acc + 1) tokens. Trim to the cap
            # so we never overshoot max_new_tokens by 1.
            if len(accepted_tokens) > max_new_tokens:
                accepted_tokens = accepted_tokens[:max_new_tokens]
                break

            target_pending_input = torch.tensor(
                [[int(emitted)]], device=target_device, dtype=input_ids.dtype,
            )
            draft_pending_input = (
                target_pending_input
                if draft_device == target_device
                else target_pending_input.to(draft_device, non_blocking=True)
            )

            # ---- EOS handling ----
            if eos_id is not None:
                if int(emitted) == int(eos_id) or int(eos_id) in accepted_draft:
                    if int(eos_id) in accepted_tokens:
                        eos_pos = accepted_tokens.index(int(eos_id))
                        accepted_tokens = accepted_tokens[:eos_pos]
                    stopped_eos = True
                    break

    n_new = len(accepted_tokens)
    output_text = tokenizer.decode(accepted_tokens, skip_special_tokens=True)

    tpot_ms = ((full_timer.elapsed_ms - ttft_timer.elapsed_ms) / max(n_new - 1, 1)) if n_new > 1 else 0.0
    tokens_per_sec = n_new / full_timer.elapsed_s if full_timer.elapsed_s > 0 else 0.0
    alpha = total_accepted / total_proposed if total_proposed > 0 else 0.0
    b_eff = total_accepted / n_verify_steps if n_verify_steps > 0 else 0.0

    return {
        "latency_s": round(full_timer.elapsed_s, 4),
        "ttft_ms": round(ttft_timer.elapsed_ms, 2),
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
    }


def _save_verify_logs(results: list[dict], log_path: Path) -> None:
    """Persist per-sample verify logs to a JSON file (sample_id keyed)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {r["sample_id"]: r.get("verify_log", []) for r in results}
    with open(log_path, "w") as f:
        json.dump(payload, f)


def run_speculative_grid(
    data: dict[str, list[dict]],
    draft_label: str,
    k: int,
    regime_name: str,
    target_model=None,
    target_tokenizer=None,
    draft_model=None,
    draft_tokenizer=None,
    seed: int = SEED,
) -> list[dict]:
    """Run speculative decoding for all samples with one (draft, k, regime) config."""
    from baseline import load_target_model

    regime = REGIMES[regime_name]
    if target_model is None:
        target_model, target_tokenizer = load_target_model()
    if draft_model is None:
        draft_model, draft_tokenizer = load_draft_model(draft_label)

    results = []
    total = sum(len(v) for v in data.values())
    done = 0
    regime_short = "det" if regime_name == "deterministic" else "stoch"

    for task_name, samples in data.items():
        max_new_tokens = DATASETS[task_name]["max_new_tokens"]
        for sample in samples:
            done += 1
            set_seed(seed)

            out = speculative_decode_sample(
                target_model, draft_model, target_tokenizer,
                sample["prompt"], max_new_tokens, k,
                regime.temperature, regime.top_p,
            )

            row = {
                "sample_id": sample["sample_id"],
                "task": task_name,
                "draft": draft_label,
                "k": k,
                "regime": regime_name,
                "seed": seed,
                **out,
            }
            results.append(row)

            if done % 50 == 0 or done == total:
                print(f"  [spec {draft_label} k={k} {regime_short}] {done}/{total}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"spec_{draft_label}_k{k}_{regime_short}.csv"
    csv_rows = [{key: val for key, val in r.items() if key != "verify_log"} for r in results]
    write_csv(csv_path, csv_rows)
    log_path = VERIFY_LOG_DIR / f"spec_{draft_label}_k{k}_{regime_short}_seed{seed}.json"
    _save_verify_logs(results, log_path)
    print(f"  Saved -> {csv_path}")
    print(f"  Verify logs -> {log_path}")
    return results


def run_stability_analysis(
    data: dict[str, list[dict]],
    draft_label: str,
    k: int,
    regime_name: str,
    target_model=None,
    target_tokenizer=None,
    draft_model=None,
    draft_tokenizer=None,
) -> list[dict]:
    """Re-run a config with multiple seeds for stability analysis (Phase 4)."""
    regime_short = "det" if regime_name == "deterministic" else "stoch"
    all_results = []

    for seed in STABILITY_SEEDS:
        print(f"\n  Stability run: seed={seed}")
        results = run_speculative_grid(
            data, draft_label, k, regime_name,
            target_model, target_tokenizer,
            draft_model, draft_tokenizer,
            seed=seed,
        )
        STABILITY_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = STABILITY_DIR / f"spec_{draft_label}_k{k}_{regime_short}_seed{seed}.csv"
        csv_rows = [{key: val for key, val in r.items() if key != "verify_log"} for r in results]
        write_csv(csv_path, csv_rows)
        all_results.append({"seed": seed, "results": results})

    return all_results
