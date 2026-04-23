"""
Speculative decoding runner (Phases 3–4).

Implements the draft-then-verify loop:
  1. Draft model generates k tokens autoregressively.
  2. Target model verifies all k tokens in a single forward pass.
  3. Accept the longest matching prefix; resample from the divergence point.

Supports both deterministic and stochastic regimes.
"""

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


def _get_quant_kwargs():
    """Return model loading kwargs based on QUANT_MODE config."""
    kwargs, _ = get_quant_kwargs()
    return kwargs


def load_draft_model(draft_label: str):
    """Load a draft model and tokenizer."""
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


def _draft_generate(
    model,
    input_ids: torch.Tensor,
    k: int,
    gen_kwargs: dict,
) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """
    Generate k draft tokens with KV-cache decoding.

    Returns:
        draft_ids: full sequence (prefix + proposed draft tokens)
        q_token_probs: q(y_i) for each proposed token under draft policy
        q_step_logits: draft logits used at each speculative step (k, vocab)
    """
    q_token_probs: list[float] = []
    q_step_logits: list[torch.Tensor] = []
    proposed_tokens: list[torch.Tensor] = []

    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        for step in range(k):
            q_step_logits.append(logits.squeeze(0).detach())

            next_token, probs = sample_next_token(logits, gen_kwargs)
            q_prob = probs.gather(1, next_token).squeeze(1)
            q_token_probs.append(float(q_prob.item()))

            proposed_tokens.append(next_token)

            if step < k - 1:
                outputs = model(
                    next_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

    proposed = torch.cat(proposed_tokens, dim=-1)
    draft_ids = torch.cat([input_ids, proposed], dim=-1)
    return draft_ids, q_token_probs, torch.stack(q_step_logits, dim=0)


def _verify_and_accept(
    target_model,
    input_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    n_input: int,
    gen_kwargs: dict,
    q_token_probs: list[float],
    q_step_logits: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """
    Verify draft tokens with target model.

    Returns:
        accepted_ids: full sequence with accepted + 1 bonus token
        n_proposed: number of draft tokens proposed
        n_accepted: number of draft tokens accepted (before divergence)
    """
    draft_tokens = draft_ids[0, n_input:]
    k = len(draft_tokens)

    n_accepted = 0
    with torch.inference_mode():
        # Block verification: one target forward over prefix+draft block.
        outputs = target_model(draft_ids)
        # logits indices [n_input - 1 .. n_input + k - 1] correspond to
        # distributions for draft tokens [0..k-1] and the bonus token at index k.
        verify_logits = outputs.logits[0, n_input - 1: n_input + k, :]

        for i in range(k):
            logits_i = verify_logits[i].unsqueeze(0)
            draft_token = draft_tokens[i].item()

            if gen_kwargs.get("do_sample", False):
                # Paper-faithful stochastic acceptance:
                # accept with probability min(1, p_i(y_i) / q_i(y_i)).
                p_probs = probs_from_logits(logits_i, gen_kwargs)
                p_token = float(p_probs[0, draft_token].item())
                q_token = max(float(q_token_probs[i]), 1e-12)
                accept_prob = min(1.0, p_token / q_token)

                if float(torch.rand(1).item()) < accept_prob:
                    n_accepted += 1
                else:
                    # Rejection correction: sample from residual max(p - q, 0).
                    q_logits_i = q_step_logits[i].unsqueeze(0).to(p_probs.device)
                    q_probs = probs_from_logits(q_logits_i, gen_kwargs)
                    residual = torch.clamp(p_probs - q_probs, min=0.0)
                    residual_mass = float(residual.sum().item())
                    if residual_mass <= 1e-12:
                        corrected = p_probs
                    else:
                        corrected = residual / residual.sum(dim=-1, keepdim=True)

                    corrected_token = torch.multinomial(corrected, 1)
                    accepted_seq = draft_ids[:, : n_input + n_accepted]
                    final_ids = torch.cat(
                        [accepted_seq, corrected_token.to(accepted_seq.device)],
                        dim=-1,
                    )
                    return final_ids, k, n_accepted
            else:
                # Deterministic: accept if argmax matches; on mismatch emit target token.
                target_token = logits_i.argmax(dim=-1, keepdim=True)
                if int(target_token.item()) == draft_token:
                    n_accepted += 1
                else:
                    accepted_seq = draft_ids[:, : n_input + n_accepted]
                    final_ids = torch.cat([accepted_seq, target_token.to(accepted_seq.device)], dim=-1)
                    return final_ids, k, n_accepted

        # All proposed tokens accepted; sample one bonus token from target.
        accepted_seq = draft_ids[:, : n_input + n_accepted]
        bonus_logits = verify_logits[k].unsqueeze(0)
        if gen_kwargs.get("do_sample", False):
            bonus_probs = probs_from_logits(bonus_logits, gen_kwargs)
            bonus_token = torch.multinomial(bonus_probs, 1)
        else:
            bonus_token = bonus_logits.argmax(dim=-1, keepdim=True)

    final_ids = torch.cat([accepted_seq, bonus_token], dim=-1)
    return final_ids, k, n_accepted


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
    """
    Run speculative decoding for a single sample.
    """
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

    current_ids = input_ids.clone()
    target_device = input_ids.device
    draft_device = next(draft_model.parameters()).device
    total_proposed = 0
    total_accepted = 0
    n_verify_steps = 0
    verify_log = []

    # --- TTFT: first verify step ---
    ttft_timer = GPUTimer()
    full_timer = GPUTimer()

    with full_timer:
        while current_ids.shape[1] - n_input < max_new_tokens:
            remaining = max_new_tokens - (current_ids.shape[1] - n_input)
            effective_k = min(k, remaining)
            if effective_k <= 0:
                break

            # Move to draft device only when needed.
            if draft_device == target_device:
                draft_input = current_ids
            else:
                draft_input = current_ids.to(draft_device, non_blocking=True)

            # Draft phase
            draft_ids, q_token_probs, q_step_logits = _draft_generate(
                draft_model,
                draft_input,
                effective_k,
                gen_kwargs,
            )

            # Move back to target device for verification
            if draft_device != target_device:
                draft_ids = draft_ids.to(target_device, non_blocking=True)

            # Verify phase
            if n_verify_steps == 0:
                ttft_timer.__enter__()

            new_ids, n_proposed, n_accepted = _verify_and_accept(
                target_model,
                current_ids,
                draft_ids,
                current_ids.shape[1],
                gen_kwargs,
                q_token_probs,
                q_step_logits,
            )

            if n_verify_steps == 0:
                ttft_timer.__exit__(None, None, None)

            total_proposed += n_proposed
            total_accepted += n_accepted
            n_verify_steps += 1
            verify_log.append({
                "step": n_verify_steps,
                "proposed": n_proposed,
                "accepted": n_accepted,
            })

            current_ids = new_ids

            # Check for EOS
            if tokenizer.eos_token_id is not None:
                new_tokens = current_ids[0, n_input:]
                if tokenizer.eos_token_id in new_tokens.tolist():
                    eos_pos = new_tokens.tolist().index(tokenizer.eos_token_id)
                    current_ids = current_ids[:, : n_input + eos_pos]
                    break

    new_ids = current_ids[0, n_input:]
    n_new = len(new_ids)
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

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
        "output_text": output_text,
        "verify_log": verify_log,
    }


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
    """
    Run speculative decoding for all samples with a specific (draft, k, regime) config.
    Returns list of per-sample result dicts.
    """
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
                **{key: val for key, val in out.items() if key != "verify_log"},
            }
            results.append(row)

            if done % 50 == 0 or done == total:
                print(f"  [spec {draft_label} k={k} {regime_short}] {done}/{total}")

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"spec_{draft_label}_k{k}_{regime_short}.csv"
    write_csv(csv_path, results)
    print(f"  Saved → {csv_path}")
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
    """
    Re-run a specific config with multiple seeds for stability analysis (Phase 4).
    Returns list of per-seed result dicts.
    """
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
        # Save per-seed
        STABILITY_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = STABILITY_DIR / f"spec_{draft_label}_k{k}_{regime_short}_seed{seed}.csv"
        write_csv(csv_path, results)
        all_results.append({"seed": seed, "results": results})

    return all_results
