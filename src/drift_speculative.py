"""
DriftDiffuse speculative decoding loop.

Drop-in alternative to src/speculative.py that uses a parallel diffusion
drafter instead of an autoregressive draft model.

Key design points:
- Drafter proposes k tokens in `n_denoise_steps` (typically 3-5) instead of k
  autoregressive forwards.
- Two acceptance strategies (selected via `accept_mode`):
    * "block": single accept-or-reject of the whole k-block, with token-level
              fallback on rejection (provably unbiased, see notes).
    * "token": Leviathan-style per-token rejection sampling, using a pseudo-
               likelihood factorization q(x_i | x_<i, ctx) obtained by re-
               running the drifter with each prefix prefix revealed (cheap
               because it's a single batched forward of size k).
- Target verifier is reused unchanged from speculative.py: a single forward
  pass over the proposed block returns k+1 logit rows.

Output schema mirrors speculative_decode_sample so existing aggregation
(metrics.py, evaluate.py, runtime.ensure_*_results) works without changes.
The CSV gets two extra columns: `n_denoise_steps`, `block_accepted`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from config import REGIMES, DATASETS, RESULTS_DIR, SEED
from diffusion.drifter import DriftDiffuser
from diffusion.sampler import iterative_unmask
from sampling import probs_from_logits
from utils import set_seed, GPUTimer, write_csv


def _crop_cache(past_key_values, length: int):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(length)
        return past_key_values
    return tuple(tuple(t[..., :length, :] for t in layer) for layer in past_key_values)


@torch.inference_mode()
def _drifter_pseudo_q(
    drifter: DriftDiffuser,
    context_ids: torch.Tensor,        # (1, L_ctx)
    block_tokens: torch.Tensor,       # (1, k) — the *committed* draft tokens
    temperature: float,
) -> torch.Tensor:
    """
    Pseudo-likelihood factorization q(x_i | x_<i, ctx) for each block position.

    Implementation: build a (k, L_ctx + k) input where row i has the first i
    block positions revealed and positions [i..k) masked. Run the drifter once
    over this batch and read the i-th block-position logits from each row.

    Returns: (1, k, V) probability tensor.
    """
    cfg = drifter.cfg
    device = context_ids.device
    k = block_tokens.shape[1]
    L_ctx = context_ids.shape[1]

    ctx_batch = context_ids.expand(k, -1).contiguous()                 # (k, L_ctx)
    block_batch = block_tokens.expand(k, -1).contiguous().clone()      # (k, k)
    mask_flags = torch.zeros((k, k), dtype=torch.bool, device=device)
    for i in range(k):
        mask_flags[i, i:] = True
    # For consistency at inference with the iterative_unmask call, give frozen
    # positions t=0 and masked positions t=1 (the lowest-noise step).
    t_per_pos = torch.where(mask_flags, torch.ones_like(mask_flags, dtype=torch.long),
                            torch.zeros_like(mask_flags, dtype=torch.long))

    logits = drifter(ctx_batch, block_batch, mask_flags, t_per_pos)    # (k, k, V)
    # We want q(x_i | x_<i): row i, position i.
    diag_logits = logits[torch.arange(k, device=device), torch.arange(k, device=device), :]  # (k, V)
    if temperature > 0.0:
        diag_logits = diag_logits / max(temperature, 1e-6)
    q = F.softmax(diag_logits, dim=-1).unsqueeze(0)                    # (1, k, V)
    return q


def _block_accept_log_ratio(
    p_probs: torch.Tensor,    # (k, V_target)
    q_probs: torch.Tensor,    # (k, V_draft)  (may differ in V)
    block_ids: torch.Tensor,  # (k,)
) -> float:
    """log(prod_i p(x_i)/q(x_i)) with vocab alignment + numerical floor."""
    common_vocab = min(p_probs.shape[-1], q_probs.shape[-1])
    p = p_probs[:, :common_vocab]
    q = q_probs[:, :common_vocab]
    # Renormalize over the common vocabulary for a fair ratio.
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    p_x = p.gather(1, block_ids.view(-1, 1).clamp(max=common_vocab - 1)).squeeze(1).clamp(min=1e-30)
    q_x = q.gather(1, block_ids.view(-1, 1).clamp(max=common_vocab - 1)).squeeze(1).clamp(min=1e-30)
    return float((p_x.log() - q_x.log()).sum().item())


def drift_decode_sample(
    target_model,
    drifter: DriftDiffuser,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    k: int,
    temperature: float,
    top_p: float,
    n_denoise_steps: int = 3,
    accept_mode: str = "block",        # "block" or "token"
    drifter_ctx_len: int | None = None,
) -> dict:
    """Run DriftDiffuse speculative decoding for one sample."""
    assert accept_mode in ("block", "token")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(target_model.device)
    n_input = input_ids.shape[1]

    gen_kwargs = (
        {"do_sample": False}
        if temperature == 0.0
        else {"do_sample": True, "temperature": temperature, "top_p": top_p}
    )
    sample_mode = bool(gen_kwargs.get("do_sample", False))

    target_device = input_ids.device
    drifter_device = next(drifter.parameters()).device
    eos_id = tokenizer.eos_token_id

    drifter_ctx_len = drifter_ctx_len or (drifter.cfg.max_ctx_len - drifter.cfg.k_max - 8)

    target_past = None
    target_pending_input = input_ids
    accepted_tokens: list[int] = []

    total_proposed = 0
    total_accepted = 0
    n_verify_steps = 0
    block_accept_count = 0
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

            # ---- Build drifter context (last drifter_ctx_len tokens of prefix) ----
            full_prefix = torch.cat(
                [input_ids[0], torch.tensor(accepted_tokens, device=target_device, dtype=input_ids.dtype)],
                dim=0,
            )
            if full_prefix.shape[0] >= drifter_ctx_len:
                ctx_for_drifter = full_prefix[-drifter_ctx_len:]
            else:
                # Left-pad with EOS so the drifter always sees fixed-size context.
                pad_id = eos_id if eos_id is not None else 0
                pad_n = drifter_ctx_len - full_prefix.shape[0]
                ctx_for_drifter = torch.cat([
                    torch.full((pad_n,), pad_id, device=target_device, dtype=input_ids.dtype),
                    full_prefix,
                ], dim=0)
            ctx_for_drifter = ctx_for_drifter.unsqueeze(0).to(drifter_device)

            # ---- Drafter: parallel k-token proposal ----
            draft_block, _ = iterative_unmask(
                drifter,
                ctx_for_drifter,
                effective_k,
                n_steps=n_denoise_steps,
                temperature=temperature if sample_mode else 0.0,
            )  # (1, k)

            # Pseudo-likelihood factorization for token-mode acceptance.
            if accept_mode == "token" or sample_mode:
                q_probs = _drifter_pseudo_q(
                    drifter, ctx_for_drifter, draft_block, temperature if sample_mode else 1.0,
                )[0]  # (k, V_draft)
            else:
                q_probs = None

            # ---- Verify with the target model (single forward) ----
            draft_block_t = draft_block.to(target_device)
            target_input = torch.cat([target_pending_input, draft_block_t], dim=-1)

            if n_verify_steps == 0:
                ttft_timer.__enter__()

            target_outputs = target_model(
                target_input,
                past_key_values=target_past,
                use_cache=True,
            )
            target_past = target_outputs.past_key_values
            verify_logits = target_outputs.logits[0, -(effective_k + 1):, :]   # (k+1, V_target)

            if n_verify_steps == 0:
                ttft_timer.__exit__(None, None, None)

            block_ids_flat = draft_block_t.squeeze(0)
            block_target_logits = verify_logits[:effective_k, :]               # (k, V_target)
            target_probs = probs_from_logits(block_target_logits, gen_kwargs)  # (k, V_target)

            # ---- Acceptance ----
            if accept_mode == "block" and sample_mode and q_probs is not None:
                # Block accept/reject (Bernoulli on whole block)
                log_ratio = _block_accept_log_ratio(target_probs, q_probs, block_ids_flat)
                u = float(torch.rand(1).item())
                accept_block = (log_ratio >= 0.0) or (u < min(1.0, float(torch.exp(torch.tensor(log_ratio)).item())))
                if accept_block:
                    n_acc = effective_k
                    block_accept_count += 1
                    bonus_logits = verify_logits[effective_k].unsqueeze(0)
                    if sample_mode:
                        bonus_probs = probs_from_logits(bonus_logits, gen_kwargs)
                        emitted = int(torch.multinomial(bonus_probs, 1).item())
                    else:
                        emitted = int(bonus_logits.argmax(dim=-1).item())
                else:
                    # Fallback to token-level for an honest partial acceptance.
                    n_acc, emitted = _token_accept_fallback(
                        target_probs, q_probs, block_ids_flat, verify_logits, gen_kwargs,
                    )
            else:
                # Either deterministic mode (just compare argmax) or accept_mode == "token".
                n_acc, emitted = _token_accept_fallback(
                    target_probs,
                    q_probs if sample_mode else None,
                    block_ids_flat,
                    verify_logits,
                    gen_kwargs,
                )
                if n_acc == effective_k:
                    block_accept_count += 1

            total_proposed += effective_k
            total_accepted += n_acc
            n_verify_steps += 1
            verify_log.append({
                "step": n_verify_steps,
                "proposed": effective_k,
                "accepted": n_acc,
                "abs_position": n_input + len(accepted_tokens),
            })

            # ---- KV cache crop ----
            keep_len = n_input + len(accepted_tokens) + n_acc
            target_past = _crop_cache(target_past, keep_len)

            accepted_draft = block_ids_flat[:n_acc].tolist()
            accepted_tokens.extend(int(t) for t in accepted_draft)
            accepted_tokens.append(int(emitted))
            if len(accepted_tokens) > max_new_tokens:
                accepted_tokens = accepted_tokens[:max_new_tokens]
                break

            target_pending_input = torch.tensor(
                [[int(emitted)]], device=target_device, dtype=input_ids.dtype,
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
    block_acc_rate = block_accept_count / max(n_verify_steps, 1)

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
        "n_denoise_steps": n_denoise_steps,
        "block_accept_rate": round(block_acc_rate, 4),
        "stopped_eos": stopped_eos,
        "output_text": output_text,
        "verify_log": verify_log,
    }


def _token_accept_fallback(
    target_probs: torch.Tensor,        # (k, V_target)
    q_probs: torch.Tensor | None,      # (k, V_draft)  or None for greedy
    block_ids: torch.Tensor,           # (k,)
    verify_logits: torch.Tensor,       # (k+1, V_target)
    gen_kwargs: dict,
) -> tuple[int, int]:
    """Standard Leviathan rejection sampling as a fallback path."""
    k = block_ids.shape[0]
    sample_mode = bool(gen_kwargs.get("do_sample", False))

    for i in range(k):
        d_tok = int(block_ids[i].item())
        p_row = target_probs[i:i+1]
        if not sample_mode:
            target_tok = int(p_row.argmax(dim=-1).item())
            if target_tok == d_tok:
                continue
            return target_tok, i

        common_vocab = min(p_row.shape[-1], q_probs.shape[-1])
        p_tok = float(p_row[0, min(d_tok, common_vocab - 1)].item())
        q_tok = max(float(q_probs[i, min(d_tok, common_vocab - 1)].item()), 1e-12)
        accept_prob = min(1.0, p_tok / q_tok)
        if float(torch.rand(1).item()) < accept_prob:
            continue
        # Reject -> sample from residual max(p - q, 0)
        p_c = p_row[..., :common_vocab].clone()
        q_c = q_probs[i:i+1, :common_vocab].clone()
        p_c = p_c / p_c.sum().clamp(min=1e-12)
        q_c = q_c / q_c.sum().clamp(min=1e-12)
        residual = torch.clamp(p_c - q_c, min=0.0)
        if float(residual.sum().item()) <= 1e-12:
            corrected = p_c
        else:
            corrected = residual / residual.sum(dim=-1, keepdim=True)
        emitted = int(torch.multinomial(corrected, 1).item())
        return emitted, i

    bonus_logits = verify_logits[k].unsqueeze(0)
    if sample_mode:
        bonus_probs = probs_from_logits(bonus_logits, gen_kwargs)
        emitted = int(torch.multinomial(bonus_probs, 1).item())
    else:
        emitted = int(bonus_logits.argmax(dim=-1).item())
    return emitted, k


def run_drift_grid(
    data: dict[str, list[dict]],
    drifter: DriftDiffuser,
    target_model,
    target_tokenizer,
    k: int,
    regime_name: str,
    n_denoise_steps: int = 3,
    accept_mode: str = "block",
    seed: int = SEED,
    label: str = "drift",
) -> list[dict]:
    """Run the full eval suite for one (k, regime, n_denoise) config."""
    regime = REGIMES[regime_name]
    regime_short = "det" if regime_name == "deterministic" else "stoch"
    results = []

    for task_name, samples in data.items():
        max_new = DATASETS[task_name]["max_new_tokens"]
        for sample in samples:
            set_seed(seed)
            out = drift_decode_sample(
                target_model,
                drifter,
                target_tokenizer,
                sample["prompt"],
                max_new,
                k,
                regime.temperature,
                regime.top_p,
                n_denoise_steps=n_denoise_steps,
                accept_mode=accept_mode,
            )
            results.append({
                "sample_id": sample["sample_id"],
                "task": task_name,
                "draft": label,
                "k": k,
                "regime": regime_name,
                "seed": seed,
                "n_denoise_steps": n_denoise_steps,
                "accept_mode": accept_mode,
                **{key: val for key, val in out.items() if key != "verify_log"},
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"drift_{label}_n{n_denoise_steps}_{accept_mode}_k{k}_{regime_short}.csv"
    write_csv(csv_path, results)
    print(f"Saved -> {csv_path}")
    return results
