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
import os
from collections import deque

import torch

from config import (
    DATASETS,
    REGIMES,
    RESULTS_DIR,
    SEED,
    GPU_PREALLOCATE_STEP_BUFFERS,
    GPU_USE_STABLE_STEP_SHAPES,
    GPU_TRY_CUDA_GRAPHS,
    GPU_REQUIRE_CUDA_GRAPHS,
)
from sampling import sample_next_token_and_prob
from speculative import (
    VERIFY_LOG_DIR,
    _crop_cache,
    _draft_generate,
    _verify_block,
    load_draft_model,
)
from utils import GPUTimer, set_seed, write_csv


def _cuda_graph_possible(
    target_device: torch.device,
    step_input_buffer: torch.Tensor | None,
    stable_step_shapes: bool,
    target_model=None,
) -> bool:
    if os.environ.get("SPECDEC_FORCE_QWEN_CUDA_GRAPHS", "0") != "1":
        model_type = str(getattr(getattr(target_model, "config", None), "model_type", "")).lower()
        if model_type in {"qwen2", "qwen2_moe"}:
            return False

    return bool(
        GPU_TRY_CUDA_GRAPHS
        and torch.cuda.is_available()
        and target_device.type == "cuda"
        and step_input_buffer is not None
        and stable_step_shapes
    )


def _graph_cache_supported(past_key_values) -> bool:
    return past_key_values is not None and hasattr(past_key_values, "crop")


def _first_oov_pos(tokens_1d: torch.Tensor, vocab_size: int) -> int | None:
    if tokens_1d.numel() == 0:
        return None
    invalid = (tokens_1d < 0) | (tokens_1d >= int(vocab_size))
    if not bool(invalid.any()):
        return None
    return int(torch.nonzero(invalid, as_tuple=False)[0].item())


def _safe_fallback_token_id(vocab_size: int, preferred_id: int | None = None) -> int:
    if vocab_size <= 0:
        return 0
    if preferred_id is not None and 0 <= int(preferred_id) < int(vocab_size):
        return int(preferred_id)
    return 0


def _sanitize_token_ids(
    tokens: torch.Tensor,
    vocab_size: int,
    fallback_id: int,
) -> tuple[torch.Tensor, int]:
    if tokens.numel() == 0 or vocab_size <= 0:
        return tokens, 0
    invalid = (tokens < 0) | (tokens >= int(vocab_size))
    if not bool(invalid.any()):
        return tokens, 0
    fixed = tokens.clone()
    fixed[invalid] = int(fallback_id)
    return fixed, int(invalid.sum().item())


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


def _build_missing_prefix_tokens(
    input_ids: torch.Tensor,
    accepted_tokens: list[int],
    prefix_len: int,
) -> torch.Tensor:
    """Return committed prefix tokens not yet covered by a draft cache."""
    n_input = input_ids.shape[1]
    committed_len = n_input + len(accepted_tokens)
    if prefix_len < 0 or prefix_len > committed_len:
        raise ValueError(f"Invalid draft prefix length {prefix_len} for committed length {committed_len}")

    pieces: list[torch.Tensor] = []
    if prefix_len < n_input:
        pieces.append(input_ids[:, prefix_len:n_input])

    accepted_start = max(prefix_len - n_input, 0)
    if accepted_start < len(accepted_tokens):
        accepted_suffix = torch.tensor(
            accepted_tokens[accepted_start:],
            device=input_ids.device,
            dtype=input_ids.dtype,
        ).unsqueeze(0)
        pieces.append(accepted_suffix)

    if not pieces:
        return input_ids[:, :0]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=-1)


def _prepare_draft_pending_input(
    input_ids: torch.Tensor,
    accepted_tokens: list[int],
    draft_prefix_len: int,
    target_pending_input: torch.Tensor,
    draft_device: torch.device,
) -> torch.Tensor:
    """Reuse the target pending slice when possible; otherwise send only the missing suffix."""
    n_input = input_ids.shape[1]
    committed_len = n_input + len(accepted_tokens)
    target_pending_prefix_len = committed_len - target_pending_input.shape[1]

    if draft_prefix_len == target_pending_prefix_len:
        pending = target_pending_input
    else:
        pending = _build_missing_prefix_tokens(input_ids, accepted_tokens, draft_prefix_len)

    if pending.device == draft_device:
        return pending
    return pending.to(draft_device, non_blocking=True)


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
    base_k: int = 4,
    k_choices: tuple[int, ...] = (3, 4, 5),
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
    target_vocab_size = int(getattr(getattr(target_model, "config", None), "vocab_size", 0) or 0)
    target_fallback_id = _safe_fallback_token_id(target_vocab_size, eos_id)

    target_past = None
    target_pending_input = input_ids
    one_token_buf = torch.empty((1, 1), device=target_device, dtype=input_ids.dtype)
    step_input_buffer = None
    if GPU_PREALLOCATE_STEP_BUFFERS:
        step_input_buffer = torch.empty((1, max(k_choices) + 1), device=target_device, dtype=input_ids.dtype)
    stable_step_shapes = bool(GPU_USE_STABLE_STEP_SHAPES and step_input_buffer is not None)

    graph_capture_note = "disabled"
    if GPU_TRY_CUDA_GRAPHS:
        graph_capture_note = "pending"

    graph_state = {
        "enabled": _cuda_graph_possible(target_device, step_input_buffer, stable_step_shapes, target_model),
        "captured": False,
        "failed": False,
        "graph": None,
        "input": None,
        "logits": None,
        "past": None,
        "cache_id": None,
        "max_window": int(max(k_choices) + 1),
    }

    draft_cache = {primary_label: None, rescue_label: None}
    draft_prefix_lens = {primary_label: 0, rescue_label: 0}
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
                last_active_label = active_label

            draft_model = draft_models[active_label]
            draft_device = next(draft_model.parameters()).device
            draft_vocab_size = int(getattr(getattr(draft_model, "config", None), "vocab_size", 0) or 0)
            draft_fallback_id = _safe_fallback_token_id(draft_vocab_size, eos_id)

            target_pending_input, n_fix_target_pending = _sanitize_token_ids(
                target_pending_input,
                target_vocab_size,
                target_fallback_id,
            )
            draft_pending_input = _prepare_draft_pending_input(
                input_ids,
                accepted_tokens,
                draft_prefix_lens[active_label],
                target_pending_input,
                draft_device,
            )
            draft_pending_input, n_fix_draft_pending = _sanitize_token_ids(
                draft_pending_input,
                draft_vocab_size,
                draft_fallback_id,
            )

            draft_tokens, q_token_probs, q_step_logits, draft_past = _draft_generate(
                draft_model,
                draft_pending_input,
                draft_cache[active_label],
                k_now,
                gen_kwargs,
                keep_step_logits,
            )

            draft_tokens_t = draft_tokens if draft_tokens.device == target_device else draft_tokens.to(target_device, non_blocking=True)
            safe_k = k_now
            oov_pos = None
            if target_vocab_size > 0:
                oov_pos = _first_oov_pos(draft_tokens_t.squeeze(0)[:k_now], target_vocab_size)
                if oov_pos is not None:
                    safe_k = int(oov_pos)

            if safe_k == 0:
                with torch.inference_mode():
                    target_out = target_model(
                        target_pending_input,
                        past_key_values=target_past,
                        use_cache=True,
                    )
                target_past = target_out.past_key_values
                bonus_logits = target_out.logits[:, -1, :]
                if gen_kwargs.get("do_sample", False):
                    bonus_probs = torch.softmax(bonus_logits.float() / max(float(gen_kwargs.get("temperature", 1.0)), 1e-5), dim=-1)
                    emitted = int(torch.multinomial(bonus_probs, 1).item())
                else:
                    emitted = int(bonus_logits.argmax(dim=-1).item())
                n_acc = 0

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
                        "oov_reject": True,
                        "target_pending_sanitized": n_fix_target_pending,
                        "draft_pending_sanitized": n_fix_draft_pending,
                    }
                )

                keep_len = n_input + len(accepted_tokens)
                target_past = _crop_cache(target_past, keep_len)
                draft_cache[active_label] = _crop_cache(draft_past, keep_len)
                draft_prefix_lens[active_label] = keep_len

                accepted_tokens.append(int(emitted))
                if len(accepted_tokens) > max_new_tokens:
                    accepted_tokens = accepted_tokens[:max_new_tokens]
                    break

                one_token_buf.fill_(int(emitted))
                target_pending_input = one_token_buf

                if eos_id is not None and int(emitted) == int(eos_id):
                    if int(eos_id) in accepted_tokens:
                        eos_pos = accepted_tokens.index(int(eos_id))
                        accepted_tokens = accepted_tokens[:eos_pos]
                    stopped_eos = True
                    break
                continue
            use_buffered_step = (
                step_input_buffer is not None
                and target_pending_input.shape[1] == 1
                and target_past is not None
            )
            fixed_shape_step = bool(use_buffered_step and stable_step_shapes)

            if use_buffered_step:
                step_input_buffer[:, :1].copy_(target_pending_input)
                step_input_buffer[:, 1:safe_k + 1].copy_(draft_tokens_t[:, :safe_k])
                if fixed_shape_step and safe_k < max(k_choices):
                    pad_id = int(eos_id) if eos_id is not None else int(target_pending_input[0, 0].item())
                    step_input_buffer[:, safe_k + 1:max(k_choices) + 1].fill_(pad_id)
                    target_input = step_input_buffer
                    verify_window = max(k_choices) + 1
                else:
                    target_input = step_input_buffer[:, :safe_k + 1]
                    verify_window = safe_k + 1
            else:
                target_input = torch.cat([target_pending_input, draft_tokens_t[:, :safe_k]], dim=-1)
                verify_window = safe_k + 1

            target_input, n_fix_target_input = _sanitize_token_ids(
                target_input,
                target_vocab_size,
                target_fallback_id,
            )
            if target_vocab_size > 0:
                clamp_max = int(target_vocab_size - 1)
                target_input_clamped = torch.clamp(target_input, min=0, max=clamp_max)
                n_clamped = int((target_input_clamped != target_input).sum().item())
                target_input = target_input_clamped
            else:
                n_clamped = 0

            if n_verify_steps == 0 and ttft_start is not None:
                ttft_start.record()

            can_try_graph = bool(
                graph_state["enabled"]
                and not graph_state["failed"]
                and fixed_shape_step
                and _graph_cache_supported(target_past)
            )

            if (
                can_try_graph
                and graph_state["captured"]
                and graph_state.get("cache_id") != id(target_past)
            ):
                graph_state["captured"] = False
                graph_state["graph"] = None
                graph_state["input"] = None
                graph_state["logits"] = None
                graph_state["past"] = None
                graph_state["cache_id"] = None
                graph_capture_note = "recapture_pending"

            if can_try_graph and not graph_state["captured"]:
                try:
                    graph_input = torch.empty(
                        (1, graph_state["max_window"]),
                        device=target_device,
                        dtype=input_ids.dtype,
                    )
                    graph_input.copy_(target_input)

                    warmup_stream = torch.cuda.Stream(device=target_device)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(2):
                            target_model(
                                graph_input,
                                past_key_values=target_past,
                                use_cache=True,
                            )
                    torch.cuda.current_stream(device=target_device).wait_stream(warmup_stream)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_outputs = target_model(
                            graph_input,
                            past_key_values=target_past,
                            use_cache=True,
                        )
                        graph_logits = graph_outputs.logits
                        graph_past = graph_outputs.past_key_values

                    graph_state["graph"] = graph
                    graph_state["input"] = graph_input
                    graph_state["logits"] = graph_logits
                    graph_state["past"] = graph_past
                    graph_state["cache_id"] = id(target_past)
                    graph_state["captured"] = True
                    graph_capture_note = "captured"
                except Exception:
                    graph_state["failed"] = True
                    graph_capture_note = "capture_failed_fallback_eager"
                    if GPU_REQUIRE_CUDA_GRAPHS:
                        raise RuntimeError("CUDA Graph capture failed and GPU_REQUIRE_CUDA_GRAPHS=True")

            if can_try_graph and graph_state["captured"]:
                try:
                    graph_state["input"].copy_(target_input)
                    graph_state["graph"].replay()
                    target_past = graph_state["past"]
                    verify_logits = graph_state["logits"][0, -verify_window:, :]
                    if fixed_shape_step and safe_k < max(k_choices):
                        verify_logits = verify_logits[:safe_k + 1, :]
                except Exception:
                    graph_state["failed"] = True
                    graph_capture_note = "replay_failed_fallback_eager"
                    if GPU_REQUIRE_CUDA_GRAPHS:
                        raise RuntimeError("CUDA Graph replay failed and GPU_REQUIRE_CUDA_GRAPHS=True")
                    with torch.inference_mode():
                        target_out = target_model(
                            target_input,
                            past_key_values=target_past,
                            use_cache=True,
                        )
                    target_past = target_out.past_key_values
                    verify_logits = target_out.logits[0, -verify_window:, :]
                    if fixed_shape_step and safe_k < max(k_choices):
                        verify_logits = verify_logits[:safe_k + 1, :]
            else:
                with torch.inference_mode():
                    target_out = target_model(
                        target_input,
                        past_key_values=target_past,
                        use_cache=True,
                    )
                target_past = target_out.past_key_values
                verify_logits = target_out.logits[0, -verify_window:, :]
                if fixed_shape_step and safe_k < max(k_choices):
                    verify_logits = verify_logits[:safe_k + 1, :]

            if n_verify_steps == 0 and ttft_end is not None:
                ttft_end.record()

            emitted, n_acc = _verify_block(
                verify_logits,
                draft_tokens.squeeze(0)[:safe_k],
                (q_token_probs[:safe_k] if q_token_probs is not None else None),
                (q_step_logits[:safe_k] if q_step_logits is not None else None),
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
                    "oov_reject": bool(oov_pos is not None),
                    "target_pending_sanitized": n_fix_target_pending,
                    "draft_pending_sanitized": n_fix_draft_pending,
                    "target_input_sanitized": n_fix_target_input,
                    "target_input_clamped": n_clamped,
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
            draft_prefix_lens[active_label] = keep_len

            accepted_draft = draft_tokens.squeeze(0)[:n_acc].tolist()
            accepted_tokens.extend(int(t) for t in accepted_draft)
            accepted_tokens.append(int(emitted))

            if len(accepted_tokens) > max_new_tokens:
                accepted_tokens = accepted_tokens[:max_new_tokens]
                break

            one_token_buf.fill_(int(emitted))
            target_pending_input = one_token_buf

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
        "gpu_stable_step_shapes": stable_step_shapes,
        "gpu_graph_capture": graph_capture_note,
    }


def run_acsd_grid(
    data: dict[str, list[dict]],
    regime_name: str,
    target_model,
    target_tokenizer,
    draft_models: dict[str, object] | None = None,
    primary_label: str = "0.5B",
    rescue_label: str = "1.5B",
    base_k: int = 4,
    k_choices: tuple[int, ...] = (3, 4, 5),
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
