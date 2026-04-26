"""
Iterative confidence-weighted unmasking (MaskGIT-style) for the DriftDiffuser.

Given a fully-masked block of length k, run T denoising steps. At each step,
keep tokens whose softmax confidence exceeds a per-step threshold determined
by the schedule; mask the rest and re-predict.

Returns the final block tokens plus per-position logits (for the spec verifier
and for pseudo-likelihood factorization in drift_speculative.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def iterative_unmask(
    drifter,
    context_ids: torch.Tensor,   # (1, L_ctx)
    k: int,
    n_steps: int | None = None,
    confidence_floor: float = 0.0,
    early_exit_tau: float = 0.95,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        block_ids: (1, k) long tensor — sampled tokens.
        block_logits: (1, k, V) tensor — final-step logits per position.
    """
    cfg = drifter.cfg
    n_steps = n_steps or cfg.n_steps
    device = context_ids.device
    B = context_ids.shape[0]
    assert B == 1, "iterative_unmask currently expects batch size 1"

    block_ids = torch.zeros((B, k), dtype=torch.long, device=device)
    mask_flags = torch.ones((B, k), dtype=torch.bool, device=device)

    final_logits = None
    for step in range(n_steps, 0, -1):
        t_per_pos = torch.full((B, k), step, dtype=torch.long, device=device)
        # For frozen positions we set t=0 so the time-emb contribution is the same
        # zeroth bucket every step (stable conditioning).
        t_per_pos = torch.where(mask_flags, t_per_pos, torch.zeros_like(t_per_pos))

        logits = drifter(context_ids, block_ids, mask_flags, t_per_pos)  # (1,k,V)
        final_logits = logits

        if not mask_flags.any():
            break

        if temperature == 0.0:
            sampled = logits.argmax(dim=-1)
            confidence = F.softmax(logits, dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            sampled = torch.multinomial(probs.view(-1, cfg.vocab_size), 1).view(B, k)
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        # Update masked positions with the new sample; keep frozen ones intact.
        block_ids = torch.where(mask_flags, sampled, block_ids)

        # Decide which positions to freeze this step. Strategy:
        #   - keep proportional to (1 - step/n_steps): few frozen early, many late
        #   - among masked positions, freeze the highest-confidence first
        n_masked = int(mask_flags.sum().item())
        keep_frac = 1.0 - (step - 1) / n_steps
        n_to_freeze = max(1, int(round(keep_frac * n_masked)))

        masked_conf = confidence.clone()
        masked_conf[~mask_flags] = -1.0
        # Always freeze any position above early-exit threshold.
        early_exit = mask_flags & (confidence >= early_exit_tau)
        if int(early_exit.sum().item()) >= n_to_freeze:
            mask_flags = mask_flags & ~early_exit
        else:
            # Pick top-n_to_freeze masked positions by confidence to freeze.
            flat = masked_conf.view(-1)
            top_idx = torch.topk(flat, k=min(n_to_freeze, flat.numel()), largest=True).indices
            freeze = torch.zeros_like(flat, dtype=torch.bool)
            freeze[top_idx] = True
            freeze = freeze.view(B, k) & mask_flags & (confidence >= confidence_floor)
            mask_flags = mask_flags & ~freeze

    if mask_flags.any():
        # Final fallback: argmax remaining masked positions from the last logits.
        residual = final_logits.argmax(dim=-1)
        block_ids = torch.where(mask_flags, residual, block_ids)

    return block_ids, final_logits
