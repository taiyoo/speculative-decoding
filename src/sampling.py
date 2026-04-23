"""
Shared sampling utilities for baseline and speculative decoding.
"""

from __future__ import annotations

import torch


def probs_from_logits(logits: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
    """
    Convert logits to a probability distribution under the active sampling policy.

    For stochastic decoding this applies temperature + top-p filtering.
    For deterministic decoding this returns a one-hot distribution at argmax.
    """
    if not gen_kwargs.get("do_sample", False):
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    temperature = max(float(gen_kwargs.get("temperature", 1.0)), 1e-5)
    top_p = float(gen_kwargs.get("top_p", 1.0))
    scaled_logits = logits.float() / temperature

    if top_p >= 1.0:
        return torch.softmax(scaled_logits, dim=-1)

    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep the first token above threshold so top-p always has at least one token.
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    filtered_sorted_probs = torch.softmax(sorted_logits, dim=-1)
    probs = torch.zeros_like(filtered_sorted_probs)
    probs.scatter_(1, sorted_indices, filtered_sorted_probs)
    return probs


def sample_next_token(logits: torch.Tensor, gen_kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample or select the next token under the shared sampling policy."""
    probs = probs_from_logits(logits, gen_kwargs)
    if gen_kwargs.get("do_sample", False):
        next_token = torch.multinomial(probs, 1)
    else:
        next_token = probs.argmax(dim=-1, keepdim=True)
    return next_token, probs
