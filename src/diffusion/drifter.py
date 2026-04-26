"""
DriftDiffuser model: small bidirectional transformer that, given a context
prefix and k masked block positions, predicts logits for all k positions in
parallel after a few denoising steps.

Design choices:
- Self-contained PyTorch module (no HF dependency for the model itself).
- Vocab size = target tokenizer vocab (default 152064 for Qwen2.5-7B). The
  drifter shares the target vocab so verify-time logits align directly.
- A learned MASK embedding sits alongside the token embedding table; mask
  positions in the input use this vector instead of a token id.
- Time embedding is added to mask positions only.
- Bidirectional attention over (context + block); causal masking would defeat
  the parallel-prediction property.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DrifterConfig:
    vocab_size: int = 152064          # Qwen2.5-7B target vocab
    hidden: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 4
    max_ctx_len: int = 768            # 512 ctx + room for block + slack
    k_max: int = 16
    n_steps: int = 8                  # diffusion steps T
    dropout: float = 0.0
    tie_embeddings: bool = True


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class _Block(nn.Module):
    def __init__(self, cfg: DrifterConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden // cfg.n_heads
        self.qkv = nn.Linear(cfg.hidden, cfg.hidden * 3, bias=False)
        self.out = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.norm1 = _RMSNorm(cfg.hidden)
        self.norm2 = _RMSNorm(cfg.hidden)
        ffn_hidden = cfg.hidden * cfg.ffn_mult
        self.fc1 = nn.Linear(cfg.hidden, ffn_hidden, bias=False)
        self.fc2 = nn.Linear(ffn_hidden, cfg.hidden, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Bidirectional (no mask)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self._attn(self.norm1(x)))
        x = x + self.drop(self.fc2(F.silu(self.fc1(self.norm2(x)))))
        return x


class DriftDiffuser(nn.Module):
    """
    Inputs:
        context_ids : (B, L_ctx) long, token ids of accepted prefix
        block_ids   : (B, k) long, current token ids in the diffusion block.
                      Positions still masked use `mask_id` (a sentinel; here we
                      use vocab_size so it's out-of-range for the embedding
                      table and gets routed to the mask embedding).
        mask_flags  : (B, k) bool — True means position is currently masked.
        t_per_pos   : (B, k) long, per-position diffusion step in [1, n_steps].

    Output:
        logits      : (B, k, vocab_size) — predictions for every block position.
    """

    def __init__(self, cfg: DrifterConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.mask_emb = nn.Parameter(torch.zeros(cfg.hidden))
        self.pos_emb = nn.Embedding(cfg.max_ctx_len + cfg.k_max, cfg.hidden)
        self.time_emb = nn.Embedding(cfg.n_steps + 1, cfg.hidden)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = _RMSNorm(cfg.hidden)
        if cfg.tie_embeddings:
            self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight
        else:
            self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.mask_emb, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb.weight, std=0.02)

    def forward(
        self,
        context_ids: torch.Tensor,
        block_ids: torch.Tensor,
        mask_flags: torch.Tensor,
        t_per_pos: torch.Tensor,
    ) -> torch.Tensor:
        B, L_ctx = context_ids.shape
        k = block_ids.shape[1]
        assert k <= self.cfg.k_max
        assert L_ctx + k <= self.pos_emb.num_embeddings

        # Embed context tokens straightforwardly.
        ctx_e = self.tok_emb(context_ids)  # (B, L_ctx, D)

        # Block: token embedding for unmasked, mask embedding for masked.
        safe_ids = block_ids.clamp(min=0, max=self.cfg.vocab_size - 1)
        blk_e = self.tok_emb(safe_ids)  # (B, k, D)
        mask_vec = self.mask_emb.view(1, 1, -1).expand(B, k, -1)
        blk_e = torch.where(mask_flags.unsqueeze(-1), mask_vec, blk_e)

        # Add time embedding only on block positions.
        blk_e = blk_e + self.time_emb(t_per_pos)

        # Concatenate context + block, add positional embeddings.
        x = torch.cat([ctx_e, blk_e], dim=1)  # (B, L_ctx + k, D)
        positions = torch.arange(L_ctx + k, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        block_h = x[:, L_ctx:, :]
        logits = self.head(block_h)  # (B, k, V)
        return logits

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
