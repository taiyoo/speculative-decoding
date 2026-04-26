"""
Position-conditional masked-diffusion noise schedule.

Standard masked-diffusion ELBO uses a single t per sequence and unmasks all
positions at the same rate. We bias later positions to stay masked longer:

    t_i = clip(t + lambda * (i / k_max), 0, T)

This is the "drift" — early positions converge first, matching the geometric
acceptance asymmetry alpha^i that speculative-decoding verification produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DriftSchedule:
    n_steps: int = 8           # T
    k_max: int = 16
    drift_lambda: float = 2.0  # 0.0 = symmetric (no drift); larger = later positions noisier longer

    def sample_t_per_position(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns (B, k_max) integer tensor of per-position diffusion steps.
        """
        # Per-sequence base step in [1, T]
        t_base = torch.randint(1, self.n_steps + 1, (batch_size, 1), device=device).float()
        i = torch.arange(self.k_max, device=device).float().unsqueeze(0)
        # Drift offset, scaled in step units
        offset = self.drift_lambda * (i / max(self.k_max - 1, 1)) * (self.n_steps / 4.0)
        t = (t_base + offset).clamp(min=1.0, max=float(self.n_steps)).long()
        return t

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Linear masking schedule: p_mask(t) = t / T. t in [1, T].
        Returns a tensor with the same shape as t.
        """
        return t.float() / float(self.n_steps)
