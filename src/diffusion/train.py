"""
Distillation training for DriftDiffuser using the cached baseline outputs.

We treat each row of `baseline_deterministic.csv` / `baseline_stochastic.csv`
as a (prompt, target_continuation) pair. Tokenized with the target tokenizer
so vocab ids align with the verifier at inference time.

Training objective:
    For each window (ctx_chunk, block_chunk) of length (L_ctx, k_max):
      - sample per-position diffusion steps t_i via DriftSchedule
      - mask each position i with probability mask_prob(t_i)
      - cross-entropy loss on masked positions only, predicting the true token

Validation:
    On a held-out slice, measure
      - per-position accuracy (matches greedy 7B output at that position)
      - block accept rate using iterative_unmask + argmax
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import RESULTS_DIR, TARGET_MODEL_ID
from .drifter import DriftDiffuser, DrifterConfig
from .schedule import DriftSchedule
from .sampler import iterative_unmask


CHECKPOINT_DIR = RESULTS_DIR / "drifter_ckpt"


# ---------- Dataset ----------------------------------------------------------

@dataclass
class DistillExample:
    context_ids: torch.Tensor   # (L_ctx,)
    block_ids: torch.Tensor     # (k_max,)


class BaselineDistillDataset(Dataset):
    """
    Builds (context, block) windows from cached baseline outputs.

    Each row's prompt + output_text is tokenized once and chopped into
    overlapping windows so a single 256-token output yields multiple training
    examples.
    """

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        ctx_len: int = 512,
        k_max: int = 16,
        stride: int | None = None,
    ):
        self.ctx_len = ctx_len
        self.k_max = k_max
        self.stride = stride or k_max  # non-overlapping by default

        examples: list[DistillExample] = []
        for row in rows:
            prompt = row.get("prompt") or ""
            cont = row.get("output_text") or ""
            if not cont.strip():
                continue

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            cont_ids = tokenizer(cont, add_special_tokens=False)["input_ids"]
            full = prompt_ids + cont_ids
            # Walk through positions where we can slice a (ctx_len, k_max) window
            # such that the block lies inside the continuation.
            block_start_min = max(self.ctx_len, len(prompt_ids))
            block_start_max = len(full) - self.k_max
            if block_start_max < block_start_min:
                continue
            for start in range(block_start_min, block_start_max + 1, self.stride):
                ctx = full[start - self.ctx_len:start]
                block = full[start:start + self.k_max]
                if len(ctx) != self.ctx_len or len(block) != self.k_max:
                    continue
                examples.append(DistillExample(
                    context_ids=torch.tensor(ctx, dtype=torch.long),
                    block_ids=torch.tensor(block, dtype=torch.long),
                ))
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DistillExample:
        return self.examples[idx]


def _collate(batch: list[DistillExample]) -> dict:
    return {
        "context_ids": torch.stack([b.context_ids for b in batch], dim=0),
        "block_ids": torch.stack([b.block_ids for b in batch], dim=0),
    }


# ---------- Loss / step ------------------------------------------------------

def _drift_step(
    model: DriftDiffuser,
    schedule: DriftSchedule,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    ctx = batch["context_ids"].to(device, non_blocking=True)
    blk = batch["block_ids"].to(device, non_blocking=True)
    B, k = blk.shape

    t_per_pos = schedule.sample_t_per_position(B, device)[:, :k]
    mask_prob = schedule.mask_prob(t_per_pos)
    mask_flags = torch.rand_like(mask_prob) < mask_prob
    # Force at least one masked position per row so loss is non-degenerate.
    none_masked = ~mask_flags.any(dim=1)
    if none_masked.any():
        forced = torch.zeros_like(mask_flags)
        forced[none_masked, 0] = True
        mask_flags = mask_flags | forced

    # Where masked, feed mask vector (we use vocab_size as sentinel; the model
    # routes any value matching mask_flags to the mask_emb).
    feed_block = torch.where(mask_flags, torch.full_like(blk, model.cfg.vocab_size - 1), blk)

    logits = model(ctx, feed_block, mask_flags, t_per_pos)  # (B,k,V)
    loss = F.cross_entropy(
        logits.reshape(-1, model.cfg.vocab_size),
        blk.reshape(-1),
        reduction="none",
    ).reshape(B, k)
    loss = (loss * mask_flags.float()).sum() / mask_flags.float().sum().clamp(min=1.0)
    return loss


# ---------- Validation -------------------------------------------------------

@torch.inference_mode()
def validate(
    model: DriftDiffuser,
    loader: DataLoader,
    device: torch.device,
    n_steps: int = 4,
    max_batches: int = 16,
) -> dict:
    model.eval()
    pos_correct = torch.zeros(model.cfg.k_max, dtype=torch.long, device=device)
    pos_total = torch.zeros(model.cfg.k_max, dtype=torch.long, device=device)
    block_match = 0
    block_total = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        ctx = batch["context_ids"].to(device)
        gold_block = batch["block_ids"].to(device)
        B, k = gold_block.shape
        for b in range(B):
            pred, _ = iterative_unmask(model, ctx[b:b+1], k, n_steps=n_steps, temperature=0.0)
            match = (pred[0] == gold_block[b])
            pos_correct[:k] += match.long()
            pos_total[:k] += 1
            block_match += int(match.all().item())
            block_total += 1
    model.train()
    per_pos_acc = (pos_correct.float() / pos_total.float().clamp(min=1)).cpu().tolist()
    return {
        "per_position_accuracy": per_pos_acc,
        "alpha_proxy": float(np.mean(per_pos_acc)) if per_pos_acc else 0.0,
        "block_match_rate": block_match / max(block_total, 1),
        "n_samples": block_total,
    }


# ---------- Top-level train --------------------------------------------------

def _load_baseline_rows(regimes: list[str], data: dict) -> list[dict]:
    """
    Load baseline CSVs and re-attach the prompt for each row by joining on
    sample_id against the loaded `data` dict (keyed by task).
    """
    sid_to_prompt: dict[str, str] = {}
    for task, samples in data.items():
        for s in samples:
            sid_to_prompt[s["sample_id"]] = s["prompt"]

    rows: list[dict] = []
    for regime in regimes:
        csv = RESULTS_DIR / f"baseline_{regime}.csv"
        if not csv.exists():
            print(f"  skip (not found): {csv}")
            continue
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            sid = r["sample_id"]
            prompt = sid_to_prompt.get(sid, "")
            if not prompt:
                continue
            rows.append({"sample_id": sid, "prompt": prompt, "output_text": r["output_text"]})
    return rows


def train_drifter(
    data: dict,
    target_tokenizer,
    cfg: DrifterConfig | None = None,
    schedule: DriftSchedule | None = None,
    regimes: tuple[str, ...] = ("deterministic",),
    val_split: float = 0.05,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    n_epochs: int = 1,
    log_every: int = 50,
    val_every: int = 500,
    max_steps: int | None = None,
    grad_clip: float = 1.0,
    device: str | None = None,
    save_path: Path | None = None,
    resume_from: Path | None = None,
) -> dict:
    """
    Train the DriftDiffuser via distillation on cached baseline outputs.

    Returns a metrics dict including the final validation block-match rate.
    """
    cfg = cfg or DrifterConfig(vocab_size=len(target_tokenizer))
    schedule = schedule or DriftSchedule(n_steps=cfg.n_steps, k_max=cfg.k_max)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Building distillation dataset from {regimes} ...")
    rows = _load_baseline_rows(list(regimes), data)
    if not rows:
        raise RuntimeError("No baseline output rows found. Run Phase 2 first.")

    full_ds = BaselineDistillDataset(rows, target_tokenizer, ctx_len=cfg.max_ctx_len - cfg.k_max - 8, k_max=cfg.k_max)
    n_total = len(full_ds)
    if n_total < 16:
        raise RuntimeError(f"Distillation set too small ({n_total}). Need more baseline outputs.")
    n_val = max(8, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"  total windows: {n_total}  train: {n_train}  val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)

    model = DriftDiffuser(cfg).to(device_t)
    print(f"  drifter params: {model.num_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device_t.type == "cuda")

    start_step = 0
    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device_t)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        print(f"  resumed from {resume_from} at step {start_step}")

    history = {"train_loss": [], "val": []}
    step = start_step
    started = time.perf_counter()
    for epoch in range(n_epochs):
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_t.type, enabled=device_t.type == "cuda", dtype=torch.bfloat16):
                loss = _drift_step(model, schedule, batch, device_t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            step += 1
            if step % log_every == 0:
                elapsed = time.perf_counter() - started
                print(f"  step {step:6d} | loss {loss.item():.4f} | {step/max(elapsed,1):.2f} steps/s")
                history["train_loss"].append({"step": step, "loss": float(loss.item())})

            if step % val_every == 0:
                val_metrics = validate(model, val_loader, device_t, n_steps=max(1, cfg.n_steps // 2))
                print(f"  [val @ {step}] block_match={val_metrics['block_match_rate']:.3f} alpha~{val_metrics['alpha_proxy']:.3f}")
                history["val"].append({"step": step, **val_metrics})

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    final_val = validate(model, val_loader, device_t, n_steps=max(1, cfg.n_steps // 2), max_batches=64)
    history["final_val"] = final_val
    print(f"FINAL val: block_match={final_val['block_match_rate']:.3f}  alpha_proxy={final_val['alpha_proxy']:.3f}")

    save_path = Path(save_path or (CHECKPOINT_DIR / "drifter_latest.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "cfg": cfg.__dict__,
        "schedule": schedule.__dict__,
        "step": step,
        "history": history,
    }, save_path)
    print(f"Saved checkpoint -> {save_path}")

    return {"path": str(save_path), "step": step, "val": final_val, "history": history}


def load_drifter(checkpoint_path: Path | str, device: str | None = None) -> tuple[DriftDiffuser, DriftSchedule]:
    """Load a saved drifter for use by drift_speculative."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device_t)
    cfg = DrifterConfig(**ckpt["cfg"])
    schedule = DriftSchedule(**ckpt["schedule"])
    model = DriftDiffuser(cfg).to(device_t)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, schedule
