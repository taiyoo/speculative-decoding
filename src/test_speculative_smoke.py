"""
Correctness + smoke test for the rewritten KV-cache speculative loop.

Two checks:
  1. CORRECTNESS — In deterministic regime, the speculative decoder must
     emit *the same token stream* as the AR baseline. If KV-cache cropping
     or position indexing is off, this will diverge after the first reject.
  2. SPEEDUP    — End-to-end wall-clock for speculative on a real prompt
     should beat the AR baseline. Prints S so we can see the gain.

Usage (from project root):
    .venv/bin/python -m src.test_speculative_smoke  --quick
or with a tiny model pair (no GPU needed for the correctness check, just slow):
    .venv/bin/python -m src.test_speculative_smoke  --tiny
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `src/` importable when run as a script rather than a module.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sampling import sample_next_token  # noqa: E402
from speculative import speculative_decode_sample  # noqa: E402
from utils import set_seed  # noqa: E402


PROMPT = (
    "You are a careful math assistant. Solve the problem step by step "
    "and end with Final Answer: <number>.\n"
    "Question: A train travels 60 miles in 1.5 hours. What is its average speed in mph?"
)


def ar_decode(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[list[int], float]:
    """Plain AR greedy decoding with KV cache. Returns (token_ids, latency_s)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    eos_id = tokenizer.eos_token_id

    generated: list[int] = []
    gen_kwargs = {"do_sample": False}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values

        for _ in range(max_new_tokens):
            next_token, _ = sample_next_token(logits, gen_kwargs)
            tok = int(next_token.item())
            if eos_id is not None and tok == int(eos_id):
                break
            generated.append(tok)
            outputs = model(next_token, past_key_values=past, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return generated, elapsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Use Qwen2.5-1.5B as target / 0.5B as draft (fast).")
    p.add_argument("--tiny", action="store_true",
                   help="Use sshleifer/tiny-gpt2-style pair (CPU-friendly correctness check).")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--full", action="store_true",
                   help="Use the real 7B target / 0.5B draft from config.py.")
    args = p.parse_args()

    if args.tiny:
        target_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        draft_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    elif args.quick:
        target_id = "Qwen/Qwen2.5-1.5B-Instruct"
        draft_id = "Qwen/Qwen2.5-0.5B-Instruct"
    elif args.full:
        from config import TARGET_MODEL_ID, DRAFT_MODELS
        target_id = TARGET_MODEL_ID
        draft_id = DRAFT_MODELS["0.5B"]
    else:
        # Default: quick.
        target_id = "Qwen/Qwen2.5-1.5B-Instruct"
        draft_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"target = {target_id}")
    print(f"draft  = {draft_id}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading target...")
    target = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    ).eval()
    print("Loading draft...")
    draft = AutoModelForCausalLM.from_pretrained(
        draft_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    ).eval()

    # Warm-up.
    set_seed(42)
    _ = ar_decode(target, tok, PROMPT, max_new_tokens=4)

    # ---- 1) Correctness ----
    print("\n[1/2] Correctness: AR vs speculative (deterministic) must agree token-for-token")
    set_seed(42)
    ar_tokens, ar_t = ar_decode(target, tok, PROMPT, max_new_tokens=args.max_new)

    set_seed(42)
    spec = speculative_decode_sample(
        target, draft, tok, PROMPT, args.max_new, args.k,
        temperature=0.0, top_p=1.0,
    )
    spec_tokens = tok(spec["output_text"], add_special_tokens=False)["input_ids"]
    # Use raw decoded text comparison as the canonical check.
    ar_text = tok.decode(ar_tokens, skip_special_tokens=True)
    ok = spec["output_text"] == ar_text
    print(f"  AR   text: {ar_text!r}")
    print(f"  SPEC text: {spec['output_text']!r}")
    print(f"  match: {ok}")
    if not ok:
        # Diagnostics: find first divergence.
        i = 0
        while i < min(len(ar_text), len(spec["output_text"])) and ar_text[i] == spec["output_text"][i]:
            i += 1
        print(f"  diverges at char {i}: AR={ar_text[i:i+30]!r} | SPEC={spec['output_text'][i:i+30]!r}")
        return 1

    # ---- 2) Wall-clock speedup ----
    print(f"\n[2/2] Speedup: AR vs speculative (k={args.k})")
    print(f"  AR latency:   {ar_t*1000:.1f} ms  ({len(ar_tokens)} tokens)")
    print(f"  SPEC latency: {spec['latency_s']*1000:.1f} ms  ({spec['num_tokens']} tokens)")
    s = ar_t / spec["latency_s"] if spec["latency_s"] > 0 else 0.0
    print(f"  S = {s:.3f}x  | alpha = {spec['alpha']:.3f}  | B_eff = {spec['B_eff']:.2f}")
    print(f"  verify steps = {spec['n_verify_steps']}, proposed = {spec['total_proposed']}, accepted = {spec['total_accepted']}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
