"""
Data loader: load, sample, and freeze evaluation subsets for GSM8K, MMLU, CNN/DM.

Phase 1 — all sampling uses seed=42; manifests are written to manifests/.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from config import DATASETS, MANIFESTS_DIR, PROMPTS, SEED
from utils import set_seed


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_gsm8k(cfg: dict) -> list[dict]:
    """Load GSM8K test split, sample n, return list of dicts."""
    ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["split"])
    indices = list(range(len(ds)))
    random.shuffle(indices)
    sampled = indices[: cfg["n_samples"]]
    sampled.sort()  # deterministic order after shuffle
    rows = []
    for idx in sampled:
        item = ds[idx]
        rows.append({
            "sample_id": f"gsm8k_{idx}",
            "index": idx,
            "task": "gsm8k",
            "question": item["question"],
            "answer": item["answer"],
            "prompt": PROMPTS["gsm8k"].format(question=item["question"]),
        })
    return rows


def _load_mmlu(cfg: dict) -> list[dict]:
    """
    Load MMLU test split for 5 chosen subjects, sample 100 per subject.
    """
    rows = []
    for subject in cfg["subjects"]:
        ds = load_dataset(cfg["hf_name"], subject, split=cfg["split"])
        indices = list(range(len(ds)))
        random.shuffle(indices)
        sampled = indices[: cfg["samples_per_subject"]]
        sampled.sort()
        for idx in sampled:
            item = ds[idx]
            choices = item["choices"]
            label_idx = item["answer"]
            rows.append({
                "sample_id": f"mmlu_{subject}_{idx}",
                "index": idx,
                "task": "mmlu",
                "subject": subject,
                "question": item["question"],
                "choices": choices,
                "answer_index": label_idx,
                "answer_letter": "ABCD"[label_idx],
                "prompt": PROMPTS["mmlu"].format(
                    question=item["question"],
                    A=choices[0],
                    B=choices[1],
                    C=choices[2],
                    D=choices[3],
                ),
            })
    return rows


def _load_cnndm(cfg: dict) -> list[dict]:
    """Load CNN/DailyMail test split, sample n."""
    ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["split"])
    indices = list(range(len(ds)))
    random.shuffle(indices)
    sampled = indices[: cfg["n_samples"]]
    sampled.sort()
    rows = []
    for idx in sampled:
        item = ds[idx]
        rows.append({
            "sample_id": f"cnndm_{idx}",
            "index": idx,
            "task": "cnndm",
            "article": item["article"],
            "reference": item["highlights"],
            "prompt": PROMPTS["cnndm"].format(article=item["article"]),
        })
    return rows


# ── Public API ────────────────────────────────────────────────────────────────

LOADERS = {
    "gsm8k": _load_gsm8k,
    "mmlu": _load_mmlu,
    "cnndm": _load_cnndm,
}


def load_all_datasets() -> dict[str, list[dict]]:
    """
    Load and sample all three datasets with seed=42.
    Returns { task_name: [sample_dicts] }.
    """
    set_seed(SEED)
    data = {}
    for task_name, loader_fn in LOADERS.items():
        cfg = DATASETS[task_name]
        data[task_name] = loader_fn(cfg)
        print(f"  [{task_name}] loaded {len(data[task_name])} samples")
    return data


def freeze_manifests(data: dict[str, list[dict]]) -> None:
    """
    Save sample-ID manifests to manifests/*.json so every subsequent run
    uses identical data without re-sampling.
    """
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    for task_name, rows in data.items():
        manifest = [r["sample_id"] for r in rows]
        path = MANIFESTS_DIR / f"{task_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  [{task_name}] manifest frozen → {path}  ({len(manifest)} ids)")


def save_full_data(data: dict[str, list[dict]]) -> None:
    """
    Save the full sampled data (prompts + references) so later phases don't
    need to re-download datasets.
    """
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    for task_name, rows in data.items():
        path = MANIFESTS_DIR / f"{task_name}_data.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"  [{task_name}] full data saved → {path}")


def load_from_manifests() -> dict[str, list[dict]]:
    """
    Load previously-frozen data from manifests/*_data.json.
    Use this in Phases 2-6 so you don't re-download.
    """
    data = {}
    for task_name in LOADERS:
        path = MANIFESTS_DIR / f"{task_name}_data.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest data not found: {path}. Run Phase 1 first."
            )
        with open(path, "r", encoding="utf-8") as f:
            data[task_name] = json.load(f)
        print(f"  [{task_name}] loaded {len(data[task_name])} samples from manifest")
    return data


def verify_tokenizer_compatibility() -> bool:
    """
    Verify all three Qwen 2.5 models share the same tokenizer vocabulary.
    Returns True if compatible.
    """
    from transformers import AutoTokenizer
    from config import TARGET_MODEL_ID, DRAFT_MODELS

    model_ids = [TARGET_MODEL_ID] + list(DRAFT_MODELS.values())
    vocabs = {}
    for mid in model_ids:
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        vocabs[mid] = tok.vocab_size
        print(f"  {mid}: vocab_size = {tok.vocab_size}")

    sizes = list(vocabs.values())
    if len(set(sizes)) == 1:
        print(f"  ✓ All tokenizers share vocab_size = {sizes[0]}")
        return True
    else:
        print(f"  ✗ Tokenizer mismatch: {vocabs}")
        return False


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Data preparation")
    parser.add_argument("--verify-tokenizer", action="store_true",
                        help="Also verify tokenizer compatibility (requires model download)")
    parser.add_argument("--smoke-test", type=int, default=0,
                        help="Run N-sample smoke test with target model")
    args = parser.parse_args()

    print("Phase 1: Loading and sampling datasets…")
    data = load_all_datasets()

    print("\nFreezing manifests…")
    freeze_manifests(data)

    print("\nSaving full data…")
    save_full_data(data)

    if args.verify_tokenizer:
        print("\nVerifying tokenizer compatibility…")
        verify_tokenizer_compatibility()

    if args.smoke_test > 0:
        print(f"\nRunning {args.smoke_test}-sample smoke test…")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from config import TARGET_MODEL_ID
        from utils import GPUTimer

        tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        gsm_samples = data["gsm8k"][: args.smoke_test]
        for i, sample in enumerate(gsm_samples):
            inputs = tok(sample["prompt"], return_tensors="pt").to(model.device)
            timer = GPUTimer()
            with timer:
                out = model.generate(
                    **inputs,
                    max_new_tokens=DATASETS["gsm8k"]["max_new_tokens"],
                    do_sample=False,
                )
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  [{i+1}/{args.smoke_test}] {timer.elapsed_ms:.0f} ms, "
                  f"{out.shape[1] - inputs['input_ids'].shape[1]} tokens, "
                  f"output[:80]: {text[:80]!r}")

    print("\n✓ Phase 1 complete.")
