"""
Quality evaluation: exact-match (GSM8K), letter-match (MMLU), ROUGE-L (CNN/DM).

Used in Phase 2 (baseline quality) and Phase 5 (speculative vs baseline comparison).
"""

import re
from rouge_score import rouge_scorer


# ── GSM8K: extract final numeric answer ──────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from GSM8K model output.
    Looks for 'Final Answer: <number>' first, then falls back to last number.
    """
    # Try structured pattern first
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*:\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in text
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def extract_gsm8k_gold(answer_text: str) -> str:
    """Extract the gold answer from GSM8K answer field ('#### <number>')."""
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    return answer_text.strip()


def gsm8k_exact_match(predicted: str, gold_answer: str) -> bool:
    pred = extract_gsm8k_answer(predicted)
    gold = extract_gsm8k_gold(gold_answer)
    if pred is None:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip() == gold.strip()


# ── MMLU: letter match ───────────────────────────────────────────────────────

_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_mmlu_letter(text: str) -> str | None:
    """Extract the first A/B/C/D letter from model output."""
    text_clean = text.strip()
    # Check if starts with a letter
    if text_clean and text_clean[0] in "ABCD":
        return text_clean[0]
    m = _LETTER_RE.search(text_clean)
    return m.group(1) if m else None


def mmlu_letter_match(predicted: str, gold_letter: str) -> bool:
    pred = extract_mmlu_letter(predicted)
    return pred is not None and pred == gold_letter


# ── CNN/DailyMail: ROUGE-L ──────────────────────────────────────────────────

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def cnndm_rouge_l(predicted: str, reference: str) -> float:
    """Compute ROUGE-L F1 score."""
    scores = _scorer.score(reference, predicted)
    return scores["rougeL"].fmeasure


# ── Unified evaluation ──────────────────────────────────────────────────────

def evaluate_sample(task: str, predicted: str, sample: dict) -> dict:
    """
    Evaluate a single sample, returning a quality dict.
    """
    if task == "gsm8k":
        correct = gsm8k_exact_match(predicted, sample["answer"])
        return {"metric": "exact_match", "score": 1.0 if correct else 0.0}
    elif task == "mmlu":
        correct = mmlu_letter_match(predicted, sample["answer_letter"])
        return {"metric": "letter_match", "score": 1.0 if correct else 0.0}
    elif task == "cnndm":
        score = cnndm_rouge_l(predicted, sample["reference"])
        return {"metric": "rouge_l", "score": round(score, 4)}
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_results(
    results: list[dict],
    data: dict[str, list[dict]],
) -> dict[str, float]:
    """
    Evaluate a full results list against the ground truth data.
    Returns { task: aggregate_score }.
    """
    # Build lookup: sample_id → sample dict
    lookup = {}
    for task_name, samples in data.items():
        for s in samples:
            lookup[s["sample_id"]] = s

    # Aggregate per task
    task_scores: dict[str, list[float]] = {}
    for row in results:
        sid = row["sample_id"]
        task = row["task"]
        if sid not in lookup:
            continue
        sample = lookup[sid]
        ev = evaluate_sample(task, row["output_text"], sample)
        task_scores.setdefault(task, []).append(ev["score"])

    # Compute means
    summary = {}
    for task, scores in task_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        summary[task] = round(avg * 100, 2)  # percentage
        metric_name = {"gsm8k": "exact_match", "mmlu": "letter_match", "cnndm": "rouge_l"}[task]
        print(f"  {task} ({metric_name}): {summary[task]:.2f}%  (n={len(scores)})")
    return summary
