"""Scoring sampled GSM8K answers, and pass@k."""

from __future__ import annotations

import math
import re


def gold_answer(text: str) -> str:
    """GSM8K puts the final answer after '####'."""
    return text.split("####")[-1].strip().replace(",", "")


def extract(text: str) -> str | None:
    """The model's final answer: the last \\boxed{}, else the last number."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = boxed[-1] if boxed else None
    if cand is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").replace("$", "").strip().rstrip(".")
    m = re.search(r"-?\d+\.?\d*", cand)
    return m.group(0) if m else cand


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from Chen et al. 2021: n samples, c of them correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
