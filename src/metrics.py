"""
Macro-averaged F_beta scoring, exactly as specified in the challenge PDF:

    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

computed PER Source-1 entity, then averaged across all Source-1 entities.
A singleton (no true matches) scores 1.0 if predicted empty, 0.0 if any
match is (incorrectly) predicted for it.
"""
from __future__ import annotations

from typing import Iterable


def f_beta_for_entity(predicted: set[str], truth: set[str], beta: float = 0.5) -> float:
    if not truth:
        # singleton: full credit for predicting empty, zero otherwise
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(
    predictions: dict[str, set[str]],
    truths: dict[str, set[str]],
    all_s1_ids: Iterable[str],
    beta: float = 0.5,
) -> float:
    """Macro-average F_beta over every S1 id in `all_s1_ids` (ids missing
    from either dict are treated as empty predictions / empty truth)."""
    scores = []
    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, set())
        truth = truths.get(s1_id, set())
        scores.append(f_beta_for_entity(pred, truth, beta=beta))
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k_blocking(
    candidates: dict[str, list[str]], truths: dict[str, set[str]], all_s1_ids: Iterable[str]
) -> float:
    """Blocking-quality metric: fraction of true matches that appear
    somewhere in the candidate list (the 'recall ceiling' the PDF asks
    you to report for candidate_pairs.tsv)."""
    total_true, total_found = 0, 0
    for s1_id in all_s1_ids:
        truth = truths.get(s1_id, set())
        if not truth:
            continue
        cand_set = set(candidates.get(s1_id, []))
        total_true += len(truth)
        total_found += len(truth & cand_set)
    return total_found / total_true if total_true else 1.0
