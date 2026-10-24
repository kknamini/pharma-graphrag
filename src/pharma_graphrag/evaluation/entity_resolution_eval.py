"""Score entity-resolver output against the hand-labeled gold set.

Pure metrics logic, independent of how the predictions were produced - a single
EntityResolver.resolve() call per row, or a precomputed batch from a threshold sweep -
so it's reusable by both a single default-config check and the full grid sweep.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pharma_graphrag.ingestion.entity_resolution import ResolutionResult


@dataclass
class ERMetrics:
    overall: Counter
    by_tier: dict[str, Counter]
    errors: list[tuple[str, str, str, str | None, str]]  # outcome, tier, condition, predicted, gold

    def precision_recall_f1(self, counts: Counter | None = None) -> tuple[float, float, float]:
        counts = counts if counts is not None else self.overall
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        if precision == precision and recall == recall and (precision + recall):  # not NaN
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")
        return precision, recall, f1


def classify(predicted_id: str | None, gold_id: str) -> str:
    """Return one of TP / FP / FN / TN for a single row."""
    is_match = gold_id not in ("N/A", "", "UNSURE")
    if predicted_id is None:
        return "FN" if is_match else "TN"
    if not is_match:
        return "FP"
    return "TP" if predicted_id == gold_id else "FP"


def load_gold_rows(gold_path: Path) -> list[dict]:
    with open(gold_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def evaluate_results(
    gold_rows: list[dict], results_by_condition: dict[str, ResolutionResult]
) -> ERMetrics:
    """Score a set of resolver outputs against gold labels.

    `results_by_condition` maps condition_raw -> ResolutionResult. UNSURE gold rows
    are excluded from scoring, same as the original single-config evaluation.
    """
    overall: Counter = Counter()
    by_tier: dict[str, Counter] = defaultdict(Counter)
    errors = []

    for row in gold_rows:
        gold_id = row["gold_disease_id"].strip()
        if gold_id == "UNSURE":
            continue
        condition = row["condition_raw"]
        result = results_by_condition[condition]

        outcome = classify(result.disease_id, gold_id)
        overall[outcome] += 1
        by_tier[result.tier][outcome] += 1
        if outcome in ("FP", "FN"):
            errors.append((outcome, result.tier, condition, result.disease_id, gold_id))

    return ERMetrics(overall=overall, by_tier=dict(by_tier), errors=errors)
