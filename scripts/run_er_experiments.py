"""Grid-sweep the entity resolver's thresholds and embedding models against the gold
set, to see how ER performance moves across the parameter space before committing to
final defaults.

For each embedding model: encode once (the expensive part - a model forward pass over
every condition string and every disease name/synonym), then compute the best
embedding match per condition once. Every (string_sim_threshold,
embedding_sim_threshold) pair is then decided cheaply against those cached scores, so
the sweep doesn't re-run model inference per threshold combination.

Output: results/er_sweep_<timestamp>.csv, one row per
(embedding_model, string_sim_threshold, embedding_sim_threshold) combination with
overall and per-tier precision/recall/F1. Narrowing down to final defaults from this
is a separate, later step - this script's only job is producing the numbers.
"""

import csv
import itertools
from collections import Counter
from datetime import datetime, timezone

from pharma_graphrag.config import PROJECT_ROOT
from pharma_graphrag.evaluation.entity_resolution_eval import evaluate_results, load_gold_rows
from pharma_graphrag.ingestion.embedding_models import EMBEDDING_MODELS, load_embedding_model
from pharma_graphrag.ingestion.entity_resolution import (
    CandidateEmbeddingIndex,
    ConditionScores,
    decide,
    load_disease_refs,
    score_condition,
)

GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "gold_er_set.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

STRING_SIM_THRESHOLDS = [75, 80, 85, 90, 95]
EMBEDDING_SIM_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
EMBEDDING_MODEL_NAMES = list(EMBEDDING_MODELS.keys())

TIER_FIELDNAMES = ["exact_synonym", "string_sim", "embedding"]
FIELDNAMES = [
    "embedding_model",
    "string_sim_threshold",
    "embedding_sim_threshold",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "tp",
    "fp",
    "fn",
    "tn",
    *[f"{tier}_precision" for tier in TIER_FIELDNAMES],
    *[f"{tier}_n" for tier in TIER_FIELDNAMES],
]


def round_or_blank(value: float) -> float | str:
    return round(value, 4) if value == value else ""  # NaN != NaN


def main():
    refs = load_disease_refs()
    gold_rows = load_gold_rows(GOLD_PATH)
    conditions = [row["condition_raw"] for row in gold_rows]

    # Threshold-independent scores (exact + string match): computed once, total.
    base_scores = {c: score_condition(c, refs) for c in conditions}

    rows_out = []
    for model_name in EMBEDDING_MODEL_NAMES:
        print(f"Encoding with {model_name}...")
        model = load_embedding_model(model_name)
        index = CandidateEmbeddingIndex(model, refs)
        embedding_matches = dict(zip(conditions, index.best_match_batch(conditions)))

        scores_for_model = {
            c: ConditionScores(
                condition_raw=c,
                exact_match=base_scores[c].exact_match,
                string_match=base_scores[c].string_match,
                embedding_match=embedding_matches[c],
            )
            for c in conditions
        }

        for string_thresh, embedding_thresh in itertools.product(
            STRING_SIM_THRESHOLDS, EMBEDDING_SIM_THRESHOLDS
        ):
            results_by_condition = {
                c: decide(scores_for_model[c], string_thresh, embedding_thresh) for c in conditions
            }
            metrics = evaluate_results(gold_rows, results_by_condition)
            precision, recall, f1 = metrics.precision_recall_f1()

            row = {
                "embedding_model": model_name,
                "string_sim_threshold": string_thresh,
                "embedding_sim_threshold": embedding_thresh,
                "overall_precision": round_or_blank(precision),
                "overall_recall": round_or_blank(recall),
                "overall_f1": round_or_blank(f1),
                "tp": metrics.overall["TP"],
                "fp": metrics.overall["FP"],
                "fn": metrics.overall["FN"],
                "tn": metrics.overall["TN"],
            }
            for tier in TIER_FIELDNAMES:
                tier_counts = metrics.by_tier.get(tier, Counter())
                tier_total = tier_counts["TP"] + tier_counts["FP"]
                tier_precision = tier_counts["TP"] / tier_total if tier_total else float("nan")
                row[f"{tier}_precision"] = round_or_blank(tier_precision)
                row[f"{tier}_n"] = sum(tier_counts.values())
            rows_out.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"er_sweep_{timestamp}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} configurations to {out_path}")


if __name__ == "__main__":
    main()
