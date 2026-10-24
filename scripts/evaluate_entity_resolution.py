"""Evaluate the entity resolver against the hand-labeled gold set.

Re-runs the current resolver fresh against every condition string in the gold set
(rather than trusting the predicted_* columns baked into the sheet when it was first
generated) and reports precision/recall/F1 overall and broken down by tier, so
threshold/scoring changes can be measured against real numbers instead of guesses.
"""

import csv
from collections import Counter, defaultdict

from pharma_graphrag.config import DATA_RAW_DIR
from pharma_graphrag.ingestion.entity_resolution import EntityResolver, load_disease_refs

GOLD_PATH = DATA_RAW_DIR.parent / "gold" / "gold_er_set.csv"


def load_gold_rows():
    with open(GOLD_PATH, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classify(predicted_id: str | None, gold_id: str):
    """Return one of TP / FP / FN / TN for a single row."""
    is_match = gold_id not in ("N/A", "", "UNSURE")
    if predicted_id is None:
        return "FN" if is_match else "TN"
    if not is_match:
        return "FP"
    return "TP" if predicted_id == gold_id else "FP"


def main():
    refs = load_disease_refs()
    resolver = EntityResolver(refs)
    gold_rows = load_gold_rows()

    unsure = [r for r in gold_rows if r["gold_disease_id"].strip() == "UNSURE"]
    scored_rows = [r for r in gold_rows if r["gold_disease_id"].strip() != "UNSURE"]

    overall = Counter()
    by_tier = defaultdict(Counter)
    errors = []

    for row in scored_rows:
        condition = row["condition_raw"]
        gold_id = row["gold_disease_id"].strip()
        result = resolver.resolve(condition, use_embedding_fallback=True)

        outcome = classify(result.disease_id, gold_id)
        overall[outcome] += 1
        by_tier[result.tier][outcome] += 1

        if outcome in ("FP", "FN"):
            errors.append((outcome, result.tier, condition, result.disease_id, gold_id))

    def prf(counts: Counter):
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        return precision, recall, f1

    print(f"Gold rows scored: {len(scored_rows)}  (excluded {len(unsure)} UNSURE rows)")
    print()
    print("=== Overall ===")
    p, r, f1 = prf(overall)
    print(f"TP={overall['TP']} FP={overall['FP']} FN={overall['FN']} TN={overall['TN']}")
    print(f"precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")
    print()

    print("=== By tier (tier = what the resolver actually used for that row) ===")
    for tier in ["exact_synonym", "string_sim", "embedding", "unresolved"]:
        counts = by_tier[tier]
        if not sum(counts.values()):
            continue
        p, r, f1 = prf(counts)
        print(f"{tier:15} n={sum(counts.values()):4} TP={counts['TP']:3} FP={counts['FP']:3} "
              f"FN={counts['FN']:3} TN={counts['TN']:3}  precision={p:.3f}")
    print()

    print(f"=== Errors ({len(errors)} total) ===")
    for outcome, tier, condition, predicted, gold in errors:
        pred_str = predicted if predicted else "unresolved"
        print(f"  [{outcome}] tier={tier:13} {condition!r}")
        print(f"        predicted={pred_str}  gold={gold}")


if __name__ == "__main__":
    main()
