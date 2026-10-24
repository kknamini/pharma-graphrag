"""Evaluate the entity resolver's current default configuration against the gold set.

Re-runs the resolver fresh against every condition string in the gold set (rather
than trusting the predicted_* columns baked into the sheet when it was first
generated) and reports precision/recall/F1 overall and broken down by tier.

For comparing many threshold/embedding-model configurations at once, use
scripts/run_er_experiments.py instead - this script is a quick single-config check.
"""

from pharma_graphrag.config import PROJECT_ROOT
from pharma_graphrag.evaluation.entity_resolution_eval import evaluate_results, load_gold_rows
from pharma_graphrag.ingestion.entity_resolution import EntityResolver, load_disease_refs

GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "gold_er_set.csv"


def main():
    refs = load_disease_refs()
    resolver = EntityResolver(refs)
    gold_rows = load_gold_rows(GOLD_PATH)

    unsure_count = sum(1 for r in gold_rows if r["gold_disease_id"].strip() == "UNSURE")
    results_by_condition = {
        row["condition_raw"]: resolver.resolve(row["condition_raw"], use_embedding_fallback=True)
        for row in gold_rows
    }
    metrics = evaluate_results(gold_rows, results_by_condition)

    print(f"Gold rows scored: {len(gold_rows) - unsure_count}  (excluded {unsure_count} UNSURE rows)")
    print()
    print("=== Overall ===")
    precision, recall, f1 = metrics.precision_recall_f1()
    o = metrics.overall
    print(f"TP={o['TP']} FP={o['FP']} FN={o['FN']} TN={o['TN']}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    print()

    print("=== By tier (tier = what the resolver actually used for that row) ===")
    for tier in ["exact_synonym", "string_sim", "embedding", "unresolved"]:
        counts = metrics.by_tier.get(tier)
        if not counts:
            continue
        p, r, f1 = metrics.precision_recall_f1(counts)
        n = sum(counts.values())
        print(
            f"{tier:15} n={n:4} TP={counts['TP']:3} FP={counts['FP']:3} "
            f"FN={counts['FN']:3} TN={counts['TN']:3}  precision={p:.3f}"
        )
    print()

    print(f"=== Errors ({len(metrics.errors)} total) ===")
    for outcome, tier, condition, predicted, gold in metrics.errors:
        pred_str = predicted if predicted else "unresolved"
        print(f"  [{outcome}] tier={tier:13} {condition!r}")
        print(f"        predicted={pred_str}  gold={gold}")


if __name__ == "__main__":
    main()
