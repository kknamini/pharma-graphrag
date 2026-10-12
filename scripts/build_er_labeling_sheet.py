"""Build a CSV for hand-labeling entity resolution.

Collects every unique condition string across the cached ClinicalTrials.gov trials,
runs the current EntityResolver over each one, and writes a CSV with the resolver's
guess alongside blank columns for a human-assigned gold label.

Run after ingestion/open_targets.py and ingestion/clinical_trials.py have populated
data/raw/. Output: data/processed/er_labeling_sheet.csv
"""

import csv
import json

from pharma_graphrag.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from pharma_graphrag.ingestion.entity_resolution import EntityResolver, load_disease_refs

FIELDNAMES = [
    "condition_raw",
    "frequency",
    "source_query_buckets",
    "example_nct_id",
    "example_brief_title",
    "predicted_disease_id",
    "predicted_disease_name",
    "predicted_tier",
    "predicted_score",
    "gold_disease_id",
    "gold_notes",
]


def iter_trial_conditions():
    for path in sorted(DATA_RAW_DIR.glob("clinical_trials_*.json")):
        bucket = path.stem.removeprefix("clinical_trials_")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for study in payload.get("studies", []):
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            conditions = protocol.get("conditionsModule", {}).get("conditions", [])
            for condition in conditions:
                yield condition, bucket, identification.get("nctId", ""), identification.get("briefTitle", "")


def main():
    refs = load_disease_refs()
    resolver = EntityResolver(refs)

    unique_conditions: dict[str, dict] = {}
    for condition, bucket, nct_id, brief_title in iter_trial_conditions():
        entry = unique_conditions.setdefault(
            condition,
            {"frequency": 0, "buckets": set(), "example_nct_id": nct_id, "example_brief_title": brief_title},
        )
        entry["frequency"] += 1
        entry["buckets"].add(bucket)

    rows = []
    for condition, entry in unique_conditions.items():
        result = resolver.resolve(condition)
        rows.append(
            {
                "condition_raw": condition,
                "frequency": entry["frequency"],
                "source_query_buckets": "|".join(sorted(entry["buckets"])),
                "example_nct_id": entry["example_nct_id"],
                "example_brief_title": entry["example_brief_title"],
                "predicted_disease_id": result.disease_id or "",
                "predicted_disease_name": result.disease_name or "",
                "predicted_tier": result.tier,
                "predicted_score": f"{result.score:.1f}" if result.score is not None else "",
                "gold_disease_id": "",
                "gold_notes": "",
            }
        )
    rows.sort(key=lambda r: (-r["frequency"], r["condition_raw"].lower()))

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "er_labeling_sheet.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} unique condition strings to {out_path}")
    print("\nDisease ID legend:")
    for ref in refs:
        print(f"  {ref.disease_id}  ->  {ref.name}")


if __name__ == "__main__":
    main()
