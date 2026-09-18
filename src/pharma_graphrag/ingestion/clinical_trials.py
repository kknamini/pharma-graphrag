"""Pull trials for each disease from the ClinicalTrials.gov API v2 and save them raw."""

import json

import requests

from pharma_graphrag.config import DATA_RAW_DIR, DISEASES

API_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials(condition: str) -> dict:
    resp = requests.get(API_URL, params={"query.cond": condition, "pageSize": 100})
    resp.raise_for_status()
    return resp.json()


def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for condition in DISEASES:
        data = fetch_trials(condition)

        slug = condition.lower().replace(" ", "_").replace("'", "")
        out_path = DATA_RAW_DIR / f"clinical_trials_{slug}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"Saved trials for {condition} -> {out_path}")


if __name__ == "__main__":
    main()
