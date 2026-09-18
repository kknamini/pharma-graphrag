"""Pull disease -> target -> drug data from the Open Targets GraphQL API and save it raw."""

import json

import requests

from pharma_graphrag.config import DATA_RAW_DIR, DISEASES

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

SEARCH_QUERY = """
query search($q: String!) {
  search(queryString: $q, entityNames: ["disease"]) {
    hits { id name entity }
  }
}
"""

DISEASE_QUERY = """
query diseaseInfo($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: { size: 50, index: 0 }) {
      count
      rows {
        score
        target { id approvedSymbol }
      }
    }
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug { id name drugType maximumClinicalStage }
      }
    }
  }
}
"""


def resolve_efo_id(disease_name: str) -> str:
    resp = requests.post(API_URL, json={"query": SEARCH_QUERY, "variables": {"q": disease_name}})
    resp.raise_for_status()
    hits = resp.json()["data"]["search"]["hits"]
    return hits[0]["id"]


def fetch_disease_data(efo_id: str) -> dict:
    resp = requests.post(API_URL, json={"query": DISEASE_QUERY, "variables": {"efoId": efo_id}})
    resp.raise_for_status()
    return resp.json()


def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for disease_name in DISEASES:
        efo_id = resolve_efo_id(disease_name)
        data = fetch_disease_data(efo_id)

        out_path = DATA_RAW_DIR / f"open_targets_{efo_id}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"Saved {disease_name} ({efo_id}) -> {out_path}")


if __name__ == "__main__":
    main()
