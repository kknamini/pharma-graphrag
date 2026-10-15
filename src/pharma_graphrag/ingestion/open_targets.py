"""Pull disease -> target -> drug data from the Open Targets GraphQL API and save it raw."""

import asyncio
import json

import httpx

from pharma_graphrag.config import DATA_RAW_DIR, DISEASES

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
MAX_CONCURRENT_REQUESTS = 5

# Open Targets' free-text search ranking is not stable: for "multiple sclerosis" it
# sometimes ranks a sub-concept (EFO_0803536, "multiple sclerosis symptom measurement")
# above the base disease term. Worse, the base "multiple sclerosis" entity itself
# (MONDO_0005301) is not resolvable via disease(efoId:...) at all in the current API -
# only MS subtypes are. We use relapsing-remitting MS (the most common subtype, ~85%
# of patients at diagnosis) as the Open Targets-side proxy for "multiple sclerosis".
# ClinicalTrials.gov matching still targets the broader "multiple sclerosis" string.
DISEASE_ID_OVERRIDES = {
    "multiple sclerosis": "MONDO_0005314",  # relapsing-remitting multiple sclerosis
}

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
    synonyms {
      relation
      terms
    }
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


async def resolve_efo_id(
    disease_name: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore
) -> str:
    if disease_name in DISEASE_ID_OVERRIDES:
        return DISEASE_ID_OVERRIDES[disease_name]

    async with semaphore:
        response = await client.post(
            API_URL, json={"query": SEARCH_QUERY, "variables": {"q": disease_name}}
        )
        response.raise_for_status()
        hits = response.json()["data"]["search"]["hits"]

    for hit in hits:
        if hit["name"].lower() == disease_name.lower():
            return hit["id"]
    return hits[0]["id"]


async def fetch_disease_data(
    efo_id: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        response = await client.post(
            API_URL, json={"query": DISEASE_QUERY, "variables": {"efoId": efo_id}}
        )
        response.raise_for_status()
        return response.json()


async def fetch_disease(
    disease_name: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore
) -> tuple[str, str, dict]:
    efo_id = await resolve_efo_id(disease_name, client, semaphore)
    data = await fetch_disease_data(efo_id, client, semaphore)
    return disease_name, efo_id, data


async def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(
                fetch_disease(disease_name, client, semaphore)
                for disease_name in DISEASES
            )
        )

    for disease_name, efo_id, data in results:
        out_path = DATA_RAW_DIR / f"open_targets_{efo_id}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"Saved {disease_name} ({efo_id}) -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
