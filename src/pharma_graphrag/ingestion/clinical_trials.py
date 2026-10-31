"""Pull trials for each disease from the ClinicalTrials.gov API v2 and save them raw."""

import asyncio
import json

import httpx

from pharma_graphrag.config import DATA_RAW_DIR, DISEASES

API_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_CONCURRENT_REQUESTS = 5


async def fetch_trials(
    condition: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore
) -> tuple[str, dict]:
    async with semaphore:
        response = await client.get(
            API_URL, params={"query.cond": condition, "pageSize": 100}
        )
        response.raise_for_status()
        return condition, response.json()


async def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(fetch_trials(condition, client, semaphore) for condition in DISEASES)
        )

    for condition, data in results:
        slug = condition.lower().replace(" ", "_").replace("'", "")
        out_path = DATA_RAW_DIR / f"clinical_trials_{slug}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"Saved trials for {condition} -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
