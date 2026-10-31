from pathlib import Path

# Folder locations
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Disease names
DISEASES = [
    "rheumatoid arthritis",
    "psoriasis",
    "Crohn's disease",
    "multiple sclerosis",
    "Parkinson's disease",
]

# API URLs and request settings
OPEN_TARGETS_API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_CONCURRENT_REQUESTS = 5

# Disease ID overrides for specific diseases
DISEASE_ID_OVERRIDES = {
    "multiple sclerosis": "MONDO_0005314",
}

# Open Targets GraphQL queries
OPEN_TARGETS_SEARCH_QUERY = """
query search($q: String!) {
    search(queryString: $q, entityNames: ["disease"]) {
        hits { id name entity }
    }
}
"""

OPEN_TARGETS_DISEASE_QUERY = """
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

# Entity resolution thresholds and settings
STRING_SIM_THRESHOLD = 88.0
EMBEDDING_SIM_THRESHOLD = 0.60
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EXACT_SYNONYM_RELATIONS = {"hasExactSynonym"}

ER_LABELING_SHEET_FIELDNAMES = [
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
