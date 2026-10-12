from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DISEASES = [
    "rheumatoid arthritis",
    "psoriasis",
    "Crohn's disease",
    "multiple sclerosis",
    "Parkinson's disease",
]
