"""Resolve free-text ClinicalTrials.gov condition strings to Open Targets disease nodes.

Tiered cascade, each tier only tried if the previous one didn't produce a match:
  1. exact_synonym - normalized string matches the disease name or an EFO "exact
     synonym" (e.g. "RA" -> rheumatoid arthritis), via a dictionary built from the
     Open Targets disease data cached in data/raw/.
  2. string_sim    - fuzzy string similarity (token-set ratio + Jaro-Winkler) against
     the same name/synonym pool, for spelling variants and reordered qualifiers
     (e.g. "Crohn's Disease, Moderate to Severe").
  3. embedding     - sentence-embedding cosine similarity, for true synonyms with no
     string overlap (e.g. abbreviations or clinical terminology swaps the first two
     tiers can't catch). Uses a generic sentence-transformers model by default; a
     biomedical-specific model (e.g. SapBERT) would be the upgrade with more time.

Every result carries the tier and score that produced it, so match quality can be
audited and thresholds calibrated against a hand-labeled gold set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from pharma_graphrag.config import (
    DATA_RAW_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_SIM_THRESHOLD,
    EXACT_SYNONYM_RELATIONS,
    STRING_SIM_THRESHOLD,
)

# Only "exact" synonyms are treated as true equivalence for tier 1; "related" /
# "narrow" / "broad" synonyms are not reliably interchangeable with the disease itself.


@dataclass
class DiseaseRef:
    disease_id: str
    name: str
    exact_synonyms: list[str] = field(default_factory=list)


@dataclass
class ResolutionResult:
    condition_raw: str
    disease_id: str | None
    disease_name: str | None
    tier: str  # "exact_synonym" | "string_sim" | "embedding" | "unresolved"
    score: float | None


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_disease_refs(raw_dir: Path = DATA_RAW_DIR) -> list[DiseaseRef]:
    refs = []
    for path in sorted(raw_dir.glob("open_targets_*.json")):
        disease = json.loads(path.read_text(encoding="utf-8"))["data"]["disease"]
        if disease is None:
            print(f"Skipping {path.name}: Open Targets returned no disease for this ID")
            continue
        exact_terms = [
            term
            for group in disease.get("synonyms") or []
            if group["relation"] in EXACT_SYNONYM_RELATIONS
            for term in group["terms"]
        ]
        refs.append(DiseaseRef(disease["id"], disease["name"], exact_terms))
    return refs


def _string_similarity(a: str, b: str) -> float:
    token_score = fuzz.token_set_ratio(a, b)
    jaro_score = JaroWinkler.normalized_similarity(a, b) * 100
    return max(token_score, jaro_score)


class EntityResolver:
    def __init__(self, disease_refs: list[DiseaseRef]):
        self.disease_refs = disease_refs
        self._exact_lookup: dict[str, DiseaseRef] = {}
        for ref in disease_refs:
            self._exact_lookup[normalize(ref.name)] = ref
            for synonym in ref.exact_synonyms:
                self._exact_lookup[normalize(synonym)] = ref

        self._embedding_model = None
        self._candidate_refs: list[DiseaseRef] = []
        self._candidate_embeddings = None

    def resolve(
        self, condition_raw: str, use_embedding_fallback: bool = True
    ) -> ResolutionResult:
        normalized = normalize(condition_raw)

        exact_ref = self._exact_lookup.get(normalized)
        if exact_ref is not None:
            return ResolutionResult(
                condition_raw,
                exact_ref.disease_id,
                exact_ref.name,
                "exact_synonym",
                100.0,
            )

        string_ref, string_score = self._best_string_match(normalized)
        if string_ref is not None and string_score >= STRING_SIM_THRESHOLD:
            return ResolutionResult(
                condition_raw,
                string_ref.disease_id,
                string_ref.name,
                "string_sim",
                string_score,
            )

        if use_embedding_fallback:
            embed_ref, embed_score = self._best_embedding_match(condition_raw)
            if embed_ref is not None and embed_score >= EMBEDDING_SIM_THRESHOLD:
                return ResolutionResult(
                    condition_raw,
                    embed_ref.disease_id,
                    embed_ref.name,
                    "embedding",
                    embed_score,
                )

        return ResolutionResult(condition_raw, None, None, "unresolved", None)

    def _best_string_match(
        self, normalized_condition: str
    ) -> tuple[DiseaseRef | None, float]:
        best_ref, best_score = None, 0.0
        for ref in self.disease_refs:
            for candidate in [ref.name, *ref.exact_synonyms]:
                score = _string_similarity(normalized_condition, normalize(candidate))
                if score > best_score:
                    best_ref, best_score = ref, score
        return best_ref, best_score

    def _ensure_embeddings(self) -> None:
        if self._embedding_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        candidate_texts = []
        for ref in self.disease_refs:
            for text in [ref.name, *ref.exact_synonyms]:
                self._candidate_refs.append(ref)
                candidate_texts.append(text)
        self._candidate_embeddings = self._embedding_model.encode(
            candidate_texts, normalize_embeddings=True
        )

    def _best_embedding_match(
        self, condition_raw: str
    ) -> tuple[DiseaseRef | None, float]:
        self._ensure_embeddings()
        condition_embedding = self._embedding_model.encode(
            condition_raw, normalize_embeddings=True
        )
        similarities = self._candidate_embeddings @ condition_embedding
        best_idx = int(similarities.argmax())
        return self._candidate_refs[best_idx], float(similarities[best_idx])
