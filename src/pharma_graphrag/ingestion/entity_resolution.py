"""Resolve free-text ClinicalTrials.gov condition strings to Open Targets disease nodes.

Tiered cascade, each tier only tried if the previous one didn't produce a match:
  1. exact_synonym - normalized string matches the disease name or an EFO "exact
     synonym" (e.g. "RA" -> rheumatoid arthritis), via a dictionary built from the
     Open Targets disease data cached in data/raw/.
  2. string_sim    - fuzzy string similarity (token-set ratio + Jaro-Winkler) against
     the same name/synonym pool, for spelling variants and reordered qualifiers
     (e.g. "Crohn's Disease, Moderate to Severe").
  3. embedding     - sentence-embedding cosine similarity, for true synonyms with no
     string overlap that the first two tiers can't catch.

Every result carries the tier and score that produced it, so match quality can be
audited and thresholds calibrated against a hand-labeled gold set.

Scoring vs. deciding: computing a condition's exact/string/embedding scores is the
expensive part (especially the embedding score, which needs a model forward pass);
turning those scores into a final match given one specific threshold pair is cheap.
score_condition() and CandidateEmbeddingIndex do the expensive part once; decide()
does the cheap part, so sweeping many threshold combinations (see
scripts/run_er_experiments.py) doesn't re-run any model inference. EntityResolver
wraps both together into one convenience call for everyday (non-sweep) use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from pharma_graphrag.config import DATA_RAW_DIR
from pharma_graphrag.ingestion.embedding_models import EmbeddingModel, load_embedding_model

DEFAULT_STRING_SIM_THRESHOLD = 88.0  # 0-100 scale
DEFAULT_EMBEDDING_SIM_THRESHOLD = 0.60  # cosine similarity, 0-1 scale
DEFAULT_EMBEDDING_MODEL_NAME = "minilm"

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


@dataclass
class ConditionScores:
    """Raw, threshold-independent scores for one condition string."""

    condition_raw: str
    exact_match: DiseaseRef | None
    string_match: tuple[DiseaseRef, float] | None
    embedding_match: tuple[DiseaseRef, float] | None = None


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


def _candidate_texts(disease_refs: list[DiseaseRef]) -> tuple[list[DiseaseRef], list[str]]:
    """Flatten disease name + exact synonyms into parallel (ref, text) lists."""
    refs, texts = [], []
    for ref in disease_refs:
        for text in [ref.name, *ref.exact_synonyms]:
            refs.append(ref)
            texts.append(text)
    return refs, texts


def best_exact_match(condition_raw: str, disease_refs: list[DiseaseRef]) -> DiseaseRef | None:
    normalized = normalize(condition_raw)
    for ref in disease_refs:
        candidates = [ref.name, *ref.exact_synonyms]
        if normalized in (normalize(c) for c in candidates):
            return ref
    return None


def best_string_match(condition_raw: str, disease_refs: list[DiseaseRef]) -> tuple[DiseaseRef, float] | None:
    normalized_condition = normalize(condition_raw)
    best_ref, best_score = None, 0.0
    for ref in disease_refs:
        for candidate in [ref.name, *ref.exact_synonyms]:
            score = _string_similarity(normalized_condition, normalize(candidate))
            if score > best_score:
                best_ref, best_score = ref, score
    return (best_ref, best_score) if best_ref is not None else None


def score_condition(condition_raw: str, disease_refs: list[DiseaseRef]) -> ConditionScores:
    """Compute the threshold-independent exact/string scores for one condition.

    Embedding scores aren't included here - they're model-specific and expensive, so
    they're computed separately via CandidateEmbeddingIndex and attached afterward.
    """
    return ConditionScores(
        condition_raw=condition_raw,
        exact_match=best_exact_match(condition_raw, disease_refs),
        string_match=best_string_match(condition_raw, disease_refs),
    )


class CandidateEmbeddingIndex:
    """Precomputed candidate (disease name/synonym) embeddings for one model, reused
    across every condition string scored against that model."""

    def __init__(self, embedding_model: EmbeddingModel, disease_refs: list[DiseaseRef]):
        self.refs, texts = _candidate_texts(disease_refs)
        self.embeddings = embedding_model.encode(texts)
        self._embedding_model = embedding_model

    def best_match(self, condition_raw: str) -> tuple[DiseaseRef, float]:
        condition_embedding = self._embedding_model.encode([condition_raw])[0]
        similarities = self.embeddings @ condition_embedding
        best_idx = int(similarities.argmax())
        return self.refs[best_idx], float(similarities[best_idx])

    def best_match_batch(self, condition_strings: list[str]) -> list[tuple[DiseaseRef, float]]:
        condition_embeddings = self._embedding_model.encode(condition_strings)
        similarities = condition_embeddings @ self.embeddings.T  # (n_conditions, n_candidates)
        best_idx = similarities.argmax(axis=1)
        return [(self.refs[i], float(similarities[row, i])) for row, i in enumerate(best_idx)]


def decide(
    scores: ConditionScores,
    string_sim_threshold: float = DEFAULT_STRING_SIM_THRESHOLD,
    embedding_sim_threshold: float = DEFAULT_EMBEDDING_SIM_THRESHOLD,
) -> ResolutionResult:
    """Apply one specific threshold pair to precomputed scores to get a final match."""
    if scores.exact_match is not None:
        ref = scores.exact_match
        return ResolutionResult(scores.condition_raw, ref.disease_id, ref.name, "exact_synonym", 100.0)

    if scores.string_match is not None:
        ref, score = scores.string_match
        if score >= string_sim_threshold:
            return ResolutionResult(scores.condition_raw, ref.disease_id, ref.name, "string_sim", score)

    if scores.embedding_match is not None:
        ref, score = scores.embedding_match
        if score >= embedding_sim_threshold:
            return ResolutionResult(scores.condition_raw, ref.disease_id, ref.name, "embedding", score)

    return ResolutionResult(scores.condition_raw, None, None, "unresolved", None)


class EntityResolver:
    """Convenience single-call resolver for everyday (non-sweep) use.

    For sweeping many threshold/model combinations, use score_condition() +
    CandidateEmbeddingIndex + decide() directly so scores are computed once and
    decisions are computed cheaply per configuration - see
    scripts/run_er_experiments.py.
    """

    def __init__(
        self,
        disease_refs: list[DiseaseRef],
        string_sim_threshold: float = DEFAULT_STRING_SIM_THRESHOLD,
        embedding_sim_threshold: float = DEFAULT_EMBEDDING_SIM_THRESHOLD,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    ):
        self.disease_refs = disease_refs
        self.string_sim_threshold = string_sim_threshold
        self.embedding_sim_threshold = embedding_sim_threshold
        self.embedding_model_name = embedding_model_name
        self._embedding_index: CandidateEmbeddingIndex | None = None

    def _ensure_embedding_index(self) -> CandidateEmbeddingIndex:
        if self._embedding_index is None:
            model = load_embedding_model(self.embedding_model_name)
            self._embedding_index = CandidateEmbeddingIndex(model, self.disease_refs)
        return self._embedding_index

    def resolve(self, condition_raw: str, use_embedding_fallback: bool = True) -> ResolutionResult:
        scores = score_condition(condition_raw, self.disease_refs)

        needs_embedding = scores.exact_match is None and (
            scores.string_match is None or scores.string_match[1] < self.string_sim_threshold
        )
        if use_embedding_fallback and needs_embedding:
            scores.embedding_match = self._ensure_embedding_index().best_match(condition_raw)

        return decide(scores, self.string_sim_threshold, self.embedding_sim_threshold)
