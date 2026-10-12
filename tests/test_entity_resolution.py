import pytest

from pharma_graphrag.ingestion.entity_resolution import DiseaseRef, EntityResolver


@pytest.fixture
def resolver():
    refs = [
        DiseaseRef("MONDO_0008383", "rheumatoid arthritis", ["RA", "Rheumatoid disease"]),
        DiseaseRef("MONDO_0005083", "psoriasis", ["Psoriasis vulgaris"]),
    ]
    return EntityResolver(refs)


def test_exact_synonym_match(resolver):
    result = resolver.resolve("RA", use_embedding_fallback=False)
    assert result.disease_id == "MONDO_0008383"
    assert result.tier == "exact_synonym"


def test_exact_match_is_case_and_punctuation_insensitive(resolver):
    result = resolver.resolve("  rheumatoid ARTHRITIS!! ", use_embedding_fallback=False)
    assert result.disease_id == "MONDO_0008383"
    assert result.tier == "exact_synonym"


def test_string_similarity_catches_reordered_qualifier(resolver):
    result = resolver.resolve("Arthritis, Rheumatoid, Active", use_embedding_fallback=False)
    assert result.disease_id == "MONDO_0008383"
    assert result.tier == "string_sim"


def test_unrelated_condition_is_unresolved_without_embedding_fallback(resolver):
    result = resolver.resolve("Osteoarthritis of the knee", use_embedding_fallback=False)
    assert result.tier == "unresolved"
    assert result.disease_id is None
