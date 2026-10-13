# Pharma GraphRAG

A GraphRAG evaluation project over pharma target/disease/drug/trial data (Open Targets +
ClinicalTrials.gov), built to show *when and why* graph traversal beats vector-only RAG —
with precision@k/recall@k and LLM-as-judge numbers, broken out by single-hop vs.
multi-hop questions, not just a demo that works once.

**Status:** data ingestion and entity resolution are built; graph loading, retrieval
pipelines, and evaluation are in progress. This README is updated as each part lands —
see below for what's covered so far.

## Data sources

- **Open Targets Platform** (GraphQL) — disease → associated targets → known drugs →
  mechanism of action.
- **ClinicalTrials.gov API v2** (REST) — trial → conditions, interventions, sponsor,
  phase, brief summary.

Scope: 5 diseases across immunology, oncology, and neuroscience — rheumatoid arthritis,
psoriasis, Crohn's disease, multiple sclerosis, Parkinson's disease.

## Entity resolution

Open Targets identifies diseases with EFO/MONDO ontology IDs. ClinicalTrials.gov
identifies them with free-text `conditions` strings per trial, with no controlled
vocabulary. Linking the two is the entity resolution problem this project solves.

### Why this isn't generic record linkage

This is **ontology/concept normalization** — matching free text to a fixed, known
vocabulary (the 5 Open Targets disease nodes) — not general two-table deduplication,
where neither side is authoritative. That distinction drove the design: a small
deterministic cascade rather than a trained classifier, since the target vocabulary is
tiny and there's no large pool of labeled pairs to train on.

### The tiered cascade

Implemented in `src/pharma_graphrag/ingestion/entity_resolution.py`. Each tier is only
tried if the previous one didn't produce a match, and every result is tagged with the
tier and score that produced it:

1. **Exact synonym match** — the normalized condition string against the disease name
   plus Open Targets' own curated `hasExactSynonym` terms (e.g. `"RA"` resolves directly
   to rheumatoid arthritis via EFO's synonym list — not a hand-built lookup table).
2. **String similarity** — token-set ratio + Jaro-Winkler (`rapidfuzz`), threshold
   88/100. Needed because trial condition strings carry reordered qualifiers, e.g.
   `"Crohn's Disease, Moderate to Severe"`.
3. **Embedding fallback** — `sentence-transformers/all-MiniLM-L6-v2` cosine similarity,
   threshold 0.60, for true synonyms with no string overlap that the first two tiers
   can't catch.

### Theoretical grounding

The classical foundation for this problem is **Fellegi-Sunter probabilistic record
linkage** — a comparison vector scored by log-likelihood ratios (m/u probabilities),
thresholded into match / possible-match / non-match. The tiered cascade here is that
model's hand-thresholded special case: full EM-fitted Fellegi-Sunter wasn't used because
there's no labeled pair data and the scale (~344 unique condition strings) doesn't
warrant it. At larger scale, EM-fitted Fellegi-Sunter would also be the principled
*unsupervised* way to estimate the matcher's own error rate — worth naming as the
next step up, not used here by choice.

### What actually happened when this ran

The embedding tier turned out to be the weakest in practice: it matched `"Ulcerative
Colitis"` to Crohn's disease and `"Osteoarthritis"` to rheumatoid arthritis at 0.6–0.7
cosine similarity — related diseases, not the same disease. Generic sentence embeddings
capture topical similarity, not clinical identity. The fix with more time: a
biomedical-specific embedding model trained on synonym pairs (e.g. SapBERT, trained on
UMLS) instead of a generic one.

## Known data limitations

- **Open Targets' free-text search is not reliably rankable.** Re-resolving "multiple
  sclerosis" by name returned a sub-concept (`EFO_0803536`, "multiple sclerosis symptom
  measurement") above the actual disease. `resolve_efo_id()` now prefers an exact
  case-insensitive name match among search hits over blindly taking the top result.
- **The base "multiple sclerosis" entity is unresolvable in Open Targets' current
  platform release** — `disease(efoId: "MONDO_0005301")` returns null; only MS subtypes
  resolve. Relapsing-remitting MS (`MONDO_0005314`, the most common subtype at
  diagnosis) is used as the Open Targets-side proxy for target/drug data. ClinicalTrials.gov
  matching still targets the broader "multiple sclerosis" string via the string-similarity
  tier. Pinned in `DISEASE_ID_OVERRIDES` in `open_targets.py`.

## Evaluating entity resolution without ground-truth labels

End-to-end retrieval and LLM-judge metrics alone aren't enough to evaluate entity
resolution — a bad final answer could be ER's fault, the retrieval template's fault, or
the judge's fault, and those metrics can't tell you which.

- **A small hand-labeled gold set for ER specifically** (distinct from the retrieval
  gold QA set) is cheap at this scale — about 344 unique condition strings — and is the
  actual standard approach, not an exotic workaround. Built via
  `scripts/build_er_labeling_sheet.py` → `data/processed/er_labeling_sheet.csv`.
- **Label-free diagnostics** worth running alongside it: match-score distribution shape
  (a good matcher produces a bimodal distribution, not a smeared blob), stratified
  clerical-review sampling across confidence tiers, per-tier coverage/match-rate
  tracking, and structural graph sanity checks (orphan disease nodes, one disease
  over-absorbing unrelated conditions).

### Hand-labeling process

1. Read `condition_raw` against the 5-disease legend first. Most rows resolve on sight —
   either a direct keyword/abbreviation match, or an obviously unrelated term (`"Healthy"`,
   `"Fatigue"`, `"Anxiety"`).
2. The real judgment calls are a small residual of *related-but-distinct* disease-family
   terms — psoriatic arthritis, axial spondyloarthritis, ankylosing spondylitis, avascular
   necrosis, inflammatory bowel disease / ulcerative colitis. These default to `NONE`
   unless the project scope is deliberately expanded to include them as their own disease
   nodes.
3. `source_query_buckets` (which disease query pulled the trial in) is a more useful
   secondary signal than `example_brief_title` — trial titles describe the intervention,
   not the disease, so they rarely help disambiguate.
4. `NONE` vs. `UNSURE` are kept distinct: `NONE` means confidently out of scope; `UNSURE`
   flags genuine ambiguity. The resolver's own `unresolved` tier is not auto-copied as
   `NONE` — it's independently checked, since it sometimes reflects a real threshold miss
   (a false negative) rather than a true non-match, which is exactly the signal used to
   recalibrate tier thresholds afterward.

### What NONE means downstream

`NONE` on one condition string does not drop its trial — it only means that string
doesn't produce a `STUDIES` edge to a disease node. A trial is excluded from graph
loading only if *all* of its listed conditions resolve to `NONE` (it would otherwise be
an orphan node, unreachable from any disease-anchored multi-hop query).

The non-trivial `NONE` rate seen in this data (~70% of unique condition strings) is an
expected consequence of how the data was fetched, not a flaw: ClinicalTrials.gov's
`query.cond` is a relevance search, not an exact filter, and is deliberately
recall-oriented — it pulls in comorbidities and tangentially related trials by design.
Entity resolution is the precision filter that sits downstream of that intentionally
loose fetch.

## What's next

Graph loading (`graph/`), retrieval pipelines (`retrieval/`), and the evaluation
framework (`evaluation/`) — see `graphrag_project_plan.md` for the full build plan and
day-by-day breakdown.
