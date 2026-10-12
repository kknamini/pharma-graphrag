# Pharma GraphRAG Evaluation Project — Build Plan

**Goal:** Build a small, defensible GraphRAG system over pharma target/disease/drug/trial data, with a rigorous evaluation framework (precision@k/recall@k + LLM-as-judge) comparing GraphRAG against a vector-only RAG baseline — broken out by single-hop vs. multi-hop questions.

**Why this project exists (keep this framing in the README and in your head for interviews):** most GraphRAG demos show retrieval working once and stop. The point of this project is the eval — showing *when and why* graph traversal beats vector search, with numbers, not vibes.

---

## Data sources (both free, no API key required)

- **Open Targets Platform** (GraphQL) — `https://api.platform.opentargets.org/api/v4/graphql`
  Disease → associated targets → known drugs → mechanism of action.
- **ClinicalTrials.gov API v2** (REST) — `https://clinicaltrials.gov/api/v2/studies`
  Trial → conditions, interventions, sponsor, phase, brief summary.

**Scope:** 5–8 diseases across AbbVie's real therapeutic areas — immunology, oncology, neuroscience (e.g., rheumatoid arthritis, psoriasis, Crohn's disease, multiple sclerosis, Parkinson's). Small and thematically deliberate beats large and generic.

**Entity resolution note:** Open Targets uses EFO IDs; ClinicalTrials.gov uses free-text condition names. Matching these is a real record-linkage problem — build it deliberately (exact match → normalized string match → fallback fuzzy match with a logged confidence score) and document the decision. This directly maps to the "entity resolution, record linkage" line in the job posting.

---

## Graph schema

```
(:Disease {efoId, name})
(:Target {ensemblId, symbol})
(:Drug {chemblId, name, isApproved})
(:Trial {nctId, phase, status, briefSummary})

(:Disease)-[:ASSOCIATED_WITH {score}]->(:Target)
(:Drug)-[:TARGETS {mechanismOfAction}]->(:Target)
(:Drug)-[:INDICATED_FOR]->(:Disease)
(:Trial)-[:TESTS]->(:Drug)
(:Trial)-[:STUDIES]->(:Disease)
```

Labeled-property graph, built in **Neo4j** (AuraDB free tier or local Docker), queried with **Cypher**. Be ready to explain in the interview why labeled-property fit better here than RDF/SPARQL (schema is small, relationships are the main object of interest, no need for formal ontology reasoning at this scale) — this is a stronger answer than "I just picked the one I knew."

---

## Python package structure

```
pharma_graphrag/
├── pyproject.toml
├── .env.example                  # NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ANTHROPIC_API_KEY
├── README.md                     # final write-up: schema diagram, methodology, results table
│
├── src/pharma_graphrag/
│   ├── __init__.py
│   ├── config.py                 # env/config loading
│   │
│   ├── ingestion/
│   │   ├── open_targets.py       # GraphQL client + fetch_disease_drugs(), fetch_target_associations()
│   │   ├── clinical_trials.py    # REST client + fetch_trials_for_condition()
│   │   └── entity_resolution.py  # match_condition_to_efo() with confidence scoring
│   │
│   ├── graph/
│   │   ├── schema.py             # Cypher constraint/index DDL as constants
│   │   ├── loader.py             # batch upsert nodes/relationships into Neo4j
│   │   └── client.py             # thin Neo4j driver wrapper (session mgmt, retries)
│   │
│   ├── retrieval/
│   │   ├── vector_baseline.py    # embed trial/drug text, similarity search (local embeddings — see note below)
│   │   ├── graph_rag.py          # Cypher traversal retrieval (multi-hop query templates)
│   │   └── hybrid.py             # optional: graph traversal + vector re-ranking
│   │
│   ├── generation/
│   │   └── answer.py             # calls Claude API with retrieved context → generates answer
│   │
│   ├── evaluation/
│   │   ├── gold_set.py           # loads/validates gold QA set (question, expected_entities, expected_answer, hop_type)
│   │   ├── retrieval_metrics.py  # precision_at_k(), recall_at_k()
│   │   ├── llm_judge.py          # rubric-based scoring via Claude API (faithfulness, relevance, correctness)
│   │   └── run_eval.py           # orchestrates: run both pipelines over gold set → metrics table
│   │
│   └── cli.py                    # `pharma-graphrag ingest|load-graph|ask|eval`
│
├── data/
│   ├── raw/                      # cached API responses (don't re-hit APIs every run)
│   ├── processed/
│   └── gold_qa_set.jsonl         # 20–30 hand-written questions, tagged single_hop / multi_hop
│
├── tests/
│   ├── test_entity_resolution.py
│   ├── test_retrieval_metrics.py
│   └── test_graph_loader.py
│
└── results/
    ├── eval_results.csv          # per-question scores, both pipelines
    └── summary_by_hop_type.csv   # the headline comparison table
```

**On embeddings:** use a local model (e.g., `sentence-transformers`) for the vector baseline rather than a paid embedding API — keeps cost near zero and avoids a second vendor dependency. Reserve API spend for the LLM-as-judge calls, which are the part that actually needs a strong model.

**On the CLI:** a `cli.py` with clear subcommands (`ingest`, `load-graph`, `ask`, `eval`) makes the project demoable in an interview — you can literally run it live if asked.

---

## Day-by-day plan

**Day 1 — Data + graph model**
- `ingestion/open_targets.py`, `ingestion/clinical_trials.py`: fetch and cache raw data for your chosen disease set
- `ingestion/entity_resolution.py`: build and test the condition→EFO matching logic
- `graph/schema.py`, `graph/loader.py`: define constraints, load into Neo4j
- Checkpoint: can you run a Cypher query that answers "which drugs target genes associated with rheumatoid arthritis and are being tested in an active trial?" — a genuine multi-hop question

**Day 2 — Retrieval pipelines**
- `retrieval/vector_baseline.py`: embed trial summaries + drug mechanism text, build similarity search
- `retrieval/graph_rag.py`: write 3–5 Cypher query templates covering both single-hop and multi-hop patterns
- Checkpoint: both pipelines return results for the same set of test questions

**Day 3 — Evaluation framework**
- `evaluation/gold_set.py`: write 20–30 questions by hand, tag each `single_hop`/`multi_hop`, specify expected entities/answers
- `evaluation/retrieval_metrics.py`: precision@k, recall@k against expected entities
- `evaluation/llm_judge.py`: design the rubric explicitly (e.g., faithfulness — is the answer supported by retrieved context; correctness — does it match ground truth) before writing the scoring prompt
- `evaluation/run_eval.py`: run both pipelines over the full gold set, produce `results/summary_by_hop_type.csv`
- Checkpoint: a results table showing GraphRAG's advantage (or lack thereof) concentrated in multi-hop questions

**Day 4 — Write-up (buffer)**
- README: schema diagram, methodology, entity-resolution approach, results table, and 3–4 sentences on what you'd do with more time (e.g., scale to more diseases, add SHACL-style validation rules, try a hybrid re-ranker)
- This README doubles as your interview script — write it like you're explaining the project to the interviewer, not just documenting code

---

## Interview-ready talking points this project should produce

- Why labeled-property graph over RDF for this use case
- How you approached entity resolution and what confidence-scoring tradeoffs you made
- Why you built a baseline instead of just GraphRAG alone (comparability is the point)
- How you designed the LLM-judge rubric and what failure modes you saw the judge itself have
- One concrete example where graph traversal answered something vector search couldn't
