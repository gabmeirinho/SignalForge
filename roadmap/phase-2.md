## Phase 2 — Make chunking and retrieval independently comparable

**Outcome:** The same corpus can be indexed and queried under isolated retrieval profiles so chunking changes are measured rather than guessed.

### Commit P2-C1 — Introduce explicit chunking strategies

- **Purpose:** Turn the current paragraph character splitter into one interchangeable strategy among meaningful baselines.
- **Scope:** Define a `ChunkingStrategy` protocol and validated, serializable profile; retain paragraph-window behavior as the compatibility default; add fixed-token and section-aware parent/child strategies; emit stable chunk IDs plus parent, heading, character, and token metadata; apply the same interface to filings and generic documents.
- **Touchpoints:** `signalforge/sections.py`, `signalforge/ingestion.py`, `signalforge/source_ingestion.py`, new `signalforge/chunking.py`, `signalforge/models.py`, `signalforge/storage.py`, `tests/test_sections.py`, `tests/test_ingestion.py`, `tests/test_source_ingestion.py`, new `tests/test_chunking.py`.
- **Depends on:** P1-C1.
- **Done when:** All three strategies satisfy size/overlap invariants, preserve traceability to source text, handle oversized paragraphs and tables deterministically, and the default produces backward-compatible chunks.
- **Verify:** `uv run pytest tests/test_chunking.py tests/test_sections.py tests/test_ingestion.py tests/test_source_ingestion.py`

### Commit P2-C2 — Version and isolate retrieval profiles

- **Purpose:** Prevent chunks or embeddings from different experiments from contaminating one another.
- **Scope:** Define a retrieval-profile fingerprint from chunker name/config, embedding model, retrieval mode, and reranker; use it in chunk metadata, embedding records, and Qdrant collection aliases/names; add an experiment index builder that can reuse unchanged artifacts and never overwrites the active production profile.
- **Touchpoints:** `signalforge/models.py`, `signalforge/storage.py`, `signalforge/vectorization.py`, `signalforge/vector_store.py`, `signalforge/config.py`, new `signalforge/evaluation/indexes.py`, new Alembic migration after `0004_add_experiment_runs.py`, `tests/test_embedding_status.py`, `tests/test_vector_store.py`, `tests/test_migrations.py`.
- **Depends on:** P2-C1, P1-C3.
- **Done when:** Two profiles can coexist for identical source documents; every result reports its profile fingerprint; rerunning an unchanged profile is idempotent; partial builds cannot become the production alias; stale experimental collections have an explicit safe cleanup path.
- **Verify:** `uv run pytest tests/test_embedding_status.py tests/test_vector_store.py tests/test_migrations.py`

### Commit P2-C3 — Add dense, hybrid, and reranked retrieval adapters

- **Purpose:** Compare chunking separately from retrieval architecture and expose realistic quality/latency trade-offs.
- **Scope:** Keep dense search as the baseline; add Qdrant dense+sparse fusion and an optional reranking stage behind a common retriever interface; record candidate count, stage timings, fusion method, and reranker identity; preserve metadata filters and comparison/trend balancing across all modes.
- **Touchpoints:** `signalforge/vector_store.py`, `signalforge/rag_service.py`, `signalforge/config.py`, new `signalforge/retrieval.py`, `signalforge/cli/search.py`, `tests/test_retrieval.py`, `tests/test_vector_store.py`, `tests/test_rag_service.py`.
- **Depends on:** P2-C2.
- **Done when:** A retrieval profile selects dense, hybrid, or reranked execution without branching in callers; all modes honor ticker/section/year/source filters; tests demonstrate fusion and reranking with deterministic vectors; the baseline remains the default.
- **Verify:** `uv run pytest tests/test_retrieval.py tests/test_vector_store.py tests/test_rag_service.py`

**Exit check:** At least three chunking profiles and three retrieval modes can be built side by side and each returned chunk is attributable to one immutable profile.
