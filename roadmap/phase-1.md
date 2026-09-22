## Phase 1 — Establish reproducible experiments and telemetry

**Outcome:** SignalForge can describe an experiment completely, replay a representative dataset, and persist comparable quality, latency, token, cost, and failure data without changing the production query path.

### Commit P1-C1 — Define a versioned evaluation dataset

- **Purpose:** Replace the planner-only fixture with a benchmark contract that can evaluate retrieval and answers.
- **Scope:** Add a schema and loader for versioned cases containing the question, query class, expected plan, relevant evidence identifiers or passages, answer facts, and rubric tags; seed 20–30 cases across lookup, synthesis, comparison, trend, missing-evidence, and conflicting-source scenarios; document how a completed `query_run` is promoted only after human review.
- **Touchpoints:** `tests/fixtures/planner_golden.json`, new `evaluation/datasets/`, new `signalforge/evaluation/dataset.py`, `tests/test_planner_evaluator.py`, new `tests/test_evaluation_dataset.py`, `README.md`.
- **Depends on:** None.
- **Done when:** Invalid or duplicate cases fail validation; every seeded case has stable evidence labels and a query class; existing planner cases remain runnable through the new loader; dataset version and content hash are printed by evaluators.
- **Verify:** `uv run pytest tests/test_evaluation_dataset.py tests/test_planner_evaluator.py`

### Commit P1-C2 — Capture end-to-end run telemetry

- **Purpose:** Make production query runs usable as evidence for performance and cost decisions.
- **Scope:** Time planning, retrieval, and answer generation separately; record provider/model, prompt version, token usage when supplied, estimated cost with a versioned price snapshot, retry/fallback counts, and terminal error category; emit OpenTelemetry spans through a disabled-by-default exporter configured to send traces to Langfuse via OTLP; add nullable fields through an Alembic migration and preserve compatibility with SQLite and Postgres.
- **Touchpoints:** `pyproject.toml`, `signalforge/rag_service.py`, `signalforge/query_planner.py`, `signalforge/answer_generator.py`, `signalforge/models.py`, `signalforge/storage.py`, `signalforge/api.py`, `signalforge/config.py`, new `signalforge/telemetry.py`, new `alembic/versions/0003_add_query_run_telemetry.py`, `tests/test_api.py`, `tests/test_generic_data_model.py`, `tests/test_migrations.py`.
- **Depends on:** None.
- **Done when:** A completed or failed query stores per-stage and total latency, exact token usage or an explicit unavailable value, price-catalog version, estimated cost or an explicit unavailable value, model/prompt identities, and fallback/error metadata; spans use stable attributes and export when configured; no secret, full prompt, or source body is stored in telemetry.
- **Verify:** `uv run pytest tests/test_api.py tests/test_generic_data_model.py tests/test_migrations.py && uv run ruff check signalforge tests`

### Commit P1-C3 — Persist experiment runs and case results

- **Purpose:** Give offline comparisons a durable, auditable record instead of console-only pass/fail output.
- **Scope:** Add experiment and experiment-case tables plus storage helpers for immutable configuration snapshots, dataset hashes, Git SHA, timestamps, status, metrics, judge records, latency, tokens, cost, and errors; support resume after interruption without duplicating completed case/config pairs.
- **Touchpoints:** `signalforge/models.py`, `signalforge/storage.py`, new `signalforge/evaluation/models.py`, new `alembic/versions/0004_add_experiment_runs.py`, `tests/test_generic_data_model.py`, `tests/test_migrations.py`.
- **Depends on:** P1-C1, P1-C2.
- **Done when:** Experiment configuration and results round-trip on SQLite and Postgres; uniqueness prevents duplicate case/config results; interrupted runs can resume; stored results identify dataset content, code revision, models, prompts, index, and random settings.
- **Verify:** `uv run pytest tests/test_generic_data_model.py tests/test_migrations.py && uv run pytest -m postgres tests/test_postgres_storage.py`

**Exit check:** A fixed dataset and configuration can produce a persistent experiment record whose inputs, code version, timings, token use, cost basis, failures, and outputs are reproducible.
