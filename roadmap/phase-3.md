## Phase 3 — Build a calibrated model and pipeline evaluator

**Outcome:** One command produces a trustworthy leaderboard and Pareto frontier across chunkers, retrievers, planner models, answer models, and judge models.

### Commit P3-C1 — Implement deterministic retrieval metrics

- **Purpose:** Measure evidence discovery before judging generated prose.
- **Scope:** Match retrieved chunks to labeled evidence using stable source spans/IDs; calculate recall@k, precision@k, MRR, nDCG, source coverage, and empty-result/error rates by query class; report macro averages and per-case failures; update the existing retrieval CLI to emit JSON as well as readable output.
- **Touchpoints:** `signalforge/cli/evaluate_retrieval.py`, new `signalforge/evaluation/retrieval_metrics.py`, `signalforge/evaluation/dataset.py`, `tests/test_retrieval_evaluator.py`, new `tests/test_retrieval_metrics.py`.
- **Depends on:** P1-C1, P2-C3.
- **Done when:** Metric implementations pass hand-calculated fixtures; irrelevant high-ranked results reduce ranking scores; missing expected evidence reduces recall; results are sliced by query class and stored against the experiment profile.
- **Verify:** `uv run pytest tests/test_retrieval_evaluator.py tests/test_retrieval_metrics.py`

### Commit P3-C2 — Add a structured LLM-as-judge harness

- **Purpose:** Score groundedness and answer usefulness while making judge uncertainty visible.
- **Scope:** Create an independent judge adapter with structured output for citation correctness, faithfulness to supplied evidence, factual completeness against labeled facts, relevance, and abstention quality; version prompts/rubrics; blind candidate identities; randomize pairwise order; retry invalid judge output; store score, rationale, raw structured result, judge identity, latency, tokens, and cost.
- **Touchpoints:** `signalforge/cli/evaluate_answers.py`, new `signalforge/evaluation/judge.py`, new `prompts/answer_judge.md`, `signalforge/evaluation/models.py`, `tests/test_answer_generator.py`, new `tests/test_llm_judge.py`.
- **Depends on:** P1-C3, P3-C1.
- **Done when:** Judge output is schema-validated; candidate and judge models are independently selectable; evidence is the only factual context supplied to the judge; position-swapped pairwise judgments are retained; failures become explicit missing scores rather than zeros.
- **Verify:** `uv run pytest tests/test_llm_judge.py tests/test_answer_generator.py`

### Commit P3-C3 — Calibrate judge scores with human labels

- **Purpose:** Demonstrate that automated evaluation correlates with a documented human standard.
- **Scope:** Add a blinded review export/import workflow for a stratified subset; include two human ratings per answer and adjudication for disagreements; calculate agreement and judge-to-human correlation/confusion by rubric; define a minimum calibration gate and document limitations such as verbosity, self-preference, and shared-provider bias.
- **Touchpoints:** new `signalforge/cli/export_review_batch.py`, new `signalforge/cli/import_review_batch.py`, new `signalforge/evaluation/calibration.py`, `evaluation/datasets/`, new `tests/test_judge_calibration.py`, `README.md`.
- **Depends on:** P3-C2.
- **Done when:** Review files contain no candidate model identity; malformed or incomplete imports fail clearly; agreement and calibration metrics are reproducible; uncalibrated judges are labeled as such and cannot supply the release-gating score.
- **Verify:** `uv run pytest tests/test_judge_calibration.py`

### Commit P3-C4 — Run matrices and select the Pareto frontier

- **Purpose:** Turn raw measurements into an evidence-backed model and architecture decision.
- **Scope:** Add a YAML experiment manifest and runner for chunker × retriever × planner × answer-model configurations with bounded concurrency, timeout, retry, budget, resume, and cache controls; calculate bootstrap confidence intervals; flag regressions against a named baseline; emit CSV/JSON/Markdown leaderboards and non-dominated configurations for quality, p95 latency, cost/query, and failure rate.
- **Touchpoints:** new `evaluation/experiments/`, new `signalforge/evaluation/runner.py`, new `signalforge/evaluation/report.py`, new `signalforge/cli/run_experiment.py`, `Makefile`, new `tests/test_experiment_runner.py`, new `tests/test_evaluation_report.py`, `README.md`.
- **Depends on:** P3-C1, P3-C3.
- **Done when:** A dry run works without paid model calls; interrupted runs resume; a hard dollar budget stops new calls safely; reports include dataset/config/code identities and uncertainty; no single weighted score hides quality, latency, cost, or reliability.
- **Verify:** `uv run pytest tests/test_experiment_runner.py tests/test_evaluation_report.py && uv run python -m signalforge.cli.run_experiment --help`

### Commit P3-C5 — Add a deterministic evaluation smoke gate

- **Purpose:** Catch retrieval and orchestration regressions in CI without flaky or unbounded model spending.
- **Scope:** Add a small frozen dataset using fake model adapters and deterministic embeddings; gate on deterministic metrics and schema/lifecycle behavior; make live-model benchmarks opt-in and publish their artifact separately rather than blocking every commit.
- **Touchpoints:** new `.github/workflows/evaluation.yml`, `tests/fixtures/`, `Makefile`, `README.md`.
- **Depends on:** P3-C4.
- **Done when:** CI runs with no API keys, has a fixed runtime and zero provider cost, fails on an intentional retrieval regression, and documents the separate command for scheduled/manual live evaluation.
- **Verify:** `make test && make lint && cd frontend && npm test -- --run && npm run build`

**Exit check:** A saved report names the best quality configuration, the cheapest acceptable configuration, and the fastest acceptable configuration, with calibrated quality measures and reproducible provenance.
