## Phase 4 — Turn evaluation results into a bounded agentic system

**Outcome:** SignalForge uses measured evidence to select and, when necessary, revise its research path under explicit quality, latency, and cost limits.

### Commit P4-C1 — Implement a typed research state machine

- **Purpose:** Make the application genuinely agentic through observable decisions and feedback, not role-playing agents.
- **Scope:** Model the workflow as classify → plan → retrieve → grade evidence → optionally rewrite/retrieve once → answer → verify citations; define typed state and tool contracts; cap steps, retries, retrieved context, wall time, and estimated spend; persist every decision and transition on the existing query run.
- **Touchpoints:** `signalforge/rag_service.py`, `signalforge/query_planner.py`, `signalforge/answer_generator.py`, `signalforge/storage.py`, new `signalforge/research_agent.py`, `signalforge/api.py`, `tests/test_rag_service.py`, new `tests/test_research_agent.py`.
- **Depends on:** P3-C4.
- **Done when:** Simple questions take the direct path; insufficient evidence triggers at most one corrective retrieval; exhausted budgets yield a cited partial answer or explicit abstention; replaying stored fake-model decisions is deterministic; each transition exposes its reason and resource use.
- **Verify:** `uv run pytest tests/test_research_agent.py tests/test_rag_service.py tests/test_api.py`

### Commit P4-C2 — Derive and enforce an adaptive routing policy

- **Purpose:** Select the lowest-cost configuration predicted to meet service objectives for each query class.
- **Scope:** Build a versioned policy artifact from experiment results; route by query class and workload constraints; require minimum quality and maximum p95 latency/failure rate; support an explicit baseline override and safe fallback; log predicted versus observed outcomes for later recalibration.
- **Touchpoints:** new `signalforge/routing.py`, `signalforge/config.py`, `signalforge/research_agent.py`, `signalforge/storage.py`, `signalforge/api.py`, new `tests/test_routing.py`, `README.md`.
- **Depends on:** P4-C1.
- **Done when:** The same policy artifact produces deterministic choices; every route cites its experiment evidence; missing/stale policy data selects the safe baseline; per-request budget and latency constraints are enforced; observed drift is queryable.
- **Verify:** `uv run pytest tests/test_routing.py tests/test_research_agent.py tests/test_api.py`

### Commit P4-C3 — Add shadow evaluation and release safeguards

- **Purpose:** Validate a candidate policy on real traffic without silently degrading user answers.
- **Scope:** Add sampled asynchronous shadow execution with redaction, concurrency/rate limits, circuit breaking, and separate budgets; compare candidate and active policies using deterministic checks plus the calibrated judge; define promotion/rollback gates and retention controls; never return shadow output to the user.
- **Touchpoints:** `signalforge/api.py`, `signalforge/worker.py`, `signalforge/models.py`, `signalforge/storage.py`, new `signalforge/shadow_evaluation.py`, new Alembic migration after the retrieval-profile migration, `tests/test_worker.py`, new `tests/test_shadow_evaluation.py`, `compose.yaml`, `.env.example`.
- **Depends on:** P4-C2.
- **Done when:** Sampling can be disabled instantly; queue pressure cannot slow the primary response path; budgets and circuit breakers halt shadow calls; promotion requires configured quality/reliability gates; rollback restores the prior policy artifact without reindexing.
- **Verify:** `uv run pytest tests/test_shadow_evaluation.py tests/test_worker.py tests/test_api.py`

### Commit P4-C4 — Expose experiments and agent traces in the console

- **Purpose:** Make the portfolio’s engineering decisions legible in a short live demo.
- **Scope:** Add read-only APIs and a frontend view for experiment status, configuration comparison, Pareto frontier, per-case drill-down, and query state transitions; clearly label measured versus estimated values and active versus shadow runs; add accessible loading, empty, error, and narrow-screen states.
- **Touchpoints:** `signalforge/api.py`, `frontend/src/api.ts`, `frontend/src/types.ts`, `frontend/src/App.tsx`, new `frontend/src/components/ExperimentDashboard.tsx`, new `frontend/src/components/AgentTrace.tsx`, `frontend/src/styles.css`, `tests/test_api.py`, `frontend/src/test/App.test.tsx`.
- **Depends on:** P4-C3.
- **Done when:** A reviewer can identify why a model/profile was selected, inspect each agent transition, compare quality/p95 latency/cost/failure rate, and open a failed case without running a new experiment; UI tests cover all states.
- **Verify:** `uv run pytest tests/test_api.py && cd frontend && npm test -- --run && npm run build`

**Exit check:** A recorded demo shows a production query being routed under a budget, its bounded corrective loop and trace, and the experiment evidence supporting that routing choice.
