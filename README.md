# SignalForge

SignalForge is a local market intelligence pipeline for company research. It ingests SEC EDGAR 10-K filings and approved company source feeds, indexes the content locally with Qdrant, and answers research questions with cited evidence.

It is built for experimenting with retrieval, query planning, and answer generation over a local SQLite database and vector index.

## What It Does

- Downloads and parses SEC 10-K filings by ticker.
- Extracts key 10-K sections such as Business, Risk Factors, MD&A, and Market Risk.
- Discovers candidate company sources such as blogs, newsrooms, investor relations pages, RSS feeds, and Atom feeds.
- Lets you approve, reject, list, or manually add sources before ingestion.
- Ingests approved source articles into generic documents.
- Embeds filing and document chunks into a local Qdrant store.
- Plans research queries with DeepSeek when configured, or a local fallback when not.
- Generates cited answers from retrieved evidence.
- Includes a FastAPI backend and React research console.

## Requirements

- Python 3.11+
- `uv`
- Node.js and npm, only needed for the frontend
- SEC EDGAR user-agent details for filing downloads
- Optional: DeepSeek API key for LLM planning and answer generation

The default vector store uses embedded `qdrant-client`, so no external Qdrant server is required.

## Setup

Install Python dependencies:

```bash
uv sync --extra dev
```

Create a local `.env` file:

```bash
SEC_COMPANY_NAME="Your Name or App Name"
SEC_EMAIL="you@example.com"

# Optional
DEEPSEEK_API_KEY="your_deepseek_api_key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Without `DEEPSEEK_API_KEY`, SignalForge still supports local rule-based planning and extractive evidence output.

## Research Workflow

Ingest a recent 10-K:

```bash
uv run python -m signalforge.cli.ingest --ticker NVDA --limit 1
```

Discover candidate company sources. Candidates are saved for review, but they are not ingested automatically:

```bash
uv run python -m signalforge.cli.discover_sources --ticker NVDA --website-domain nvidia.com
```

Review candidate, approved, rejected, and manual sources:

```bash
uv run python -m signalforge.cli.list_sources --ticker NVDA
```

Approve or reject discovered sources:

```bash
uv run python -m signalforge.cli.approve_source --source-id 1
uv run python -m signalforge.cli.reject_source --source-id 2
```

Ingest approved sources:

```bash
uv run python -m signalforge.cli.ingest_sources --ticker NVDA --limit-per-source 5
```

Build or refresh the mixed SEC/document vector index:

```bash
uv run python -m signalforge.cli.vectorize
```

Run the background worker loop for approved source ingestion and vectorization:

```bash
uv run python -m signalforge.worker
```

Ask a mixed-corpus research question:

```bash
uv run python -m signalforge.cli.answer_query \
  "What is NVIDIA saying recently about AI infrastructure and supply constraints?" \
  --show-plan \
  --show-chunks
```

Answers cite retrieved evidence with normalized labels. SEC evidence is labeled by ticker, filing year, filing item, and chunk. Web document evidence is labeled by source name, publication date when available, and article title.

## Evaluation dataset

`evaluation/datasets/benchmark-v1.json` is the version 1.0 offline benchmark.
Each case has a question, query class, planner context and expected plan, stable
evidence labels and passages, answer facts, and rubric tags. The 24 seeded cases
cover lookup, synthesis, comparison, trend, missing evidence, and conflicting
sources. Their passages are **synthetic fixture text**, not SEC filing claims;
use them with a fixture index when evaluating retrieval or answers. The original
`tests/fixtures/planner_golden.json` remains accepted as a planner-only legacy
dataset via `--dataset`.

The three evaluators validate their dataset before running and print its version
and SHA-256 content hash. The hash uses canonical JSON, so whitespace changes do
not create a new dataset identity. A change to case content should also bump the
dataset version and retain the old file for reproducible comparisons.

```bash
uv run python -m signalforge.cli.evaluate_planner --dataset tests/fixtures/planner_golden.json
uv run python -m signalforge.cli.evaluate_retrieval --dataset evaluation/datasets/benchmark-v1.json
uv run python -m signalforge.cli.evaluate_answers --dataset evaluation/datasets/benchmark-v1.json
```

To promote a completed `query_run` into the benchmark:

1. Inspect its saved question, plan, answer, and retrieval metadata alongside the
   actual source passages. Confirm that the source is allowed in the benchmark
   and record a stable source or fixture label for every relevant passage.
2. Have a human reviewer verify the expected plan, independently check each
   answer fact against the passages, assign a query class and rubric tags, and
   document any missing or conflicting evidence. Do not treat the model answer
   or its citations as ground truth by themselves.
3. Add the reviewed case to a **new versioned dataset file**, validate it with
   `uv run pytest tests/test_evaluation_dataset.py tests/test_planner_evaluator.py`,
   and record the printed content hash with experiment results. A `query_run` is
   not promoted merely because it completed successfully.

## Source Lifecycle

Source discovery follows this flow:

```text
ticker -> discover candidate sources -> approve or reject sources
-> ingest approved sources -> vectorize documents -> answer with citations
```

Discovery uses deterministic heuristics such as official-domain matching, common blog/newsroom/investor-relations paths, source subdomains, reachable pages, page titles, and RSS/Atom links.

Manual source registration is available for demos and edge cases:

```bash
uv run python -m signalforge.cli.add_source \
  --ticker NVDA \
  --name "NVIDIA Newsroom" \
  --url "https://nvidianews.nvidia.com/" \
  --source-type newsroom \
  --ownership official \
  --trust-level high
```

## Query telemetry

Completed and failed API query runs save planning, retrieval, answer, and total
latency; planner and answer provider/model/prompt versions; fallback and retry
counts; terminal error category; token usage status and counts; and an estimated
USD cost. A missing provider usage response is recorded as `unavailable`. A
retry with an unreported attempt is `partial` because that attempt may have
consumed tokens. In either case the cost is unavailable, never inferred
from text length.

The price snapshot is `deepseek-2026-09-22`, based on the
[DeepSeek pricing table](https://api-docs.deepseek.com/quick_start/pricing/).
It records the UTC peak/off-peak tier and uses cache-miss pricing if the provider
does not report cached prompt tokens. Estimates may differ from the provider's
bill. Unknown models have an unavailable cost.

OpenTelemetry export is disabled by default. To send content-free query and stage
spans to [Langfuse's OTLP endpoint](https://langfuse.com/integrations/native/opentelemetry),
set:

```bash
SIGNALFORGE_TRACING_ENABLED=1
SIGNALFORGE_OTLP_TRACES_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
```

Use the endpoint for your Langfuse region or self-hosted instance. Trace
attributes contain timings, counts, model and prompt identifiers, and status;
they do not contain the question, prompts, source bodies, or credentials.

To verify export against Langfuse Cloud, create a project and its API key pair
in **Project Settings → API Keys**, put the four variables above in `.env`, and
start the API with `make api`. To verify against a local stack instead, start it
as described in [Running the Local Langfuse Stack](#running-the-local-langfuse-stack).
Send a query:

```bash
curl -sS http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are NVIDIA risk factors?","include_source_text":false}'
```

Telemetry is stored on the `query_runs` row for the returned `query_run_id` and
is also returned in the `telemetry` object of the query response. Saved runs,
including their telemetry, can be read back with `GET /api/query-runs` and
`GET /api/query-runs/{id}`, which also back the frontend history view. In the matching
Langfuse project, open **Tracing** and select the latest `signalforge.query`
trace. It contains `signalforge.planning`, `signalforge.retrieval`, and
`signalforge.answer` child spans. The exporter sends the Langfuse v4 ingestion
header, so traces should appear shortly after export.
If the API has no `DEEPSEEK_API_KEY`, token usage and cost are unavailable, while
stage timings and fallback counts are still recorded.

### Verifying export on Langfuse v4

Langfuse v4 serves reads from a new observations-first data model. Trace data is
written to the `events_full` and `events_core` ClickHouse tables; the v3
`traces` and `observations` tables are legacy and stay empty on a v4 deployment
unless it runs the `legacy` or `dual` write mode. Use the Observations API v2 to
confirm traces arrived:

```bash
curl -sS -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  'http://127.0.0.1:3000/api/public/v2/observations?limit=5'
```

The deprecated `GET /api/public/traces` endpoint is not a valid check on v4.

The worker's `DUAL WRITE` log line reporting `No partitions available for
processing` is expected on a v4 deployment and does not indicate a failure. That
job only moves data for older SDKs during a `legacy` or `dual` mode migration.

### Running the Local Langfuse Stack

The stack lives in [`infra/langfuse/`](infra/langfuse/) under its own Compose
project. It needs the git-ignored `infra/langfuse/.env` to exist; it holds the
generated passwords, initial user, and API keys.

From the repository root:

```bash
docker compose -p signalforge-langfuse \
  -f infra/langfuse/compose.yaml \
  --env-file infra/langfuse/.env up -d
```

Then open **http://localhost:3000** and log in as `local@signalforge.test`. The
password is readable with:

```bash
awk -F= '$1 == "LANGFUSE_INIT_USER_PASSWORD" { print $2 }' infra/langfuse/.env
```

The `SignalForge Evaluation` project and its API key pair are created on startup.
Point the root `.env` at the local endpoint:

```bash
SIGNALFORGE_TRACING_ENABLED=1
SIGNALFORGE_OTLP_TRACES_ENDPOINT=http://127.0.0.1:3000/api/public/otel/v1/traces
```

Stop the stack while keeping its volumes with `down` in place of `up -d`.

#### Check the Docker context first

This stack is often invisible to a bare `docker ps`, because it may run on a
different Docker context than the one your CLI currently selects. `docker ps`
showing no containers does not mean the stack is down; port 3000 may still be
serving. Check which context owns it before starting a second copy:

```bash
docker context ls
docker context inspect --format '{{.Name}}' default

# If the stack is on another context, pass --context explicitly:
docker --context default compose -p signalforge-langfuse \
  -f infra/langfuse/compose.yaml \
  --env-file infra/langfuse/.env ps
```

Running `up -d` against the wrong context starts a duplicate stack that
collides on the published ports and splits trace data across two databases.

#### Recreating the stack

`--force-recreate` restarts containers but keeps volumes. Recreate the worker
last, or restart it separately, so it does not hold stale sockets to the
Postgres and Redis containers it was replaced alongside:

```bash
docker --context default compose -p signalforge-langfuse \
  -f infra/langfuse/compose.yaml \
  --env-file infra/langfuse/.env up -d --force-recreate

docker --context default compose -p signalforge-langfuse \
  -f infra/langfuse/compose.yaml \
  --env-file infra/langfuse/.env up -d --force-recreate \
  langfuse-worker langfuse-web
```

A worker left running against replaced dependencies logs
`Socket timeout. Expecting data, but didn't receive any in 30000ms` on every
queue job, including `event-propagation-queue`.

The separate SignalForge Docker Compose API cannot reach this stack at
`127.0.0.1:3000` from inside its container, because that address is the
container's own loopback. Use a locally run `make api` process with this stack,
or configure a container-reachable Langfuse endpoint.

## Local App

The lightweight local mode uses SQLite and embedded Qdrant paths by default:

```bash
SIGNALFORGE_DB_PATH=data/signalforge.sqlite3
SIGNALFORGE_QDRANT_PATH=data/qdrant
```

Run the API:

```bash
uv run uvicorn signalforge.api:app --reload --port 8000
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend calls `http://localhost:8000` by default.

You can use the Make targets for the same local workflow:

```bash
make api
make frontend
make worker
```

Run one background ingestion/vectorization cycle without starting the infinite
worker loop:

```bash
make worker-once
```

Inspect local index state from the terminal:

```bash
make index-state
```

## Postgres Development

Start Postgres and Qdrant through Docker Compose, then run local Python processes
against those services:

```bash
docker compose up -d postgres qdrant
export SIGNALFORGE_DATABASE_URL=postgresql+psycopg://signalforge:signalforge@localhost:15432/signalforge
export SIGNALFORGE_QDRANT_URL=http://localhost:6333
uv run alembic -c alembic.ini upgrade head
make api
```

In another shell, run the frontend:

```bash
make frontend
```

Run a one-shot worker cycle against Postgres/Qdrant:

```bash
make worker-once
```

## Docker Images

Build each application component independently:

```bash
docker build -f Dockerfile.api -t signalforge-api .
docker build -f Dockerfile.worker -t signalforge-worker .
docker build -f frontend/Dockerfile -t signalforge-frontend ./frontend
```

Run the API image:

```bash
docker run --rm -p 8000:8000 \
  -e SIGNALFORGE_DATABASE_URL=postgresql+psycopg://user:password@host.docker.internal:5432/signalforge \
  -e SIGNALFORGE_QDRANT_URL=http://host.docker.internal:6333 \
  signalforge-api
```

Run the worker image with the same database and Qdrant configuration:

```bash
docker run --rm \
  -e SIGNALFORGE_DATABASE_URL=postgresql+psycopg://user:password@host.docker.internal:5432/signalforge \
  -e SIGNALFORGE_QDRANT_URL=http://host.docker.internal:6333 \
  signalforge-worker
```

Run the frontend image and configure the API URL at container startup:

```bash
docker run --rm -p 8080:80 \
  -e SIGNALFORGE_API_BASE_URL=http://localhost:8000 \
  signalforge-frontend
```

Open `http://localhost:8080`.

## Docker Compose

Run the full local stack:

```bash
docker compose up --build
```

This starts:

- Postgres on `localhost:15432`
- Qdrant on `localhost:6333`
- API on `http://localhost:8000`
- Frontend on `http://localhost:8080`
- Worker background ingestion/vectorization loop

The stack uses named Docker volumes for Postgres data, Qdrant data, processed
article text, and raw SEC filing downloads. Stop the stack while keeping data:

```bash
docker compose down
```

Stop the stack and delete persisted local Docker data:

```bash
docker compose down -v
```

Copy `.env.example` to `.env` if you want to override ports, credentials, models,
worker settings, or optional DeepSeek/SEC values. Docker Compose reads `.env`
automatically.

Useful checks:

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/api/index
docker compose logs -f api
docker compose logs -f worker
```

Equivalent Make targets:

```bash
make compose-up
make compose-logs
make compose-down
```

## SQLite To Postgres Migration

Move an existing local SQLite database into Postgres with a dry run first:

```bash
uv run python -m signalforge.cli.migrate_sqlite_to_postgres \
  --sqlite-path data/signalforge.sqlite3 \
  --postgres-url postgresql+psycopg://signalforge:signalforge@localhost:15432/signalforge
```

If the dry run reports the expected row counts, run the migration:

```bash
uv run python -m signalforge.cli.migrate_sqlite_to_postgres \
  --sqlite-path data/signalforge.sqlite3 \
  --postgres-url postgresql+psycopg://signalforge:signalforge@localhost:15432/signalforge \
  --execute
```

The migration preserves IDs and timestamps, copies document metadata into Postgres
JSONB, validates row counts after import, and resets Postgres sequences. The
target database must be empty unless you pass `--replace`, which deletes existing
target rows before importing:

```bash
uv run python -m signalforge.cli.migrate_sqlite_to_postgres \
  --sqlite-path data/signalforge.sqlite3 \
  --postgres-url postgresql+psycopg://signalforge:signalforge@localhost:15432/signalforge \
  --execute \
  --replace
```

## Useful Commands

```bash
# Run database migrations for the configured database
uv run alembic -c alembic.ini upgrade head

# Ingest an existing local SEC download without hitting EDGAR
uv run python -m signalforge.cli.ingest --ticker NVDA --no-download

# Run one approved-source ingestion pass
uv run python -m signalforge.cli.ingest_sources

# Run one worker cycle: approved-source ingestion plus vectorization
uv run python -m signalforge.cli.run_worker_once

# Semantic search over indexed chunks
uv run python -m signalforge.cli.search "AI infrastructure risks" --ticker NVDA --section 1A

# Discover sources without saving candidates
uv run python -m signalforge.cli.discover_sources --ticker NVDA --website-domain nvidia.com --dry-run

# Inspect a query plan
uv run python -m signalforge.cli.plan_query "Compare NVDA and MSFT risk factors"

# Inspect database/index state
uv run python -m signalforge.cli.index_state
```

The same operations are available as Make targets:

```bash
make migrate
make worker-once
make vectorize
make index-state
```

## API Metadata

`GET /api/index` returns the local index state used by the frontend:

- Indexed filing coverage by ticker and section.
- Approved source count.
- Candidate source count.
- Web document count.
- Per-source document counts.
- Last ingestion status and completion time.

## Configuration

Default local paths:

- SQLite database: `data/signalforge.sqlite3`
- Qdrant store: `data/qdrant`
- Raw filings: `data/raw`
- Processed filing and article text: `data/processed`

Default models:

- Embeddings: `jinaai/jina-embeddings-v2-small-en`
- Planner: `deepseek-v4-flash`
- Answer generator: `deepseek-v4-flash`

The API supports these environment overrides:

- `SIGNALFORGE_DATABASE_URL`
- `SIGNALFORGE_DB_PATH`
- `SIGNALFORGE_QDRANT_URL`
- `SIGNALFORGE_QDRANT_PATH`
- `SIGNALFORGE_COLLECTION`
- `SIGNALFORGE_EMBEDDING_MODEL`
- `SIGNALFORGE_PLANNER_MODEL`
- `SIGNALFORGE_ANSWER_MODEL`
- `SIGNALFORGE_CORS_ORIGINS`

`SIGNALFORGE_CORS_ORIGINS` is a comma-separated browser origin allowlist. By default,
the API allows the local Vite dev server and a frontend container mapped to port `8080`.

Worker-specific overrides:

- `SIGNALFORGE_WORKER_INTERVAL_SECONDS`
- `SIGNALFORGE_INGEST_LIMIT_PER_SOURCE`
- `SIGNALFORGE_ENABLE_SCHEDULED_INGESTION`
- `SIGNALFORGE_PROCESSED_DIR`
- `SIGNALFORGE_CHUNK_SIZE`
- `SIGNALFORGE_CHUNK_OVERLAP`
- `SIGNALFORGE_VECTORIZE_BATCH_SIZE`
- `SIGNALFORGE_LOG_LEVEL`

Docker Compose-specific overrides are documented in `.env.example` and include:

- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `POSTGRES_PORT`
- `QDRANT_PORT`
- `API_PORT`
- `FRONTEND_PORT`
- `SIGNALFORGE_API_BASE_URL`
- `SIGNALFORGE_IMAGE_TAG`

`SIGNALFORGE_DATABASE_URL` takes precedence over `SIGNALFORGE_DB_PATH`.
`SIGNALFORGE_QDRANT_URL` takes precedence over `SIGNALFORGE_QDRANT_PATH`.
For Postgres, use a URL such as:

```bash
SIGNALFORGE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/signalforge
```

For Qdrant server mode, use:

```bash
SIGNALFORGE_QDRANT_URL=http://localhost:6333
```

Set a custom frontend API URL with:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

The frontend Docker image reads `SIGNALFORGE_API_BASE_URL` at container startup.

## Worker Behavior

The worker runs this cycle repeatedly:

```text
load approved enabled sources -> ingest new source documents
-> chunk documents -> vectorize pending SEC/document chunks -> sleep
```

It uses `SIGNALFORGE_WORKER_INTERVAL_SECONDS` between cycles. Set
`SIGNALFORGE_ENABLE_SCHEDULED_INGESTION=false` to skip source ingestion and only
vectorize pending chunks. Use `SIGNALFORGE_INGEST_LIMIT_PER_SOURCE` to cap article
ingestion per source during demos or testing.

For operational checks, prefer a one-shot cycle before starting the loop:

```bash
uv run python -m signalforge.cli.run_worker_once
```

## Backup And Restore

For the Docker Compose Postgres database, create a compressed backup:

```bash
make backup-postgres
```

This writes:

```text
backups/signalforge.dump
```

Restore that backup into the Compose Postgres service:

```bash
make restore-postgres
```

Qdrant data is stored in the `signalforge_qdrant-data` Docker volume. For a
simple local backup, stop the stack and archive the named volume with Docker or
your host backup tooling. For SQLite local development, back up
`data/signalforge.sqlite3`, `data/qdrant`, `data/raw`, and `data/processed`.

## Tests

```bash
uv run pytest

cd frontend
npm test
```

Run the optional Postgres persistence profile against a disposable database
whose name contains `test`:

```bash
SIGNALFORGE_POSTGRES_TEST_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/signalforge_test \
  uv run pytest -m postgres
```

## Notes

SignalForge is intended for local research workflows. Generated answers are grounded in retrieved filing and document chunks, but they are not financial advice. Verify important claims against the original SEC filing or source document.
