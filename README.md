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
