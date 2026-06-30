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

## Useful Commands

```bash
# Ingest an existing local SEC download without hitting EDGAR
uv run python -m signalforge.cli.ingest --ticker NVDA --no-download

# Semantic search over indexed chunks
uv run python -m signalforge.cli.search "AI infrastructure risks" --ticker NVDA --section 1A

# Discover sources without saving candidates
uv run python -m signalforge.cli.discover_sources --ticker NVDA --website-domain nvidia.com --dry-run

# Inspect a query plan
uv run python -m signalforge.cli.plan_query "Compare NVDA and MSFT risk factors"
```

Create or update a database schema with Alembic migrations:

```bash
uv run alembic -c alembic.ini upgrade head
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

`SIGNALFORGE_DATABASE_URL` takes precedence over `SIGNALFORGE_DB_PATH`.
`SIGNALFORGE_QDRANT_URL` takes precedence over `SIGNALFORGE_QDRANT_PATH`.
For Postgres, use a URL such as:

```bash
SIGNALFORGE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/signalforge
```

Set a custom frontend API URL with:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

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
