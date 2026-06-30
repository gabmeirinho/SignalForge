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

## Basic Workflow

Ingest a recent 10-K:

```bash
uv run python -m signalforge.cli.ingest --ticker NVDA --limit 1
```

Discover company sources:

```bash
uv run python -m signalforge.cli.discover_sources --ticker NVDA --website-domain nvidia.com
```

Review and approve sources:

```bash
uv run python -m signalforge.cli.list_sources --ticker NVDA
uv run python -m signalforge.cli.approve_source --source-id 1
```

Ingest approved sources:

```bash
uv run python -m signalforge.cli.ingest_sources --ticker NVDA --limit-per-source 5
```

Build or refresh the vector index:

```bash
uv run python -m signalforge.cli.vectorize
```

Ask a question:

```bash
uv run python -m signalforge.cli.answer_query "What does NVDA say about supply chain risk?" --show-plan --show-chunks
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

# Manually add a source
uv run python -m signalforge.cli.add_source \
  --ticker NVDA \
  --name "NVIDIA Newsroom" \
  --url "https://nvidianews.nvidia.com/" \
  --source-type newsroom \
  --ownership official \
  --trust-level high

# Reject a candidate source
uv run python -m signalforge.cli.reject_source --source-id 2

# Inspect a query plan
uv run python -m signalforge.cli.plan_query "Compare NVDA and MSFT risk factors"
```

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

- `SIGNALFORGE_DB_PATH`
- `SIGNALFORGE_QDRANT_PATH`
- `SIGNALFORGE_COLLECTION`
- `SIGNALFORGE_EMBEDDING_MODEL`
- `SIGNALFORGE_PLANNER_MODEL`
- `SIGNALFORGE_ANSWER_MODEL`

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

## Notes

SignalForge is intended for local research workflows. Generated answers are grounded in retrieved filing and document chunks, but they are not financial advice. Verify important claims against the original SEC filing or source document.
