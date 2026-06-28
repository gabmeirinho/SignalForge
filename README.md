# SignalForge

SignalForge is a local market intelligence pipeline for SEC EDGAR 10-K filings. It downloads filings, extracts key sections, chunks them into a SQLite database, indexes those chunks in a local Qdrant vector store, and answers financial research questions with cited filing evidence.

The project is script-first and designed for experimentation with retrieval, query planning, and answer generation over company filings.

## Features

- Download recent SEC 10-K filings by ticker.
- Parse and clean filing text from raw EDGAR submissions.
- Extract common 10-K sections:
  - Item 1: Business
  - Item 1A: Risk Factors
  - Item 7: Management's Discussion and Analysis
  - Item 7A: Quantitative and Qualitative Disclosures About Market Risk
- Store filings, sections, chunks, and embedding status in SQLite.
- Build a local Qdrant vector index with FastEmbed embeddings.
- Plan SEC retrieval queries with DeepSeek, with a local rule-based fallback when no API key is configured.
- Generate cited answers from retrieved chunks, with extractive fallback output when LLM generation is unavailable.
- Evaluate planner, retrieval, and answer behavior against golden test cases.

## Project Structure

```text
.
|-- ingest.py              # CLI for downloading and ingesting SEC filings
|-- vectorize.py           # CLI for embedding chunks into local Qdrant
|-- search.py              # CLI for semantic search over indexed chunks
|-- plan_query.py          # CLI for inspecting query plans
|-- answer_query.py        # CLI for end-to-end question answering
|-- evaluate_planner.py    # Planner golden-case evaluation
|-- evaluate_retrieval.py  # Retrieval golden-case evaluation
|-- evaluate_answers.py    # Answer-generation golden-case evaluation
|-- ingestion.py           # Ingestion orchestration
|-- parser.py              # SEC filing text parsing
|-- sections.py            # 10-K section extraction and chunking
|-- storage.py             # SQLite schema and persistence
|-- vector_store.py        # Qdrant indexing and retrieval helpers
|-- query_planner.py       # LLM and fallback query planners
|-- answer_generator.py    # LLM and extractive answer generators
`-- tests/                 # Unit tests and golden fixtures
```

Generated data is intentionally ignored by git:

- `data/signalforge.sqlite3`
- `data/qdrant/`
- `data/raw/`
- `data/processed/`
- `sec-edgar-filings/`

## Requirements

- Python 3.11 or newer
- `uv`
- SEC EDGAR user-agent details for downloads
- Optional: DeepSeek API key for LLM query planning and answer generation

The vector store runs locally through `qdrant-client` embedded storage. No external Qdrant server is required for the default workflow.

## Setup

Install dependencies:

```bash
uv sync --extra dev
```

Create a local `.env` file:

```bash
SEC_COMPANY_NAME="Your Name or App Name"
SEC_EMAIL="you@example.com"

# Optional, enables LLM planning and answer generation.
DEEPSEEK_API_KEY="your_deepseek_api_key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

`DEEPSEEK_API_KEY` is optional. Without it, SignalForge still supports local rule-based planning and extractive evidence output, but generated answers will not use the LLM.

## Quickstart

Ingest a recent 10-K:

```bash
uv run python ingest.py --ticker NVDA --limit 1
```

Build the vector index:

```bash
uv run python vectorize.py
```

Run semantic search:

```bash
uv run python search.py "What are the main AI infrastructure risks?" --ticker NVDA --section 1A
```

Ask an end-to-end question:

```bash
uv run python answer_query.py "What does NVDA say about supply chain risk?" --show-plan --show-chunks
```

## Common Commands

Ingest an existing local SEC download without hitting EDGAR:

```bash
uv run python ingest.py --ticker NVDA --no-download
```

Ingest more than one filing:

```bash
uv run python ingest.py --ticker MSFT --limit 3
```

Use a custom database or vector-store path:

```bash
uv run python ingest.py --ticker AAPL --db-path data/custom.sqlite3
uv run python vectorize.py --db-path data/custom.sqlite3 --qdrant-path data/custom-qdrant
uv run python answer_query.py "Summarize AAPL revenue risks" --db-path data/custom.sqlite3 --qdrant-path data/custom-qdrant
```

Inspect only the planner output:

```bash
uv run python plan_query.py "Compare NVDA and MSFT risk factors in their latest filings"
```

## Evaluation

Run the unit test suite:

```bash
uv run pytest
```

Run planner golden-case evaluation:

```bash
uv run python evaluate_planner.py
```

Run retrieval golden-case evaluation:

```bash
uv run python evaluate_retrieval.py
```

Run answer evaluation:

```bash
uv run python evaluate_answers.py
```

You can run a single golden case with `--case-id`:

```bash
uv run python evaluate_answers.py --case-id latest_risk_factors
```

## Configuration

Default local paths:

- SQLite database: `data/signalforge.sqlite3`
- Qdrant store: `data/qdrant`
- Raw filings: `data/raw`
- Processed filing text: `data/processed`

Default models:

- Embeddings: `jinaai/jina-embeddings-v2-small-en`
- Planner: `deepseek-v4-flash`
- Answer generator: `deepseek-v4-flash`

Most scripts expose flags for database path, Qdrant path, collection name, model names, and limits. Run any script with `--help` for the full set of options.

## Notes

SignalForge is intended for local research workflows. Generated answers are grounded in retrieved filing chunks, but they are not financial advice. Always verify important claims against the original SEC filing.
