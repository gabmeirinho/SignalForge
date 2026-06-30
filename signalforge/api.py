from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from signalforge.answer_generator import DEFAULT_ANSWER_MODEL
from signalforge.query_planner import DEFAULT_PLANNER_MODEL
from signalforge.rag_service import QueryResponse, answer_question
from signalforge.storage import (
    connect_database,
    initialize_database,
    load_index_metadata,
    load_index_section_counts,
)
from signalforge.vector_store import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL


load_dotenv()

app = FastAPI(title="SignalForge API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class ApiConfig:
    db_path: str = "data/signalforge.sqlite3"
    qdrant_path: str = "data/qdrant"
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    planner_model: str = DEFAULT_PLANNER_MODEL
    answer_model: str = DEFAULT_ANSWER_MODEL

    @classmethod
    def from_environment(cls) -> "ApiConfig":
        import os

        return cls(
            db_path=os.getenv("SIGNALFORGE_DB_PATH", cls.db_path),
            qdrant_path=os.getenv("SIGNALFORGE_QDRANT_PATH", cls.qdrant_path),
            collection=os.getenv("SIGNALFORGE_COLLECTION", cls.collection),
            embedding_model=os.getenv("SIGNALFORGE_EMBEDDING_MODEL", cls.embedding_model),
            planner_model=os.getenv("SIGNALFORGE_PLANNER_MODEL", cls.planner_model),
            answer_model=os.getenv("SIGNALFORGE_ANSWER_MODEL", cls.answer_model),
        )


class HealthResponse(BaseModel):
    status: str
    database: bool
    qdrant_path: bool


class IndexFiling(BaseModel):
    accession_number: str
    form_type: str
    filing_date: str | None
    period_of_report: str | None
    status: str
    expected_point_count: int
    indexed_point_count: int


class IndexSection(BaseModel):
    section_id: str
    chunk_count: int


class IndexTicker(BaseModel):
    ticker: str
    company_name: str | None
    filings: list[IndexFiling]
    sections: list[IndexSection]


class IndexResponse(BaseModel):
    tickers: list[IndexTicker]
    embedding_model: str
    collection: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    include_plan: bool = True
    include_source_text: bool = True

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class SourceResponse(BaseModel):
    label: str
    score: float
    chunk_source: str
    ticker: str | None
    company_name: str | None
    filing_date: str | None
    published_at: str | None
    section_id: str | None
    section_title: str | None
    chunk_index: int | None
    accession_number: str | None
    document_id: int | None
    source_id: int | None
    source_name: str | None
    source_type: str | None
    url: str | None
    title: str | None
    text: str | None


class QueryApiResponse(BaseModel):
    question: str
    answer: str
    warnings: list[str]
    used_fallback: bool
    planner_error: str | None
    plan: dict | None
    sources: list[SourceResponse]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    config = ApiConfig.from_environment()
    return HealthResponse(
        status="ok",
        database=Path(config.db_path).exists(),
        qdrant_path=Path(config.qdrant_path).exists(),
    )


@app.get("/api/index", response_model=IndexResponse)
def index() -> IndexResponse:
    config = ApiConfig.from_environment()
    ensure_local_index(config)

    with connect_database(config.db_path) as connection:
        initialize_database(connection)
        filing_rows = load_index_metadata(
            connection,
            embedding_model=config.embedding_model,
            vector_collection=config.collection,
        )
        section_rows = load_index_section_counts(connection)

    sections_by_ticker: dict[str, list[IndexSection]] = defaultdict(list)
    for row in section_rows:
        sections_by_ticker[row["ticker"]].append(
            IndexSection(
                section_id=row["section_id"],
                chunk_count=int(row["chunk_count"]),
            )
        )

    ticker_groups: dict[str, dict] = {}
    for row in filing_rows:
        ticker = row["ticker"]
        group = ticker_groups.setdefault(
            ticker,
            {
                "ticker": ticker,
                "company_name": row["company_name"],
                "filings": [],
            },
        )
        group["filings"].append(
            IndexFiling(
                accession_number=row["accession_number"],
                form_type=row["form_type"],
                filing_date=row["filing_date"],
                period_of_report=row["period_of_report"],
                status=row["status"],
                expected_point_count=int(row["expected_point_count"]),
                indexed_point_count=int(row["indexed_point_count"]),
            )
        )

    tickers = [
        IndexTicker(
            ticker=group["ticker"],
            company_name=group["company_name"],
            filings=group["filings"],
            sections=sections_by_ticker.get(group["ticker"], []),
        )
        for group in ticker_groups.values()
    ]

    return IndexResponse(
        tickers=tickers,
        embedding_model=config.embedding_model,
        collection=config.collection,
    )


@app.post("/api/query", response_model=QueryApiResponse)
def query(request: QueryRequest) -> QueryApiResponse:
    config = ApiConfig.from_environment()

    try:
        response = answer_question(
            request.question,
            db_path=config.db_path,
            qdrant_path=config.qdrant_path,
            collection=config.collection,
            embedding_model=config.embedding_model,
            planner_model=config.planner_model,
            answer_model=config.answer_model,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {type(error).__name__}") from error

    return query_response_to_api(
        response,
        include_plan=request.include_plan,
        include_source_text=request.include_source_text,
    )


def ensure_local_index(config: ApiConfig) -> None:
    if not Path(config.db_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"SignalForge database not found at {config.db_path}",
        )
    if not Path(config.qdrant_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"SignalForge Qdrant index not found at {config.qdrant_path}",
        )


def query_response_to_api(
    response: QueryResponse,
    *,
    include_plan: bool,
    include_source_text: bool,
) -> QueryApiResponse:
    return QueryApiResponse(
        question=response.question,
        answer=response.answer,
        warnings=response.warnings,
        used_fallback=response.used_fallback,
        planner_error=response.planner_error,
        plan=response.plan.model_dump() if include_plan else None,
        sources=[
            SourceResponse(
                label=source.label,
                score=source.score,
                chunk_source=source.chunk_source,
                ticker=source.ticker,
                company_name=source.company_name,
                filing_date=source.filing_date,
                published_at=source.published_at,
                section_id=source.section_id,
                section_title=source.section_title,
                chunk_index=source.chunk_index,
                accession_number=source.accession_number,
                document_id=source.document_id,
                source_id=source.source_id,
                source_name=source.source_name,
                source_type=source.source_type,
                url=source.url,
                title=source.title,
                text=source.text if include_source_text else None,
            )
            for source in response.sources
        ],
    )
