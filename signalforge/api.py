import os
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from signalforge.config import RuntimeConfig, target_exists
from signalforge.index_health import CorpusIndexHealth, IndexHealth, check_index_health
from signalforge.rag_service import QueryResponse, answer_question
from signalforge.storage import (
    connect_database,
    initialize_database,
    list_sources,
    load_index_metadata,
    load_index_section_counts,
)
from signalforge.vector_store import create_qdrant_client


load_dotenv()

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]


def configured_cors_origins() -> list[str]:
    value = os.getenv("SIGNALFORGE_CORS_ORIGINS")
    if value is None:
        return DEFAULT_CORS_ORIGINS
    return [origin for raw_origin in value.split(",") if (origin := raw_origin.strip())]


def format_optional_datetime(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


app = FastAPI(title="SignalForge API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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


class IndexSource(BaseModel):
    id: int
    ticker: str | None
    company_name: str | None
    name: str
    url: str
    source_type: str
    ownership: str
    trust_level: str
    discovery_status: str
    enabled: bool
    confidence_score: float | None
    document_count: int
    last_ingestion_status: str | None
    last_ingestion_completed_at: str | None


class IndexSummary(BaseModel):
    indexed_filing_count: int
    approved_source_count: int
    candidate_source_count: int
    document_count: int


class IndexResponse(BaseModel):
    tickers: list[IndexTicker]
    sources: list[IndexSource]
    summary: IndexSummary
    embedding_model: str
    collection: str


class CorpusIndexHealthResponse(BaseModel):
    name: str
    postgres_expected_points: int
    postgres_ready_points: int
    postgres_embedding_records: int
    qdrant_points: int
    missing_qdrant_points: int
    extra_qdrant_points: int
    is_complete_in_postgres: bool
    is_consistent_with_qdrant: bool


class IndexHealthResponse(BaseModel):
    status: str
    collection: str
    collection_exists: bool
    embedding_model: str
    total_postgres_expected_points: int
    total_qdrant_points: int
    sec: CorpusIndexHealthResponse
    documents: CorpusIndexHealthResponse


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
    config = RuntimeConfig.from_environment()
    return HealthResponse(
        status="ok",
        database=target_exists(config.database_target),
        qdrant_path=target_exists(config.qdrant_target),
    )


@app.get("/api/index", response_model=IndexResponse)
def index() -> IndexResponse:
    config = RuntimeConfig.from_environment()
    ensure_local_index(config)

    with connect_database(config.database_target) as connection:
        initialize_database(connection)
        filing_rows = load_index_metadata(
            connection,
            embedding_model=config.embedding_model,
            vector_collection=config.collection,
        )
        section_rows = load_index_section_counts(connection)
        source_rows = list_sources(connection)

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
    sources = [
        IndexSource(
            id=int(row["id"]),
            ticker=row["ticker"],
            company_name=row["company_name"],
            name=row["name"],
            url=row["url"],
            source_type=row["source_type"],
            ownership=row["ownership"],
            trust_level=row["trust_level"],
            discovery_status=row["discovery_status"],
            enabled=bool(row["enabled"]),
            confidence_score=row["confidence_score"],
            document_count=int(row["document_count"]),
            last_ingestion_status=row["last_ingestion_status"],
            last_ingestion_completed_at=format_optional_datetime(row["last_ingestion_completed_at"]),
        )
        for row in source_rows
    ]
    summary = IndexSummary(
        indexed_filing_count=sum(
            1
            for row in filing_rows
            if row["status"] == "ready"
            and int(row["expected_point_count"]) > 0
            and int(row["indexed_point_count"]) >= int(row["expected_point_count"])
        ),
        approved_source_count=sum(
            1 for source in sources if source.discovery_status in {"approved", "manual"} and source.enabled
        ),
        candidate_source_count=sum(1 for source in sources if source.discovery_status == "candidate"),
        document_count=sum(source.document_count for source in sources),
    )

    return IndexResponse(
        tickers=tickers,
        sources=sources,
        summary=summary,
        embedding_model=config.embedding_model,
        collection=config.collection,
    )


@app.get("/api/index/health", response_model=IndexHealthResponse)
def index_health() -> IndexHealthResponse:
    config = RuntimeConfig.from_environment()
    ensure_local_index(config)

    with connect_database(config.database_target) as connection:
        initialize_database(connection)
        client = create_qdrant_client(config.qdrant_target)
        try:
            health = check_index_health(
                connection,
                client,
                collection=config.collection,
                embedding_model=config.embedding_model,
            )
        finally:
            client.close()

    return index_health_response(health)


@app.post("/api/query", response_model=QueryApiResponse)
def query(request: QueryRequest) -> QueryApiResponse:
    config = RuntimeConfig.from_environment()

    try:
        response = answer_question(
            request.question,
            db_path=config.database_target,
            qdrant_path=config.qdrant_target,
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


def ensure_local_index(config: RuntimeConfig) -> None:
    if not target_exists(config.database_target):
        raise HTTPException(
            status_code=503,
            detail=f"SignalForge database not found at {config.database_target}",
        )
    if not target_exists(config.qdrant_target):
        raise HTTPException(
            status_code=503,
            detail=f"SignalForge Qdrant index not found at {config.qdrant_target}",
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


def corpus_index_health_response(health: CorpusIndexHealth) -> CorpusIndexHealthResponse:
    return CorpusIndexHealthResponse(
        name=health.name,
        postgres_expected_points=health.postgres_expected_points,
        postgres_ready_points=health.postgres_ready_points,
        postgres_embedding_records=health.postgres_embedding_records,
        qdrant_points=health.qdrant_points,
        missing_qdrant_points=health.missing_qdrant_points,
        extra_qdrant_points=health.extra_qdrant_points,
        is_complete_in_postgres=health.is_complete_in_postgres,
        is_consistent_with_qdrant=health.is_consistent_with_qdrant,
    )


def index_health_response(health: IndexHealth) -> IndexHealthResponse:
    return IndexHealthResponse(
        status=health.status,
        collection=health.collection,
        collection_exists=health.collection_exists,
        embedding_model=health.embedding_model,
        total_postgres_expected_points=health.total_postgres_expected_points,
        total_qdrant_points=health.total_qdrant_points,
        sec=corpus_index_health_response(health.sec),
        documents=corpus_index_health_response(health.documents),
    )
