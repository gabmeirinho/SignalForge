from dataclasses import dataclass
from pathlib import Path

from signalforge.answer_generator import (
    DEFAULT_ANSWER_MODEL,
    create_answer_generator_from_environment,
    format_source_label,
)
from signalforge.query_planner import (
    DEFAULT_PLANNER_MODEL,
    PlannerContext,
    SearchPlan,
    build_planner_context,
    create_query_planner_from_environment,
)
from signalforge.storage import (
    connect_database,
    get_ready_accession_numbers,
    initialize_database,
    load_planner_metadata,
)
from signalforge.vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    SearchResult,
    create_qdrant_client,
    retrieve_chunks,
)


@dataclass(frozen=True)
class SourceChunk:
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
    text: str


@dataclass(frozen=True)
class QueryResponse:
    question: str
    answer: str
    plan: SearchPlan
    used_fallback: bool
    planner_error: str | None
    warnings: list[str]
    sources: list[SourceChunk]


def answer_question(
    question: str,
    *,
    db_path: str = "data/signalforge.sqlite3",
    qdrant_path: str = "data/qdrant",
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    planner_model: str = DEFAULT_PLANNER_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
) -> QueryResponse:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"SignalForge database not found at {db_path}")
    if not Path(qdrant_path).exists():
        raise FileNotFoundError(f"SignalForge Qdrant index not found at {qdrant_path}")

    with connect_database(db_path) as connection:
        initialize_database(connection)
        context = build_planner_context(load_planner_metadata(connection))
        planner = create_query_planner_from_environment(model=planner_model)
        planning_result = planner.create_plan(question, context)
        plan = planning_result.plan
        accessions_by_ticker = select_ready_accessions_by_ticker(
            connection,
            plan=plan,
            context=context,
            embedding_model=embedding_model,
            collection=collection,
        )

    accessions = sorted(
        {
            accession
            for ticker_accessions in accessions_by_ticker.values()
            for accession in ticker_accessions
        }
    )

    chunks = []
    if plan.tickers:
        client = create_qdrant_client(qdrant_path)
        try:
            chunks = retrieve_chunks(
                client,
                query=plan.semantic_queries[0],
                collection_name=collection,
                embedding_model=embedding_model,
                limit=plan.top_k,
                tickers=plan.tickers,
                section_ids=plan.sections,
                accession_numbers=accessions,
                accession_numbers_by_ticker=accessions_by_ticker,
                intent=plan.intent,
                time_scope=plan.time_scope,
            )
        finally:
            client.close()

    generator = create_answer_generator_from_environment(model=answer_model)
    generated = generator.generate(
        question=question,
        plan=plan,
        chunks=chunks,
        available_years_by_ticker=context.filing_years_by_ticker,
    )

    return QueryResponse(
        question=question,
        answer=generated.answer,
        plan=plan,
        used_fallback=planning_result.used_fallback,
        planner_error=planning_result.error,
        warnings=generated.warnings,
        sources=source_chunks_from_results(chunks),
    )


def select_ready_accessions_by_ticker(
    connection,
    *,
    plan: SearchPlan,
    context: PlannerContext,
    embedding_model: str,
    collection: str,
) -> dict[str, list[str]]:
    if not plan.tickers:
        return {}

    accessions_by_ticker = {}

    for ticker in plan.tickers:
        years = years_for_plan_scope(plan, ticker=ticker, context=context)
        key = ticker or ""
        accessions_by_ticker[key] = get_ready_accession_numbers(
            connection,
            embedding_model=embedding_model,
            vector_collection=collection,
            ticker=ticker,
            filing_years=years,
        )

    return accessions_by_ticker


def years_for_plan_scope(
    plan: SearchPlan,
    *,
    ticker: str | None,
    context: PlannerContext,
) -> list[int] | None:
    if plan.time_scope == "specific_years":
        return plan.filing_years

    available_years = sorted(
        (
            context.filing_years_by_ticker.get(ticker, ())
            if ticker
            else {year for years in context.filing_years_by_ticker.values() for year in years}
        ),
        reverse=True,
    )

    if plan.time_scope == "latest":
        return available_years[:1]
    if plan.time_scope == "latest_and_previous":
        return available_years[:2]
    if plan.time_scope == "all_available":
        return available_years

    return None


def source_chunks_from_results(chunks: list[SearchResult]) -> list[SourceChunk]:
    sources = []
    for index, result in enumerate(chunks, start=1):
        payload = result.payload
        sources.append(
            SourceChunk(
                label=format_source_label(index, payload),
                score=float(result.score),
                chunk_source=payload.get("chunk_source", "sec_filing"),
                ticker=payload.get("ticker"),
                company_name=payload.get("company_name"),
                filing_date=payload.get("filing_date"),
                published_at=payload.get("published_at"),
                section_id=payload.get("section_id"),
                section_title=payload.get("section_title"),
                chunk_index=payload.get("chunk_index"),
                accession_number=payload.get("accession_number"),
                document_id=payload.get("document_id"),
                source_id=payload.get("source_id"),
                source_name=payload.get("source_name"),
                source_type=payload.get("source_type"),
                url=payload.get("url"),
                title=payload.get("title"),
                text=payload.get("text", ""),
            )
        )
    return sources
