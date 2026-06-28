import argparse
import json

from dotenv import load_dotenv

from answer_generator import DEFAULT_ANSWER_MODEL, create_answer_generator_from_environment
from plan_query import build_planner_context
from query_planner import DEFAULT_PLANNER_MODEL, PlannerContext, SearchPlan, create_query_planner_from_environment
from storage import (
    connect_database,
    get_ready_accession_numbers,
    initialize_database,
    load_planner_metadata,
)
from vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    create_qdrant_client,
    retrieve_chunks,
)


def main() -> None:
    load_dotenv()
    args = parse_args()

    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        context = build_planner_context(load_planner_metadata(connection))
        planner = create_query_planner_from_environment(model=args.planner_model)
        planning_result = planner.create_plan(args.question, context)
        plan = planning_result.plan
        accessions_by_ticker = select_ready_accessions_by_ticker(
            connection,
            plan=plan,
            context=context,
            embedding_model=args.embedding_model,
            collection=args.collection,
        )

    accessions = sorted(
        {
            accession
            for ticker_accessions in accessions_by_ticker.values()
            for accession in ticker_accessions
        }
    )

    chunks = []
    if accessions:
        client = create_qdrant_client(args.qdrant_path)
        try:
            chunks = retrieve_chunks(
                client,
                query=plan.semantic_queries[0],
                collection_name=args.collection,
                embedding_model=args.embedding_model,
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

    generator = create_answer_generator_from_environment(model=args.answer_model)
    generated = generator.generate(
        question=args.question,
        plan=plan,
        chunks=chunks,
        available_years_by_ticker=context.filing_years_by_ticker,
    )

    if args.show_plan:
        print(json.dumps({"plan": plan.model_dump(), "used_fallback": planning_result.used_fallback}, indent=2))
        print()
    if args.show_chunks:
        print_chunk_summary(chunks)
        print()

    print(generated.answer)


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


def print_chunk_summary(chunks) -> None:
    if not chunks:
        print("Retrieved chunks: (none)")
        return

    print("Retrieved chunks:")
    for index, result in enumerate(chunks, start=1):
        payload = result.payload
        print(
            f"{index}. score={result.score:.4f} "
            f"{payload.get('ticker')} {payload.get('filing_date')} "
            f"Item {payload.get('section_id')} chunk {payload.get('chunk_index')} "
            f"accession={payload.get('accession_number')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a question from local SEC filing chunks.")
    parser.add_argument("question")
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    parser.add_argument("--qdrant-path", default="data/qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--show-chunks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
