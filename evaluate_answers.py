import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from answer_generator import AnswerGenerator, DEFAULT_ANSWER_MODEL
from answer_query import select_ready_accessions_by_ticker, years_for_plan_scope
from evaluate_planner import compare_plan
from query_planner import (
    DEFAULT_PLANNER_MODEL,
    DeepSeekQueryPlanner,
    PlannerContext,
    SearchPlan,
)
from storage import connect_database, initialize_database
from vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    create_qdrant_client,
    retrieve_chunks,
)


DEFAULT_DATASET = Path("tests/fixtures/planner_golden.json")
CITATION_RE = re.compile(r"\[\d+\]")


def main() -> None:
    load_dotenv()
    args = parse_args()
    cases = json.loads(args.dataset.read_text())
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise SystemExit(f"No golden case found for id: {args.case_id}")

    planner = None
    if not args.use_expected_plan:
        planner = DeepSeekQueryPlanner.from_environment(model=args.planner_model)
    generator = AnswerGenerator.from_environment(model=args.answer_model)
    retrieval_client = create_qdrant_client(args.qdrant_path)

    try:
        with connect_database(args.db_path) as connection:
            initialize_database(connection)
            passed = 0
            for case in cases:
                failures = evaluate_case(
                    case,
                    connection=connection,
                    planner=planner,
                    generator=generator,
                    retrieval_client=retrieval_client,
                    collection=args.collection,
                    embedding_model=args.embedding_model,
                    use_expected_plan=args.use_expected_plan,
                    limit=args.limit,
                )
                if failures:
                    print(f"FAIL {case['id']}")
                    for failure in failures:
                        print(f"  - {failure}")
                else:
                    passed += 1
                    print(f"PASS {case['id']}")
                print()
    finally:
        retrieval_client.close()

    total = len(cases)
    print(f"Score: {passed}/{total} ({passed / total:.0%})")
    raise SystemExit(0 if passed == total else 1)


def evaluate_case(
    case: dict,
    *,
    connection,
    planner,
    generator: AnswerGenerator,
    retrieval_client,
    collection: str,
    embedding_model: str,
    use_expected_plan: bool,
    limit: int,
) -> list[str]:
    context = planner_context_from_case(case)
    plan_result = (
        expected_plan_from_case(case, limit=limit)
        if use_expected_plan
        else planner.create_plan(case["question"], context)
    )
    plan = plan_result.plan
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
    if accessions:
        chunks = retrieve_chunks(
            retrieval_client,
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

    generated = generator.generate(
        question=case["question"],
        plan=plan,
        chunks=chunks,
        available_years_by_ticker=context.filing_years_by_ticker,
    )
    print_case_output(case, plan_result, chunks, generated.answer)

    return evaluate_answer_quality(
        answer=generated.answer,
        chunks=chunks,
        actual_plan=plan.model_dump(),
        expected=case["expected"],
        context=context,
        used_fallback=plan_result.used_fallback,
        plan_error=plan_result.error,
        use_expected_plan=use_expected_plan,
    )


def planner_context_from_case(case: dict) -> PlannerContext:
    context_data = case["context"]
    return PlannerContext(
        available_tickers=tuple(context_data["available_tickers"]),
        available_sections=tuple(context_data["available_sections"]),
        filing_years_by_ticker={
            ticker: tuple(years)
            for ticker, years in context_data["filing_years_by_ticker"].items()
        },
    )


def expected_plan_from_case(case: dict, *, limit: int):
    expected = case["expected"]
    plan = SearchPlan(
        tickers=expected.get("tickers", []),
        sections=expected.get("sections", []),
        semantic_queries=[case["question"]],
        time_scope=expected.get("time_scope", "latest"),
        filing_years=expected.get("filing_years", []),
        intent=expected.get("intent", "summary"),
        top_k=expected.get("top_k", limit),
    )
    return SimpleNamespace(plan=plan, used_fallback=False, error=None)


def evaluate_answer_quality(
    *,
    answer: str,
    chunks: list,
    actual_plan: dict,
    expected: dict,
    context: PlannerContext,
    used_fallback: bool,
    plan_error: str | None,
    use_expected_plan: bool,
) -> list[str]:
    failures = []
    if not use_expected_plan:
        failures.extend(compare_plan(actual_plan, expected))
    if used_fallback:
        failures.append(f"unexpected planner fallback: {plan_error}")

    unavailable_specific_year = (
        expected.get("time_scope") == "specific_years"
        and not expected.get("filing_years")
    )
    if unavailable_specific_year:
        if chunks:
            failures.append("unavailable-year case retrieved chunks")
        if "not available" not in answer.lower():
            failures.append("unavailable-year answer does not explain missing data")
        return failures

    if chunks and CITATION_RE.search(answer) is None:
        failures.append("answer has retrieved chunks but no citation labels")

    expected_tickers = set(expected.get("tickers") or [])
    if expected.get("intent") == "comparison" and expected_tickers:
        answer_upper = answer.upper()
        missing_tickers = {
            ticker for ticker in expected_tickers if ticker not in answer_upper
        }
        if missing_tickers:
            failures.append(f"answer missing comparison tickers: {sorted(missing_tickers)!r}")

    expected_years = set()
    if expected.get("intent") == "trend" or expected.get("time_scope") == "all_available":
        plan_for_years = SearchPlan(
            tickers=expected.get("tickers", []),
            sections=expected.get("sections", []),
            semantic_queries=["unused"],
            time_scope=expected.get("time_scope", "latest"),
            filing_years=expected.get("filing_years", []),
            intent=expected.get("intent", "summary"),
        )
        for ticker in expected.get("tickers") or [None]:
            expected_years.update(years_for_plan_scope(plan_for_years, ticker=ticker, context=context))

    if expected_years:
        missing_years = {
            year for year in expected_years if str(year) not in answer
        }
        if missing_years:
            failures.append(f"answer missing trend years: {sorted(missing_years)!r}")

    return failures


def print_case_output(case: dict, plan_result, chunks: list, answer: str) -> None:
    print(f"CASE {case['id']}")
    print(f"Question: {case['question']}")
    print(f"Plan: {json.dumps(plan_result.plan.model_dump(), sort_keys=True)}")
    print(f"Retrieved chunks: {len(chunks)}")
    print("Answer:")
    print(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated answers on golden cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Golden dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--case-id", help="Run one golden case by id.")
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    parser.add_argument("--qdrant-path", default="data/qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--use-expected-plan",
        action="store_true",
        help="Skip live planning and answer from each golden case's expected plan.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
