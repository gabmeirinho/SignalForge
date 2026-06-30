import argparse
import json
import textwrap
from pathlib import Path

from signalforge.answer_generator import format_source_label
from signalforge.storage import connect_database, get_ready_accession_numbers, initialize_database
from signalforge.vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    create_qdrant_client,
    retrieve_chunks,
)


DEFAULT_DATASET = Path("tests/fixtures/planner_golden.json")


def main() -> None:
    args = parse_args()
    cases = json.loads(args.dataset.read_text())
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise SystemExit(f"No golden case found for id: {args.case_id}")

    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        client = create_qdrant_client(args.qdrant_path)
        try:
            passed = 0
            for case in cases:
                failures, result_count = evaluate_case(
                    case,
                    connection=connection,
                    client=client,
                    model=args.model,
                    collection=args.collection,
                    limit=args.limit,
                    preview_chars=args.preview_chars,
                    show_full_text=args.full_text,
                )
                if failures:
                    print(f"FAIL {case['id']}")
                    for failure in failures:
                        print(f"  - {failure}")
                else:
                    passed += 1
                    print(f"PASS {case['id']} ({result_count} chunks)")
                print()
        finally:
            client.close()

    total = len(cases)
    print(f"Score: {passed}/{total} ({passed / total:.0%})")
    raise SystemExit(0 if passed == total else 1)


def evaluate_case(
    case: dict,
    *,
    connection,
    client,
    model: str,
    collection: str,
    limit: int,
    preview_chars: int,
    show_full_text: bool,
) -> tuple[list[str], int]:
    expected = case["expected"]
    tickers = expected.get("tickers") or []
    sections = expected.get("sections") or []
    top_k = int(expected.get("top_k") or limit)
    accessions_by_ticker = select_ready_accessions_by_ticker(
        connection,
        case=case,
        model=model,
        collection=collection,
    )
    accessions = sorted(
        {
            accession
            for ticker_accessions in accessions_by_ticker.values()
            for accession in ticker_accessions
        }
    )

    print(f"CASE {case['id']}")
    print(f"Question: {case['question']}")
    print(f"Expected plan: {json.dumps(expected, sort_keys=True)}")
    print(f"Ready accessions: {', '.join(accessions) if accessions else '(none)'}")

    if not accessions:
        if expected.get("time_scope") == "specific_years" and not expected.get("filing_years"):
            print("No retrieval expected: requested filing year is unavailable.")
            return [], 0
        return ["no ready accessions matched the expected ticker/year scope"], 0

    results = retrieve_chunks(
        client,
        query=case["question"],
        collection_name=collection,
        embedding_model=model,
        limit=max(top_k, limit),
        tickers=tickers,
        section_ids=sections,
        accession_numbers=accessions,
        accession_numbers_by_ticker=accessions_by_ticker,
        intent=expected.get("intent", "summary"),
        time_scope=expected.get("time_scope", "latest"),
    )
    print_retrieved_chunks(results, preview_chars=preview_chars, show_full_text=show_full_text)

    expected_years = sorted(
        {
            year
            for ticker in tickers or [None]
            for year in expected_years_for_ticker(
                expected,
                ticker=ticker,
                years_by_ticker=case["context"]["filing_years_by_ticker"],
            )
        }
    )
    return compare_retrieved_scope(results, expected, expected_years=expected_years), len(results)


def select_ready_accessions_by_ticker(
    connection,
    *,
    case: dict,
    model: str,
    collection: str,
) -> dict[str, list[str]]:
    expected = case["expected"]
    tickers = expected.get("tickers") or []
    years_by_ticker = case["context"]["filing_years_by_ticker"]
    accessions_by_ticker = {}

    for ticker in tickers or [None]:
        years = expected_years_for_ticker(
            expected,
            ticker=ticker,
            years_by_ticker=years_by_ticker,
        )
        key = ticker or ""
        accessions_by_ticker[key] = get_ready_accession_numbers(
            connection,
            embedding_model=model,
            vector_collection=collection,
            ticker=ticker,
            filing_years=years,
        )

    return accessions_by_ticker


def expected_years_for_ticker(
    expected: dict,
    *,
    ticker: str | None,
    years_by_ticker: dict[str, list[int]],
) -> list[int] | None:
    time_scope = expected.get("time_scope", "latest")

    if time_scope == "specific_years":
        return expected.get("filing_years") or []

    available_years = sorted(
        (
            years_by_ticker.get(ticker, [])
            if ticker
            else {year for years in years_by_ticker.values() for year in years}
        ),
        reverse=True,
    )
    if not available_years:
        return []

    if time_scope == "latest":
        return available_years[:1]
    if time_scope == "latest_and_previous":
        return available_years[:2]
    if time_scope == "all_available":
        return available_years

    return None


def compare_retrieved_scope(
    results,
    expected: dict,
    *,
    expected_years: list[int] | None = None,
) -> list[str]:
    if not results:
        return ["no chunks retrieved"]

    failures = []
    expected_tickers = set(expected.get("tickers") or [])
    expected_sections = set(expected.get("sections") or [])
    specific_years = set(expected.get("filing_years") or [])
    expected_years = set(expected_years or [])
    actual_tickers = {result.payload.get("ticker") for result in results}
    actual_years = {
        int(filing_date[:4])
        for result in results
        if (filing_date := result.payload.get("filing_date") or "")[:4].isdigit()
    }

    if expected.get("intent") == "comparison" and expected_tickers:
        missing_tickers = expected_tickers - actual_tickers
        if missing_tickers:
            failures.append(
                f"missing comparison ticker chunks: {sorted(missing_tickers)!r}"
            )

    if (
        (expected.get("intent") == "trend" or expected.get("time_scope") == "all_available")
        and expected_years
    ):
        missing_years = expected_years - actual_years
        if missing_years:
            failures.append(f"missing trend filing year chunks: {sorted(missing_years)!r}")

    if expected.get("time_scope") == "latest_and_previous" and expected_years:
        missing_years = expected_years - actual_years
        if missing_years:
            failures.append(f"missing period comparison year chunks: {sorted(missing_years)!r}")

    for index, result in enumerate(results, start=1):
        payload = result.payload
        ticker = payload.get("ticker")
        section_id = payload.get("section_id")
        filing_date = payload.get("filing_date") or ""
        filing_year = int(filing_date[:4]) if filing_date[:4].isdigit() else None

        if expected_tickers and ticker not in expected_tickers:
            failures.append(f"chunk {index}: ticker {ticker!r} not in {sorted(expected_tickers)!r}")
        if expected_sections and section_id not in expected_sections:
            failures.append(
                f"chunk {index}: section {section_id!r} not in {sorted(expected_sections)!r}"
            )
        if expected.get("time_scope") == "specific_years" and filing_year not in specific_years:
            failures.append(
                f"chunk {index}: filing year {filing_year!r} not in {sorted(specific_years)!r}"
            )

    return failures


def print_retrieved_chunks(results, *, preview_chars: int, show_full_text: bool) -> None:
    if not results:
        print("Retrieved chunks: (none)")
        return

    print("Retrieved chunks:")
    for index, result in enumerate(results, start=1):
        payload = result.payload
        text = payload.get("text", "")
        if not show_full_text and len(text) > preview_chars:
            text = f"{text[:preview_chars].rstrip()}..."

        label = format_source_label(index, payload)
        if payload.get("chunk_source") == "document":
            print(f"{index}. score={result.score:.4f} {label} url={payload.get('url') or '-'}")
        else:
            print(
                f"{index}. score={result.score:.4f} {label} "
                f"accession={payload.get('accession_number') or '-'}"
            )
        print(textwrap.indent(text, "   "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results for planner golden cases."
    )
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
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=800)
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print full retrieved chunk text instead of a preview.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
