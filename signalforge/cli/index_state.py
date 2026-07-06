import argparse
from collections import defaultdict

from signalforge.config import RuntimeConfig
from signalforge.storage import (
    connect_database,
    initialize_database,
    list_sources,
    load_index_metadata,
    load_index_section_counts,
)


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        filing_rows = load_index_metadata(
            connection,
            embedding_model=args.model,
            vector_collection=args.collection,
        )
        section_rows = load_index_section_counts(connection)
        source_rows = list_sources(connection)

    print(f"Database: {args.db_path}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.model}")
    print()

    print("Filings")
    if not filing_rows:
        print("  none")
    else:
        for row in filing_rows:
            print(
                f"  {row['ticker']} {row['form_type']} {row['filing_date'] or 'unknown-date'} "
                f"{row['accession_number']}: {row['status']} "
                f"{row['indexed_point_count']}/{row['expected_point_count']} points"
            )

    sections_by_ticker = defaultdict(list)
    for row in section_rows:
        sections_by_ticker[row["ticker"]].append(row)

    print()
    print("Sections")
    if not section_rows:
        print("  none")
    else:
        for ticker, rows in sections_by_ticker.items():
            sections = ", ".join(f"{row['section_id']}={row['chunk_count']}" for row in rows)
            print(f"  {ticker}: {sections}")

    approved_count = sum(
        1
        for source in source_rows
        if source["discovery_status"] in {"approved", "manual"} and bool(source["enabled"])
    )
    candidate_count = sum(1 for source in source_rows if source["discovery_status"] == "candidate")
    document_count = sum(int(source["document_count"]) for source in source_rows)

    print()
    print("Sources")
    print(f"  approved/manual enabled: {approved_count}")
    print(f"  candidates: {candidate_count}")
    print(f"  documents: {document_count}")


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(description="Inspect local SignalForge index state.")
    parser.add_argument("--db-path", default=config.database_target)
    parser.add_argument("--collection", default=config.collection)
    parser.add_argument("--model", default=config.embedding_model)
    return parser.parse_args()


if __name__ == "__main__":
    main()
