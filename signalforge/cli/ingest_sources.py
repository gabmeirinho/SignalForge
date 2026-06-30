import argparse

from signalforge.source_ingestion import ingest_approved_sources
from signalforge.storage import connect_database


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        results = ingest_approved_sources(
            connection,
            ticker=args.ticker,
            processed_dir=args.processed_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            limit_per_source=args.limit_per_source,
        )

    if not results:
        scope = f" for {args.ticker.upper()}" if args.ticker else ""
        print(f"No approved enabled sources found{scope}.")
        return

    for result in results:
        print(
            f"[{result.source_id}] {result.source_name}: {result.status} "
            f"({result.inserted_count} inserted, {result.skipped_count} skipped, "
            f"{result.discovered_count} discovered)"
        )
        if result.error_message:
            print(f"    error: {result.error_message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest approved RSS/blog sources.")
    parser.add_argument("--ticker", help="Optional ticker to limit approved sources.")
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--chunk-size", type=int, default=4_000)
    parser.add_argument("--overlap", type=int, default=500)
    parser.add_argument(
        "--limit-per-source",
        type=int,
        help="Optional maximum number of feed entries to ingest per source.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
