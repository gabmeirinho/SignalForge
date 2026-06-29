import argparse

from signalforge.source_discovery import discover_sources_for_ticker
from signalforge.storage import connect_database, initialize_database


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        sources = discover_sources_for_ticker(
            connection=connection,
            ticker=args.ticker,
            website_domain=args.website_domain,
            persist=not args.dry_run,
        )

    ticker = args.ticker.upper()
    if not sources:
        print(f"No candidate sources discovered for {ticker}.")
        return

    print(f"Candidate sources for {ticker}")
    print()
    for index, source in enumerate(sources, start=1):
        source_id = source.persisted_id if source.persisted_id is not None else "dry-run"
        print(f"[{index}] {source.name}")
        print(f"    {source.url}")
        print(f"    source_id: {source_id}")
        print(f"    type: {source.source_type}")
        print(f"    ownership: {source.ownership}")
        print(f"    trust_level: {source.trust_level}")
        print(f"    confidence: {source.confidence_score:.2f}")
        print(f"    reason: {source.discovery_reason}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover candidate company source pages and feeds.",
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, for example NVDA.")
    parser.add_argument(
        "--website-domain",
        help="Official company website domain, for example nvidia.com. Stored on the company row.",
    )
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe sources and print candidates without storing them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
