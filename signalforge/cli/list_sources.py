import argparse
import sys

from signalforge.config import RuntimeConfig
from signalforge.storage import (
    SOURCE_DISCOVERY_STATUSES,
    connect_database,
    initialize_database,
    list_sources,
)


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        try:
            sources = list_sources(
                connection,
                ticker=args.ticker,
                discovery_status=args.status,
                enabled=parse_enabled_filter(args.enabled),
            )
        except ValueError as error:
            print(f"Source listing failed: {error}", file=sys.stderr)
            raise SystemExit(2) from None

    if not sources:
        print("No sources found.")
        return

    for source in sources:
        confidence = (
            f"{source['confidence_score']:.2f}"
            if source["confidence_score"] is not None
            else "n/a"
        )
        ticker = source["ticker"] or "-"
        enabled = "yes" if source["enabled"] else "no"

        print(
            f"[{source['id']}] {source['name']} "
            f"({ticker}, {source['discovery_status']}, enabled={enabled})"
        )
        print(f"    {source['url']}")
        print(f"    type: {source['source_type']}")
        print(f"    ownership: {source['ownership']}")
        print(f"    trust_level: {source['trust_level']}")
        print(f"    confidence: {confidence}")
        print(f"    documents: {source['document_count']}")
        if source["last_ingestion_status"] is not None:
            completed_at = source["last_ingestion_completed_at"] or "not completed"
            print(f"    last_ingestion: {source['last_ingestion_status']} at {completed_at}")
        if source["discovery_reason"]:
            print(f"    reason: {source['discovery_reason']}")
        print()


def parse_enabled_filter(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError("--enabled must be one of: true, false, yes, no, 1, 0")


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(description="List discovered and registered sources.")
    parser.add_argument("--db-path", default=config.database_target)
    parser.add_argument("--ticker", help="Only show sources for this ticker.")
    parser.add_argument(
        "--status",
        choices=sorted(SOURCE_DISCOVERY_STATUSES),
        help="Only show sources with this discovery status.",
    )
    parser.add_argument(
        "--enabled",
        help="Only show enabled or disabled sources. Values: true, false, yes, no, 1, 0.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
