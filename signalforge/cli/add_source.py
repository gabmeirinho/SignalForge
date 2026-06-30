import argparse

from signalforge.storage import (
    SOURCE_OWNERSHIPS,
    SOURCE_TRUST_LEVELS,
    SOURCE_TYPES,
    CompanyRecord,
    SourceRecord,
    connect_database,
    initialize_database,
    load_company_by_ticker,
    upsert_company,
    upsert_source,
)


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        company_id = None
        if args.ticker:
            company = load_company_by_ticker(connection, args.ticker)
            if company is None:
                company_id = upsert_company(connection, CompanyRecord(ticker=args.ticker))
            else:
                company_id = int(company["id"])

        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name=args.name,
                url=args.url,
                source_type=args.source_type,
                ownership=args.ownership,
                trust_level=args.trust_level,
                discovery_status="manual",
                enabled=not args.disabled,
                discovery_reason="manual source registration",
            ),
        )

    print(f"Registered manual source {source_id}: {args.name}")
    print(f"    {args.url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually register a source fallback.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), required=True)
    parser.add_argument("--ticker", help="Optional ticker to associate with this source.")
    parser.add_argument("--ownership", choices=sorted(SOURCE_OWNERSHIPS), default="unknown")
    parser.add_argument("--trust-level", choices=sorted(SOURCE_TRUST_LEVELS), default="medium")
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Register the source but leave it disabled.",
    )
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    return parser.parse_args()


if __name__ == "__main__":
    main()
