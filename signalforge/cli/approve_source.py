import argparse

from signalforge.config import RuntimeConfig
from signalforge.storage import approve_source, connect_database, initialize_database


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        source = approve_source(connection, args.source_id)

    if source is None:
        raise SystemExit(f"Source not found: {args.source_id}")

    print(f"Approved source {source['id']}: {source['name']}")
    print(f"    {source['url']}")


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(description="Approve a candidate source for ingestion.")
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--db-path", default=config.database_target)
    return parser.parse_args()


if __name__ == "__main__":
    main()
