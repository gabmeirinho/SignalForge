import argparse

from signalforge.storage import connect_database, initialize_database, reject_source


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        source = reject_source(connection, args.source_id)

    if source is None:
        raise SystemExit(f"Source not found: {args.source_id}")

    print(f"Rejected source {source['id']}: {source['name']}")
    print(f"    {source['url']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reject a candidate source.")
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    return parser.parse_args()


if __name__ == "__main__":
    main()
