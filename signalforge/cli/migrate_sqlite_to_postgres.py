import argparse
import sys

from dotenv import load_dotenv

from signalforge.config import RuntimeConfig
from signalforge.sqlite_to_postgres import migrate_sqlite_to_postgres


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.execute:
        print("Dry run only. Re-run with --execute to write to Postgres.")

    try:
        result = migrate_sqlite_to_postgres(
            sqlite_path=args.sqlite_path,
            postgres_url=args.postgres_url,
            dry_run=not args.execute,
            replace=args.replace,
        )
    except Exception as error:
        print(f"SQLite to Postgres migration failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    action = "Validated" if result.dry_run else "Migrated"
    print(f"{action} {result.source_total} rows from {result.source_path}.")
    for table in result.tables:
        if result.dry_run:
            print(f"  {table.table_name}: source={table.source_count}")
        else:
            print(
                f"  {table.table_name}: "
                f"source={table.source_count} target={table.target_count}"
            )


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(
        description="Copy an existing SignalForge SQLite database into Postgres."
    )
    parser.add_argument(
        "--sqlite-path",
        default=config.db_path,
        help="Path to the source SQLite database.",
    )
    parser.add_argument(
        "--postgres-url",
        default=config.database_url,
        help="Target Postgres URL. Defaults to SIGNALFORGE_DATABASE_URL.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write to Postgres. Without this flag, only validation and row counting run.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing target rows before importing. Only valid with --execute.",
    )
    args = parser.parse_args()

    if not args.postgres_url:
        parser.error("--postgres-url or SIGNALFORGE_DATABASE_URL is required")
    if args.replace and not args.execute:
        parser.error("--replace requires --execute")

    return args


if __name__ == "__main__":
    main()
