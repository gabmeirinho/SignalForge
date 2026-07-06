from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection

from signalforge.config import sqlalchemy_database_url
from signalforge.migrations import upgrade_database


COPY_TABLES = (
    "companies",
    "filings",
    "sources",
    "documents",
    "chunks",
    "document_chunks",
    "chunk_embeddings",
    "document_chunk_embeddings",
    "embedding_runs",
    "source_ingestion_runs",
)

ID_TABLES = (
    "companies",
    "filings",
    "sources",
    "documents",
    "chunks",
    "document_chunks",
    "source_ingestion_runs",
)


@dataclass(frozen=True)
class TableMigrationResult:
    table_name: str
    source_count: int
    target_count: int | None


@dataclass(frozen=True)
class SQLiteToPostgresMigrationResult:
    source_path: str
    target_url: str
    dry_run: bool
    replaced: bool
    tables: list[TableMigrationResult]

    @property
    def source_total(self) -> int:
        return sum(table.source_count for table in self.tables)

    @property
    def target_total(self) -> int | None:
        if self.dry_run:
            return None
        return sum(table.target_count or 0 for table in self.tables)


def migrate_sqlite_to_postgres(
    *,
    sqlite_path: str | Path,
    postgres_url: str,
    dry_run: bool = True,
    replace: bool = False,
) -> SQLiteToPostgresMigrationResult:
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    target_engine = create_engine(sqlalchemy_database_url(postgres_url), future=True)

    try:
        _validate_source_schema(source)
        source_counts = _source_counts(source)

        if dry_run:
            _validate_target_connection(target_engine)
            return SQLiteToPostgresMigrationResult(
                source_path=str(sqlite_path),
                target_url=postgres_url,
                dry_run=True,
                replaced=False,
                tables=[
                    TableMigrationResult(table_name=table, source_count=count, target_count=None)
                    for table, count in source_counts.items()
                ],
            )

        upgrade_database(postgres_url)
        with target_engine.begin() as target:
            _validate_target_schema(target)
            if replace:
                _clear_target_tables(target)
            else:
                _ensure_target_empty(target)

            for table_name in COPY_TABLES:
                _copy_table(source, target, table_name)

            if target.dialect.name == "postgresql":
                _reset_postgres_sequences(target)

            target_counts = _target_counts(target)
            _validate_counts(source_counts, target_counts)

        return SQLiteToPostgresMigrationResult(
            source_path=str(sqlite_path),
            target_url=postgres_url,
            dry_run=False,
            replaced=replace,
            tables=[
                TableMigrationResult(
                    table_name=table,
                    source_count=source_counts[table],
                    target_count=target_counts[table],
                )
                for table in COPY_TABLES
            ],
        )
    finally:
        source.close()
        target_engine.dispose()


def _validate_source_schema(connection: sqlite3.Connection) -> None:
    existing_tables = {
        row["name"]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }
    missing = [table for table in COPY_TABLES if table not in existing_tables]
    if missing:
        raise ValueError(f"SQLite source is missing tables: {', '.join(missing)}")


def _validate_target_connection(engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def _validate_target_schema(connection: Connection) -> None:
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    missing = [table for table in COPY_TABLES if table not in existing_tables]
    if missing:
        raise ValueError(f"Postgres target is missing tables: {', '.join(missing)}")


def _source_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table_name: int(connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()[0])
        for table_name in COPY_TABLES
    }


def _target_counts(connection: Connection) -> dict[str, int]:
    return {
        table_name: int(connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
        for table_name in COPY_TABLES
    }


def _ensure_target_empty(connection: Connection) -> None:
    counts = _target_counts(connection)
    non_empty = [table for table, count in counts.items() if count > 0]
    if non_empty:
        raise ValueError(
            "Postgres target is not empty. Use --replace to delete existing target rows first. "
            f"Non-empty tables: {', '.join(non_empty)}"
        )


def _clear_target_tables(connection: Connection) -> None:
    table_names = ", ".join(COPY_TABLES)
    if connection.dialect.name == "postgresql":
        connection.execute(text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"))
        return

    for table_name in reversed(COPY_TABLES):
        connection.execute(text(f"DELETE FROM {table_name}"))


def _copy_table(source: sqlite3.Connection, target: Connection, table_name: str) -> None:
    rows = source.execute(f"SELECT * FROM {table_name}").fetchall()
    if not rows:
        return

    column_names = rows[0].keys()
    columns_sql = ", ".join(column_names)
    values_sql = ", ".join(f":{column}" for column in column_names)
    statement = text(f"INSERT INTO {table_name} ({columns_sql}) VALUES ({values_sql})")
    target.execute(
        statement,
        [_prepare_row(dict(row), table_name=table_name, dialect_name=target.dialect.name) for row in rows],
    )


def _prepare_row(row: dict[str, Any], *, table_name: str, dialect_name: str) -> dict[str, Any]:
    if table_name == "documents" and dialect_name == "postgresql":
        row["metadata_json"] = Jsonb(_decode_metadata_json(row["metadata_json"]))

    if table_name == "sources" and dialect_name == "postgresql":
        row["enabled"] = bool(row["enabled"])

    return row


def _decode_metadata_json(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return value


def _reset_postgres_sequences(connection: Connection) -> None:
    for table_name in ID_TABLES:
        sequence_name = connection.execute(
            text("SELECT pg_get_serial_sequence(:table_name, 'id')"),
            {"table_name": table_name},
        ).scalar_one()
        if sequence_name is None:
            continue

        max_id = connection.execute(text(f"SELECT MAX(id) FROM {table_name}")).scalar_one()
        if max_id is None:
            connection.execute(text("SELECT setval(:sequence_name, 1, false)"), {"sequence_name": sequence_name})
        else:
            connection.execute(
                text("SELECT setval(:sequence_name, :max_id, true)"),
                {"sequence_name": sequence_name, "max_id": int(max_id)},
            )


def _validate_counts(source_counts: dict[str, int], target_counts: dict[str, int]) -> None:
    mismatches = [
        f"{table}: source={source_counts[table]} target={target_counts[table]}"
        for table in COPY_TABLES
        if source_counts[table] != target_counts[table]
    ]
    if mismatches:
        raise RuntimeError("Migration row-count validation failed: " + "; ".join(mismatches))
