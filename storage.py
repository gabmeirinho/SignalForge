import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sections import TextChunk


@dataclass(frozen=True)
class FilingMetadata:
    accession_number: str
    ticker: str
    cik: str | None
    company_name: str | None
    form_type: str
    filing_date: str | None
    period_of_report: str | None
    raw_path: str
    raw_sha256: str
    clean_text_path: str


def connect_database(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            accession_number TEXT NOT NULL UNIQUE,
            ticker TEXT NOT NULL,
            cik TEXT,
            company_name TEXT,
            form_type TEXT NOT NULL,
            filing_date TEXT,
            period_of_report TEXT,
            raw_path TEXT NOT NULL,
            raw_sha256 TEXT NOT NULL,
            clean_text_path TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filing_id INTEGER NOT NULL,
            section_id TEXT NOT NULL,
            section_title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (filing_id) REFERENCES filings(id) ON DELETE CASCADE,
            UNIQUE (filing_id, section_id, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            vector_collection TEXT NOT NULL,
            vector_id TEXT NOT NULL,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (chunk_id, embedding_model, vector_collection),
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS embedding_runs (
            filing_id INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            vector_collection TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'indexing', 'ready', 'failed')
            ),
            expected_point_count INTEGER NOT NULL DEFAULT 0,
            indexed_point_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (filing_id, embedding_model, vector_collection),
            FOREIGN KEY (filing_id) REFERENCES filings(id) ON DELETE CASCADE
        );
        """
    )
    filing_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(filings)").fetchall()
    }
    if "raw_sha256" not in filing_columns:
        connection.execute("ALTER TABLE filings ADD COLUMN raw_sha256 TEXT")
    connection.commit()


def upsert_filing(connection: sqlite3.Connection, metadata: FilingMetadata) -> int:
    connection.execute(
        """
        INSERT INTO filings (
            accession_number,
            ticker,
            cik,
            company_name,
            form_type,
            filing_date,
            period_of_report,
            raw_path,
            raw_sha256,
            clean_text_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(accession_number) DO UPDATE SET
            ticker = excluded.ticker,
            cik = excluded.cik,
            company_name = excluded.company_name,
            form_type = excluded.form_type,
            filing_date = excluded.filing_date,
            period_of_report = excluded.period_of_report,
            raw_path = excluded.raw_path,
            raw_sha256 = excluded.raw_sha256,
            clean_text_path = excluded.clean_text_path,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            metadata.accession_number,
            metadata.ticker,
            metadata.cik,
            metadata.company_name,
            metadata.form_type,
            metadata.filing_date,
            metadata.period_of_report,
            metadata.raw_path,
            metadata.raw_sha256,
            metadata.clean_text_path,
        ),
    )
    connection.commit()

    row = connection.execute(
        "SELECT id FROM filings WHERE accession_number = ?",
        (metadata.accession_number,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to load filing after upsert: {metadata.accession_number}")
    return int(row["id"])


def replace_filing_chunks(
    connection: sqlite3.Connection,
    filing_id: int,
    chunks: list[TextChunk],
) -> None:
    connection.execute("DELETE FROM chunks WHERE filing_id = ?", (filing_id,))
    connection.executemany(
        """
        INSERT INTO chunks (
            filing_id,
            section_id,
            section_title,
            chunk_index,
            text,
            char_count
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                filing_id,
                chunk.section_id,
                chunk.section_title,
                chunk.chunk_index,
                chunk.text,
                len(chunk.text),
            )
            for chunk in chunks
        ],
    )
    connection.execute(
        """
        UPDATE embedding_runs
        SET
            status = 'pending',
            expected_point_count = ?,
            indexed_point_count = 0,
            error_message = NULL,
            started_at = NULL,
            completed_at = NULL,
            updated_at = ?
        WHERE filing_id = ?
        """,
        (len(chunks), _utc_now(), filing_id),
    )
    connection.commit()


def load_chunks_for_vector_index(
    connection: sqlite3.Connection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            c.id AS chunk_id,
            c.filing_id,
            c.section_id,
            c.section_title,
            c.chunk_index,
            c.text,
            f.accession_number,
            f.ticker,
            f.cik,
            f.company_name,
            f.form_type,
            f.filing_date,
            f.period_of_report
        FROM chunks AS c
        JOIN filings AS f ON f.id = c.filing_id
        LEFT JOIN embedding_runs AS er
          ON er.filing_id = f.id
         AND er.embedding_model = ?
         AND er.vector_collection = ?
        WHERE er.status IS NULL OR er.status != 'ready'
        ORDER BY f.id, c.section_id, c.chunk_index
        """,
        (embedding_model, vector_collection),
    ).fetchall()


def record_chunk_embeddings(
    connection: sqlite3.Connection,
    *,
    chunk_vector_ids: list[tuple[int, str]],
    embedding_model: str,
    vector_collection: str,
) -> None:
    embedded_at = datetime.now(UTC).isoformat()
    connection.executemany(
        """
        INSERT INTO chunk_embeddings (
            chunk_id,
            embedding_model,
            vector_collection,
            vector_id,
            embedded_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id, embedding_model, vector_collection) DO UPDATE SET
            vector_id = excluded.vector_id,
            embedded_at = excluded.embedded_at
        """,
        [
            (chunk_id, embedding_model, vector_collection, vector_id, embedded_at)
            for chunk_id, vector_id in chunk_vector_ids
        ],
    )
    connection.commit()


def set_embedding_run_status(
    connection: sqlite3.Connection,
    *,
    filing_id: int,
    embedding_model: str,
    vector_collection: str,
    status: str,
    expected_point_count: int,
    indexed_point_count: int = 0,
    error_message: str | None = None,
) -> None:
    if status not in {"pending", "indexing", "ready", "failed"}:
        raise ValueError(f"Unsupported embedding status: {status}")

    now = _utc_now()
    started_at = now if status == "indexing" else None
    completed_at = now if status in {"ready", "failed"} else None

    connection.execute(
        """
        INSERT INTO embedding_runs (
            filing_id,
            embedding_model,
            vector_collection,
            status,
            expected_point_count,
            indexed_point_count,
            error_message,
            started_at,
            completed_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filing_id, embedding_model, vector_collection) DO UPDATE SET
            status = excluded.status,
            expected_point_count = excluded.expected_point_count,
            indexed_point_count = excluded.indexed_point_count,
            error_message = excluded.error_message,
            started_at = CASE
                WHEN excluded.status = 'indexing' THEN excluded.started_at
                ELSE embedding_runs.started_at
            END,
            completed_at = excluded.completed_at,
            updated_at = excluded.updated_at
        """,
        (
            filing_id,
            embedding_model,
            vector_collection,
            status,
            expected_point_count,
            indexed_point_count,
            error_message,
            started_at,
            completed_at,
            now,
        ),
    )
    connection.commit()


def get_ready_accession_numbers(
    connection: sqlite3.Connection,
    *,
    embedding_model: str,
    vector_collection: str,
    ticker: str | None = None,
    filing_years: list[int] | None = None,
) -> list[str]:
    query = """
        SELECT f.accession_number
        FROM embedding_runs AS er
        JOIN filings AS f ON f.id = er.filing_id
        WHERE er.embedding_model = ?
          AND er.vector_collection = ?
          AND er.status = 'ready'
    """
    parameters: list[str] = [embedding_model, vector_collection]

    if ticker:
        query += " AND f.ticker = ?"
        parameters.append(ticker.upper())

    if filing_years is not None:
        if not filing_years:
            return []
        placeholders = ", ".join("?" for _ in filing_years)
        query += f" AND CAST(substr(f.filing_date, 1, 4) AS INTEGER) IN ({placeholders})"
        parameters.extend(filing_years)

    query += " ORDER BY f.accession_number"
    return [
        str(row["accession_number"])
        for row in connection.execute(query, parameters).fetchall()
    ]


def load_planner_metadata(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT DISTINCT
            f.ticker,
            f.company_name,
            f.filing_date,
            c.section_id
        FROM filings AS f
        JOIN chunks AS c ON c.filing_id = f.id
        ORDER BY f.ticker, f.filing_date, c.section_id
        """
    ).fetchall()


def update_embedding_run_progress(
    connection: sqlite3.Connection,
    *,
    filing_id: int,
    embedding_model: str,
    vector_collection: str,
    indexed_point_count: int,
) -> None:
    connection.execute(
        """
        UPDATE embedding_runs
        SET indexed_point_count = ?, updated_at = ?
        WHERE filing_id = ?
          AND embedding_model = ?
          AND vector_collection = ?
          AND status = 'indexing'
        """,
        (
            indexed_point_count,
            _utc_now(),
            filing_id,
            embedding_model,
            vector_collection,
        ),
    )
    connection.commit()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
