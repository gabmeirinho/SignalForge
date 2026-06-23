import sqlite3
from dataclasses import dataclass
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
        """
    )
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
            clean_text_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(accession_number) DO UPDATE SET
            ticker = excluded.ticker,
            cik = excluded.cik,
            company_name = excluded.company_name,
            form_type = excluded.form_type,
            filing_date = excluded.filing_date,
            period_of_report = excluded.period_of_report,
            raw_path = excluded.raw_path,
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
    connection.commit()
