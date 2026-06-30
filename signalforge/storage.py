import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from signalforge.config import sqlalchemy_database_url
from signalforge.migrations import upgrade_database
from signalforge.sections import TextChunk

SOURCE_TYPES = {
    "company_blog",
    "newsroom",
    "investor_relations",
    "industry_blog",
    "news_feed",
    "sec",
    "webpage",
}
SOURCE_OWNERSHIPS = {"official", "third_party", "unknown"}
SOURCE_TRUST_LEVELS = {"high", "medium", "low"}
SOURCE_DISCOVERY_STATUSES = {"candidate", "approved", "rejected", "manual"}
DOCUMENT_TYPES = {
    "article",
    "blog_post",
    "press_release",
    "investor_update",
    "filing",
    "webpage",
}
SOURCE_INGESTION_STATUSES = {"running", "completed", "failed", "partial"}


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


@dataclass(frozen=True)
class CompanyRecord:
    ticker: str
    name: str | None = None
    cik: str | None = None
    website_domain: str | None = None


@dataclass(frozen=True)
class SourceRecord:
    name: str
    url: str
    source_type: str
    ownership: str = "unknown"
    trust_level: str = "medium"
    discovery_status: str = "candidate"
    enabled: bool = True
    company_id: int | None = None
    confidence_score: float | None = None
    discovery_reason: str | None = None


@dataclass(frozen=True)
class DocumentRecord:
    source_id: int
    url: str
    title: str | None
    content_hash: str
    document_type: str
    author: str | None = None
    published_at: str | None = None
    fetched_at: str | None = None
    clean_text_path: str | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class GenericDocumentChunk:
    chunk_index: int
    text: str


class StorageConnection:
    def __init__(
        self,
        *,
        target: str,
        sqlite_connection: sqlite3.Connection | None = None,
        sqlalchemy_engine: Engine | None = None,
    ) -> None:
        self.target = target
        self._sqlite_connection = sqlite_connection
        self._sqlalchemy_engine = sqlalchemy_engine
        self._sqlalchemy_connection: Connection | None = None

    @property
    def dialect_name(self) -> str:
        if self._sqlite_connection is not None:
            return "sqlite"
        if self._sqlalchemy_connection is not None:
            return self._sqlalchemy_connection.dialect.name
        if self._sqlalchemy_engine is not None:
            return self._sqlalchemy_engine.dialect.name
        raise RuntimeError("Storage connection is not open")

    def __enter__(self) -> "StorageConnection":
        if self._sqlalchemy_engine is not None and self._sqlalchemy_connection is None:
            self._sqlalchemy_connection = self._sqlalchemy_engine.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.rollback()
        self.close()

    def execute(self, query: str, parameters: Iterable[Any] | dict[str, Any] = ()):
        if self._sqlite_connection is not None:
            return self._sqlite_connection.execute(query, parameters)

        connection = self._require_sqlalchemy_connection()
        statement, bound_parameters = _sqlalchemy_statement(query, parameters)
        return connection.execute(text(statement), bound_parameters).mappings()

    def executemany(self, query: str, parameters: Iterable[Iterable[Any] | dict[str, Any]]):
        if self._sqlite_connection is not None:
            return self._sqlite_connection.executemany(query, parameters)

        rows = list(parameters)
        if not rows:
            return None

        statement, bound_parameters = _sqlalchemy_many_statement(query, rows)
        return self._require_sqlalchemy_connection().execute(text(statement), bound_parameters)

    def executescript(self, script: str):
        if self._sqlite_connection is None:
            raise NotImplementedError("executescript is only used by the SQLite compatibility path")
        return self._sqlite_connection.executescript(script)

    def commit(self) -> None:
        if self._sqlite_connection is not None:
            self._sqlite_connection.commit()
            return
        self._require_sqlalchemy_connection().commit()

    def rollback(self) -> None:
        if self._sqlite_connection is not None:
            self._sqlite_connection.rollback()
            return
        if self._sqlalchemy_connection is not None:
            self._sqlalchemy_connection.rollback()

    def close(self) -> None:
        if self._sqlite_connection is not None:
            self._sqlite_connection.close()
            return
        if self._sqlalchemy_connection is not None:
            self._sqlalchemy_connection.close()
            self._sqlalchemy_connection = None
        if self._sqlalchemy_engine is not None:
            self._sqlalchemy_engine.dispose()

    def _require_sqlalchemy_connection(self) -> Connection:
        if self._sqlalchemy_connection is None:
            if self._sqlalchemy_engine is None:
                raise RuntimeError("Storage connection is not open")
            self._sqlalchemy_connection = self._sqlalchemy_engine.connect()
        return self._sqlalchemy_connection


def connect_database(path: str | Path) -> StorageConnection:
    target = str(path)
    parsed = urlparse(target)

    if parsed.scheme in {"", "file", "sqlite"}:
        db_path = _sqlite_path_from_target(target)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        return StorageConnection(target=str(db_path), sqlite_connection=connection)

    engine = create_engine(sqlalchemy_database_url(target), future=True)
    return StorageConnection(target=target, sqlalchemy_engine=engine)


def _sqlite_path_from_target(target: str | Path) -> Path:
    target_text = str(target)
    parsed = urlparse(target_text)

    if parsed.scheme == "":
        return Path(target_text)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))

    if parsed.scheme == "sqlite":
        if parsed.netloc and parsed.netloc != "localhost":
            raise ValueError(f"Unsupported SQLite database URL host: {parsed.netloc}")
        return Path(unquote(parsed.path))

    raise NotImplementedError(
        "SIGNALFORGE_DATABASE_URL is configured for a non-SQLite database, "
        "but Postgres access is introduced in a later migration phase."
    )


def initialize_database(connection: StorageConnection) -> None:
    if connection.dialect_name != "sqlite":
        upgrade_database(connection.target)
        return

    migration_target = _migration_target_for_empty_database(connection)
    if migration_target is not None:
        upgrade_database(migration_target)
        return

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

        CREATE TABLE IF NOT EXISTS document_chunk_embeddings (
            document_chunk_id INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            vector_collection TEXT NOT NULL,
            vector_id TEXT NOT NULL,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (document_chunk_id, embedding_model, vector_collection),
            FOREIGN KEY (document_chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
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

        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            name TEXT,
            cik TEXT,
            website_domain TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            name TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            source_type TEXT NOT NULL CHECK (
                source_type IN (
                    'company_blog',
                    'newsroom',
                    'investor_relations',
                    'industry_blog',
                    'news_feed',
                    'sec',
                    'webpage'
                )
            ),
            ownership TEXT NOT NULL CHECK (
                ownership IN ('official', 'third_party', 'unknown')
            ),
            trust_level TEXT NOT NULL CHECK (
                trust_level IN ('high', 'medium', 'low')
            ),
            discovery_status TEXT NOT NULL CHECK (
                discovery_status IN ('candidate', 'approved', 'rejected', 'manual')
            ),
            enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            confidence_score REAL CHECK (
                confidence_score IS NULL
                OR (confidence_score >= 0.0 AND confidence_score <= 1.0)
            ),
            discovery_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sources_company_id
            ON sources(company_id);
        CREATE INDEX IF NOT EXISTS idx_sources_discovery_status
            ON sources(discovery_status);

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            author TEXT,
            published_at TEXT,
            fetched_at TEXT NOT NULL,
            clean_text_path TEXT,
            content_hash TEXT NOT NULL,
            document_type TEXT NOT NULL CHECK (
                document_type IN (
                    'article',
                    'blog_post',
                    'press_release',
                    'investor_update',
                    'filing',
                    'webpage'
                )
            ),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE,
            UNIQUE (source_id, url)
        );

        CREATE INDEX IF NOT EXISTS idx_documents_source_id
            ON documents(source_id);
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash
            ON documents(content_hash);

        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE (document_id, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS source_ingestion_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'completed', 'failed', 'partial')
            ),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            discovered_count INTEGER NOT NULL DEFAULT 0,
            inserted_count INTEGER NOT NULL DEFAULT 0,
            skipped_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_source_ingestion_runs_source_id
            ON source_ingestion_runs(source_id);
        """
    )
    filing_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(filings)").fetchall()
    }
    if "raw_sha256" not in filing_columns:
        connection.execute("ALTER TABLE filings ADD COLUMN raw_sha256 TEXT")
    connection.commit()


def _migration_target_for_empty_database(connection: StorageConnection) -> str | None:
    user_tables = {
        row["name"]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
              AND name != 'alembic_version'
            """
        ).fetchall()
    }
    if user_tables:
        return None

    database_path = connection.execute("PRAGMA database_list").fetchone()["file"]
    if not database_path:
        return None
    return database_path


def upsert_company(connection: sqlite3.Connection, company: CompanyRecord) -> int:
    ticker = company.ticker.upper()
    connection.execute(
        """
        INSERT INTO companies (
            ticker,
            name,
            cik,
            website_domain
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            name = excluded.name,
            cik = excluded.cik,
            website_domain = excluded.website_domain,
            updated_at = CURRENT_TIMESTAMP
        """,
        (ticker, company.name, company.cik, company.website_domain),
    )
    connection.commit()

    row = connection.execute("SELECT id FROM companies WHERE ticker = ?", (ticker,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to load company after upsert: {ticker}")
    return int(row["id"])


def load_company_by_ticker(connection: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT *
        FROM companies
        WHERE ticker = ?
        """,
        (ticker.upper(),),
    ).fetchone()


def load_latest_filing_company_metadata(
    connection: sqlite3.Connection,
    ticker: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT ticker, company_name, cik
        FROM filings
        WHERE ticker = ?
        ORDER BY filing_date DESC, id DESC
        LIMIT 1
        """,
        (ticker.upper(),),
    ).fetchone()


def upsert_source(connection: sqlite3.Connection, source: SourceRecord) -> int:
    _validate_source(source)
    connection.execute(
        """
        INSERT INTO sources (
            company_id,
            name,
            url,
            source_type,
            ownership,
            trust_level,
            discovery_status,
            enabled,
            confidence_score,
            discovery_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            company_id = excluded.company_id,
            name = excluded.name,
            source_type = excluded.source_type,
            ownership = excluded.ownership,
            trust_level = excluded.trust_level,
            discovery_status = excluded.discovery_status,
            enabled = excluded.enabled,
            confidence_score = excluded.confidence_score,
            discovery_reason = excluded.discovery_reason,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            source.company_id,
            source.name,
            source.url,
            source.source_type,
            source.ownership,
            source.trust_level,
            source.discovery_status,
            source.enabled,
            source.confidence_score,
            source.discovery_reason,
        ),
    )
    connection.commit()

    row = connection.execute("SELECT id FROM sources WHERE url = ?", (source.url,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to load source after upsert: {source.url}")
    return int(row["id"])


def list_sources(
    connection: sqlite3.Connection,
    *,
    ticker: str | None = None,
    discovery_status: str | None = None,
    enabled: bool | None = None,
) -> list[sqlite3.Row]:
    if discovery_status is not None:
        _validate_choice("discovery_status", discovery_status, SOURCE_DISCOVERY_STATUSES)

    query = """
        SELECT
            s.*,
            c.ticker,
            c.name AS company_name,
            COUNT(DISTINCT d.id) AS document_count,
            latest_run.status AS last_ingestion_status,
            latest_run.completed_at AS last_ingestion_completed_at
        FROM sources AS s
        LEFT JOIN companies AS c ON c.id = s.company_id
        LEFT JOIN documents AS d ON d.source_id = s.id
        LEFT JOIN source_ingestion_runs AS latest_run
          ON latest_run.id = (
              SELECT sir.id
              FROM source_ingestion_runs AS sir
              WHERE sir.source_id = s.id
              ORDER BY COALESCE(sir.completed_at, sir.started_at) DESC, sir.id DESC
              LIMIT 1
          )
        WHERE 1 = 1
    """
    parameters: list[object] = []

    if ticker is not None:
        query += " AND c.ticker = ?"
        parameters.append(ticker.upper())

    if discovery_status is not None:
        query += " AND s.discovery_status = ?"
        parameters.append(discovery_status)

    if enabled is not None:
        query += " AND s.enabled = ?"
        parameters.append(enabled)

    query += """
        GROUP BY s.id
        ORDER BY
            c.ticker IS NULL,
            c.ticker,
            s.discovery_status,
            s.confidence_score DESC,
            s.name
    """
    return connection.execute(query, parameters).fetchall()


def load_source(connection: sqlite3.Connection, source_id: int) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT
            s.*,
            c.ticker,
            c.name AS company_name
        FROM sources AS s
        LEFT JOIN companies AS c ON c.id = s.company_id
        WHERE s.id = ?
        """,
        (source_id,),
    ).fetchone()


def approve_source(connection: sqlite3.Connection, source_id: int) -> sqlite3.Row | None:
    connection.execute(
        """
        UPDATE sources
        SET
            discovery_status = 'approved',
            enabled = TRUE,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (source_id,),
    )
    connection.commit()
    return load_source(connection, source_id)


def reject_source(connection: sqlite3.Connection, source_id: int) -> sqlite3.Row | None:
    connection.execute(
        """
        UPDATE sources
        SET
            discovery_status = 'rejected',
            enabled = FALSE,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (source_id,),
    )
    connection.commit()
    return load_source(connection, source_id)


def upsert_document(connection: sqlite3.Connection, document: DocumentRecord) -> int:
    _validate_document(document)
    fetched_at = document.fetched_at or _utc_now()
    metadata_value = (
        json.dumps(document.metadata or {}, sort_keys=True)
        if connection.dialect_name == "sqlite"
        else document.metadata or {}
    )
    connection.execute(
        """
        INSERT INTO documents (
            source_id,
            url,
            title,
            author,
            published_at,
            fetched_at,
            clean_text_path,
            content_hash,
            document_type,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, url) DO UPDATE SET
            title = excluded.title,
            author = excluded.author,
            published_at = excluded.published_at,
            fetched_at = excluded.fetched_at,
            clean_text_path = excluded.clean_text_path,
            content_hash = excluded.content_hash,
            document_type = excluded.document_type,
            metadata_json = excluded.metadata_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            document.source_id,
            document.url,
            document.title,
            document.author,
            document.published_at,
            fetched_at,
            document.clean_text_path,
            document.content_hash,
            document.document_type,
            metadata_value,
        ),
    )
    connection.commit()

    row = connection.execute(
        "SELECT id FROM documents WHERE source_id = ? AND url = ?",
        (document.source_id, document.url),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to load document after upsert: {document.url}")
    return int(row["id"])


def replace_document_chunks(
    connection: sqlite3.Connection,
    document_id: int,
    chunks: list[GenericDocumentChunk],
) -> None:
    connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
    connection.executemany(
        """
        INSERT INTO document_chunks (
            document_id,
            chunk_index,
            text,
            char_count
        )
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                document_id,
                chunk.chunk_index,
                chunk.text,
                len(chunk.text),
            )
            for chunk in chunks
        ],
    )
    connection.commit()


def create_source_ingestion_run(connection: sqlite3.Connection, source_id: int) -> int:
    started_at = _utc_now()
    cursor = connection.execute(
        """
        INSERT INTO source_ingestion_runs (
            source_id,
            status,
            started_at
        )
        VALUES (?, 'running', ?)
        RETURNING id
        """,
        (source_id, started_at),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Failed to create source ingestion run for source: {source_id}")
    connection.commit()
    return int(row["id"])


def complete_source_ingestion_run(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    status: str,
    discovered_count: int = 0,
    inserted_count: int = 0,
    skipped_count: int = 0,
    error_message: str | None = None,
) -> None:
    if status not in SOURCE_INGESTION_STATUSES - {"running"}:
        raise ValueError(f"Unsupported completed ingestion status: {status}")

    connection.execute(
        """
        UPDATE source_ingestion_runs
        SET
            status = ?,
            completed_at = ?,
            discovered_count = ?,
            inserted_count = ?,
            skipped_count = ?,
            error_message = ?
        WHERE id = ?
        """,
        (
            status,
            _utc_now(),
            discovered_count,
            inserted_count,
            skipped_count,
            error_message,
            run_id,
        ),
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


def load_document_chunks_for_vector_index(
    connection: sqlite3.Connection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            dc.id AS document_chunk_id,
            dc.document_id,
            dc.chunk_index,
            dc.text,
            d.source_id,
            d.url,
            d.title,
            d.author,
            d.published_at,
            d.fetched_at,
            d.document_type,
            s.name AS source_name,
            s.source_type,
            s.ownership,
            s.trust_level,
            c.ticker,
            c.name AS company_name
        FROM document_chunks AS dc
        JOIN documents AS d ON d.id = dc.document_id
        JOIN sources AS s ON s.id = d.source_id
        LEFT JOIN companies AS c ON c.id = s.company_id
        LEFT JOIN document_chunk_embeddings AS dce
          ON dce.document_chunk_id = dc.id
         AND dce.embedding_model = ?
         AND dce.vector_collection = ?
        WHERE dce.document_chunk_id IS NULL
          AND s.enabled = ?
          AND s.discovery_status IN ('approved', 'manual')
        ORDER BY d.id, dc.chunk_index
        """,
        (embedding_model, vector_collection, True),
    ).fetchall()


def record_document_chunk_embeddings(
    connection: sqlite3.Connection,
    *,
    chunk_vector_ids: list[tuple[int, str]],
    embedding_model: str,
    vector_collection: str,
) -> None:
    embedded_at = datetime.now(UTC).isoformat()
    connection.executemany(
        """
        INSERT INTO document_chunk_embeddings (
            document_chunk_id,
            embedding_model,
            vector_collection,
            vector_id,
            embedded_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(document_chunk_id, embedding_model, vector_collection) DO UPDATE SET
            vector_id = excluded.vector_id,
            embedded_at = excluded.embedded_at
        """,
        [
            (chunk_id, embedding_model, vector_collection, vector_id, embedded_at)
            for chunk_id, vector_id in chunk_vector_ids
        ],
    )
    connection.commit()


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


def load_index_metadata(
    connection: sqlite3.Connection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            f.ticker,
            f.company_name,
            f.accession_number,
            f.form_type,
            f.filing_date,
            f.period_of_report,
            COALESCE(er.status, 'pending') AS status,
            COALESCE(er.expected_point_count, 0) AS expected_point_count,
            COALESCE(er.indexed_point_count, 0) AS indexed_point_count
        FROM filings AS f
        LEFT JOIN embedding_runs AS er
          ON er.filing_id = f.id
         AND er.embedding_model = ?
         AND er.vector_collection = ?
        ORDER BY f.ticker, f.filing_date DESC, f.accession_number
        """,
        (embedding_model, vector_collection),
    ).fetchall()


def load_index_section_counts(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            f.ticker,
            c.section_id,
            COUNT(*) AS chunk_count
        FROM chunks AS c
        JOIN filings AS f ON f.id = c.filing_id
        GROUP BY f.ticker, c.section_id
        ORDER BY f.ticker, c.section_id
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


def _validate_source(source: SourceRecord) -> None:
    _validate_choice("source_type", source.source_type, SOURCE_TYPES)
    _validate_choice("ownership", source.ownership, SOURCE_OWNERSHIPS)
    _validate_choice("trust_level", source.trust_level, SOURCE_TRUST_LEVELS)
    _validate_choice(
        "discovery_status",
        source.discovery_status,
        SOURCE_DISCOVERY_STATUSES,
    )
    if source.confidence_score is not None and not 0.0 <= source.confidence_score <= 1.0:
        raise ValueError("confidence_score must be between 0.0 and 1.0")


def _validate_document(document: DocumentRecord) -> None:
    _validate_choice("document_type", document.document_type, DOCUMENT_TYPES)


def _validate_choice(field_name: str, value: str, supported_values: set[str]) -> None:
    if value not in supported_values:
        allowed = ", ".join(sorted(supported_values))
        raise ValueError(f"Unsupported {field_name}: {value}. Expected one of: {allowed}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sqlalchemy_statement(
    query: str,
    parameters: Iterable[Any] | dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if isinstance(parameters, dict):
        return query, parameters

    values = tuple(parameters)
    return _replace_qmark_parameters(query, values)


def _sqlalchemy_many_statement(
    query: str,
    rows: list[Iterable[Any] | dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    first = rows[0]
    if isinstance(first, dict):
        return query, rows  # type: ignore[return-value]

    statement, first_parameters = _replace_qmark_parameters(query, tuple(first))
    converted_rows = [first_parameters]
    for row in rows[1:]:
        if isinstance(row, dict):
            raise TypeError("executemany parameters must be consistently positional or named")
        _, parameters = _replace_qmark_parameters(query, tuple(row))
        converted_rows.append(parameters)
    return statement, converted_rows


def _replace_qmark_parameters(query: str, values: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    parts = query.split("?")
    expected_count = len(parts) - 1
    if expected_count != len(values):
        raise ValueError(f"Expected {expected_count} SQL parameters, received {len(values)}")

    statement = parts[0]
    parameters = {}
    for index, value in enumerate(values):
        name = f"p{index}"
        statement += f":{name}{parts[index + 1]}"
        parameters[name] = value
    return statement, parameters
