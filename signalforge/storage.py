from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from signalforge.config import sqlalchemy_database_url
from signalforge.migrations import upgrade_database
from signalforge.models import (
    Chunk,
    ChunkEmbedding,
    Company,
    Document,
    DocumentChunk,
    DocumentChunkEmbedding,
    EmbeddingRun,
    Filing,
    Source,
    SourceIngestionRun,
)
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
        sqlalchemy_engine: Engine | None = None,
    ) -> None:
        self.target = target
        self._sqlalchemy_engine = sqlalchemy_engine or create_engine(
            sqlalchemy_database_url(target),
            future=True,
        )
        self._sqlalchemy_connection: Connection | None = None
        self._session_factory = sessionmaker(future=True)
        self.session: Session | None = None

    @property
    def dialect_name(self) -> str:
        if self._sqlalchemy_connection is not None:
            return self._sqlalchemy_connection.dialect.name
        return self._sqlalchemy_engine.dialect.name

    def __enter__(self) -> "StorageConnection":
        if self._sqlalchemy_connection is None:
            self._sqlalchemy_connection = self._sqlalchemy_engine.connect()
        if self.session is None:
            self.session = self._session_factory(bind=self._sqlalchemy_connection)
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
        session = self._require_session()
        statement, bound_parameters = _sqlalchemy_statement(query, parameters)
        return session.execute(text(statement), bound_parameters).mappings()

    def executemany(self, query: str, parameters: Iterable[Iterable[Any] | dict[str, Any]]):
        rows = list(parameters)
        if not rows:
            return None

        statement, bound_parameters = _sqlalchemy_many_statement(query, rows)
        return self._require_session().execute(text(statement), bound_parameters)

    def executescript(self, script: str):
        raise NotImplementedError("executescript is not supported by SQLAlchemy storage")

    def commit(self) -> None:
        self._require_session().commit()

    def rollback(self) -> None:
        if self.session is not None:
            self.session.rollback()

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        if self._sqlalchemy_connection is not None:
            self._sqlalchemy_connection.close()
            self._sqlalchemy_connection = None
        self._sqlalchemy_engine.dispose()

    def _require_sqlalchemy_connection(self) -> Connection:
        if self._sqlalchemy_connection is None:
            self._sqlalchemy_connection = self._sqlalchemy_engine.connect()
        return self._sqlalchemy_connection

    def _require_session(self) -> Session:
        if self.session is None:
            self._require_sqlalchemy_connection()
            self.session = self._session_factory(bind=self._sqlalchemy_connection)
        return self.session


def connect_database(path: str | Path) -> StorageConnection:
    target = str(path)
    parsed = urlparse(target)

    if parsed.scheme in {"", "file", "sqlite"}:
        db_path = _sqlite_path_from_target(target)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return StorageConnection(target=str(db_path))

    return StorageConnection(target=target)


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
    upgrade_database(connection.target)


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


def upsert_company(connection: StorageConnection, company: CompanyRecord) -> int:
    session = connection._require_session()
    ticker = company.ticker.upper()
    record = session.execute(select(Company).where(Company.ticker == ticker)).scalar_one_or_none()
    if record is None:
        record = Company(ticker=ticker)
        session.add(record)
    record.name = company.name
    record.cik = company.cik
    record.website_domain = company.website_domain
    record.updated_at = _utc_now_datetime()
    session.commit()
    return int(record.id)


def load_company_by_ticker(connection: StorageConnection, ticker: str) -> Any | None:
    return connection.execute(
        """
        SELECT *
        FROM companies
        WHERE ticker = ?
        """,
        (ticker.upper(),),
    ).fetchone()


def load_latest_filing_company_metadata(
    connection: StorageConnection,
    ticker: str,
) -> Any | None:
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


def upsert_source(connection: StorageConnection, source: SourceRecord) -> int:
    _validate_source(source)
    session = connection._require_session()
    record = session.execute(select(Source).where(Source.url == source.url)).scalar_one_or_none()
    if record is None:
        record = Source(url=source.url)
        session.add(record)
    record.company_id = source.company_id
    record.name = source.name
    record.source_type = source.source_type
    record.ownership = source.ownership
    record.trust_level = source.trust_level
    record.discovery_status = source.discovery_status
    record.enabled = source.enabled
    record.confidence_score = source.confidence_score
    record.discovery_reason = source.discovery_reason
    record.updated_at = _utc_now_datetime()
    session.commit()
    return int(record.id)


def list_sources(
    connection: StorageConnection,
    *,
    ticker: str | None = None,
    discovery_status: str | None = None,
    enabled: bool | None = None,
) -> list[Any]:
    if discovery_status is not None:
        _validate_choice("discovery_status", discovery_status, SOURCE_DISCOVERY_STATUSES)

    query = """
        SELECT
            s.*,
            c.ticker,
            c.name AS company_name,
            COALESCE(document_counts.document_count, 0) AS document_count,
            latest_run.status AS last_ingestion_status,
            latest_run.completed_at AS last_ingestion_completed_at
        FROM sources AS s
        LEFT JOIN companies AS c ON c.id = s.company_id
        LEFT JOIN (
            SELECT source_id, COUNT(*) AS document_count
            FROM documents
            GROUP BY source_id
        ) AS document_counts ON document_counts.source_id = s.id
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
        ORDER BY
            c.ticker IS NULL,
            c.ticker,
            s.discovery_status,
            s.confidence_score DESC,
            s.name
    """
    return connection.execute(query, parameters).fetchall()


def load_source(connection: StorageConnection, source_id: int) -> Any | None:
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


def approve_source(connection: StorageConnection, source_id: int) -> Any | None:
    session = connection._require_session()
    source = session.get(Source, source_id)
    if source is not None:
        source.discovery_status = "approved"
        source.enabled = True
        source.updated_at = _utc_now_datetime()
        session.commit()
    return load_source(connection, source_id)


def reject_source(connection: StorageConnection, source_id: int) -> Any | None:
    session = connection._require_session()
    source = session.get(Source, source_id)
    if source is not None:
        source.discovery_status = "rejected"
        source.enabled = False
        source.updated_at = _utc_now_datetime()
        session.commit()
    return load_source(connection, source_id)


def upsert_document(connection: StorageConnection, document: DocumentRecord) -> int:
    _validate_document(document)
    fetched_at = _coerce_datetime(document.fetched_at) or _utc_now_datetime()
    metadata = document.metadata or {}
    session = connection._require_session()
    record = session.execute(
        select(Document).where(Document.source_id == document.source_id, Document.url == document.url)
    ).scalar_one_or_none()
    if record is None:
        record = Document(source_id=document.source_id, url=document.url)
        session.add(record)
    record.title = document.title
    record.author = document.author
    record.published_at = document.published_at
    record.fetched_at = fetched_at
    record.clean_text_path = document.clean_text_path
    record.content_hash = document.content_hash
    record.document_type = document.document_type
    record.metadata_json = metadata
    record.updated_at = _utc_now_datetime()
    session.commit()
    return int(record.id)


def replace_document_chunks(
    connection: StorageConnection,
    document_id: int,
    chunks: list[GenericDocumentChunk],
) -> None:
    session = connection._require_session()
    session.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete(
        synchronize_session=False
    )
    session.add_all(
        [
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                char_count=len(chunk.text),
            )
            for chunk in chunks
        ]
    )
    session.commit()


def create_source_ingestion_run(connection: StorageConnection, source_id: int) -> int:
    session = connection._require_session()
    run = SourceIngestionRun(
        source_id=source_id,
        status="running",
        started_at=_utc_now_datetime(),
    )
    session.add(run)
    session.commit()
    return int(run.id)


def complete_source_ingestion_run(
    connection: StorageConnection,
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

    session = connection._require_session()
    run = session.get(SourceIngestionRun, run_id)
    if run is None:
        raise RuntimeError(f"Failed to load source ingestion run: {run_id}")
    run.status = status
    run.completed_at = _utc_now_datetime()
    run.discovered_count = discovered_count
    run.inserted_count = inserted_count
    run.skipped_count = skipped_count
    run.error_message = error_message
    session.commit()


def upsert_filing(connection: StorageConnection, metadata: FilingMetadata) -> int:
    session = connection._require_session()
    record = session.execute(
        select(Filing).where(Filing.accession_number == metadata.accession_number)
    ).scalar_one_or_none()
    if record is None:
        record = Filing(accession_number=metadata.accession_number)
        session.add(record)
    record.ticker = metadata.ticker
    record.cik = metadata.cik
    record.company_name = metadata.company_name
    record.form_type = metadata.form_type
    record.filing_date = metadata.filing_date
    record.period_of_report = metadata.period_of_report
    record.raw_path = metadata.raw_path
    record.raw_sha256 = metadata.raw_sha256
    record.clean_text_path = metadata.clean_text_path
    record.updated_at = _utc_now_datetime()
    session.commit()
    return int(record.id)


def replace_filing_chunks(
    connection: StorageConnection,
    filing_id: int,
    chunks: list[TextChunk],
) -> None:
    session = connection._require_session()
    session.query(Chunk).filter(Chunk.filing_id == filing_id).delete(synchronize_session=False)
    session.add_all(
        [
            Chunk(
                filing_id=filing_id,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                char_count=len(chunk.text),
            )
            for chunk in chunks
        ]
    )
    now = _utc_now_datetime()
    for run in session.execute(
        select(EmbeddingRun).where(EmbeddingRun.filing_id == filing_id)
    ).scalars():
        run.status = "pending"
        run.expected_point_count = len(chunks)
        run.indexed_point_count = 0
        run.error_message = None
        run.started_at = None
        run.completed_at = None
        run.updated_at = now
    session.commit()


def load_chunks_for_vector_index(
    connection: StorageConnection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> list[Any]:
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
    connection: StorageConnection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> list[Any]:
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
    connection: StorageConnection,
    *,
    chunk_vector_ids: list[tuple[int, str]],
    embedding_model: str,
    vector_collection: str,
) -> None:
    session = connection._require_session()
    embedded_at = _utc_now_datetime()
    for chunk_id, vector_id in chunk_vector_ids:
        record = session.get(
            DocumentChunkEmbedding,
            (chunk_id, embedding_model, vector_collection),
        )
        if record is None:
            record = DocumentChunkEmbedding(
                document_chunk_id=chunk_id,
                embedding_model=embedding_model,
                vector_collection=vector_collection,
            )
            session.add(record)
        record.vector_id = vector_id
        record.embedded_at = embedded_at
    session.commit()


def record_chunk_embeddings(
    connection: StorageConnection,
    *,
    chunk_vector_ids: list[tuple[int, str]],
    embedding_model: str,
    vector_collection: str,
) -> None:
    session = connection._require_session()
    embedded_at = _utc_now_datetime()
    for chunk_id, vector_id in chunk_vector_ids:
        record = session.get(ChunkEmbedding, (chunk_id, embedding_model, vector_collection))
        if record is None:
            record = ChunkEmbedding(
                chunk_id=chunk_id,
                embedding_model=embedding_model,
                vector_collection=vector_collection,
            )
            session.add(record)
        record.vector_id = vector_id
        record.embedded_at = embedded_at
    session.commit()


def reset_sec_index_metadata(
    connection: StorageConnection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> None:
    session = connection._require_session()
    now = _utc_now_datetime()
    runs = session.execute(
        select(EmbeddingRun).where(
            EmbeddingRun.embedding_model == embedding_model,
            EmbeddingRun.vector_collection == vector_collection,
        )
    ).scalars()
    for run in runs:
        expected_count = session.query(Chunk).filter(Chunk.filing_id == run.filing_id).count()
        run.status = "pending"
        run.expected_point_count = expected_count
        run.indexed_point_count = 0
        run.error_message = None
        run.started_at = None
        run.completed_at = None
        run.updated_at = now
    session.query(ChunkEmbedding).filter(
        ChunkEmbedding.embedding_model == embedding_model,
        ChunkEmbedding.vector_collection == vector_collection,
    ).delete(synchronize_session=False)
    session.commit()


def reset_document_index_metadata(
    connection: StorageConnection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> None:
    session = connection._require_session()
    session.query(DocumentChunkEmbedding).filter(
        DocumentChunkEmbedding.embedding_model == embedding_model,
        DocumentChunkEmbedding.vector_collection == vector_collection,
    ).delete(synchronize_session=False)
    session.commit()


def set_embedding_run_status(
    connection: StorageConnection,
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

    now_dt = _utc_now_datetime()
    started_at = now_dt if status == "indexing" else None
    completed_at = now_dt if status in {"ready", "failed"} else None

    session = connection._require_session()
    run = session.get(EmbeddingRun, (filing_id, embedding_model, vector_collection))
    if run is None:
        run = EmbeddingRun(
            filing_id=filing_id,
            embedding_model=embedding_model,
            vector_collection=vector_collection,
        )
        session.add(run)
    run.status = status
    run.expected_point_count = expected_point_count
    run.indexed_point_count = indexed_point_count
    run.error_message = error_message
    if status == "indexing":
        run.started_at = started_at
    run.completed_at = completed_at
    run.updated_at = now_dt
    session.commit()


def get_ready_accession_numbers(
    connection: StorageConnection,
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


def load_planner_metadata(connection: StorageConnection) -> list[Any]:
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
    connection: StorageConnection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> list[Any]:
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


def load_index_section_counts(connection: StorageConnection) -> list[Any]:
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


def load_index_health_counts(
    connection: StorageConnection,
    *,
    embedding_model: str,
    vector_collection: str,
) -> dict[str, int]:
    row = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM chunks) AS sec_expected_points,
            (
                SELECT COALESCE(SUM(er.expected_point_count), 0)
                FROM embedding_runs AS er
                WHERE er.embedding_model = ?
                  AND er.vector_collection = ?
                  AND er.status = 'ready'
            ) AS sec_ready_points,
            (
                SELECT COUNT(*)
                FROM chunk_embeddings AS ce
                WHERE ce.embedding_model = ?
                  AND ce.vector_collection = ?
            ) AS sec_embedding_records,
            (
                SELECT COUNT(*)
                FROM document_chunks AS dc
                JOIN documents AS d ON d.id = dc.document_id
                JOIN sources AS s ON s.id = d.source_id
                WHERE s.enabled = ?
                  AND s.discovery_status IN ('approved', 'manual')
            ) AS document_expected_points,
            (
                SELECT COUNT(*)
                FROM document_chunk_embeddings AS dce
                WHERE dce.embedding_model = ?
                  AND dce.vector_collection = ?
            ) AS document_embedding_records
        """,
        (
            embedding_model,
            vector_collection,
            embedding_model,
            vector_collection,
            True,
            embedding_model,
            vector_collection,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to load index health counts")

    return {key: int(row[key] or 0) for key in row.keys()}


def update_embedding_run_progress(
    connection: StorageConnection,
    *,
    filing_id: int,
    embedding_model: str,
    vector_collection: str,
    indexed_point_count: int,
) -> None:
    session = connection._require_session()
    run = session.get(EmbeddingRun, (filing_id, embedding_model, vector_collection))
    if run is not None and run.status == "indexing":
        run.indexed_point_count = indexed_point_count
        run.updated_at = _utc_now_datetime()
        session.commit()


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


def _utc_now_datetime() -> datetime:
    return datetime.now(UTC)


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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
