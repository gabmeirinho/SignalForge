import json
import os
from urllib.parse import urlparse

import pytest
from sqlalchemy import create_engine, text

from signalforge.config import sqlalchemy_database_url
from signalforge.sections import TextChunk
from signalforge.sqlite_to_postgres import migrate_sqlite_to_postgres
from signalforge.storage import (
    CompanyRecord,
    DocumentRecord,
    FilingMetadata,
    GenericDocumentChunk,
    SourceRecord,
    complete_source_ingestion_run,
    connect_database,
    create_source_ingestion_run,
    initialize_database,
    record_chunk_embeddings,
    record_document_chunk_embeddings,
    replace_document_chunks,
    replace_filing_chunks,
    set_embedding_run_status,
    upsert_company,
    upsert_document,
    upsert_filing,
    upsert_source,
)


SIGNALFORGE_TABLES = (
    "query_runs",
    "research_sessions",
    "source_ingestion_runs",
    "document_chunk_embeddings",
    "document_chunks",
    "documents",
    "sources",
    "companies",
    "embedding_runs",
    "chunk_embeddings",
    "chunks",
    "filings",
    "alembic_version",
)


@pytest.fixture
def postgres_database_url():
    database_url = os.environ["SIGNALFORGE_POSTGRES_TEST_DATABASE_URL"]
    parsed = urlparse(database_url)
    database_name = parsed.path.rsplit("/", 1)[-1].lower()
    if "test" not in database_name:
        pytest.skip(
            "SIGNALFORGE_POSTGRES_TEST_DATABASE_URL must point to a disposable "
            "database whose name contains 'test'"
        )

    reset_postgres_database(database_url)
    try:
        yield database_url
    finally:
        reset_postgres_database(database_url)


def test_migrate_sqlite_to_postgres_dry_run_counts_source_rows(tmp_path):
    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    create_sample_sqlite_database(source_path)

    result = migrate_sqlite_to_postgres(
        sqlite_path=source_path,
        postgres_url=str(target_path),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.target_total is None
    assert result.source_total == 16
    assert {table.table_name: table.source_count for table in result.tables} == {
        "companies": 1,
        "filings": 1,
        "sources": 1,
        "documents": 1,
        "chunks": 2,
        "document_chunks": 2,
        "chunk_embeddings": 2,
        "document_chunk_embeddings": 2,
        "embedding_runs": 1,
        "source_ingestion_runs": 1,
        "research_sessions": 1,
        "query_runs": 1,
    }


def test_migrate_sqlite_to_postgres_executes_and_preserves_ids_and_counts(tmp_path):
    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    create_sample_sqlite_database(source_path)

    result = migrate_sqlite_to_postgres(
        sqlite_path=source_path,
        postgres_url=str(target_path),
        dry_run=False,
    )

    assert result.dry_run is False
    assert result.source_total == result.target_total == 16

    with connect_database(target_path) as connection:
        company = connection.execute("SELECT * FROM companies").fetchone()
        document = connection.execute("SELECT * FROM documents").fetchone()
        query_run = connection.execute("SELECT * FROM query_runs").fetchone()
        chunks = connection.execute("SELECT * FROM chunks ORDER BY id").fetchall()

    assert company["id"] == 1
    assert document["id"] == 1
    assert json.loads(document["metadata_json"]) == {"tags": ["ai"], "ticker": "NVDA"}
    assert json.loads(query_run["planned_query_json"]) == {
        "semantic_queries": ["NVDA risk factors"]
    }
    assert [chunk["id"] for chunk in chunks] == [1, 2]


def test_migrate_sqlite_to_postgres_requires_empty_target_unless_replace(tmp_path):
    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    create_sample_sqlite_database(source_path)
    migrate_sqlite_to_postgres(sqlite_path=source_path, postgres_url=str(target_path), dry_run=False)

    with pytest.raises(ValueError, match="target is not empty"):
        migrate_sqlite_to_postgres(
            sqlite_path=source_path,
            postgres_url=str(target_path),
            dry_run=False,
        )

    result = migrate_sqlite_to_postgres(
        sqlite_path=source_path,
        postgres_url=str(target_path),
        dry_run=False,
        replace=True,
    )

    assert result.replaced is True
    assert result.target_total == 16


@pytest.mark.postgres
def test_migrate_sqlite_to_postgres_sets_postgres_jsonb_and_sequences(
    tmp_path,
    postgres_database_url,
):
    source_path = tmp_path / "source.sqlite3"
    create_sample_sqlite_database(source_path)

    migrate_sqlite_to_postgres(
        sqlite_path=source_path,
        postgres_url=postgres_database_url,
        dry_run=False,
    )

    with connect_database(postgres_database_url) as connection:
        document = connection.execute("SELECT * FROM documents").fetchone()
        query_run = connection.execute("SELECT * FROM query_runs").fetchone()
        company_id = upsert_company(
            connection,
            CompanyRecord(ticker="AMD", name="Advanced Micro Devices"),
        )

    assert document["metadata_json"] == {"tags": ["ai"], "ticker": "NVDA"}
    assert query_run["planned_query_json"] == {"semantic_queries": ["NVDA risk factors"]}
    assert company_id == 2


def reset_postgres_database(database_url: str) -> None:
    engine = create_engine(sqlalchemy_database_url(database_url), future=True)
    try:
        with engine.begin() as connection:
            for table_name in SIGNALFORGE_TABLES:
                connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
    finally:
        engine.dispose()


def create_sample_sqlite_database(path):
    with connect_database(path) as connection:
        initialize_database(connection)
        company_id = upsert_company(
            connection,
            CompanyRecord(ticker="NVDA", name="NVIDIA CORP", website_domain="nvidia.com"),
        )
        filing_id = upsert_filing(
            connection,
            FilingMetadata(
                accession_number="0001045810-26-000021",
                ticker="NVDA",
                cik="1045810",
                company_name="NVIDIA CORP",
                form_type="10-K",
                filing_date="2026-02-25",
                period_of_report="2026-01-25",
                raw_path="data/raw/sec-edgar-filings/NVDA/10-K/full-submission.txt",
                raw_sha256="a" * 64,
                clean_text_path="data/processed/clean_text/NVDA/10-K/clean-text.txt",
            ),
        )
        replace_filing_chunks(
            connection,
            filing_id,
            [
                TextChunk("1A", "Risk Factors", 0, "Risk text"),
                TextChunk("7", "MD&A", 0, "Management discussion"),
            ],
        )
        set_embedding_run_status(
            connection,
            filing_id=filing_id,
            embedding_model="test-model",
            vector_collection="test-collection",
            status="ready",
            expected_point_count=2,
            indexed_point_count=2,
        )
        sec_rows = connection.execute("SELECT id FROM chunks ORDER BY id").fetchall()
        record_chunk_embeddings(
            connection,
            chunk_vector_ids=[(int(row["id"]), f"sec-vector-{index}") for index, row in enumerate(sec_rows)],
            embedding_model="test-model",
            vector_collection="test-collection",
        )

        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="NVIDIA Newsroom",
                url="https://nvidianews.nvidia.com/",
                source_type="newsroom",
                ownership="official",
                trust_level="high",
                discovery_status="approved",
            ),
        )
        document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://nvidianews.nvidia.com/news/example/",
                title="AI Infrastructure Update",
                content_hash="b" * 64,
                document_type="press_release",
                metadata={"ticker": "NVDA", "tags": ["ai"]},
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [
                GenericDocumentChunk(chunk_index=0, text="First web evidence."),
                GenericDocumentChunk(chunk_index=1, text="Second web evidence."),
            ],
        )
        document_rows = connection.execute("SELECT id FROM document_chunks ORDER BY id").fetchall()
        record_document_chunk_embeddings(
            connection,
            chunk_vector_ids=[
                (int(row["id"]), f"document-vector-{index}") for index, row in enumerate(document_rows)
            ],
            embedding_model="test-model",
            vector_collection="test-collection",
        )
        run_id = create_source_ingestion_run(connection, source_id)
        complete_source_ingestion_run(
            connection,
            run_id=run_id,
            status="completed",
            discovered_count=1,
            inserted_count=1,
        )
        connection.execute(
            """
            INSERT INTO research_sessions (
                session_key,
                title,
                metadata_json
            )
            VALUES (?, ?, ?)
            """,
            (
                "session-1",
                "NVDA research",
                json.dumps({"ticker": "NVDA"}, sort_keys=True),
            ),
        )
        connection.execute(
            """
            INSERT INTO query_runs (
                research_session_id,
                question,
                status,
                planner_model,
                answer_model,
                embedding_model,
                vector_collection,
                planned_query_json,
                retrieval_metadata_json,
                answer_text,
                started_at,
                completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "What changed in NVDA risk factors?",
                "completed",
                "planner",
                "answer",
                "test-model",
                "test-collection",
                json.dumps({"semantic_queries": ["NVDA risk factors"]}, sort_keys=True),
                json.dumps({"chunks": [1, 2]}, sort_keys=True),
                "Answer",
                "2026-07-06T00:00:00+00:00",
                "2026-07-06T00:00:00+00:00",
            ),
        )
        connection.commit()
