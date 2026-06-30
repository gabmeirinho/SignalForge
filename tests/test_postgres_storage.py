import os
from urllib.parse import urlparse

import pytest
from sqlalchemy import create_engine, text

from signalforge.config import sqlalchemy_database_url
from signalforge.sections import TextChunk
from signalforge.storage import (
    CompanyRecord,
    DocumentRecord,
    FilingMetadata,
    GenericDocumentChunk,
    SourceRecord,
    approve_source,
    complete_source_ingestion_run,
    connect_database,
    create_source_ingestion_run,
    get_ready_accession_numbers,
    initialize_database,
    list_sources,
    load_chunks_for_vector_index,
    load_document_chunks_for_vector_index,
    load_index_metadata,
    load_index_section_counts,
    record_chunk_embeddings,
    record_document_chunk_embeddings,
    reject_source,
    replace_document_chunks,
    replace_filing_chunks,
    set_embedding_run_status,
    upsert_company,
    upsert_document,
    upsert_filing,
    upsert_source,
)


pytestmark = pytest.mark.postgres

SIGNALFORGE_TABLES = (
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


def test_postgres_source_lifecycle_and_document_deduplication(postgres_database_url):
    with connect_database(postgres_database_url) as connection:
        initialize_database(connection)

        company_id = upsert_company(
            connection,
            CompanyRecord(ticker="nvda", name="Old NVIDIA", website_domain="old.example"),
        )
        updated_company_id = upsert_company(
            connection,
            CompanyRecord(ticker="NVDA", name="NVIDIA CORP", website_domain="nvidia.com"),
        )
        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=updated_company_id,
                name="NVIDIA Blog",
                url="https://blogs.nvidia.com/feed/",
                source_type="news_feed",
                ownership="official",
                trust_level="high",
                discovery_status="candidate",
                confidence_score=0.9,
            ),
        )

        approved = approve_source(connection, source_id)
        rejected = reject_source(connection, source_id)

        document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://blogs.nvidia.com/blog/example/",
                title="Original title",
                content_hash="a" * 64,
                document_type="blog_post",
                metadata={"ticker": "NVDA", "tags": ["ai"]},
            ),
        )
        updated_document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://blogs.nvidia.com/blog/example/",
                title="Updated title",
                content_hash="b" * 64,
                document_type="blog_post",
                metadata={"ticker": "NVDA", "tags": ["infrastructure"]},
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [
                GenericDocumentChunk(chunk_index=0, text="First Postgres document chunk."),
                GenericDocumentChunk(chunk_index=1, text="Second Postgres document chunk."),
            ],
        )

        source = connection.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        document = connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        chunks = connection.execute(
            """
            SELECT *
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            (document_id,),
        ).fetchall()

    assert updated_company_id == company_id
    assert approved["discovery_status"] == "approved"
    assert bool(approved["enabled"]) is True
    assert rejected["discovery_status"] == "rejected"
    assert bool(rejected["enabled"]) is False
    assert updated_document_id == document_id
    assert bool(source["enabled"]) is False
    assert document["title"] == "Updated title"
    assert document["metadata_json"] == {"ticker": "NVDA", "tags": ["infrastructure"]}
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]


def test_postgres_embedding_status_index_metadata_and_mixed_retrieval(
    postgres_database_url,
):
    with connect_database(postgres_database_url) as connection:
        initialize_database(connection)

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
                raw_path="raw.txt",
                raw_sha256="sha",
                clean_text_path="clean.txt",
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
        sec_rows = load_chunks_for_vector_index(
            connection,
            embedding_model="test-model",
            vector_collection="test-collection",
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
        record_chunk_embeddings(
            connection,
            chunk_vector_ids=[(int(row["chunk_id"]), f"sec-vector-{index}") for index, row in enumerate(sec_rows)],
            embedding_model="test-model",
            vector_collection="test-collection",
        )

        company_id = upsert_company(connection, CompanyRecord(ticker="NVDA", name="NVIDIA CORP"))
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
                content_hash="c" * 64,
                document_type="press_release",
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [GenericDocumentChunk(chunk_index=0, text="Web document evidence.")],
        )
        document_rows = load_document_chunks_for_vector_index(
            connection,
            embedding_model="test-model",
            vector_collection="test-collection",
        )
        record_document_chunk_embeddings(
            connection,
            chunk_vector_ids=[
                (int(document_rows[0]["document_chunk_id"]), "document-vector")
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

        ready_accessions = get_ready_accession_numbers(
            connection,
            embedding_model="test-model",
            vector_collection="test-collection",
            ticker="nvda",
            filing_years=[2026],
        )
        index_rows = load_index_metadata(
            connection,
            embedding_model="test-model",
            vector_collection="test-collection",
        )
        section_rows = load_index_section_counts(connection)
        sources = list_sources(connection)
        remaining_document_rows = load_document_chunks_for_vector_index(
            connection,
            embedding_model="test-model",
            vector_collection="test-collection",
        )

    assert ready_accessions == ["0001045810-26-000021"]
    assert len(sec_rows) == 2
    assert {row["section_id"] for row in section_rows} == {"1A", "7"}
    assert index_rows[0]["status"] == "ready"
    assert index_rows[0]["indexed_point_count"] == 2
    assert document_rows[0]["source_name"] == "NVIDIA Newsroom"
    assert document_rows[0]["ticker"] == "NVDA"
    assert remaining_document_rows == []
    assert sources[0]["document_count"] == 1
    assert sources[0]["last_ingestion_status"] == "completed"


def reset_postgres_database(database_url: str) -> None:
    engine = create_engine(sqlalchemy_database_url(database_url), future=True)
    try:
        with engine.begin() as connection:
            for table_name in SIGNALFORGE_TABLES:
                connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
    finally:
        engine.dispose()
