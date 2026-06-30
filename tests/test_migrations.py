from alembic import command
from sqlalchemy import create_engine, inspect

from signalforge.config import sqlalchemy_database_url
from signalforge.migrations import alembic_config as make_alembic_config
from signalforge.storage import (
    CompanyRecord,
    DocumentRecord,
    GenericDocumentChunk,
    SourceRecord,
    connect_database,
    initialize_database,
    list_sources,
    replace_document_chunks,
    upsert_company,
    upsert_document,
    upsert_source,
)


EXPECTED_TABLES = {
    "alembic_version",
    "filings",
    "chunks",
    "chunk_embeddings",
    "document_chunk_embeddings",
    "embedding_runs",
    "companies",
    "sources",
    "documents",
    "document_chunks",
    "source_ingestion_runs",
}


def test_initial_migration_creates_current_schema(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    cfg = make_alembic_config(db_path)

    command.upgrade(cfg, "head")

    engine = create_engine(sqlalchemy_database_url(str(db_path)))
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) == EXPECTED_TABLES
    assert {index["name"] for index in inspector.get_indexes("sources")} >= {
        "idx_sources_company_id",
        "idx_sources_discovery_status",
    }
    assert {index["name"] for index in inspector.get_indexes("documents")} >= {
        "idx_documents_source_id",
        "idx_documents_content_hash",
    }
    assert {index["name"] for index in inspector.get_indexes("source_ingestion_runs")} >= {
        "idx_source_ingestion_runs_source_id",
    }

    command.downgrade(cfg, "base")

    remaining_tables = set(inspect(engine).get_table_names())
    assert remaining_tables <= {"alembic_version"}


def test_migrated_sqlite_database_supports_existing_storage_functions(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    command.upgrade(make_alembic_config(db_path), "head")

    with connect_database(db_path) as connection:
        company_id = upsert_company(
            connection,
            CompanyRecord(ticker="NVDA", name="NVIDIA CORP", website_domain="nvidia.com"),
        )
        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="NVIDIA Blog",
                url="https://blogs.nvidia.com/feed/",
                source_type="news_feed",
                ownership="official",
                trust_level="high",
                discovery_status="approved",
                confidence_score=0.95,
            ),
        )
        document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://blogs.nvidia.com/blog/example/",
                title="Example",
                content_hash="a" * 64,
                document_type="blog_post",
                metadata={"topic": "ai"},
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [GenericDocumentChunk(chunk_index=0, text="AI infrastructure update")],
        )

        sources = list_sources(connection)

    assert len(sources) == 1
    assert sources[0]["name"] == "NVIDIA Blog"
    assert sources[0]["document_count"] == 1


def test_initialize_database_uses_migrations_for_new_file_database(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"

    with connect_database(db_path) as connection:
        initialize_database(connection)
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert version["version_num"] == "0001_initial_schema"
