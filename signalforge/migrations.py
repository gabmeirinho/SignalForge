from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from signalforge.config import sqlalchemy_database_url


LEGACY_INITIAL_REVISION = "0001_initial_schema"
LEGACY_INITIAL_SCHEMA = {
    "filings": {
        "id",
        "accession_number",
        "ticker",
        "cik",
        "company_name",
        "form_type",
        "filing_date",
        "period_of_report",
        "raw_path",
        "raw_sha256",
        "clean_text_path",
        "created_at",
        "updated_at",
    },
    "chunks": {
        "id",
        "filing_id",
        "section_id",
        "section_title",
        "chunk_index",
        "text",
        "char_count",
        "created_at",
        "updated_at",
    },
    "chunk_embeddings": {
        "chunk_id",
        "embedding_model",
        "vector_collection",
        "vector_id",
        "embedded_at",
    },
    "embedding_runs": {
        "filing_id",
        "embedding_model",
        "vector_collection",
        "status",
        "expected_point_count",
        "indexed_point_count",
        "error_message",
        "started_at",
        "completed_at",
        "updated_at",
    },
    "companies": {
        "id",
        "ticker",
        "name",
        "cik",
        "website_domain",
        "created_at",
        "updated_at",
    },
    "sources": {
        "id",
        "company_id",
        "name",
        "url",
        "source_type",
        "ownership",
        "trust_level",
        "discovery_status",
        "enabled",
        "confidence_score",
        "discovery_reason",
        "created_at",
        "updated_at",
    },
    "documents": {
        "id",
        "source_id",
        "url",
        "title",
        "author",
        "published_at",
        "fetched_at",
        "clean_text_path",
        "content_hash",
        "document_type",
        "metadata_json",
        "created_at",
        "updated_at",
    },
    "document_chunks": {
        "id",
        "document_id",
        "chunk_index",
        "text",
        "char_count",
        "created_at",
        "updated_at",
    },
    "document_chunk_embeddings": {
        "document_chunk_id",
        "embedding_model",
        "vector_collection",
        "vector_id",
        "embedded_at",
    },
    "source_ingestion_runs": {
        "id",
        "source_id",
        "status",
        "started_at",
        "completed_at",
        "discovered_count",
        "inserted_count",
        "skipped_count",
        "error_message",
    },
}


def upgrade_database(target: str | Path, revision: str = "head") -> None:
    cfg = alembic_config(target)
    if _is_unversioned_legacy_sqlite(target):
        command.stamp(cfg, LEGACY_INITIAL_REVISION)
    command.upgrade(cfg, revision)


def alembic_config(target: str | Path) -> Config:
    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_database_url(str(target)))
    return cfg


def _is_unversioned_legacy_sqlite(target: str | Path) -> bool:
    engine = create_engine(sqlalchemy_database_url(str(target)), future=True)
    try:
        if engine.dialect.name != "sqlite":
            return False

        with engine.connect() as connection:
            inspector = inspect(connection)
            table_names = set(inspector.get_table_names())
            if "alembic_version" in table_names:
                recorded_revision = connection.execute(
                    text("SELECT version_num FROM alembic_version LIMIT 1")
                ).scalar_one_or_none()
                if recorded_revision is not None:
                    return False

            application_tables = table_names - {"alembic_version"}
            if application_tables != set(LEGACY_INITIAL_SCHEMA):
                return False

            return all(
                {column["name"] for column in inspector.get_columns(table_name)} == expected_columns
                for table_name, expected_columns in LEGACY_INITIAL_SCHEMA.items()
            )
    finally:
        engine.dispose()
