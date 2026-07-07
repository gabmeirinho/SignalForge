from datetime import UTC, datetime

from alembic import command
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError

from signalforge.config import sqlalchemy_database_url
from signalforge.migrations import alembic_config as make_alembic_config
from signalforge.models import Base, QueryRun, ResearchSession
from signalforge.storage import (
    CompanyRecord,
    DocumentRecord,
    GenericDocumentChunk,
    SourceRecord,
    StorageConnection,
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
    "research_sessions",
    "query_runs",
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
    assert {index["name"] for index in inspector.get_indexes("query_runs")} >= {
        "idx_query_runs_research_session_id",
        "idx_query_runs_status",
        "idx_query_runs_started_at",
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

    assert version["version_num"] == "0002_add_research_sessions_query_runs"


def test_sqlalchemy_connection_path_supports_existing_storage_functions(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    engine = create_engine(sqlalchemy_database_url(str(db_path)), future=True)

    with StorageConnection(target=str(db_path), sqlalchemy_engine=engine) as connection:
        initialize_database(connection)
        company_id = upsert_company(connection, CompanyRecord(ticker="AMD", name="AMD"))
        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="AMD Newsroom",
                url="https://example.com/amd",
                source_type="newsroom",
                discovery_status="approved",
            ),
        )
        document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://example.com/amd/article",
                title="AMD Update",
                content_hash="b" * 64,
                document_type="article",
                metadata={"category": "news"},
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [GenericDocumentChunk(chunk_index=0, text="AMD product update")],
        )

        sources = list_sources(connection, enabled=True)

    assert len(sources) == 1
    assert sources[0]["ticker"] == "AMD"
    assert sources[0]["document_count"] == 1


def test_orm_metadata_matches_storage_schema_tables():
    assert set(Base.metadata.tables) == EXPECTED_TABLES - {"alembic_version"}

    sources = Base.metadata.tables["sources"]
    source_index_names = {index.name for index in sources.indexes}
    assert source_index_names >= {"idx_sources_company_id", "idx_sources_discovery_status"}
    assert any("source_type" in str(constraint.sqltext) for constraint in sources.constraints if hasattr(constraint, "sqltext"))

    documents = Base.metadata.tables["documents"]
    document_index_names = {index.name for index in documents.indexes}
    assert document_index_names >= {"idx_documents_source_id", "idx_documents_content_hash"}
    assert any(
        {"source_id", "url"} == {column.name for column in constraint.columns}
        for constraint in documents.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    )

    ingestion_runs = Base.metadata.tables["source_ingestion_runs"]
    assert {index.name for index in ingestion_runs.indexes} >= {
        "idx_source_ingestion_runs_source_id"
    }

    query_runs = Base.metadata.tables["query_runs"]
    assert {index.name for index in query_runs.indexes} >= {
        "idx_query_runs_research_session_id",
        "idx_query_runs_status",
        "idx_query_runs_started_at",
    }
    assert any("status" in str(constraint.sqltext) for constraint in query_runs.constraints if hasattr(constraint, "sqltext"))


def test_query_session_orm_schema_round_trips_json_and_enforces_status(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    started_at = datetime.now(UTC)

    with connect_database(db_path) as connection:
        initialize_database(connection)
        if connection.session is None:
            raise AssertionError("StorageConnection did not create a SQLAlchemy session")

        session = connection.session
        research_session = ResearchSession(
            session_key="session-1",
            title="NVDA risks",
            metadata_json={"ticker": "NVDA", "workflow": "research"},
        )
        session.add(research_session)
        session.flush()

        query_run = QueryRun(
            research_session_id=research_session.id,
            question="What changed in NVDA risk factors?",
            status="completed",
            planner_model="planner",
            answer_model="answer",
            embedding_model="embedding",
            vector_collection="collection",
            planned_query_json={"semantic_queries": ["NVDA risk factors"]},
            retrieval_metadata_json={"chunks": [1, 2]},
            answer_text="Answer",
            started_at=started_at,
            completed_at=started_at,
        )
        session.add(query_run)
        session.commit()

        loaded_session = session.get(ResearchSession, research_session.id)
        loaded_query = session.get(QueryRun, query_run.id)

        assert loaded_session is not None
        assert loaded_query is not None
        assert loaded_session.metadata_json == {"ticker": "NVDA", "workflow": "research"}
        assert loaded_query.research_session_id == loaded_session.id
        assert loaded_query.planned_query_json == {"semantic_queries": ["NVDA risk factors"]}
        assert loaded_query.retrieval_metadata_json == {"chunks": [1, 2]}

        session.add(
            QueryRun(
                question="Invalid status",
                status="queued",
                planned_query_json={},
                retrieval_metadata_json={},
                started_at=started_at,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
