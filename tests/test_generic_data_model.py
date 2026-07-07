import json

import pytest

from signalforge.storage import (
    CompanyRecord,
    DocumentRecord,
    GenericDocumentChunk,
    SourceRecord,
    complete_query_run,
    complete_source_ingestion_run,
    connect_database,
    create_query_run,
    create_source_ingestion_run,
    fail_query_run,
    get_or_create_research_session,
    initialize_database,
    list_query_runs,
    load_query_run,
    load_research_session,
    replace_document_chunks,
    load_document_chunks_for_vector_index,
    record_document_chunk_embeddings,
    upsert_company,
    upsert_document,
    upsert_source,
)


def test_initialize_database_adds_generic_model_tables_without_dropping_sec_tables(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }

    assert {
        "filings",
        "chunks",
        "companies",
        "sources",
        "documents",
        "document_chunks",
        "source_ingestion_runs",
    }.issubset(tables)


def test_generic_company_source_document_and_chunk_persistence(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)

        company_id = upsert_company(
            connection,
            CompanyRecord(
                ticker="nvda",
                name="NVIDIA Corp",
                cik="0001045810",
                website_domain="nvidia.com",
            ),
        )
        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="NVIDIA Blog",
                url="https://blogs.nvidia.com/",
                source_type="company_blog",
                ownership="official",
                trust_level="high",
                discovery_status="approved",
                confidence_score=0.95,
                discovery_reason="official nvidia.com blog path",
            ),
        )
        document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://blogs.nvidia.com/blog/example/",
                title="Example AI Infrastructure Update",
                author="NVIDIA",
                published_at="2026-06-01T12:00:00+00:00",
                fetched_at="2026-06-02T12:00:00+00:00",
                clean_text_path="data/processed/web/nvda/example.txt",
                content_hash="b" * 64,
                document_type="blog_post",
                metadata={"ticker": "NVDA", "tags": ["ai", "infrastructure"]},
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [
                GenericDocumentChunk(chunk_index=0, text="First chunk."),
                GenericDocumentChunk(chunk_index=1, text="Second chunk."),
            ],
        )

        company = connection.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
        source = connection.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        document = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        chunks = connection.execute(
            """
            SELECT *
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            (document_id,),
        ).fetchall()

    assert company["ticker"] == "NVDA"
    assert source["enabled"] == 1
    assert source["discovery_status"] == "approved"
    assert document["document_type"] == "blog_post"
    assert json.loads(document["metadata_json"]) == {
        "tags": ["ai", "infrastructure"],
        "ticker": "NVDA",
    }
    assert [row["char_count"] for row in chunks] == [12, 13]


def test_document_chunk_embedding_tracking_filters_indexed_chunks(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        company_id = upsert_company(connection, CompanyRecord(ticker="NVDA", name="NVIDIA CORP"))
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
            ),
        )
        document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://blogs.nvidia.com/blog/example/",
                title="Example",
                content_hash="d" * 64,
                document_type="blog_post",
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [GenericDocumentChunk(chunk_index=0, text="Document chunk.")],
        )

        rows = load_document_chunks_for_vector_index(
            connection,
            embedding_model="test-model",
            vector_collection="test-collection",
        )
        record_document_chunk_embeddings(
            connection,
            chunk_vector_ids=[(int(rows[0]["document_chunk_id"]), "vector-id")],
            embedding_model="test-model",
            vector_collection="test-collection",
        )
        remaining = load_document_chunks_for_vector_index(
            connection,
            embedding_model="test-model",
            vector_collection="test-collection",
        )

    assert len(rows) == 1
    assert rows[0]["source_name"] == "NVIDIA Blog"
    assert rows[0]["ticker"] == "NVDA"
    assert remaining == []


def test_generic_model_upserts_replace_existing_rows(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        company_id = upsert_company(connection, CompanyRecord(ticker="MSFT", name="Old"))
        updated_company_id = upsert_company(
            connection,
            CompanyRecord(ticker="msft", name="Microsoft", website_domain="microsoft.com"),
        )
        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=updated_company_id,
                name="Old Source",
                url="https://blogs.microsoft.com/",
                source_type="company_blog",
            ),
        )
        updated_source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=updated_company_id,
                name="Microsoft Blog",
                url="https://blogs.microsoft.com/",
                source_type="newsroom",
                ownership="official",
                trust_level="high",
                discovery_status="manual",
                enabled=False,
            ),
        )

        companies = connection.execute("SELECT * FROM companies").fetchall()
        sources = connection.execute("SELECT * FROM sources").fetchall()

    assert updated_company_id == company_id
    assert updated_source_id == source_id
    assert len(companies) == 1
    assert companies[0]["name"] == "Microsoft"
    assert len(sources) == 1
    assert sources[0]["name"] == "Microsoft Blog"
    assert sources[0]["enabled"] == 0


def test_source_ingestion_run_tracks_completion_counts(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        source_id = upsert_source(
            connection,
            SourceRecord(
                name="Industry Feed",
                url="https://example.com/feed",
                source_type="news_feed",
                ownership="third_party",
            ),
        )
        run_id = create_source_ingestion_run(connection, source_id)
        complete_source_ingestion_run(
            connection,
            run_id=run_id,
            status="partial",
            discovered_count=5,
            inserted_count=3,
            skipped_count=2,
            error_message="one document failed",
        )

        run = connection.execute(
            "SELECT * FROM source_ingestion_runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    assert run["source_id"] == source_id
    assert run["status"] == "partial"
    assert run["completed_at"] is not None
    assert run["discovered_count"] == 5
    assert run["inserted_count"] == 3
    assert run["skipped_count"] == 2
    assert run["error_message"] == "one document failed"


def test_query_run_lifecycle_persists_completed_history(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        session_id = get_or_create_research_session(
            connection,
            session_key="local-session",
            title="NVDA research",
            metadata={"ticker": "NVDA"},
        )
        same_session_id = get_or_create_research_session(
            connection,
            session_key="local-session",
            title="Updated title",
            metadata={"ticker": "NVDA", "workflow": "research"},
        )
        run_id = create_query_run(
            connection,
            research_session_id=session_id,
            question="What changed in NVDA risk factors?",
            planner_model="planner-v1",
            answer_model="answer-v1",
            embedding_model="embedding-v1",
            vector_collection="collection-v1",
        )
        complete_query_run(
            connection,
            run_id=run_id,
            answer_text="Risk factors changed.",
            planned_query={"semantic_queries": ["NVDA risk factors"], "top_k": 8},
            retrieval_metadata={
                "sources": [{"label": "[1]", "score": 0.81, "ticker": "NVDA"}],
                "warnings": [],
            },
        )

        session_record = load_research_session(connection, session_id)
        loaded = load_query_run(connection, run_id)
        runs = list_query_runs(connection, research_session_id=session_id)

    assert same_session_id == session_id
    assert session_record is not None
    assert session_record.title == "Updated title"
    assert session_record.metadata == {"ticker": "NVDA", "workflow": "research"}
    assert loaded is not None
    assert loaded.id == run_id
    assert loaded.research_session_id == session_id
    assert loaded.question == "What changed in NVDA risk factors?"
    assert loaded.status == "completed"
    assert loaded.answer_text == "Risk factors changed."
    assert loaded.error_message is None
    assert loaded.planner_model == "planner-v1"
    assert loaded.answer_model == "answer-v1"
    assert loaded.embedding_model == "embedding-v1"
    assert loaded.vector_collection == "collection-v1"
    assert loaded.planned_query == {"semantic_queries": ["NVDA risk factors"], "top_k": 8}
    assert loaded.retrieval_metadata == {
        "sources": [{"label": "[1]", "score": 0.81, "ticker": "NVDA"}],
        "warnings": [],
    }
    assert loaded.started_at is not None
    assert loaded.completed_at is not None
    assert [run.id for run in runs] == [run_id]


def test_query_run_lifecycle_persists_failed_history_and_lists_recent_first(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        session_id = get_or_create_research_session(
            connection,
            session_key="local-session",
        )
        completed_run_id = create_query_run(
            connection,
            research_session_id=session_id,
            question="Completed question",
        )
        complete_query_run(
            connection,
            run_id=completed_run_id,
            answer_text="Completed answer",
        )
        failed_run_id = create_query_run(
            connection,
            research_session_id=session_id,
            question="Failed question",
            embedding_model="embedding-v1",
            vector_collection="collection-v1",
        )
        fail_query_run(
            connection,
            run_id=failed_run_id,
            error_message="RAG query failed: RuntimeError",
            retrieval_metadata={"warnings": ["failed before retrieval"]},
            planner_model="planner-v1",
            answer_model="answer-v1",
        )

        failed = load_query_run(connection, failed_run_id)
        all_runs = list_query_runs(connection, limit=10)
        limited_runs = list_query_runs(connection, limit=1)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.answer_text is None
    assert failed.error_message == "RAG query failed: RuntimeError"
    assert failed.planner_model == "planner-v1"
    assert failed.answer_model == "answer-v1"
    assert failed.embedding_model == "embedding-v1"
    assert failed.vector_collection == "collection-v1"
    assert failed.retrieval_metadata == {"warnings": ["failed before retrieval"]}
    assert [run.id for run in all_runs] == [failed_run_id, completed_run_id]
    assert [run.id for run in limited_runs] == [failed_run_id]


def test_query_run_helpers_validate_missing_records_and_limits(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)

        assert load_research_session(connection, 404) is None
        assert load_query_run(connection, 404) is None
        with pytest.raises(RuntimeError, match="Failed to load query run"):
            complete_query_run(connection, run_id=404, answer_text="Answer")
        with pytest.raises(ValueError, match="limit must be greater than zero"):
            list_query_runs(connection, limit=0)


def test_generic_model_rejects_unsupported_values(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        with pytest.raises(ValueError, match="Unsupported source_type"):
            upsert_source(
                connection,
                SourceRecord(
                    name="Bad Source",
                    url="https://example.com/",
                    source_type="social_media",
                ),
            )

        source_id = upsert_source(
            connection,
            SourceRecord(
                name="Web Page",
                url="https://example.com/",
                source_type="webpage",
            ),
        )
        with pytest.raises(ValueError, match="Unsupported document_type"):
            upsert_document(
                connection,
                DocumentRecord(
                    source_id=source_id,
                    url="https://example.com/a",
                    title="Bad Document",
                    content_hash="c" * 64,
                    document_type="transcript",
                ),
            )
