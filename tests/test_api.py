from fastapi.testclient import TestClient

from signalforge.api import app
from signalforge.query_planner import SearchPlan
from signalforge.rag_service import QueryResponse, SourceChunk
from signalforge.sections import TextChunk
from signalforge.storage import (
    CompanyRecord,
    DocumentRecord,
    FilingMetadata,
    SourceRecord,
    complete_source_ingestion_run,
    connect_database,
    create_source_ingestion_run,
    initialize_database,
    replace_filing_chunks,
    set_embedding_run_status,
    upsert_company,
    upsert_document,
    upsert_filing,
    upsert_source,
)


def test_health_reports_local_paths(monkeypatch, tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    qdrant_path = tmp_path / "qdrant"
    db_path.touch()
    qdrant_path.mkdir()
    monkeypatch.setenv("SIGNALFORGE_DB_PATH", str(db_path))
    monkeypatch.setenv("SIGNALFORGE_QDRANT_PATH", str(qdrant_path))

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": True,
        "qdrant_path": True,
    }


def test_health_accepts_qdrant_server_url(monkeypatch, tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    db_path.touch()
    monkeypatch.setenv("SIGNALFORGE_DB_PATH", str(db_path))
    monkeypatch.setenv("SIGNALFORGE_QDRANT_URL", "http://localhost:6333")

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["qdrant_path"] is True


def test_index_returns_filings_and_section_counts(monkeypatch, tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    monkeypatch.setenv("SIGNALFORGE_DB_PATH", str(db_path))
    monkeypatch.setenv("SIGNALFORGE_QDRANT_PATH", str(qdrant_path))
    monkeypatch.setenv("SIGNALFORGE_EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("SIGNALFORGE_COLLECTION", "test-collection")

    with connect_database(db_path) as connection:
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
                TextChunk("1A", "Risk Factors", 1, "More risk text"),
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
        company_id = upsert_company(
            connection,
            CompanyRecord(ticker="NVDA", name="NVIDIA CORP", website_domain="nvidia.com"),
        )
        approved_source_id = upsert_source(
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
        upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="NVIDIA Newsroom",
                url="https://nvidianews.nvidia.com/",
                source_type="newsroom",
                ownership="official",
                trust_level="high",
                discovery_status="candidate",
                confidence_score=0.9,
            ),
        )
        upsert_document(
            connection,
            DocumentRecord(
                source_id=approved_source_id,
                url="https://blogs.nvidia.com/blog/example/",
                title="Example",
                content_hash="a" * 64,
                document_type="blog_post",
            ),
        )
        run_id = create_source_ingestion_run(connection, approved_source_id)
        complete_source_ingestion_run(
            connection,
            run_id=run_id,
            status="completed",
            discovered_count=1,
            inserted_count=1,
        )

    response = TestClient(app).get("/api/index")

    assert response.status_code == 200
    payload = response.json()
    assert payload["embedding_model"] == "test-model"
    assert payload["collection"] == "test-collection"
    assert payload["tickers"][0]["ticker"] == "NVDA"
    assert payload["tickers"][0]["filings"][0]["status"] == "ready"
    assert payload["tickers"][0]["sections"] == [{"section_id": "1A", "chunk_count": 2}]
    assert payload["summary"] == {
        "indexed_filing_count": 1,
        "approved_source_count": 1,
        "candidate_source_count": 1,
        "document_count": 1,
    }
    assert payload["sources"][0]["name"] == "NVIDIA Blog"
    assert payload["sources"][0]["document_count"] == 1
    assert payload["sources"][0]["last_ingestion_status"] == "completed"
    assert payload["sources"][1]["name"] == "NVIDIA Newsroom"


def test_query_validates_and_shapes_response(monkeypatch, tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    qdrant_path = tmp_path / "qdrant"
    db_path.touch()
    qdrant_path.mkdir()
    monkeypatch.setenv("SIGNALFORGE_DB_PATH", str(db_path))
    monkeypatch.setenv("SIGNALFORGE_QDRANT_PATH", str(qdrant_path))

    def fake_answer_question(question, **kwargs):
        return QueryResponse(
            question=question,
            answer="NVIDIA cites supply-chain risk [1].",
            plan=SearchPlan(
                tickers=["NVDA"],
                sections=["1A"],
                semantic_queries=["NVIDIA risk factors"],
                time_scope="latest",
                intent="summary",
                top_k=5,
            ),
            used_fallback=False,
            planner_error=None,
            warnings=[],
            sources=[
                SourceChunk(
                    label="[1] NVDA 2026 Item 1A chunk 3",
                    score=0.82,
                    chunk_source="sec_filing",
                    ticker="NVDA",
                    company_name="NVIDIA CORP",
                    filing_date="2026-02-25",
                    published_at=None,
                    section_id="1A",
                    section_title="Risk Factors",
                    chunk_index=3,
                    accession_number="0001045810-26-000021",
                    document_id=None,
                    source_id=None,
                    source_name=None,
                    source_type=None,
                    url=None,
                    title=None,
                    text="Supply-chain risk text.",
                )
            ],
        )

    monkeypatch.setattr("signalforge.api.answer_question", fake_answer_question)
    client = TestClient(app)

    invalid_response = client.post("/api/query", json={"question": "  x "})
    response = client.post(
        "/api/query",
        json={
            "question": "  What are NVIDIA's risks? ",
            "include_plan": False,
            "include_source_text": False,
        },
    )

    assert invalid_response.status_code == 422
    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "What are NVIDIA's risks?"
    assert payload["plan"] is None
    assert payload["sources"][0]["text"] is None
    assert payload["sources"][0]["ticker"] == "NVDA"


def test_query_passes_through_warnings_and_fallback_state(monkeypatch, tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    qdrant_path = tmp_path / "qdrant"
    db_path.touch()
    qdrant_path.mkdir()
    monkeypatch.setenv("SIGNALFORGE_DB_PATH", str(db_path))
    monkeypatch.setenv("SIGNALFORGE_QDRANT_PATH", str(qdrant_path))

    def fake_answer_question(question, **kwargs):
        return QueryResponse(
            question=question,
            answer="Retrieved evidence is unavailable.",
            plan=SearchPlan(
                tickers=[],
                sections=[],
                semantic_queries=[question],
                time_scope="latest",
                intent="summary",
                top_k=5,
            ),
            used_fallback=True,
            planner_error="DEEPSEEK_API_KEY is not set; used local rule-based planner",
            warnings=["no retrieved evidence", "llm answer generation unavailable"],
            sources=[],
        )

    monkeypatch.setattr("signalforge.api.answer_question", fake_answer_question)

    response = TestClient(app).post(
        "/api/query",
        json={"question": "Summarize Intel risk factors."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["used_fallback"] is True
    assert payload["planner_error"] == "DEEPSEEK_API_KEY is not set; used local rule-based planner"
    assert payload["warnings"] == ["no retrieved evidence", "llm answer generation unavailable"]
