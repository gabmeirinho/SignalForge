import pytest

from signalforge.rag_service import answer_question, source_chunks_from_results
from signalforge.config import RuntimeConfig
from signalforge.storage import connect_database, initialize_database
from signalforge.telemetry import QueryTelemetry
from signalforge.vector_store import SearchResult


def test_answer_question_reports_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="database not found"):
        answer_question(
            "What are NVIDIA's latest risks?",
            db_path=str(tmp_path / "missing.sqlite3"),
            qdrant_path=str(tmp_path / "qdrant"),
        )


def test_answer_question_records_all_stages_without_model_usage(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    db_path = tmp_path / "signalforge.sqlite3"
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    with connect_database(db_path) as connection:
        initialize_database(connection)

    response = answer_question(
        "What are NVIDIA's risks?",
        db_path=str(db_path),
        qdrant_path=str(qdrant_path),
    )
    metrics = response.telemetry.storage_values()
    assert response.telemetry.status == "completed"
    assert all(metrics[key] is not None for key in (
        "planning_ms", "retrieval_ms", "answer_ms", "total_ms"
    ))
    assert metrics["token_usage_status"] == "unavailable"
    assert metrics["fallback_count"] == 2
    assert metrics["planner_prompt_version"] == "planner-v1.3.0"
    assert metrics["answer_prompt_version"] == "answer-v1.0.0"


def test_answer_question_preserves_partial_timings_on_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    db_path = tmp_path / "signalforge.sqlite3"
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    with connect_database(db_path) as connection:
        initialize_database(connection)

    def fail_retrieval(*args, **kwargs):
        raise RuntimeError("source body must not reach telemetry")

    monkeypatch.setattr("signalforge.rag_service.select_ready_accessions_by_ticker", fail_retrieval)
    metrics = QueryTelemetry(config=RuntimeConfig())
    with pytest.raises(RuntimeError, match="source body"):
        answer_question(
            "What are NVIDIA's risks?",
            db_path=str(db_path),
            qdrant_path=str(qdrant_path),
            telemetry=metrics,
        )

    values = metrics.storage_values()
    assert values["planning_ms"] is not None
    assert values["retrieval_ms"] is not None
    assert values["answer_ms"] is None
    assert values["total_ms"] is not None
    assert values["error_category"] == "RuntimeError"
    assert "source body" not in str(values)


def test_source_chunks_from_results_normalizes_payload_metadata():
    sources = source_chunks_from_results(
        [
            SearchResult(
                score=0.82,
                payload={
                    "ticker": "NVDA",
                    "company_name": "NVIDIA CORP",
                    "filing_date": "2026-02-25",
                    "section_id": "1A",
                    "section_title": "Risk Factors",
                    "chunk_index": 3,
                    "accession_number": "0001045810-26-000021",
                    "text": "Supply-chain risk text.",
                },
            )
        ]
    )

    assert sources[0].label == "[1] NVDA 2026 Item 1A chunk 3"
    assert sources[0].score == 0.82
    assert sources[0].company_name == "NVIDIA CORP"
    assert sources[0].text == "Supply-chain risk text."
