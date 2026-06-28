import pytest

from rag_service import answer_question, source_chunks_from_results
from vector_store import SearchResult


def test_answer_question_reports_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="database not found"):
        answer_question(
            "What are NVIDIA's latest risks?",
            db_path=str(tmp_path / "missing.sqlite3"),
            qdrant_path=str(tmp_path / "qdrant"),
        )


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
