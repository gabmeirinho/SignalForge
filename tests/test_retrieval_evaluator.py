from types import SimpleNamespace

import evaluate_retrieval


def test_unavailable_specific_year_case_passes_without_retrieval(monkeypatch, capsys):
    case = {
        "id": "unavailable_year",
        "question": "Show me Microsoft's risk factors from 2026.",
        "context": {
            "filing_years_by_ticker": {
                "MSFT": [2025, 2024, 2023, 2022, 2021],
            }
        },
        "expected": {
            "tickers": ["MSFT"],
            "sections": ["1A"],
            "time_scope": "specific_years",
            "filing_years": [],
            "intent": "summary",
        },
    }

    def no_ready_accessions(*args, **kwargs):
        return []

    monkeypatch.setattr(evaluate_retrieval, "get_ready_accession_numbers", no_ready_accessions)

    failures, result_count = evaluate_retrieval.evaluate_case(
        case,
        connection=SimpleNamespace(),
        client=SimpleNamespace(),
        model="test-model",
        collection="test_chunks",
        limit=5,
        preview_chars=100,
        show_full_text=False,
    )

    assert failures == []
    assert result_count == 0
    assert "No retrieval expected" in capsys.readouterr().out
