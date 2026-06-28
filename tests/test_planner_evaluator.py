import json
from pathlib import Path

import pytest

from signalforge.cli.evaluate_planner import compare_plan


DATASET_PATH = Path(__file__).parent / "fixtures" / "planner_golden.json"


def test_golden_dataset_has_unique_valid_cases():
    cases = json.loads(DATASET_PATH.read_text())

    assert cases
    assert len({case["id"] for case in cases}) == len(cases)
    expected_context = {
        "available_tickers": [
            "AMD",
            "AMZN",
            "AVGO",
            "GOOGL",
            "META",
            "MSFT",
            "MU",
            "NVDA",
            "QCOM",
        ],
        "available_sections": ["1", "1A", "7", "7A"],
        "filing_years_by_ticker": {
            "AMD": [2026, 2025, 2024, 2023, 2022],
            "AMZN": [2026, 2025, 2024, 2023, 2022],
            "AVGO": [2025, 2024, 2023, 2022, 2021],
            "GOOGL": [2026, 2025, 2024, 2023, 2022],
            "META": [2026, 2025, 2024, 2023, 2022],
            "MSFT": [2025, 2024, 2023, 2022, 2021],
            "MU": [2025, 2024, 2023, 2022, 2021],
            "NVDA": [2026, 2025, 2024, 2023, 2022],
            "QCOM": [2025, 2024, 2023, 2022, 2021],
        },
    }
    for case in cases:
        assert case["question"].strip()
        assert case["context"] == expected_context
        assert set(case["expected"]).issubset(
            {
                "tickers",
                "sections",
                "time_scope",
                "filing_years",
                "intent",
            }
        )
        assert {"tickers", "sections", "time_scope", "intent"}.issubset(
            case["expected"]
        )


def test_compare_plan_ignores_ticker_and_section_order():
    actual = {
        "tickers": ["NVDA", "AMD"],
        "sections": ["7", "1A"],
        "time_scope": "latest",
        "filing_years": [2024, 2022],
        "intent": "comparison",
    }
    expected = {
        "tickers": ["AMD", "NVDA"],
        "sections": ["1A", "7"],
        "time_scope": "latest",
        "filing_years": [2022, 2024],
        "intent": "comparison",
    }

    assert compare_plan(actual, expected) == []


@pytest.mark.parametrize(
    ("field", "actual_value", "expected_value"),
    [
        ("tickers", ["AMD"], ["NVDA"]),
        ("sections", ["1"], ["1A"]),
        ("time_scope", "latest", "all_available"),
        ("filing_years", [2025], [2024]),
        ("intent", "summary", "trend"),
    ],
)
def test_compare_plan_reports_mismatches(field, actual_value, expected_value):
    actual = {field: actual_value}
    expected = {field: expected_value}

    assert compare_plan(actual, expected) == [
        f"{field}: expected {expected_value!r}, got {actual_value!r}"
    ]
