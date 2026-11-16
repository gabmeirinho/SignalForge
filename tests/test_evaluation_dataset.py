import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from signalforge.evaluation.dataset import DEFAULT_DATASET, load_dataset


LEGACY_DATASET = Path(__file__).parent / "fixtures/planner_golden.json"


def test_seeded_dataset_has_complete_contract_and_all_query_classes():
    dataset = load_dataset()

    assert dataset.version == "1.0"
    assert 20 <= len(dataset.cases) <= 30
    assert {case["query_class"] for case in dataset.cases} == {
        "lookup", "synthesis", "comparison", "trend", "missing_evidence", "conflicting_source"
    }
    labels = [item["label"] for case in dataset.cases for item in case["evidence"]]
    assert len(labels) == len(set(labels))
    assert all(label.startswith("fixture:") for label in labels)
    assert all(case["answer_facts"] and case["rubric_tags"] for case in dataset.cases)
    assert all(case["expected"]["time_scope"] for case in dataset.cases)


def test_hash_is_stable_across_json_formatting(tmp_path):
    raw = json.loads(DEFAULT_DATASET.read_text())
    reformatted = tmp_path / "formatted.json"
    reformatted.write_text(json.dumps(raw, separators=(",", ":")))

    assert load_dataset(reformatted).content_hash == load_dataset().content_hash
    assert len(load_dataset().content_hash) == hashlib.sha256().digest_size * 2


@pytest.mark.parametrize("change", ["duplicate_case", "duplicate_label", "blank_fact", "unknown_class", "missing_evidence"])
def test_invalid_v1_dataset_fails_validation(tmp_path, change):
    raw = json.loads(DEFAULT_DATASET.read_text())
    raw = copy.deepcopy(raw)
    if change == "duplicate_case":
        raw["cases"].append(copy.deepcopy(raw["cases"][0]))
    elif change == "duplicate_label":
        raw["cases"][1]["evidence"][0]["label"] = raw["cases"][0]["evidence"][0]["label"]
    elif change == "blank_fact":
        raw["cases"][0]["answer_facts"] = [" "]
    elif change == "unknown_class":
        raw["cases"][0]["query_class"] = "unsupported"
    else:
        raw["cases"][0]["evidence"] = []
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValidationError):
        load_dataset(path)


def test_legacy_planner_cases_remain_runnable_through_loader():
    legacy = json.loads(LEGACY_DATASET.read_text())
    dataset = load_dataset(LEGACY_DATASET)

    assert dataset.version == "planner-legacy-v0"
    assert [(case["id"], case["expected"]) for case in dataset.cases] == [
        (case["id"], case["expected"]) for case in legacy
    ]


def test_legacy_duplicate_ids_fail(tmp_path):
    legacy = json.loads(LEGACY_DATASET.read_text())
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([legacy[0], legacy[0]]))

    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


def test_malformed_legacy_case_fails_validation(tmp_path):
    legacy = json.loads(LEGACY_DATASET.read_text())
    legacy[0]["expected"]["time_scope"] = "sometime"
    path = tmp_path / "invalid_legacy.json"
    path.write_text(json.dumps(legacy))

    with pytest.raises(ValidationError):
        load_dataset(path)


V2_DATASET = DEFAULT_DATASET.parent / "benchmark-v2.json"


def _v2_case(raw: dict, case_id: str) -> dict:
    return next(case for case in raw["cases"] if case["id"] == case_id)


def test_v2_dataset_loads_with_corpus_and_keeps_v1_plans():
    v1 = load_dataset()
    v2 = load_dataset(V2_DATASET)

    assert v2.version == "2.0"
    assert len(v2.corpus) >= 100
    assert all(passage["label"].startswith("fx:") for passage in v2.corpus)
    assert [case["id"] for case in v2.cases] == [case["id"] for case in v1.cases]
    assert [case["expected"] for case in v2.cases] == [case["expected"] for case in v1.cases]
    assert {case["query_class"] for case in v2.cases} == {case["query_class"] for case in v1.cases}


def test_v2_missing_evidence_cases_have_no_relevant_passages():
    v2 = load_dataset(V2_DATASET)

    for case in v2.cases:
        if case["query_class"] == "missing_evidence":
            assert case["relevant"] == []
        else:
            assert any(item["grade"] == 2 for item in case["relevant"])


def test_v2_hash_is_stable_across_json_formatting(tmp_path):
    reformatted = tmp_path / "formatted.json"
    reformatted.write_text(json.dumps(json.loads(V2_DATASET.read_text()), separators=(",", ":")))

    assert load_dataset(reformatted).content_hash == load_dataset(V2_DATASET).content_hash


def _mutate_v2(raw: dict, change: str) -> None:
    if change == "unknown_label":
        _v2_case(raw, "latest_risk_factors")["relevant"].append({"label": "fx:missing", "grade": 1})
    elif change == "out_of_scope_label":
        _v2_case(raw, "latest_risk_factors")["relevant"].append(
            {"label": "fx:nvda-2025-1a-supply-concentration", "grade": 1}
        )
    elif change == "missing_comparison_ticker":
        case = _v2_case(raw, "company_comparison")
        case["relevant"] = [item for item in case["relevant"] if not item["label"].startswith("fx:amd-")]
    elif change == "missing_trend_year":
        case = _v2_case(raw, "historical_range_trend")
        case["relevant"] = [item for item in case["relevant"] if item["label"] != "fx:amd-2023-7a-currency"]
    elif change == "too_few_distractors":
        raw["corpus"] = [p for p in raw["corpus"] if p["label"] != "fx:nvda-2026-1a-intro"]
    elif change == "duplicate_corpus_label":
        raw["corpus"].append(copy.deepcopy(raw["corpus"][0]))
    elif change == "missing_evidence_with_relevant":
        _v2_case(raw, "unavailable_year")["relevant"] = [
            {"label": "fx:msft-2025-7a-foreign-exchange", "grade": 1}
        ]
    elif change == "no_required_passage":
        for item in _v2_case(raw, "business_overview")["relevant"]:
            item["grade"] = 1
    elif change == "unavailable_filing":
        raw["corpus"].append(
            {"label": "fx:nvda-2019-1a-old", "ticker": "NVDA", "section_id": "1A",
             "filing_year": 2019, "text": "An old filing."}
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unknown_label", "unknown corpus label"),
        ("out_of_scope_label", "outside the expected plan's retrieval scope"),
        ("missing_comparison_ticker", "no required passage for ticker AMD"),
        ("missing_trend_year", "required passages cover years"),
        ("too_few_distractors", "in-scope distractors"),
        ("duplicate_corpus_label", "duplicate corpus label"),
        ("missing_evidence_with_relevant", "must not list relevant passages"),
        ("no_required_passage", "required \\(grade 2\\)"),
        ("unavailable_filing", "not an available filing"),
    ],
)
def test_invalid_v2_dataset_fails_validation(tmp_path, change, message):
    raw = json.loads(V2_DATASET.read_text())
    _mutate_v2(raw, change)
    path = tmp_path / "invalid_v2.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValidationError, match=message):
        load_dataset(path)
