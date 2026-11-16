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
