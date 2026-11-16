import json

import pytest

from signalforge.evaluation.dataset import DEFAULT_DATASET, EvaluationDatasetV2, load_dataset
from signalforge.evaluation.fixture_index import (
    DEFAULT_FIXTURE_DATASET,
    FixtureIndex,
    build_fixture_index,
    fixture_collection_name,
    fixture_payloads,
    retrieve_for_case,
)
from signalforge.vector_store import DEFAULT_EMBEDDING_MODEL, create_qdrant_client


def _case(dataset, case_id: str) -> dict:
    return next(case for case in dataset.cases if case["id"] == case_id)


def test_payloads_match_sec_chunk_shape_and_are_unique():
    dataset = load_dataset(DEFAULT_FIXTURE_DATASET)
    payloads = fixture_payloads(dataset, embedding_model="test-model")

    assert len(payloads) == len(dataset.corpus)
    assert len({p["chunk_id"] for p in payloads}) == len(payloads)
    positions = {(p["accession_number"], p["section_id"], p["chunk_index"]) for p in payloads}
    assert len(positions) == len(payloads)
    first = payloads[0]
    assert first["chunk_source"] == "sec_filing"
    assert first["label"] == dataset.corpus[0]["label"]
    assert first["accession_number"] == f"fixture-{first['ticker']}-{dataset.corpus[0]['filing_year']}"
    assert first["filing_date"].startswith(str(dataset.corpus[0]["filing_year"]))
    assert first["dataset_hash"] == dataset.content_hash


def test_payloads_require_a_corpus():
    with pytest.raises(ValueError, match="corpus"):
        fixture_payloads(load_dataset(DEFAULT_DATASET), embedding_model="test-model")


def test_collection_name_changes_with_dataset_and_model():
    name = fixture_collection_name("a" * 64, "jinaai/jina-embeddings-v2-small-en")

    assert name == "eval_aaaaaaaaaaaa_jinaai_jina_embeddings_v2_small_en"
    assert fixture_collection_name("b" * 64, "jinaai/jina-embeddings-v2-small-en") != name
    assert fixture_collection_name("a" * 64, "other/model") != name


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("latest_risk_factors", {"NVDA": ["fixture-NVDA-2026"]}),
        ("previous_period", {"NVDA": ["fixture-NVDA-2026", "fixture-NVDA-2025"]}),
        (
            "historical_range_trend",
            {"AMD": ["fixture-AMD-2024", "fixture-AMD-2023", "fixture-AMD-2022"]},
        ),
        ("unavailable_year", {"MSFT": []}),
        ("unavailable_company", {}),
    ],
)
def test_accessions_follow_the_expected_year_scope(case_id, expected):
    dataset = load_dataset(DEFAULT_FIXTURE_DATASET)
    accessions = {}
    for passage in dataset.corpus:
        accessions.setdefault(passage["ticker"], {})[passage["filing_year"]] = (
            f"fixture-{passage['ticker']}-{passage['filing_year']}"
        )
    fixture = FixtureIndex(
        qdrant_path="unused",
        collection="unused",
        embedding_model="unused",
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        passage_count=len(dataset.corpus),
        accessions=accessions,
        reused=False,
    )

    assert fixture.accessions_by_ticker(_case(dataset, case_id)) == expected


@pytest.fixture(scope="module")
def built_fixture(tmp_path_factory):
    """Embed the real v2 corpus once; skip if the embedding model cannot be loaded."""
    dataset = load_dataset(DEFAULT_FIXTURE_DATASET)
    qdrant_path = tmp_path_factory.mktemp("fixture_qdrant") / "qdrant"
    client = create_qdrant_client(qdrant_path)
    try:
        fixture = build_fixture_index(
            dataset, client=client, qdrant_path=qdrant_path, embedding_model=DEFAULT_EMBEDDING_MODEL
        )
    except Exception as error:  # model download or load failure, e.g. offline
        client.close()
        pytest.skip(f"embedding model unavailable: {error}")
    yield dataset, client, fixture
    client.close()


def test_build_indexes_every_passage_and_reuses_a_complete_collection(built_fixture):
    dataset, client, fixture = built_fixture

    assert not fixture.reused
    assert client.count(fixture.collection, exact=True).count == len(dataset.corpus)
    again = build_fixture_index(
        dataset,
        client=client,
        qdrant_path=fixture.qdrant_path,
        embedding_model=fixture.embedding_model,
    )
    assert again.reused
    assert again.collection == fixture.collection


def test_retrieval_stays_inside_each_case_scope(built_fixture):
    dataset, client, fixture = built_fixture
    validated = EvaluationDatasetV2.model_validate(json.loads(DEFAULT_FIXTURE_DATASET.read_text()))
    corpus = {passage.label: passage for passage in validated.corpus}

    for case_model, case in zip(validated.cases, dataset.cases, strict=True):
        results = retrieve_for_case(client, fixture, case, limit=5)
        labels = [result.payload["label"] for result in results]

        if not case_model.expected_plan.tickers or not any(
            case_model.in_scope(passage) for passage in corpus.values()
        ):
            assert labels == [], case["id"]
            continue
        assert labels, case["id"]
        assert len(labels) == len(set(labels)), case["id"]
        assert all(case_model.in_scope(corpus[label]) for label in labels), case["id"]
