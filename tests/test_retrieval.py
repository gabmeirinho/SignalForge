from types import SimpleNamespace

from sections import TextChunk
from storage import (
    FilingMetadata,
    connect_database,
    get_ready_accession_numbers,
    initialize_database,
    replace_filing_chunks,
    set_embedding_run_status,
    upsert_filing,
)
from vector_store import retrieve_chunks, semantic_search


MODEL = "test-model"
COLLECTION = "test_chunks"


def test_ready_accessions_can_be_filtered_to_expected_filing_years(tmp_path):
    with connect_database(tmp_path / "test.sqlite3") as connection:
        initialize_database(connection)
        _insert_ready_filing(
            connection,
            accession_number="0001045810-26-000021",
            ticker="NVDA",
            filing_date="2026-02-25",
        )
        _insert_ready_filing(
            connection,
            accession_number="0001045810-25-000023",
            ticker="NVDA",
            filing_date="2025-02-26",
        )
        _insert_ready_filing(
            connection,
            accession_number="0000002488-26-000012",
            ticker="AMD",
            filing_date="2026-02-04",
        )

        latest_nvda = get_ready_accession_numbers(
            connection,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            ticker="nvda",
            filing_years=[2026],
        )
        historical_nvda = get_ready_accession_numbers(
            connection,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            ticker="NVDA",
            filing_years=[2025],
        )
        unavailable_year = get_ready_accession_numbers(
            connection,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            ticker="NVDA",
            filing_years=[2024],
        )

    assert latest_nvda == ["0001045810-26-000021"]
    assert historical_nvda == ["0001045810-25-000023"]
    assert unavailable_year == []


def test_semantic_search_returns_chunks_matching_expected_scope_metadata():
    client = FakeVectorClient(
        [
            {
                "accession_number": "0001045810-26-000021",
                "ticker": "NVDA",
                "filing_date": "2026-02-25",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "2026 supplier concentration risk.",
            },
            {
                "accession_number": "0001045810-25-000023",
                "ticker": "NVDA",
                "filing_date": "2025-02-26",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "2025 supplier concentration risk.",
            },
            {
                "accession_number": "0001045810-25-000023",
                "ticker": "NVDA",
                "filing_date": "2025-02-26",
                "section_id": "7",
                "chunk_index": 0,
                "text": "2025 management discussion.",
            },
            {
                "accession_number": "0000002488-25-000009",
                "ticker": "AMD",
                "filing_date": "2025-02-05",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "2025 AMD risk factors.",
            },
        ]
    )

    results = semantic_search(
        client,
        query="supplier concentration risks",
        collection_name=COLLECTION,
        embedding_model=MODEL,
        limit=5,
        ticker="nvda",
        section_id="1A",
        accession_numbers=["0001045810-25-000023"],
    )

    assert client.embedding_model == MODEL
    assert client.last_query.collection_name == COLLECTION
    assert len(results) == 1
    assert [result.payload["filing_date"][:4] for result in results] == ["2025"]
    assert {result.payload["ticker"] for result in results} == {"NVDA"}
    assert {result.payload["section_id"] for result in results} == {"1A"}
    assert {result.payload["accession_number"] for result in results} == {
        "0001045810-25-000023"
    }


def test_semantic_search_skips_query_when_expected_accession_scope_is_empty():
    client = FakeVectorClient([])

    results = semantic_search(
        client,
        query="missing year",
        collection_name=COLLECTION,
        embedding_model=MODEL,
        accession_numbers=[],
    )

    assert results == []
    assert client.query_count == 0


def test_comparison_retrieval_balances_results_by_ticker():
    client = FakeVectorClient(
        [
            {
                "accession_number": "0001045810-26-000021",
                "ticker": "NVDA",
                "filing_date": "2026-02-25",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "NVDA risk 0.",
                "score": 0.99,
            },
            {
                "accession_number": "0001045810-26-000021",
                "ticker": "NVDA",
                "filing_date": "2026-02-25",
                "section_id": "1A",
                "chunk_index": 1,
                "text": "NVDA risk 1.",
                "score": 0.98,
            },
            {
                "accession_number": "0001045810-26-000021",
                "ticker": "NVDA",
                "filing_date": "2026-02-25",
                "section_id": "1A",
                "chunk_index": 2,
                "text": "NVDA risk 2.",
                "score": 0.97,
            },
            {
                "accession_number": "0000002488-26-000018",
                "ticker": "AMD",
                "filing_date": "2026-02-04",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "AMD risk 0.",
                "score": 0.70,
            },
            {
                "accession_number": "0000002488-26-000018",
                "ticker": "AMD",
                "filing_date": "2026-02-04",
                "section_id": "1A",
                "chunk_index": 1,
                "text": "AMD risk 1.",
                "score": 0.69,
            },
            {
                "accession_number": "0000002488-26-000018",
                "ticker": "AMD",
                "filing_date": "2026-02-04",
                "section_id": "1A",
                "chunk_index": 2,
                "text": "AMD risk 2.",
                "score": 0.68,
            },
        ]
    )

    results = retrieve_chunks(
        client,
        query="Compare NVIDIA and AMD's latest risk factors.",
        collection_name=COLLECTION,
        embedding_model=MODEL,
        limit=5,
        tickers=["NVDA", "AMD"],
        section_ids=["1A"],
        accession_numbers_by_ticker={
            "NVDA": ["0001045810-26-000021"],
            "AMD": ["0000002488-26-000018"],
        },
        intent="comparison",
    )

    tickers = [result.payload["ticker"] for result in results]
    assert set(tickers) == {"NVDA", "AMD"}
    assert tickers.count("NVDA") == 3
    assert tickers.count("AMD") == 3
    assert client.query_count == 2


def test_trend_retrieval_balances_results_by_accession_year():
    client = FakeVectorClient(
        [
            {
                "accession_number": "qcom-2021",
                "ticker": "QCOM",
                "filing_date": "2021-11-03",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "QCOM 2021 risks.",
                "score": 0.70,
            },
            {
                "accession_number": "qcom-2022",
                "ticker": "QCOM",
                "filing_date": "2022-11-02",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "QCOM 2022 risks.",
                "score": 0.99,
            },
            {
                "accession_number": "qcom-2023",
                "ticker": "QCOM",
                "filing_date": "2023-11-01",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "QCOM 2023 risks.",
                "score": 0.65,
            },
            {
                "accession_number": "qcom-2024",
                "ticker": "QCOM",
                "filing_date": "2024-11-06",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "QCOM 2024 risks.",
                "score": 0.80,
            },
            {
                "accession_number": "qcom-2025",
                "ticker": "QCOM",
                "filing_date": "2025-11-05",
                "section_id": "1A",
                "chunk_index": 0,
                "text": "QCOM 2025 risks.",
                "score": 0.79,
            },
        ]
    )

    results = retrieve_chunks(
        client,
        query="How have Qualcomm's risk factors changed over time?",
        collection_name=COLLECTION,
        embedding_model=MODEL,
        limit=5,
        tickers=["QCOM"],
        section_ids=["1A"],
        accession_numbers=["qcom-2021", "qcom-2022", "qcom-2023", "qcom-2024", "qcom-2025"],
        intent="trend",
        time_scope="all_available",
    )

    years = {result.payload["filing_date"][:4] for result in results}
    assert years == {"2021", "2022", "2023", "2024", "2025"}
    assert client.query_count == 5


def _insert_ready_filing(connection, *, accession_number: str, ticker: str, filing_date: str) -> None:
    filing_id = upsert_filing(
        connection,
        FilingMetadata(
            accession_number=accession_number,
            ticker=ticker,
            cik=None,
            company_name=f"{ticker} Corp",
            form_type="10-K",
            filing_date=filing_date,
            period_of_report=filing_date,
            raw_path="raw.txt",
            raw_sha256="a" * 64,
            clean_text_path="clean.txt",
        ),
    )
    replace_filing_chunks(
        connection,
        filing_id,
        [
            TextChunk(
                section_id="1A",
                section_title="Risk Factors",
                chunk_index=0,
                text=f"{ticker} risks for {filing_date[:4]}.",
            )
        ],
    )
    set_embedding_run_status(
        connection,
        filing_id=filing_id,
        embedding_model=MODEL,
        vector_collection=COLLECTION,
        status="ready",
        expected_point_count=1,
        indexed_point_count=1,
    )


class FakeVectorClient:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.embedding_model = None
        self.last_query = None
        self.query_count = 0

    def set_model(self, embedding_model: str) -> None:
        self.embedding_model = embedding_model

    def query_points(self, **kwargs):
        self.query_count += 1
        self.last_query = SimpleNamespace(**kwargs)
        filtered_payloads = [
            payload
            for payload in self.payloads
            if _payload_matches_filter(payload, kwargs["query_filter"])
        ][: kwargs["limit"]]
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=payload.get("score", 1.0 - (index * 0.01)),
                    payload=payload,
                )
                for index, payload in enumerate(filtered_payloads)
            ]
        )


def _payload_matches_filter(payload: dict, query_filter) -> bool:
    if query_filter is None:
        return True

    for condition in query_filter.must:
        match_value = getattr(condition.match, "value", None)
        match_any = getattr(condition.match, "any", None)
        payload_value = payload.get(condition.key)

        if match_value is not None and payload_value != match_value:
            return False
        if match_any is not None and payload_value not in match_any:
            return False

    return True
