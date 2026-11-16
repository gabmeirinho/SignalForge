"""Build a test-only vector index from a version 2.0 benchmark corpus.

The fixture index holds only the benchmark's synthetic passages, so retrieval can
be scored against the dataset's relevance labels. Each passage is stored with the
same payload shape as a real SEC filing chunk (``chunk_source``, ``ticker``,
``section_id``, ``filing_date``, ``accession_number``, ``chunk_id``) plus its
``label``, which lets the unmodified ``retrieve_chunks`` search it.

Real retrieval narrows filing years through accession numbers looked up in the
database. The fixture index has no database, so each (ticker, filing year) in the
corpus gets a synthetic accession number, and ``FixtureIndex`` resolves a case's
year scope to those numbers instead.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from signalforge.evaluation.dataset import DEFAULT_DATASET, LoadedDataset
from signalforge.sections import SECTION_TITLES
from signalforge.vector_store import (
    SearchResult,
    count_points_by_payload,
    ensure_collection,
    make_vector_id,
    retrieve_chunks,
)


DEFAULT_FIXTURE_DATASET = DEFAULT_DATASET.parent / "benchmark-v2.json"
DEFAULT_FIXTURE_QDRANT_PATH = "data/fixture_qdrant"
FIXTURE_FORM_TYPE = "10-K"


@dataclass(frozen=True)
class FixtureIndex:
    qdrant_path: str
    collection: str
    embedding_model: str
    dataset_version: str
    dataset_hash: str
    passage_count: int
    # ticker -> filing year -> synthetic accession number
    accessions: dict[str, dict[int, str]]
    reused: bool

    def accessions_by_ticker(self, case: dict) -> dict[str, list[str]]:
        """Accession numbers in the case's expected ticker and year scope.

        Mirrors the database lookup in ``evaluate_retrieval``: only filings that
        exist in the index are returned, so an unavailable year yields no accessions.
        """
        expected = case["expected"]
        years_by_ticker = case["context"]["filing_years_by_ticker"]
        result = {}
        for ticker in expected.get("tickers") or []:
            years = scope_years(expected, years_by_ticker.get(ticker, []))
            filings = self.accessions.get(ticker, {})
            result[ticker] = [filings[year] for year in sorted(years, reverse=True) if year in filings]
        return result


def scope_years(expected: dict, available_years: list[int]) -> list[int]:
    """Filing years an expected plan covers; mirrors ``years_for_plan_scope``."""
    time_scope = expected.get("time_scope", "latest")
    if time_scope == "specific_years":
        return list(expected.get("filing_years") or [])
    years = sorted(available_years, reverse=True)
    if time_scope == "latest":
        return years[:1]
    if time_scope == "latest_and_previous":
        return years[:2]
    return years


def fixture_accession_number(ticker: str, filing_year: int) -> str:
    return f"fixture-{ticker}-{filing_year}"


def fixture_collection_name(dataset_hash: str, embedding_model: str) -> str:
    """Name the collection after the dataset content and model, so either change rebuilds it."""
    model_slug = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    return f"eval_{dataset_hash[:12]}_{model_slug}"


def fixture_payloads(dataset: LoadedDataset, *, embedding_model: str) -> list[dict]:
    """Payloads for every corpus passage, shaped like SEC filing chunk payloads."""
    if not dataset.corpus:
        raise ValueError("a fixture index needs a dataset with a corpus (version 2.0 or later)")

    chunk_counters: dict[tuple[str, int, str], int] = defaultdict(int)
    payloads = []
    for passage in dataset.corpus:
        ticker = passage["ticker"]
        year = passage["filing_year"]
        section_id = passage["section_id"]
        key = (ticker, year, section_id)
        chunk_index = chunk_counters[key]
        chunk_counters[key] += 1
        payloads.append(
            {
                "chunk_source": "sec_filing",
                "chunk_id": passage["label"],
                "label": passage["label"],
                "fixture": True,
                "dataset_hash": dataset.content_hash,
                "accession_number": fixture_accession_number(ticker, year),
                "ticker": ticker,
                "company_name": None,
                "form_type": FIXTURE_FORM_TYPE,
                "filing_date": f"{year}-01-01",
                "section_id": section_id,
                "section_title": SECTION_TITLES.get(section_id, f"Item {section_id}"),
                "chunk_index": chunk_index,
                "text": passage["text"],
                "embedding_model": embedding_model,
            }
        )
    return payloads


def build_fixture_index(
    dataset: LoadedDataset,
    *,
    client: QdrantClient,
    qdrant_path: str | Path,
    embedding_model: str,
    rebuild: bool = False,
) -> FixtureIndex:
    """Embed the corpus into its own collection, reusing it when already complete."""
    payloads = fixture_payloads(dataset, embedding_model=embedding_model)
    collection = fixture_collection_name(dataset.content_hash, embedding_model)

    existing = count_points_by_payload(
        client,
        collection_name=collection,
        payload={"dataset_hash": dataset.content_hash},
    )
    reused = existing == len(payloads) and not rebuild
    if not reused:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        ensure_collection(client, collection_name=collection, embedding_model=embedding_model)
        points = [
            models.PointStruct(
                id=make_vector_id(payload["accession_number"], payload["section_id"], payload["chunk_index"]),
                vector=models.Document(text=payload["text"], model=embedding_model),
                payload=payload,
            )
            for payload in payloads
        ]
        result = client.upsert(collection_name=collection, points=points, wait=True)
        if result.status != models.UpdateStatus.COMPLETED:
            raise RuntimeError(f"Qdrant upsert did not complete for {collection}: {result.status}")

    accessions: dict[str, dict[int, str]] = defaultdict(dict)
    for payload in payloads:
        year = int(payload["filing_date"][:4])
        accessions[payload["ticker"]][year] = payload["accession_number"]

    return FixtureIndex(
        qdrant_path=str(qdrant_path),
        collection=collection,
        embedding_model=embedding_model,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        passage_count=len(payloads),
        accessions=dict(accessions),
        reused=reused,
    )


def retrieve_for_case(
    client: QdrantClient,
    fixture: FixtureIndex,
    case: dict,
    *,
    query: str | None = None,
    limit: int = 5,
) -> list[SearchResult]:
    """Run real retrieval against the fixture index with the case's expected plan.

    Cases with no tickers or no in-scope filings return nothing, as the query
    pipeline does when the plan resolves to no ready filings.
    """
    expected = case["expected"]
    accessions_by_ticker = fixture.accessions_by_ticker(case)
    accessions = sorted({a for ticker_accessions in accessions_by_ticker.values() for a in ticker_accessions})
    if not accessions:
        return []
    return retrieve_chunks(
        client,
        query=query or case["question"],
        collection_name=fixture.collection,
        embedding_model=fixture.embedding_model,
        limit=limit,
        tickers=expected.get("tickers") or [],
        section_ids=expected.get("sections") or [],
        accession_numbers=accessions,
        accession_numbers_by_ticker=accessions_by_ticker,
        intent=expected.get("intent", "summary"),
        time_scope=expected.get("time_scope", "latest"),
    )
