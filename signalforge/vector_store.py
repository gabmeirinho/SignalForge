import uuid
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable, Iterable

from qdrant_client import QdrantClient, models


DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"
DEFAULT_COLLECTION = "sec_chunks"


@dataclass(frozen=True)
class SearchResult:
    score: float
    payload: dict


def create_qdrant_client(path: str | Path) -> QdrantClient:
    qdrant_path = Path(path)
    qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    embedding_model: str,
) -> None:
    client.set_model(embedding_model)
    vector_size = client.get_embedding_size(embedding_model)

    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        existing_size = collection.config.params.vectors.size
        if existing_size != vector_size:
            raise ValueError(
                f"Collection {collection_name!r} has vector size {existing_size}, "
                f"but model {embedding_model!r} produces {vector_size}"
            )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def index_chunks(
    client: QdrantClient,
    *,
    rows: Iterable,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 16,
    progress_callback: Callable[[int], None] | None = None,
) -> list[tuple[int, str]]:
    rows = list(rows)
    ensure_collection(
        client,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    indexed = []
    rows_by_accession = defaultdict(list)
    for row in rows:
        rows_by_accession[row["accession_number"]].append(row)

    for accession_number, filing_rows in rows_by_accession.items():
        current_vector_ids = set()
        indexed_count = 0

        for start in range(0, len(filing_rows), batch_size):
            batch = filing_rows[start : start + batch_size]
            points = []

            for row in batch:
                vector_id = make_vector_id(
                    accession_number=accession_number,
                    section_id=row["section_id"],
                    chunk_index=row["chunk_index"],
                )
                payload = {
                    "chunk_source": "sec_filing",
                    "chunk_id": row["chunk_id"],
                    "filing_id": row["filing_id"],
                    "accession_number": accession_number,
                    "ticker": row["ticker"],
                    "cik": row["cik"],
                    "company_name": row["company_name"],
                    "form_type": row["form_type"],
                    "filing_date": row["filing_date"],
                    "period_of_report": row["period_of_report"],
                    "section_id": row["section_id"],
                    "section_title": row["section_title"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "embedding_model": embedding_model,
                }
                points.append(
                    models.PointStruct(
                        id=vector_id,
                        vector=models.Document(text=row["text"], model=embedding_model),
                        payload=payload,
                    )
                )
                current_vector_ids.add(vector_id)
                indexed.append((int(row["chunk_id"]), vector_id))

            result = client.upsert(collection_name=collection_name, points=points, wait=True)
            if result.status != models.UpdateStatus.COMPLETED:
                raise RuntimeError(
                    f"Qdrant upsert did not complete for {accession_number}: {result.status}"
                )
            indexed_count += len(points)
            if progress_callback:
                progress_callback(indexed_count)

        delete_obsolete_points(
            client,
            collection_name=collection_name,
            accession_number=accession_number,
            current_vector_ids=current_vector_ids,
        )
        stored_vector_ids = fetch_vector_ids_for_accession(
            client,
            collection_name=collection_name,
            accession_number=accession_number,
        )
        if stored_vector_ids != current_vector_ids:
            raise RuntimeError(
                f"Qdrant verification failed for {accession_number}: "
                f"expected {len(current_vector_ids)} points, found {len(stored_vector_ids)}"
            )

    return indexed


def index_document_chunks(
    client: QdrantClient,
    *,
    rows: Iterable,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 16,
    progress_callback: Callable[[int], None] | None = None,
) -> list[tuple[int, str]]:
    rows = list(rows)
    ensure_collection(
        client,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    indexed = []
    rows_by_document = defaultdict(list)
    for row in rows:
        rows_by_document[row["document_id"]].append(row)

    for document_id, document_rows in rows_by_document.items():
        current_vector_ids = set()
        indexed_count = 0

        for start in range(0, len(document_rows), batch_size):
            batch = document_rows[start : start + batch_size]
            points = []

            for row in batch:
                vector_id = make_document_vector_id(
                    document_id=int(document_id),
                    chunk_index=int(row["chunk_index"]),
                )
                payload = document_payload_from_row(row, embedding_model=embedding_model)
                points.append(
                    models.PointStruct(
                        id=vector_id,
                        vector=models.Document(text=row["text"], model=embedding_model),
                        payload=payload,
                    )
                )
                current_vector_ids.add(vector_id)
                indexed.append((int(row["document_chunk_id"]), vector_id))

            result = client.upsert(collection_name=collection_name, points=points, wait=True)
            if result.status != models.UpdateStatus.COMPLETED:
                raise RuntimeError(
                    f"Qdrant upsert did not complete for document {document_id}: {result.status}"
                )
            indexed_count += len(points)
            if progress_callback:
                progress_callback(indexed_count)

        delete_obsolete_document_points(
            client,
            collection_name=collection_name,
            document_id=int(document_id),
            current_vector_ids=current_vector_ids,
        )
        stored_vector_ids = fetch_vector_ids_for_document(
            client,
            collection_name=collection_name,
            document_id=int(document_id),
        )
        if stored_vector_ids != current_vector_ids:
            raise RuntimeError(
                f"Qdrant verification failed for document {document_id}: "
                f"expected {len(current_vector_ids)} points, found {len(stored_vector_ids)}"
            )

    return indexed


def document_payload_from_row(row, *, embedding_model: str) -> dict:
    return {
        "chunk_source": "document",
        "document_chunk_id": row["document_chunk_id"],
        "document_id": row["document_id"],
        "source_id": row["source_id"],
        "source_name": row["source_name"],
        "source_type": row["source_type"],
        "ownership": row["ownership"],
        "trust_level": row["trust_level"],
        "url": row["url"],
        "title": row["title"],
        "author": row["author"],
        "published_at": row["published_at"],
        "fetched_at": row["fetched_at"],
        "document_type": row["document_type"],
        "ticker": row["ticker"],
        "company_name": row["company_name"],
        "chunk_index": row["chunk_index"],
        "text": row["text"],
        "embedding_model": embedding_model,
    }


def delete_obsolete_points(
    client: QdrantClient,
    *,
    collection_name: str,
    accession_number: str,
    current_vector_ids: set[str],
) -> set[str]:
    existing_vector_ids = fetch_vector_ids_for_accession(
        client,
        collection_name=collection_name,
        accession_number=accession_number,
    )
    obsolete_vector_ids = existing_vector_ids - current_vector_ids

    if obsolete_vector_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=sorted(obsolete_vector_ids)),
            wait=True,
        )

    return obsolete_vector_ids


def delete_obsolete_document_points(
    client: QdrantClient,
    *,
    collection_name: str,
    document_id: int,
    current_vector_ids: set[str],
) -> set[str]:
    existing_vector_ids = fetch_vector_ids_for_document(
        client,
        collection_name=collection_name,
        document_id=document_id,
    )
    obsolete_vector_ids = existing_vector_ids - current_vector_ids

    if obsolete_vector_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=sorted(obsolete_vector_ids)),
            wait=True,
        )

    return obsolete_vector_ids


def fetch_vector_ids_for_accession(
    client: QdrantClient,
    *,
    collection_name: str,
    accession_number: str,
    page_size: int = 256,
) -> set[str]:
    accession_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="accession_number",
                match=models.MatchValue(value=accession_number),
            )
        ]
    )
    vector_ids = set()
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=accession_filter,
            limit=page_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        vector_ids.update(str(record.id) for record in records)

        if next_offset is None:
            break
        offset = next_offset

    return vector_ids


def fetch_vector_ids_for_document(
    client: QdrantClient,
    *,
    collection_name: str,
    document_id: int,
    page_size: int = 256,
) -> set[str]:
    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="chunk_source",
                match=models.MatchValue(value="document"),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
        ]
    )
    vector_ids = set()
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=document_filter,
            limit=page_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        vector_ids.update(str(record.id) for record in records)

        if next_offset is None:
            break
        offset = next_offset

    return vector_ids


def semantic_search(
    client: QdrantClient,
    *,
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 5,
    ticker: str | None = None,
    section_id: str | None = None,
    accession_numbers: list[str] | None = None,
    chunk_source: str | None = None,
) -> list[SearchResult]:
    client.set_model(embedding_model)
    conditions = []

    if chunk_source:
        conditions.append(
            models.FieldCondition(key="chunk_source", match=models.MatchValue(value=chunk_source))
        )
    if ticker:
        conditions.append(
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))
        )
    if section_id:
        conditions.append(
            models.FieldCondition(key="section_id", match=models.MatchValue(value=section_id.upper()))
        )
    if accession_numbers is not None:
        if not accession_numbers:
            return []
        conditions.append(
            models.FieldCondition(
                key="accession_number",
                match=models.MatchAny(any=accession_numbers),
            )
        )

    query_filter = models.Filter(must=conditions) if conditions else None
    response = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=query, model=embedding_model),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    return [
        SearchResult(score=point.score, payload=point.payload or {}) for point in response.points
    ]


def retrieve_chunks(
    client: QdrantClient,
    *,
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 5,
    tickers: list[str] | None = None,
    section_ids: list[str] | None = None,
    accession_numbers: list[str] | None = None,
    accession_numbers_by_ticker: dict[str, list[str]] | None = None,
    intent: str = "summary",
    time_scope: str = "latest",
) -> list[SearchResult]:
    tickers = [ticker.upper() for ticker in tickers or []]
    section_ids = [section_id.upper() for section_id in section_ids or []]

    if intent == "comparison" and len(tickers) > 1:
        return retrieve_comparison_chunks(
            client,
            query=query,
            collection_name=collection_name,
            embedding_model=embedding_model,
            limit=limit,
            tickers=tickers,
            section_ids=section_ids,
            accession_numbers=accession_numbers,
            accession_numbers_by_ticker=accession_numbers_by_ticker,
        )

    if (
        accession_numbers
        and len(accession_numbers) > 1
        and (intent == "trend" or time_scope in {"all_available", "latest_and_previous"})
    ):
        return retrieve_period_chunks(
            client,
            query=query,
            collection_name=collection_name,
            embedding_model=embedding_model,
            limit=limit,
            tickers=tickers,
            section_ids=section_ids,
            accession_numbers=accession_numbers,
        )

    results = []
    for ticker in tickers or [None]:
        for section_id in section_ids or [None]:
            results.extend(
                semantic_search(
                    client,
                    query=query,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    limit=limit,
                    ticker=ticker,
                    section_id=section_id,
                    accession_numbers=accession_numbers,
                )
            )
        results.extend(
            semantic_search(
                client,
                query=query,
                collection_name=collection_name,
                embedding_model=embedding_model,
                limit=limit,
                ticker=ticker,
                chunk_source="document",
            )
        )

    return _dedupe_results(results)[:limit]


def retrieve_period_chunks(
    client: QdrantClient,
    *,
    query: str,
    collection_name: str,
    embedding_model: str,
    limit: int,
    tickers: list[str],
    section_ids: list[str],
    accession_numbers: list[str],
) -> list[SearchResult]:
    per_accession_limit = max(1, ceil(limit / len(accession_numbers)))
    results = []

    for accession_number in accession_numbers:
        accession_results = []
        for ticker in tickers or [None]:
            for section_id in section_ids or [None]:
                accession_results.extend(
                    semantic_search(
                        client,
                        query=query,
                        collection_name=collection_name,
                        embedding_model=embedding_model,
                        limit=per_accession_limit,
                        ticker=ticker,
                        section_id=section_id,
                        accession_numbers=[accession_number],
                    )
                )

        results.extend(_dedupe_results(accession_results)[:per_accession_limit])

    for ticker in tickers or [None]:
        results.extend(
            semantic_search(
                client,
                query=query,
                collection_name=collection_name,
                embedding_model=embedding_model,
                limit=limit,
                ticker=ticker,
                chunk_source="document",
            )
        )

    return _dedupe_results(results)


def retrieve_comparison_chunks(
    client: QdrantClient,
    *,
    query: str,
    collection_name: str,
    embedding_model: str,
    limit: int,
    tickers: list[str],
    section_ids: list[str],
    accession_numbers: list[str] | None,
    accession_numbers_by_ticker: dict[str, list[str]] | None,
) -> list[SearchResult]:
    per_ticker_limit = max(1, ceil(limit / len(tickers)))
    results = []

    for ticker in tickers:
        ticker_results = []
        ticker_accessions = (
            accession_numbers_by_ticker.get(ticker)
            if accession_numbers_by_ticker is not None
            else accession_numbers
        )

        for section_id in section_ids or [None]:
            ticker_results.extend(
                semantic_search(
                    client,
                    query=query,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    limit=per_ticker_limit,
                    ticker=ticker,
                    section_id=section_id,
                    accession_numbers=ticker_accessions,
                )
            )
        ticker_results.extend(
            semantic_search(
                client,
                query=query,
                collection_name=collection_name,
                embedding_model=embedding_model,
                limit=per_ticker_limit,
                ticker=ticker,
                chunk_source="document",
            )
        )

        results.extend(_dedupe_results(ticker_results)[:per_ticker_limit])

    return _dedupe_results(results)


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped = []
    seen = set()
    for result in sorted(results, key=lambda result: result.score, reverse=True):
        payload = result.payload
        key = payload.get("document_chunk_id") or payload.get("chunk_id") or (
            payload.get("chunk_source"),
            payload.get("accession_number"),
            payload.get("document_id"),
            payload.get("section_id"),
            payload.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def make_vector_id(accession_number: str, section_id: str, chunk_index: int) -> str:
    key = f"{accession_number}:{section_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def make_document_vector_id(document_id: int, chunk_index: int) -> str:
    key = f"document:{document_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
