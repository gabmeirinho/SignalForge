import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
) -> list[tuple[int, str]]:
    rows = list(rows)
    ensure_collection(
        client,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    indexed = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        points = []

        for row in batch:
            vector_id = make_vector_id(
                accession_number=row["accession_number"],
                section_id=row["section_id"],
                chunk_index=row["chunk_index"],
            )
            payload = {
                "chunk_id": row["chunk_id"],
                "filing_id": row["filing_id"],
                "accession_number": row["accession_number"],
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
            indexed.append((int(row["chunk_id"]), vector_id))

        client.upsert(collection_name=collection_name, points=points, wait=True)

    return indexed


def semantic_search(
    client: QdrantClient,
    *,
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 5,
    ticker: str | None = None,
    section_id: str | None = None,
) -> list[SearchResult]:
    client.set_model(embedding_model)
    conditions = []

    if ticker:
        conditions.append(
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))
        )
    if section_id:
        conditions.append(
            models.FieldCondition(key="section_id", match=models.MatchValue(value=section_id.upper()))
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


def make_vector_id(accession_number: str, section_id: str, chunk_index: int) -> str:
    key = f"{accession_number}:{section_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
