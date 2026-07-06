from dataclasses import dataclass

from signalforge.storage import load_index_health_counts
from signalforge.vector_store import count_points_by_payload


@dataclass(frozen=True)
class CorpusIndexHealth:
    name: str
    postgres_expected_points: int
    postgres_ready_points: int
    postgres_embedding_records: int
    qdrant_points: int

    @property
    def missing_qdrant_points(self) -> int:
        return max(0, self.postgres_embedding_records - self.qdrant_points)

    @property
    def extra_qdrant_points(self) -> int:
        return max(0, self.qdrant_points - self.postgres_embedding_records)

    @property
    def is_complete_in_postgres(self) -> bool:
        return (
            self.postgres_expected_points == self.postgres_ready_points
            and self.postgres_expected_points == self.postgres_embedding_records
        )

    @property
    def is_consistent_with_qdrant(self) -> bool:
        return self.postgres_embedding_records == self.qdrant_points


@dataclass(frozen=True)
class IndexHealth:
    status: str
    collection: str
    collection_exists: bool
    embedding_model: str
    sec: CorpusIndexHealth
    documents: CorpusIndexHealth

    @property
    def total_postgres_expected_points(self) -> int:
        return self.sec.postgres_expected_points + self.documents.postgres_expected_points

    @property
    def total_qdrant_points(self) -> int:
        return self.sec.qdrant_points + self.documents.qdrant_points


def check_index_health(
    connection,
    qdrant_client,
    *,
    collection: str,
    embedding_model: str,
) -> IndexHealth:
    postgres_counts = load_index_health_counts(
        connection,
        embedding_model=embedding_model,
        vector_collection=collection,
    )
    collection_exists = qdrant_client.collection_exists(collection)
    sec_qdrant_points = count_points_by_payload(
        qdrant_client,
        collection_name=collection,
        payload={
            "chunk_source": "sec_filing",
            "embedding_model": embedding_model,
        },
    )
    document_qdrant_points = count_points_by_payload(
        qdrant_client,
        collection_name=collection,
        payload={
            "chunk_source": "document",
            "embedding_model": embedding_model,
        },
    )

    sec = CorpusIndexHealth(
        name="sec",
        postgres_expected_points=postgres_counts["sec_expected_points"],
        postgres_ready_points=postgres_counts["sec_ready_points"],
        postgres_embedding_records=postgres_counts["sec_embedding_records"],
        qdrant_points=sec_qdrant_points,
    )
    documents = CorpusIndexHealth(
        name="documents",
        postgres_expected_points=postgres_counts["document_expected_points"],
        postgres_ready_points=postgres_counts["document_embedding_records"],
        postgres_embedding_records=postgres_counts["document_embedding_records"],
        qdrant_points=document_qdrant_points,
    )

    return IndexHealth(
        status=_index_status(collection_exists=collection_exists, sec=sec, documents=documents),
        collection=collection,
        collection_exists=collection_exists,
        embedding_model=embedding_model,
        sec=sec,
        documents=documents,
    )


def _index_status(
    *,
    collection_exists: bool,
    sec: CorpusIndexHealth,
    documents: CorpusIndexHealth,
) -> str:
    corpora = [sec, documents]
    expected_total = sum(corpus.postgres_expected_points for corpus in corpora)
    qdrant_total = sum(corpus.qdrant_points for corpus in corpora)

    if expected_total == 0 and qdrant_total == 0:
        return "empty"
    if any(not corpus.is_consistent_with_qdrant for corpus in corpora):
        return "degraded"
    if not collection_exists and expected_total > 0:
        return "building"
    if any(not corpus.is_complete_in_postgres for corpus in corpora):
        return "building"
    return "healthy"
