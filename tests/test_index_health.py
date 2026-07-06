from types import SimpleNamespace

from signalforge.index_health import check_index_health
from signalforge.sections import TextChunk
from signalforge.storage import (
    CompanyRecord,
    DocumentRecord,
    FilingMetadata,
    GenericDocumentChunk,
    SourceRecord,
    connect_database,
    initialize_database,
    record_chunk_embeddings,
    record_document_chunk_embeddings,
    replace_document_chunks,
    replace_filing_chunks,
    set_embedding_run_status,
    upsert_company,
    upsert_document,
    upsert_filing,
    upsert_source,
)


MODEL = "test-model"
COLLECTION = "test-collection"


def test_index_health_reports_degraded_when_qdrant_is_missing_ready_sec_points(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        filing_id = _insert_ready_filing(connection)

        record_chunk_embeddings(
            connection,
            chunk_vector_ids=[(1, "sec-vector-1"), (2, "sec-vector-2")],
            embedding_model=MODEL,
            vector_collection=COLLECTION,
        )
        set_embedding_run_status(
            connection,
            filing_id=filing_id,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            status="ready",
            expected_point_count=2,
            indexed_point_count=2,
        )

        health = check_index_health(
            connection,
            FakeQdrantClient(sec_points=0, document_points=0),
            collection=COLLECTION,
            embedding_model=MODEL,
        )

    assert health.status == "degraded"
    assert health.collection_exists is True
    assert health.sec.postgres_expected_points == 2
    assert health.sec.postgres_embedding_records == 2
    assert health.sec.qdrant_points == 0
    assert health.sec.missing_qdrant_points == 2


def test_index_health_reports_healthy_when_postgres_and_qdrant_counts_match(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        filing_id = _insert_ready_filing(connection)
        source_id = _insert_approved_source(connection)
        document_id = upsert_document(
            connection,
            DocumentRecord(
                source_id=source_id,
                url="https://example.com/news",
                title="Example News",
                content_hash="b" * 64,
                document_type="article",
            ),
        )
        replace_document_chunks(
            connection,
            document_id,
            [GenericDocumentChunk(chunk_index=0, text="Document evidence.")],
        )

        record_chunk_embeddings(
            connection,
            chunk_vector_ids=[(1, "sec-vector-1"), (2, "sec-vector-2")],
            embedding_model=MODEL,
            vector_collection=COLLECTION,
        )
        record_document_chunk_embeddings(
            connection,
            chunk_vector_ids=[(1, "document-vector-1")],
            embedding_model=MODEL,
            vector_collection=COLLECTION,
        )
        set_embedding_run_status(
            connection,
            filing_id=filing_id,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            status="ready",
            expected_point_count=2,
            indexed_point_count=2,
        )

        health = check_index_health(
            connection,
            FakeQdrantClient(sec_points=2, document_points=1),
            collection=COLLECTION,
            embedding_model=MODEL,
        )

    assert health.status == "healthy"
    assert health.sec.is_complete_in_postgres is True
    assert health.sec.is_consistent_with_qdrant is True
    assert health.documents.is_complete_in_postgres is True
    assert health.documents.is_consistent_with_qdrant is True
    assert health.total_postgres_expected_points == 3
    assert health.total_qdrant_points == 3


class FakeQdrantClient:
    def __init__(
        self,
        *,
        sec_points: int,
        document_points: int,
        collection_exists: bool = True,
    ) -> None:
        self.sec_points = sec_points
        self.document_points = document_points
        self._collection_exists = collection_exists

    def collection_exists(self, collection_name: str) -> bool:
        return self._collection_exists

    def count(self, *, collection_name, count_filter, exact):
        conditions = {
            condition.key: condition.match.value
            for condition in count_filter.must
        }
        if conditions["chunk_source"] == "sec_filing":
            return SimpleNamespace(count=self.sec_points)
        if conditions["chunk_source"] == "document":
            return SimpleNamespace(count=self.document_points)
        return SimpleNamespace(count=0)


def _insert_ready_filing(connection) -> int:
    filing_id = upsert_filing(
        connection,
        FilingMetadata(
            accession_number="0001045810-26-000021",
            ticker="NVDA",
            cik="1045810",
            company_name="NVIDIA CORP",
            form_type="10-K",
            filing_date="2026-02-25",
            period_of_report="2026-01-25",
            raw_path="raw.txt",
            raw_sha256="a" * 64,
            clean_text_path="clean.txt",
        ),
    )
    replace_filing_chunks(
        connection,
        filing_id,
        [
            TextChunk("1", "Business", 0, "Business overview."),
            TextChunk("1", "Business", 1, "More business overview."),
        ],
    )
    return filing_id


def _insert_approved_source(connection) -> int:
    company_id = upsert_company(connection, CompanyRecord(ticker="NVDA", name="NVIDIA CORP"))
    return upsert_source(
        connection,
        SourceRecord(
            company_id=company_id,
            name="NVIDIA Newsroom",
            url="https://example.com/feed",
            source_type="news_feed",
            discovery_status="approved",
        ),
    )
