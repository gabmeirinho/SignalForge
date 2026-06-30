import uuid

from qdrant_client import QdrantClient, models

from signalforge.vector_store import (
    delete_obsolete_points,
    document_payload_from_row,
    fetch_vector_ids_for_accession,
    make_document_vector_id,
    make_vector_id,
)


def test_make_vector_id_is_stable_and_qdrant_compatible():
    first = make_vector_id("0001045810-26-000021", "1A", 4)
    second = make_vector_id("0001045810-26-000021", "1A", 4)
    different = make_vector_id("0001045810-26-000021", "1A", 5)

    assert first == second
    assert first != different
    assert str(uuid.UUID(first)) == first


def test_make_document_vector_id_is_stable_and_qdrant_compatible():
    first = make_document_vector_id(42, 1)
    second = make_document_vector_id(42, 1)
    different = make_document_vector_id(42, 2)

    assert first == second
    assert first != different
    assert str(uuid.UUID(first)) == first


def test_document_payload_from_row_includes_mixed_corpus_metadata():
    row = {
        "document_chunk_id": 7,
        "document_id": 3,
        "source_id": 2,
        "source_name": "NVIDIA Blog",
        "source_type": "news_feed",
        "ownership": "official",
        "trust_level": "high",
        "url": "https://blogs.nvidia.com/blog/example/",
        "title": "AI Infrastructure Update",
        "author": "NVIDIA",
        "published_at": "2026-03-12T10:00:00+00:00",
        "fetched_at": "2026-03-13T10:00:00+00:00",
        "document_type": "blog_post",
        "ticker": "NVDA",
        "company_name": "NVIDIA CORP",
        "chunk_index": 0,
        "text": "AI infrastructure update.",
    }

    payload = document_payload_from_row(row, embedding_model="test-model")

    assert payload == {
        "chunk_source": "document",
        "document_chunk_id": 7,
        "document_id": 3,
        "source_id": 2,
        "source_name": "NVIDIA Blog",
        "source_type": "news_feed",
        "ownership": "official",
        "trust_level": "high",
        "url": "https://blogs.nvidia.com/blog/example/",
        "title": "AI Infrastructure Update",
        "author": "NVIDIA",
        "published_at": "2026-03-12T10:00:00+00:00",
        "fetched_at": "2026-03-13T10:00:00+00:00",
        "document_type": "blog_post",
        "ticker": "NVDA",
        "company_name": "NVIDIA CORP",
        "chunk_index": 0,
        "text": "AI infrastructure update.",
        "embedding_model": "test-model",
    }


def test_delete_obsolete_points_only_removes_stale_ids_for_accession(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    collection_name = "test_chunks"
    accession = "0001045810-26-000021"
    current_id = make_vector_id(accession, "1A", 0)
    stale_id = make_vector_id(accession, "1A", 1)
    unrelated_id = make_vector_id("0000000000-26-000001", "1A", 0)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=current_id,
                vector=[1.0, 0.0],
                payload={"accession_number": accession},
            ),
            models.PointStruct(
                id=stale_id,
                vector=[0.9, 0.1],
                payload={"accession_number": accession},
            ),
            models.PointStruct(
                id=unrelated_id,
                vector=[0.0, 1.0],
                payload={"accession_number": "0000000000-26-000001"},
            ),
        ],
        wait=True,
    )

    try:
        existing = fetch_vector_ids_for_accession(
            client,
            collection_name=collection_name,
            accession_number=accession,
            page_size=1,
        )
        deleted = delete_obsolete_points(
            client,
            collection_name=collection_name,
            accession_number=accession,
            current_vector_ids={current_id},
        )
        remaining = fetch_vector_ids_for_accession(
            client,
            collection_name=collection_name,
            accession_number=accession,
        )
        unrelated = fetch_vector_ids_for_accession(
            client,
            collection_name=collection_name,
            accession_number="0000000000-26-000001",
        )
    finally:
        client.close()

    assert existing == {current_id, stale_id}
    assert deleted == {stale_id}
    assert remaining == {current_id}
    assert unrelated == {unrelated_id}
