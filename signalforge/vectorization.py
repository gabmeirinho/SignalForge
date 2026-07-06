from collections import defaultdict
from dataclasses import dataclass

from signalforge.storage import (
    load_chunks_for_vector_index,
    load_document_chunks_for_vector_index,
    record_chunk_embeddings,
    record_document_chunk_embeddings,
    set_embedding_run_status,
    update_embedding_run_progress,
)
from signalforge.vector_store import (
    create_qdrant_client,
    index_chunks,
    index_document_chunks,
)


@dataclass(frozen=True)
class VectorizationResult:
    indexed_count: int
    sec_chunk_count: int
    document_chunk_count: int


def vectorize_pending_chunks(
    connection,
    *,
    qdrant_target: str,
    collection: str,
    embedding_model: str,
    batch_size: int = 16,
) -> VectorizationResult:
    rows = load_chunks_for_vector_index(
        connection,
        embedding_model=embedding_model,
        vector_collection=collection,
    )
    document_rows = load_document_chunks_for_vector_index(
        connection,
        embedding_model=embedding_model,
        vector_collection=collection,
    )
    if not rows and not document_rows:
        return VectorizationResult(
            indexed_count=0,
            sec_chunk_count=0,
            document_chunk_count=0,
        )

    rows_by_filing = defaultdict(list)
    for row in rows:
        rows_by_filing[int(row["filing_id"])].append(row)

    client = create_qdrant_client(qdrant_target)
    total_indexed = 0
    try:
        for filing_id, filing_rows in rows_by_filing.items():
            total_indexed += _vectorize_filing_rows(
                connection,
                client=client,
                filing_id=filing_id,
                rows=filing_rows,
                collection=collection,
                embedding_model=embedding_model,
                batch_size=batch_size,
            )

        if document_rows:
            indexed = index_document_chunks(
                client,
                rows=document_rows,
                collection_name=collection,
                embedding_model=embedding_model,
                batch_size=batch_size,
            )
            record_document_chunk_embeddings(
                connection,
                chunk_vector_ids=indexed,
                embedding_model=embedding_model,
                vector_collection=collection,
            )
            total_indexed += len(indexed)
    finally:
        client.close()

    return VectorizationResult(
        indexed_count=total_indexed,
        sec_chunk_count=len(rows),
        document_chunk_count=len(document_rows),
    )


def _vectorize_filing_rows(
    connection,
    *,
    client,
    filing_id: int,
    rows: list,
    collection: str,
    embedding_model: str,
    batch_size: int,
) -> int:
    expected_count = len(rows)
    indexed_count = 0
    set_embedding_run_status(
        connection,
        filing_id=filing_id,
        embedding_model=embedding_model,
        vector_collection=collection,
        status="indexing",
        expected_point_count=expected_count,
    )

    def update_progress(count: int) -> None:
        nonlocal indexed_count
        indexed_count = count
        update_embedding_run_progress(
            connection,
            filing_id=filing_id,
            embedding_model=embedding_model,
            vector_collection=collection,
            indexed_point_count=count,
        )

    try:
        indexed = index_chunks(
            client,
            rows=rows,
            collection_name=collection,
            embedding_model=embedding_model,
            batch_size=batch_size,
            progress_callback=update_progress,
        )
        record_chunk_embeddings(
            connection,
            chunk_vector_ids=indexed,
            embedding_model=embedding_model,
            vector_collection=collection,
        )
    except Exception as error:
        set_embedding_run_status(
            connection,
            filing_id=filing_id,
            embedding_model=embedding_model,
            vector_collection=collection,
            status="failed",
            expected_point_count=expected_count,
            indexed_point_count=indexed_count,
            error_message=str(error),
        )
        raise

    set_embedding_run_status(
        connection,
        filing_id=filing_id,
        embedding_model=embedding_model,
        vector_collection=collection,
        status="ready",
        expected_point_count=expected_count,
        indexed_point_count=expected_count,
    )
    return len(indexed)
