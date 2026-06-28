from signalforge.sections import TextChunk
from signalforge.storage import (
    FilingMetadata,
    connect_database,
    get_ready_accession_numbers,
    initialize_database,
    load_chunks_for_vector_index,
    replace_filing_chunks,
    set_embedding_run_status,
    upsert_filing,
)


MODEL = "jinaai/jina-embeddings-v2-small-en"
COLLECTION = "sec_chunks"


def test_embedding_status_controls_search_readiness_and_resets_on_reingestion(tmp_path):
    with connect_database(tmp_path / "test.sqlite3") as connection:
        initialize_database(connection)
        filing_id = upsert_filing(
            connection,
            FilingMetadata(
                accession_number="0001045810-26-000021",
                ticker="NVDA",
                cik="0001045810",
                company_name="NVIDIA CORP",
                form_type="10-K",
                filing_date="2026-02-25",
                period_of_report="2026-01-25",
                raw_path="raw.txt",
                raw_sha256="a" * 64,
                clean_text_path="clean.txt",
            ),
        )
        chunks = [
            TextChunk(
                section_id="1A",
                section_title="Risk Factors",
                chunk_index=0,
                text="Supplier risk.",
            )
        ]
        replace_filing_chunks(connection, filing_id, chunks)
        assert len(
            load_chunks_for_vector_index(
                connection,
                embedding_model=MODEL,
                vector_collection=COLLECTION,
            )
        ) == 1

        set_embedding_run_status(
            connection,
            filing_id=filing_id,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            status="failed",
            expected_point_count=1,
            error_message="Qdrant timeout",
        )
        assert len(
            load_chunks_for_vector_index(
                connection,
                embedding_model=MODEL,
                vector_collection=COLLECTION,
            )
        ) == 1
        assert get_ready_accession_numbers(
            connection,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
        ) == []

        set_embedding_run_status(
            connection,
            filing_id=filing_id,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            status="ready",
            expected_point_count=1,
            indexed_point_count=1,
        )
        assert (
            load_chunks_for_vector_index(
                connection,
                embedding_model=MODEL,
                vector_collection=COLLECTION,
            )
            == []
        )
        assert get_ready_accession_numbers(
            connection,
            embedding_model=MODEL,
            vector_collection=COLLECTION,
            ticker="nvda",
        ) == ["0001045810-26-000021"]

        replace_filing_chunks(connection, filing_id, chunks)
        assert len(
            load_chunks_for_vector_index(
                connection,
                embedding_model=MODEL,
                vector_collection=COLLECTION,
            )
        ) == 1
        run = connection.execute(
            """
            SELECT status, indexed_point_count, error_message
            FROM embedding_runs
            WHERE filing_id = ?
            """,
            (filing_id,),
        ).fetchone()

    assert run["status"] == "pending"
    assert run["indexed_point_count"] == 0
    assert run["error_message"] is None
