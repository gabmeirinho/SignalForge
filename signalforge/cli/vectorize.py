import argparse
from collections import defaultdict

from signalforge.storage import (
    connect_database,
    initialize_database,
    load_chunks_for_vector_index,
    record_chunk_embeddings,
    set_embedding_run_status,
    update_embedding_run_progress,
)
from signalforge.vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    create_qdrant_client,
    index_chunks,
)


def main() -> None:
    args = parse_args()

    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        rows = load_chunks_for_vector_index(
            connection,
            embedding_model=args.model,
            vector_collection=args.collection,
        )
        if not rows:
            print(
                "No chunks require indexing for collection "
                f"{args.collection!r} using {args.model!r}."
            )
            return

        rows_by_filing = defaultdict(list)
        for row in rows:
            rows_by_filing[int(row["filing_id"])].append(row)

        client = create_qdrant_client(args.qdrant_path)
        total_indexed = 0
        try:
            for filing_id, filing_rows in rows_by_filing.items():
                expected_count = len(filing_rows)
                indexed_count = 0
                set_embedding_run_status(
                    connection,
                    filing_id=filing_id,
                    embedding_model=args.model,
                    vector_collection=args.collection,
                    status="indexing",
                    expected_point_count=expected_count,
                )

                def update_progress(count: int) -> None:
                    nonlocal indexed_count
                    indexed_count = count
                    update_embedding_run_progress(
                        connection,
                        filing_id=filing_id,
                        embedding_model=args.model,
                        vector_collection=args.collection,
                        indexed_point_count=count,
                    )

                try:
                    indexed = index_chunks(
                        client,
                        rows=filing_rows,
                        collection_name=args.collection,
                        embedding_model=args.model,
                        batch_size=args.batch_size,
                        progress_callback=update_progress,
                    )
                    record_chunk_embeddings(
                        connection,
                        chunk_vector_ids=indexed,
                        embedding_model=args.model,
                        vector_collection=args.collection,
                    )
                except Exception as error:
                    set_embedding_run_status(
                        connection,
                        filing_id=filing_id,
                        embedding_model=args.model,
                        vector_collection=args.collection,
                        status="failed",
                        expected_point_count=expected_count,
                        indexed_point_count=indexed_count,
                        error_message=str(error),
                    )
                    raise
                else:
                    set_embedding_run_status(
                        connection,
                        filing_id=filing_id,
                        embedding_model=args.model,
                        vector_collection=args.collection,
                        status="ready",
                        expected_point_count=expected_count,
                        indexed_point_count=expected_count,
                    )
                    total_indexed += len(indexed)
        finally:
            client.close()

    print(
        f"Indexed {total_indexed} chunks into Qdrant collection "
        f"{args.collection!r} using {args.model!r}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed SQLite chunks and index them in Qdrant.")
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    parser.add_argument("--qdrant-path", default="data/qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main()
