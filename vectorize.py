import argparse

from storage import (
    connect_database,
    initialize_database,
    load_chunks_for_vector_index,
    record_chunk_embeddings,
)
from vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    create_qdrant_client,
    index_chunks,
)


def main() -> None:
    args = parse_args()

    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        rows = load_chunks_for_vector_index(connection)
        if not rows:
            raise RuntimeError("No chunks found in SQLite. Run ingest.py first.")

        client = create_qdrant_client(args.qdrant_path)
        try:
            indexed = index_chunks(
                client,
                rows=rows,
                collection_name=args.collection,
                embedding_model=args.model,
                batch_size=args.batch_size,
            )
        finally:
            client.close()
        record_chunk_embeddings(
            connection,
            chunk_vector_ids=indexed,
            embedding_model=args.model,
            vector_collection=args.collection,
        )

    print(
        f"Indexed {len(indexed)} chunks into Qdrant collection "
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
