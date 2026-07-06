import argparse

from signalforge.config import RuntimeConfig
from signalforge.storage import connect_database, initialize_database
from signalforge.vectorization import vectorize_pending_chunks


def main() -> None:
    args = parse_args()

    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        result = vectorize_pending_chunks(
            connection,
            qdrant_target=args.qdrant_path,
            collection=args.collection,
            embedding_model=args.model,
            batch_size=args.batch_size,
        )

        if result.indexed_count == 0:
            print(
                "No chunks require indexing for collection "
                f"{args.collection!r} using {args.model!r}."
            )
            return

    print(
        f"Indexed {result.indexed_count} chunks into Qdrant collection "
        f"{args.collection!r} using {args.model!r}."
    )


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(description="Embed SQLite chunks and index them in Qdrant.")
    parser.add_argument("--db-path", default=config.database_target)
    parser.add_argument("--qdrant-path", default=config.qdrant_target)
    parser.add_argument("--collection", default=config.collection)
    parser.add_argument("--model", default=config.embedding_model)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main()
