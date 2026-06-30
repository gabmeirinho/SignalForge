import argparse

from signalforge.config import RuntimeConfig
from signalforge.storage import connect_database, get_ready_accession_numbers, initialize_database
from signalforge.vector_store import (
    create_qdrant_client,
    semantic_search,
)


def main() -> None:
    args = parse_args()
    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        ready_accessions = get_ready_accession_numbers(
            connection,
            embedding_model=args.model,
            vector_collection=args.collection,
            ticker=args.ticker,
        )

    if not ready_accessions:
        scope = f" for ticker {args.ticker.upper()}" if args.ticker else ""
        raise RuntimeError(
            f"No ready vector index exists{scope}. Run vectorize.py or retry a failed run."
        )

    client = create_qdrant_client(args.qdrant_path)
    try:
        results = semantic_search(
            client,
            query=args.query,
            collection_name=args.collection,
            embedding_model=args.model,
            limit=args.limit,
            ticker=args.ticker,
            section_id=args.section,
            accession_numbers=ready_accessions,
        )
    finally:
        client.close()

    for index, result in enumerate(results, start=1):
        payload = result.payload
        print(
            f"{index}. score={result.score:.4f} "
            f"{payload.get('ticker')} Item {payload.get('section_id')} "
            f"chunk {payload.get('chunk_index')}"
        )
        print(payload.get("text", "")[: args.preview_chars])
        print()


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(description="Run semantic search over SEC filing chunks.")
    parser.add_argument("query")
    parser.add_argument("--db-path", default=config.database_target)
    parser.add_argument("--qdrant-path", default=config.qdrant_target)
    parser.add_argument("--collection", default=config.collection)
    parser.add_argument("--model", default=config.embedding_model)
    parser.add_argument("--ticker")
    parser.add_argument("--section")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    main()
