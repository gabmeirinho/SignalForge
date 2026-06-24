import argparse

from vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    create_qdrant_client,
    semantic_search,
)


def main() -> None:
    args = parse_args()
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
    parser = argparse.ArgumentParser(description="Run semantic search over SEC filing chunks.")
    parser.add_argument("query")
    parser.add_argument("--qdrant-path", default="data/qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ticker")
    parser.add_argument("--section")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    main()
