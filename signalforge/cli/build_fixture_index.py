import argparse

from signalforge.config import RuntimeConfig
from signalforge.evaluation.dataset import load_dataset
from signalforge.evaluation.fixture_index import (
    DEFAULT_FIXTURE_DATASET,
    DEFAULT_FIXTURE_QDRANT_PATH,
    build_fixture_index,
)
from signalforge.vector_store import create_qdrant_client


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    print(f"Dataset version: {dataset.version}")
    print(f"Dataset content SHA-256: {dataset.content_hash}")

    client = create_qdrant_client(args.qdrant_path)
    try:
        fixture = build_fixture_index(
            dataset,
            client=client,
            qdrant_path=args.qdrant_path,
            embedding_model=args.model,
            rebuild=args.rebuild,
        )
    finally:
        client.close()

    action = "Reused" if fixture.reused else "Built"
    filings = sum(len(years) for years in fixture.accessions.values())
    print(
        f"{action} fixture collection {fixture.collection!r} at {fixture.qdrant_path} "
        f"with {fixture.passage_count} passages across {filings} filings "
        f"using {fixture.embedding_model!r}."
    )


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(
        description="Embed a v2 benchmark corpus into a separate, test-only Qdrant collection."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_FIXTURE_DATASET))
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_FIXTURE_QDRANT_PATH,
        help="Embedded Qdrant path for the fixture index; kept apart from the real index.",
    )
    parser.add_argument("--model", default=config.embedding_model)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-embed the corpus even if a complete collection already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
