import argparse
import json

from dotenv import load_dotenv

from signalforge.config import RuntimeConfig
from signalforge.rag_service import answer_question
from signalforge.rag_service import (
    select_ready_accessions_by_ticker as select_ready_accessions_by_ticker,
)
from signalforge.rag_service import years_for_plan_scope as years_for_plan_scope


def main() -> None:
    load_dotenv()
    args = parse_args()
    response = answer_question(
        question=args.question,
        db_path=args.db_path,
        qdrant_path=args.qdrant_path,
        collection=args.collection,
        embedding_model=args.embedding_model,
        planner_model=args.planner_model,
        answer_model=args.answer_model,
    )

    if args.show_plan:
        print(
            json.dumps(
                {
                    "plan": response.plan.model_dump(),
                    "used_fallback": response.used_fallback,
                    "planner_error": response.planner_error,
                },
                indent=2,
            )
        )
        print()
    if args.show_chunks:
        print_chunk_summary(response.sources)
        print()

    print(response.answer)


def print_chunk_summary(chunks) -> None:
    if not chunks:
        print("Retrieved chunks: (none)")
        return

    print("Retrieved chunks:")
    for index, chunk in enumerate(chunks, start=1):
        if chunk.chunk_source == "document":
            print(
                f"{index}. score={chunk.score:.4f} {chunk.label} "
                f"url={chunk.url or '-'}"
            )
        else:
            print(
                f"{index}. score={chunk.score:.4f} {chunk.label} "
                f"accession={chunk.accession_number or '-'}"
            )


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(description="Answer a question from local SEC filing chunks.")
    parser.add_argument("question")
    parser.add_argument("--db-path", default=config.database_target)
    parser.add_argument("--qdrant-path", default=config.qdrant_target)
    parser.add_argument("--collection", default=config.collection)
    parser.add_argument("--embedding-model", default=config.embedding_model)
    parser.add_argument("--planner-model", default=config.planner_model)
    parser.add_argument("--answer-model", default=config.answer_model)
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--show-chunks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
