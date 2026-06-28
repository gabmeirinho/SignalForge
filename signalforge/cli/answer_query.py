import argparse
import json

from dotenv import load_dotenv

from signalforge.answer_generator import DEFAULT_ANSWER_MODEL
from signalforge.query_planner import DEFAULT_PLANNER_MODEL
from signalforge.rag_service import answer_question
from signalforge.rag_service import (
    select_ready_accessions_by_ticker as select_ready_accessions_by_ticker,
)
from signalforge.rag_service import years_for_plan_scope as years_for_plan_scope
from signalforge.vector_store import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL


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
        print(
            f"{index}. score={chunk.score:.4f} "
            f"{chunk.ticker} {chunk.filing_date} "
            f"Item {chunk.section_id} chunk {chunk.chunk_index} "
            f"accession={chunk.accession_number}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a question from local SEC filing chunks.")
    parser.add_argument("question")
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    parser.add_argument("--qdrant-path", default="data/qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--show-chunks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
