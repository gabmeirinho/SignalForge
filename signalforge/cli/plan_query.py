import argparse
import json

from dotenv import load_dotenv

from signalforge.query_planner import (
    DEFAULT_PLANNER_MODEL,
    build_planner_context,
    create_query_planner_from_environment,
)
from signalforge.storage import connect_database, initialize_database, load_planner_metadata


def main() -> None:
    load_dotenv()
    args = parse_args()

    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        context = build_planner_context(load_planner_metadata(connection))

    planner = create_query_planner_from_environment(model=args.model)
    result = planner.create_plan(args.question, context)

    output = {
        "plan": result.plan.model_dump(),
        "used_fallback": result.used_fallback,
        "error": result.error,
    }
    print(json.dumps(output, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validated SEC retrieval plan with DeepSeek."
    )
    parser.add_argument("question")
    parser.add_argument("--db-path", default="data/signalforge.sqlite3")
    parser.add_argument("--model", default=DEFAULT_PLANNER_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    main()
