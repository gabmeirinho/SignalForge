import argparse
import json

from dotenv import load_dotenv

from query_planner import (
    DEFAULT_PLANNER_MODEL,
    DeepSeekQueryPlanner,
    PlannerContext,
    SUPPORTED_SECTIONS,
)
from storage import connect_database, initialize_database, load_planner_metadata


def main() -> None:
    load_dotenv()
    args = parse_args()

    with connect_database(args.db_path) as connection:
        initialize_database(connection)
        context = build_planner_context(load_planner_metadata(connection))

    planner = DeepSeekQueryPlanner.from_environment(model=args.model)
    result = planner.create_plan(args.question, context)

    output = {
        "plan": result.plan.model_dump(),
        "used_fallback": result.used_fallback,
        "error": result.error,
    }
    print(json.dumps(output, indent=2))


def build_planner_context(rows) -> PlannerContext:
    tickers = set()
    sections = set()
    years_by_ticker: dict[str, set[int]] = {}

    for row in rows:
        ticker = str(row["ticker"]).upper()
        tickers.add(ticker)

        section_id = str(row["section_id"]).upper()
        if section_id in SUPPORTED_SECTIONS:
            sections.add(section_id)

        filing_date = row["filing_date"]
        if filing_date and len(str(filing_date)) >= 4:
            try:
                years_by_ticker.setdefault(ticker, set()).add(int(str(filing_date)[:4]))
            except ValueError:
                pass

    return PlannerContext(
        available_tickers=tuple(sorted(tickers)),
        available_sections=tuple(
            section for section in SUPPORTED_SECTIONS if section in sections
        ),
        filing_years_by_ticker={
            ticker: tuple(sorted(years, reverse=True))
            for ticker, years in sorted(years_by_ticker.items())
        },
    )


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
