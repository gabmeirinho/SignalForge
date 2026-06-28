import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from query_planner import (
    DEFAULT_PLANNER_MODEL,
    PLANNER_PROMPT_VERSION,
    PLANNER_TEMPERATURE,
    DeepSeekQueryPlanner,
    PlannerContext,
)


DEFAULT_DATASET = Path("tests/fixtures/planner_golden.json")
SET_FIELDS = {"tickers", "sections", "filing_years"}


def main() -> None:
    load_dotenv()
    args = parse_args()
    cases = json.loads(args.dataset.read_text())
    planner = DeepSeekQueryPlanner.from_environment(model=args.model)

    print(f"Model: {args.model}")
    print(f"Temperature: {PLANNER_TEMPERATURE}")
    print(f"Prompt version: {PLANNER_PROMPT_VERSION}")
    print(f"Dataset: {args.dataset}\n")

    passed = 0
    for case in cases:
        context_data = case["context"]
        context = PlannerContext(
            available_tickers=tuple(context_data["available_tickers"]),
            available_sections=tuple(context_data["available_sections"]),
            filing_years_by_ticker={
                ticker: tuple(years)
                for ticker, years in context_data["filing_years_by_ticker"].items()
            },
        )
        result = planner.create_plan(case["question"], context)
        actual = result.plan.model_dump()
        failures = compare_plan(actual, case["expected"])

        if result.used_fallback:
            failures.append(f"unexpected fallback: {result.error}")

        if failures:
            print(f"FAIL {case['id']}")
            for failure in failures:
                print(f"  - {failure}")
        else:
            passed += 1
            print(f"PASS {case['id']}")

    total = len(cases)
    print(f"\nScore: {passed}/{total} ({passed / total:.0%})")
    raise SystemExit(0 if passed == total else 1)


def compare_plan(actual: dict, expected: dict) -> list[str]:
    failures = []
    for field, expected_value in expected.items():
        actual_value = actual[field]
        matches = (
            set(actual_value) == set(expected_value)
            if field in SET_FIELDS
            else actual_value == expected_value
        )
        if not matches:
            failures.append(
                f"{field}: expected {expected_value!r}, got {actual_value!r}"
            )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the LLM planner on golden cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Golden dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--model", default=DEFAULT_PLANNER_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    main()
