import json
from pathlib import Path

from evaluate_planner import compare_plan
from query_planner import LocalQueryPlanner, PlannerContext

cases = json.loads(Path("tests/fixtures/planner_golden.json").read_text())
planner = LocalQueryPlanner()

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
        company_names_by_ticker={
            ticker: tuple(names)
            for ticker, names in context_data.get("company_names_by_ticker", {}).items()
        },
    )

    result = planner.create_plan(case["question"], context)
    actual = result.plan.model_dump()
    failures = compare_plan(actual, case["expected"])

    if failures:
        print(f"FAIL {case['id']}")
        print(f"  Actual:   {json.dumps(actual, sort_keys=True)}")
        print(f"  Expected: {json.dumps(case['expected'], sort_keys=True)}")
        for failure in failures:
            print(f"  - {failure}")
    else:
        passed += 1
        print(f"PASS {case['id']}")
    print()

total = len(cases)
print(f"Score: {passed}/{total} ({passed / total:.0%})")
