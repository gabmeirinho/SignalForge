# Planner Prompt Changelog

## planner-v1.3.0

- Added structured specific-year planning with `time_scope: "specific_years"`
  and `filing_years`.
- Added guidance for explicit filing years, year ranges, unavailable requested
  years, and oldest/earliest available filing requests.
- Added normalization for unsupported SEC item sections, market-risk vs
  risk-factor section selection, and open-ended historical trends.
- Model: deepseek-v4-flash
- Temperature: 0.0
- Dataset: tests/fixtures/planner_golden.json
- Evaluation: 19/19 on 2026-06-28

## planner-v1.2.0

- Added explicit mappings for sections 1, 1A, 7, and 7A.
- Added guidance to prefer the smallest sufficient section set.
- Added a guard against selecting section 7A unless market risk is explicit.
- Added fact-vs-summary intent guidance for reported values, metrics, amounts,
  figures, dates, and numbers.
- Model: deepseek-v4-flash
- Temperature: 0.0
- Dataset: tests/fixtures/planner_golden.json
- Evaluation: 9/9 on 2026-06-28

## Versioning Notes

- Bump the prompt version when a prompt change is expected to affect planner
  behavior.
- Record the model, temperature, dataset, and evaluation score with each prompt
  version.
- Use Git history for the exact prompt diff.
