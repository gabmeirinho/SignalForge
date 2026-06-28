# Planner Prompt Changelog

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
