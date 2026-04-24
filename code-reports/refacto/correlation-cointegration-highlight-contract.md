# Correlation and cointegration highlight contract

## Source report

- Original report: `code-reports/08-correlation-cointegration-summary-duplication.md`

## Status

Deferred for larger refactor.

## Validated issue

`correlation_matrix` and `cointegration_test` use `summary.highlights` to repeat subsets of rows already present in `data.items`. A focused fix now omits correlation highlights entirely when the result set is no larger than the highlight limit, but larger result sets and cointegration summaries still use full row copies.

## Evidence

- `src/mtdata/core/causal.py::_build_correlation_summary` returns strongest positive/negative/absolute row subsets for result sets larger than `top_n`.
- `src/mtdata/core/causal.py::_build_cointegration_summary` returns `best_pairs` and `cointegrated_pairs` as slices/subsets of the same row dictionaries used in `data.items`.
- `src/mtdata/core/causal.py` output builders place those values under `summary.highlights` while also returning canonical rows under `data.items`.

## Why this should not be fixed inline

Changing highlights from full row copies to IDs, pair labels, or compact references changes a public response contract for two tools. Consumers may currently rely on highlight rows containing all fields without joining back to `data.items`. Cointegration also has different ranking semantics than correlation, so the reference shape should be designed consistently across both tools before implementation.

## Recommended approach

1. Define a lightweight highlight reference schema, such as pair IDs plus the ranking metric:
   - `pair`
   - `left`
   - `right`
   - primary score (`correlation`, `p_value`, or `test_statistic`)
2. Apply the schema consistently to correlation and cointegration summaries.
3. Keep full row copies only behind `detail="full"` if backward compatibility is required.
4. Add migration tests documenting compact and full highlight behavior.

## Scope

- Files:
  - `src/mtdata/core/causal.py`
  - `tests/forecast/test_causal_helpers.py`
  - `tests/forecast/test_causal_extended.py`
- Symbols:
  - `_build_correlation_summary`
  - `_build_cointegration_summary`
  - `correlation_matrix`
  - `cointegration_test`
- Tests:
  - Correlation compact/full highlights
  - Cointegration compact/full highlights
  - Existing consumers of `summary.highlights`
- Docs/config/CLI/API affected:
  - MCP response contracts for both tools
  - CLI examples that display `summary.highlights`

## Risks

- Breaking clients that consume full highlight row fields directly.
- Introducing inconsistent reference schemas between correlation and cointegration.
- Losing useful context in compact highlights if references are too minimal.

## Verification plan

1. Add tests for the chosen highlight reference shape.
2. Add tests ensuring highlights can be joined back to `data.items`.
3. Run `python -m pytest tests\\forecast\\test_causal_helpers.py tests\\forecast\\test_causal_extended.py -q`.
4. Run the full pytest suite before merging the response-contract change.

## Expected impact

- Estimated LOC reduction: low.
- Complexity reduction: medium by removing duplicated summary payloads.
- Maintenance benefit: medium, especially for MCP token efficiency.
