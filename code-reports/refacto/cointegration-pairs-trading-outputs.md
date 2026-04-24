# Cointegration pairs-trading outputs

## Source report

- Original report: `code-reports/cointegration-missing-spread-and-half-life.md`

## Status

Deferred for larger refactor.

## Validated issue

The report is partially stale but still identifies real gaps. Current `cointegration_test` already estimates and returns `hedge_ratio`, `intercept`, `spread_last`, and `spread_zscore` for each evaluated pair. However, it still has no `detail` parameter, no mean-reversion half-life, no spread series output, no hedge-ratio confidence interval, no explicit spread ADF diagnostics, and no method selector beyond Engle-Granger.

## Evidence

- `src/mtdata/core/causal.py::cointegration_test` has parameters for symbols/group/timeframe/limit/transform/trend/significance/min_overlap, but no `detail` or `method` parameter.
- `src/mtdata/core/causal.py::_evaluate_cointegration_pair` calls `_fit_cointegration_hedge` and returns `hedge_ratio`, `intercept`, `spread_last`, and `spread_zscore`, so the report's claim that hedge ratio is not directly returned is stale.
- The same row payload does not include `half_life`, full `spread` series, hedge-ratio confidence bounds, or separate spread stationarity diagnostics.

## Why this should not be fixed inline

This is a feature/API expansion, not a simplification. Adding `detail`, spread series, half-life estimation, confidence intervals, and optional Johansen support affects tool schema, output size, statistical methodology, and test coverage. The half-life calculation also needs a clear, statistically correct implementation; using the Engle-Granger test statistic directly would not be sufficient.

## Recommended approach

1. Add `detail: CompactFullDetailLiteral` to control optional heavy outputs.
2. Compute half-life from a documented spread mean-reversion regression, with safeguards for non-mean-reverting or degenerate spreads.
3. Keep current compact row fields (`hedge_ratio`, `intercept`, `spread_last`, `spread_zscore`) and add heavier diagnostics only under full detail or explicit flags.
4. Add optional spread series output in full detail with row limits or compression to avoid large MCP payloads.
5. Consider method selection separately, starting with an explicit `"engle_granger"` value before adding Johansen support.

## Scope

- Files:
  - `src/mtdata/core/causal.py`
  - `tests/forecast/test_causal_helpers.py`
  - `tests/forecast/test_causal_extended.py`
- Symbols:
  - `cointegration_test`
  - `_evaluate_cointegration_pair`
  - `_fit_cointegration_hedge`
  - `_build_cointegration_summary`
- Tests:
  - Compact/full output shape
  - Half-life calculation edge cases
  - Optional spread series behavior
  - Method-parameter validation
- Docs/config/CLI/API affected:
  - MCP schema and generated CLI options
  - Causal/cointegration documentation and examples

## Risks

- Incorrect half-life calculations could mislead trading decisions.
- Full spread series can make responses very large.
- Adding Johansen support introduces dependency/method complexity.
- Changing defaults may affect existing consumers of `cointegration_test`.

## Verification plan

1. Add unit tests for half-life calculation on synthetic mean-reverting and non-mean-reverting spreads.
2. Add integration-style tests for compact and full cointegration outputs.
3. Run `python -m pytest tests\\forecast\\test_causal_helpers.py tests\\forecast\\test_causal_extended.py -q`.
4. Run the full pytest suite before merging.

## Expected impact

- Estimated LOC reduction: none; this is an API completeness refactor.
- Complexity reduction: low initially, medium if output depth becomes more explicit.
- Maintenance benefit: medium, by making cointegration output semantics clearer and more actionable.
