# Price precision output rendering

## Source report

- Original report: `code-reports/02-price-precision-loss.md`

## Status

Deferred for larger refactor.

## Validated issue

TOON output formatting still relies on field-name heuristics for price precision. Direct quote fields such as `bid`, `ask`, `open`, `high`, `low`, `close`, and `price` are protected by `_QUOTE_DECIMALS_BY_FIELD`, but several price-bearing outputs use tool-specific or generic field names that can still be rendered with too few decimals.

The nested `data_fetch_ticks` stat subset was addressed separately by making TOON formatting parent-field aware for quote stats. The remaining report spans multiple tools and needs a broader precision model.

## Evidence

- `src/mtdata/utils/minimal_output_toon.py` defines `_QUOTE_DECIMALS_BY_FIELD` as a hardcoded field-name table.
- `src/mtdata/utils/formatting.py` uses adaptive decimal selection when a field does not have an explicit precision override.
- Price-bearing names referenced by the original report, including `tp_price`, `sl_price`, `level_price`, `reference_price`, `current_price`, `fast_ma`, `slow_ma`, `zone_low`, and `zone_high`, appear across forecasting, pattern, strategy, and support/resistance code paths.
- Some suggested names, notably `value`, `entry`, and `exit`, are generic enough that forcing quote precision globally could affect unrelated numeric output.

## Why this should not be fixed inline

A safe implementation requires more than expanding a global formatter table. Hardcoding every currently observed price field would be incomplete and could over-format non-price values. The more robust approach is to propagate symbol precision or semantic price metadata from tool payloads into the rendering layer, which affects many public response shapes and tests.

## Recommended approach

Introduce a precision context for output rendering:

1. Have price-producing tools expose or annotate the relevant `digits`/precision metadata near their numeric payloads.
2. Teach TOON rendering to apply that precision context to known price subtrees instead of relying only on leaf field names.
3. Keep existing explicit `_QUOTE_DECIMALS_BY_FIELD` overrides as a fallback for payloads without precision metadata.
4. Add regression tests for each affected tool family before changing formatter behavior broadly.

## Scope

- Files:
  - `src/mtdata/utils/minimal_output_toon.py`
  - `src/mtdata/utils/formatting.py`
  - Forecast, pattern, pivot, support/resistance, strategy, and tick output builders
- Symbols:
  - `_QUOTE_DECIMALS_BY_FIELD`
  - `_column_decimals`
  - `_format_to_toon`
  - `_stringify_for_toon_value`
- Tests:
  - Existing TOON formatter tests under `tests/core/`
  - Affected tool business-logic tests under `tests/forecast/`, `tests/patterns/`, and related core tests
- Docs/config/CLI/API affected:
  - CLI output rendering for `fmt="toon"`
  - MCP tool compact output examples if documented

## Risks

- Over-formatting unrelated numeric fields if generic names are globally forced to quote precision.
- Changing compact CLI output snapshots or documentation examples.
- Inconsistent behavior if only some tools provide precision metadata.
- Potential public-output contract changes for consumers parsing TOON output.

## Verification plan

1. Add targeted formatter tests for symbol precision metadata and generic numeric fields that must not be over-formatted.
2. Add representative output tests for `pivot_compute_points`, `support_resistance_levels`, `patterns_detect`, `forecast_barrier_optimize`, `forecast_generate`, `strategy_backtest`, and `forecast_barrier_prob`.
3. Run the focused formatter and tool tests.
4. Run the full pytest suite before merging the broad renderer change.

## Expected impact

- Estimated LOC reduction: low; the main benefit is correctness and removal of ad-hoc precision exceptions over time.
- Complexity reduction: medium after precision behavior becomes centralized and context-aware.
- Maintenance benefit: high for future price-producing tools, because they should not need new formatter field-name exceptions.
