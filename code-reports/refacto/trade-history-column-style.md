# Trade history column style

## Source report

- Original report: `code-reports/column_style_missing_from_trade_history.md`

## Status

Deferred for larger refactor.

## Validated issue

`trade_get_open` and `trade_get_pending` expose `column_style: Literal["humanized", "snake_case"]`, while `trade_history` does not. This is a real inconsistency across trading table-style outputs.

## Evidence

- `src/mtdata/core/trading/requests.py` defines `column_style` on `TradeGetOpenRequest` and `TradeGetPendingRequest`.
- `src/mtdata/core/trading/requests.py` defines `TradeHistoryRequest` without `column_style`.
- `src/mtdata/core/trading/use_cases.py` applies column style while building open-position and pending-order DataFrames.
- `src/mtdata/core/trading/use_cases.py::run_trade_history` currently returns history rows as dictionaries using raw MT5/snake-case field names.
- `src/mtdata/core/trading/positions.py::normalize_trade_history_output` adds `normalized_items`, but does not rename the primary `items` columns.

## Why this should not be fixed inline

Adding the parameter is easy, but selecting the default and applying it safely is not. Matching `trade_get_open`/`trade_get_pending` would imply `column_style="humanized"` by default, which would change existing `trade_history` row keys from raw snake-case names to display labels. Keeping snake-case as the default would preserve behavior but make `trade_history` inconsistent with the existing tools' default. This is a public output-contract decision.

## Recommended approach

1. Decide whether `trade_history` should preserve raw MT5-style keys by default or align with open/pending table defaults.
2. If humanized output is adopted, provide a migration note and possibly use `detail` or an explicit opt-in first.
3. Implement a shared trading column-name helper instead of duplicating mappings in each output builder.
4. Apply the style consistently to deals, orders, and normalized history rows where appropriate.

## Scope

- Files:
  - `src/mtdata/core/trading/requests.py`
  - `src/mtdata/core/trading/use_cases.py`
  - `src/mtdata/core/trading/positions.py`
  - Trading output tests
- Symbols:
  - `TradeHistoryRequest`
  - `run_trade_history`
  - `normalize_trade_history_output`
  - `_build_trade_get_open_output`
  - `_build_trade_get_pending_output`
- Tests:
  - Trade history deals/orders output keys
  - Open/pending column style regression tests
  - CLI schema/help tests for the new request field
- Docs/config/CLI/API affected:
  - Generated CLI parameter surface for `trade_history`
  - Public MCP request schema
  - Any examples that assume snake-case history keys

## Risks

- Breaking consumers that parse `trade_history.items` by raw MT5 field names.
- Inconsistent style between `items` and `normalized_items`.
- Confusing defaults if `trade_history` diverges from open/pending tools.

## Verification plan

1. Add tests for `trade_history(column_style="snake_case")` and `trade_history(column_style="humanized")`.
2. Add tests for both `history_kind="deals"` and `history_kind="orders"`.
3. Run trading business-logic tests and CLI schema/help tests.
4. Run the full pytest suite before merging the API-surface change.

## Expected impact

- Estimated LOC reduction: low.
- Complexity reduction: medium if shared column-name mapping replaces per-tool helpers.
- Maintenance benefit: medium, by making trading tabular outputs use one naming policy.
