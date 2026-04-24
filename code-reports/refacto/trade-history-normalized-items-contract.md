# Trade history normalized items contract

## Source report

- Original report: `code-reports/08-trade-history-output-duplication.md`

## Status

Deferred for larger refactor.

## Validated issue

The report is partially stale but still identifies a real output-contract concern. Current `trade_history` no longer builds a semicolon-separated `deal_details` string; `deal_details`/`order_details` are structured mappings. However, successful `trade_history` responses still include both `items` and `normalized_items`, which can duplicate the same history rows in compact output.

## Evidence

- `src/mtdata/core/trading/positions.py::_normalize_trade_history_row` builds structured `deal_details` or `order_details` dictionaries via `_compact_non_empty_mapping`.
- `src/mtdata/core/trading/positions.py::normalize_trade_history_output` unconditionally adds `normalized_items` whenever `items` is a list and the response succeeded.
- `src/mtdata/core/trading/use_cases.py::run_trade_history` returns raw history rows as dictionaries, and normalization then adds the parallel normalized representation.

## Why this should not be fixed inline

Removing `normalized_items` from compact output or changing which representation is primary would alter a public response contract. Downstream consumers may already rely on either raw MT5-shaped `items` or the normalized history shape. The report also proposes aggregate summaries and removal of diagnostics, which requires designing a new compact history contract rather than a small deletion.

## Recommended approach

1. Define the canonical compact `trade_history` representation:
   - raw `items`,
   - normalized `items`,
   - or a small summary plus a limited row preview.
2. Keep full detail as the compatibility path for both raw and normalized row representations if needed.
3. Move diagnostic row metadata behind full detail if it is not needed for compact trade review.
4. Add a compact summary with total profit, counts, symbols, and win/loss statistics only after the row contract is settled.

## Scope

- Files:
  - `src/mtdata/core/trading/positions.py`
  - `src/mtdata/core/trading/use_cases.py`
  - `src/mtdata/core/trading/requests.py`
  - Trading tests for history deals/orders
- Symbols:
  - `normalize_trade_history_output`
  - `_normalize_trade_history_items`
  - `_normalize_trade_history_row`
  - `TradeHistoryRequest.detail`
  - `run_trade_history`
- Tests:
  - Compact and full `trade_history` response shapes
  - Deals and orders history rows
  - Existing callers that consume `trade_history` internally
- Docs/config/CLI/API affected:
  - MCP response contract for `trade_history`
  - CLI examples and output expectations

## Risks

- Breaking consumers that parse `normalized_items`.
- Losing useful normalized fields if compact output is changed too aggressively.
- Making `detail` semantics inconsistent with other trading read tools.
- Aggregate summary calculations may be misleading unless entry/exit semantics are defined carefully.

## Verification plan

1. Add tests documenting current compact/full shape before changing it.
2. Add migration tests for any retained compatibility path.
3. Run trading account/positions/history tests.
4. Run CLI formatting tests that cover trading read outputs.
5. Run the full pytest suite before merging the response-contract change.

## Expected impact

- Estimated LOC reduction: medium if duplicate compact row representations are removed.
- Complexity reduction: medium.
- Maintenance benefit: high, because `trade_history` would have one clear compact representation.
