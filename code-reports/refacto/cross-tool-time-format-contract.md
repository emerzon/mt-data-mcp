# Cross-tool time format contract

## Source report

- Original report: `code-reports/15-inconsistent-time-formats.md`

## Status

Deferred for larger refactor.

## Validated issue

Timestamp fields are not represented consistently across tools. Some payloads use display strings, some preserve raw epoch values with companion display fields, some include MT5 millisecond timestamps, and some use ISO 8601 strings. The report is partly stale for `market_ticker`, whose raw payload now includes an epoch `time` plus `time_display`, but the broader inconsistency remains valid across raw API, CLI normalization, data rows, trading history, forecasts, and market status.

## Evidence

- `src/mtdata/utils/utils.py::_format_time_minimal` formats timestamps without seconds.
- `data_fetch_candles`, `data_fetch_ticks`, `trade_history`, and forecast outputs use formatted time strings in row payloads.
- `market_ticker` carries raw epoch `time` and a `time_display` companion before CLI output normalization.
- `market_status` uses timezone-aware ISO-style strings for some fields.
- Tests in CLI/output layers explicitly cover special handling for `market_ticker` and tick/depth time fields, so timestamp behavior is a tested public contract.

## Why this should not be fixed inline

Changing time field formats affects nearly every data consumer and many snapshots/tests. ISO 8601 with seconds is a good target, but migrating all tools at once requires compatibility aliases, CLI rendering decisions, documentation, and careful handling of MT5 epoch/millisecond fields.

## Recommended approach

1. Define canonical time keys:
   - `time` for canonical ISO 8601 timestamp,
   - `time_epoch` for Unix seconds,
   - `time_msc` only where MT5 millisecond precision is intentionally exposed,
   - `time_display` only for human-focused CLI/UI display if still needed.
2. Add shared formatting helpers for ISO-with-timezone and display-with-seconds.
3. Migrate one domain at a time behind compatibility aliases.
4. Update CLI normalization so raw API contracts and CLI display behavior are explicitly separate.

## Scope

- Files:
  - `src/mtdata/utils/utils.py`
  - `src/mtdata/core/market_depth.py`
  - `src/mtdata/core/market_status.py`
  - `src/mtdata/services/data_service.py`
  - `src/mtdata/core/trading/use_cases.py`
  - Forecast output builders
  - CLI output normalization tests
- Symbols:
  - `_format_time_minimal`
  - `_format_time_minimal_local`
  - market ticker/time normalization helpers
  - data/tick/trade history row time formatting
- Tests:
  - Raw API timestamp contracts
  - CLI timestamp display contracts
  - Web API history timestamp behavior
  - Trading history millisecond precision behavior
- Docs/config/CLI/API affected:
  - MCP response contracts
  - CLI output examples
  - Web API history output examples

## Risks

- Breaking clients that parse existing display time strings.
- Confusing users if API and CLI use different canonical keys without documentation.
- Losing millisecond precision for trade history if `time_msc` is hidden too aggressively.
- Large snapshot/test churn across domains.

## Verification plan

1. Add contract tests for the chosen canonical time shape.
2. Migrate one representative data tool and one trading tool first.
3. Run focused data, trading, CLI, and web API tests after each migration.
4. Run the full pytest suite before completing the cross-tool change.

## Expected impact

- Estimated LOC reduction: low.
- Complexity reduction: high once time handling is centralized.
- Maintenance benefit: high, by making cross-tool event correlation predictable.
