# Candle forming-bar default

## Source report

- Original report: `code-reports/13-candle-data-silently-excludes-incomplete.md`

## Status

Deferred for larger refactor.

## Validated issue

`data_fetch_candles` defaults to closed candles only (`include_incomplete=False`). When the latest forming candle is available, the service trims it and returns metadata explaining the exclusion. This can surprise users who expect `limit` to include the current forming bar.

## Evidence

- `src/mtdata/core/data/requests.py::DataFetchCandlesRequest` defines `include_incomplete: bool = False`.
- `src/mtdata/services/data_service.py::fetch_candles` returns `candles_requested`, `candles_excluded`, `last_candle_open`, `incomplete_candles_skipped`, `has_forming_candle`, and a hint when a forming candle is excluded.
- `tests/services/test_data_service_coverage.py` asserts both the default exclusion behavior and `include_incomplete=True` behavior, so the current default is covered as supported behavior.
- The tool docstring explicitly states that it defaults to closed candles only.

## Why this should not be fixed inline

Changing the default to include forming candles changes trading-data semantics. Closed-candle-only output is often desirable for backtesting, signal confirmation, and avoiding repainting. Reducing metadata also changes tested response fields. A safe change needs a versioned/default-policy decision and migration guidance.

## Recommended approach

1. Decide whether the default should remain closed-candle-only or switch to active-trading behavior.
2. If changing the default, add explicit migration notes and tests for backtesting/signal use cases.
3. Consider a clearer top-level `note` while retaining detailed diagnostics in `meta` or full detail.
4. Keep `include_incomplete` explicit in examples so users know how to request forming bars.

## Scope

- Files:
  - `src/mtdata/core/data/requests.py`
  - `src/mtdata/core/data/__init__.py`
  - `src/mtdata/services/data_service.py`
  - Candle/web API tests
- Symbols:
  - `DataFetchCandlesRequest.include_incomplete`
  - `data_fetch_candles`
  - `fetch_candles`
  - web API history include-incomplete mapping
- Tests:
  - Default candle count behavior
  - Include-incomplete behavior
  - Metadata/note output shape
  - Web API history parameter behavior
- Docs/config/CLI/API affected:
  - MCP schema default
  - CLI help and examples
  - Web API docs/examples

## Risks

- Including forming candles by default may change indicator and strategy results.
- Closed-candle consumers may accidentally ingest repainting bars.
- Removing existing metadata can break clients that detect forming-candle exclusions.

## Verification plan

1. Add tests for the chosen default and explicit override.
2. Add tests for compact/full metadata or note behavior.
3. Run data service, web API history, and CLI parameter tests.
4. Run the full pytest suite before merging.

## Expected impact

- Estimated LOC reduction: low.
- Complexity reduction: low; the main benefit is clearer data semantics.
- Maintenance benefit: medium, by making forming-candle behavior more explicit.
