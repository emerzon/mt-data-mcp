# Detail default policy

## Source report

- Original report: `code-reports/09-default-detail-inconsistency.md`
- Original report: `code-reports/default-detail-level-inconsistency.md`
- Original report: `code-reports/detail_default_values_inconsistent.md`
- Original report: `code-reports/detail-default-inconsistency-read.md`

## Status

Deferred for larger refactor.

## Validated issue

Default `detail` values are inconsistent across read-oriented tools. Several reports are partially stale because some named tools now default to compact, but current code still has public tools that default to `full` while similar tools default to `compact`.

## Evidence

- `src/mtdata/core/causal.py::correlation_matrix` defaults `detail` to `full`.
- `src/mtdata/core/symbols.py::symbols_describe`, `symbols_top_markets`, and `market_scan` default `detail` to `full`.
- `src/mtdata/core/market_depth.py::market_ticker` defaults `detail` to `full`.
- `src/mtdata/core/patterns.py::support_resistance_levels` defaults `detail` to `full`, while pivot helpers default compact.
- `src/mtdata/core/trading/account.py::trade_account_info` defaults `detail` to `full`.
- `src/mtdata/core/trading/requests.py::TradeGetOpenRequest` and `TradeGetPendingRequest` default `detail` to `full`, while `TradeHistoryRequest` and `TradeSessionContextRequest` default to `compact`.
- Many other read tools, including `data_fetch_candles`, forecast requests, `market_status`, `news`, labels, indicators, pivot, report generation, and forecast task tools, default to compact.

## Why this should not be fixed inline

Changing defaults changes public behavior for frequently used tools and can break scripts that rely on full fields being present without explicitly passing `detail="full"`. The reports span multiple domains, output contracts, tests, docs, and CLI/schema expectations. A safe change needs a project-wide default policy and migration plan.

## Recommended approach

Define a documented rule for default detail levels by tool class. For read/list/status tools, prefer compact by default unless the tool has a strong reason to return full data on first call. Migrate one domain at a time, updating tests and docs so users know when to request full output explicitly.

## Scope

- Files:
  - `src/mtdata/core/causal.py`
  - `src/mtdata/core/symbols.py`
  - `src/mtdata/core/market_depth.py`
  - `src/mtdata/core/patterns.py`
  - `src/mtdata/core/trading/account.py`
  - `src/mtdata/core/trading/requests.py`
  - relevant docs and CLI/schema tests
- Symbols:
  - `correlation_matrix`
  - `symbols_describe`
  - `symbols_top_markets`
  - `market_scan`
  - `market_ticker`
  - `support_resistance_levels`
  - `trade_account_info`
  - `TradeGetOpenRequest`
  - `TradeGetPendingRequest`
- Tests:
  - Domain-specific output-shape tests for every changed tool
  - CLI/schema default tests
  - Minimal-output and output-contract tests
- Docs/config/CLI/API affected:
  - MCP schema defaults
  - generated CLI help/default behavior
  - docs examples that omit `--detail`

## Risks

- Existing clients may expect full fields by default.
- Compact output can omit request echo fields and diagnostics that scripts consume.
- Some reports disagree on which tools currently default compact or full, so the migration must use current code as source of truth.
- A project-wide default change could produce many unrelated test failures if done in one commit.

## Verification plan

- Inventory current detail defaults from source and generated schemas.
- Decide per-domain migration order.
- For each domain, update expected compact/full behavior tests before changing defaults.
- Run targeted domain tests and CLI/schema tests after each domain migration.
- Run full pytest after the final default-policy migration.

## Expected impact

- Estimated LOC reduction: small; primary impact is behavior consistency, not deletion.
- Complexity reduction: medium by making default verbosity predictable.
- Maintenance benefit: fewer user-facing surprises and clearer output-contract expectations.
