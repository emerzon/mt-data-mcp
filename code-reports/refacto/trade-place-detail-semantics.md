# Trade place detail semantics

## Source report

- Original report: `code-reports/trade-place-detail-conditional.md`

## Status

Deferred for larger refactor.

## Validated issue

`TradePlaceRequest.detail` is a dry-run preview shaping parameter, not a general response verbosity control. The name overlaps with the broader `detail` convention used by many tools, but real order placement currently returns execution output without applying the preview detail shaping.

## Evidence

- `src/mtdata/core/trading/requests.py::TradePlaceRequest` defines `detail` with values `preview`, `basic`, `full`, `compact`, and `summary`, and its description says it is a dry-run preview detail level.
- `src/mtdata/core/trading/use_cases.py::_resolve_trade_place_preview_detail` resolves this field specifically for preview output.
- `src/mtdata/core/trading/use_cases.py::_dry_run_preview` calls `_shape_trade_place_preview(preview, detail=preview_detail)`.
- Real order execution paths do not use the same detail shaping, so the parameter semantics are conditional on dry-run behavior.

## Why this should not be fixed inline

This touches real trade execution response contracts. Renaming the field would break existing callers, while applying it to live execution would require defining safe compact/full execution payloads without hiding critical order, retcode, guardrail, SL/TP, or recovery information.

## Recommended approach

1. Decide whether the field should be renamed to `preview_detail` with `detail` retained as a deprecated alias.
2. Alternatively, define live execution compact/full response shapes and apply `detail` uniformly.
3. For any live compact mode, keep critical safety and broker result fields visible.
4. Add clear conflict/deprecation messaging if both old and new field names are supported.

## Scope

- Files:
  - `src/mtdata/core/trading/requests.py`
  - `src/mtdata/core/trading/use_cases.py`
  - Trading order tests
- Symbols:
  - `TradePlaceRequest.detail`
  - `_resolve_trade_place_preview_detail`
  - `_shape_trade_place_preview`
  - live `trade_place` execution result builders
- Tests:
  - Dry-run preview detail shaping
  - Live execution compact/full shaping, if adopted
  - Alias/deprecation behavior if renamed
- Docs/config/CLI/API affected:
  - MCP schema for `trade_place`
  - Generated CLI help
  - Trading safety docs/examples

## Risks

- Hiding critical live-order execution details in compact output.
- Breaking callers that already use `detail` for dry-run previews.
- Expanding the schema with both `detail` and `preview_detail` unless migration is managed carefully.

## Verification plan

1. Add tests documenting current dry-run behavior before migration.
2. Add tests for the chosen live response shape or field alias.
3. Run trading order/use-case tests.
4. Run the full pytest suite before merging.

## Expected impact

- Estimated LOC reduction: low.
- Complexity reduction: medium if preview/live detail semantics become explicit.
- Maintenance benefit: medium, by avoiding a misleading parameter name on a real trading tool.
