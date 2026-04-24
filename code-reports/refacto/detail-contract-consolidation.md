# Detail contract consolidation

## Source report

- Original report: `code-reports/detail-parameter-inconsistency.md`
- Original report: `code-reports/detail_parameter_value_proliferation.md`
- Original report: `code-reports/detail-level-semantic-inconsistency.md`
- Original report: `code-reports/12-multiple-tools-no-detail-parameter.md`
- Original report: `code-reports/04-temporal_analyze-no-detail-parameter.md`

## Status

Deferred for larger refactor.

## Validated issue

The project uses multiple incompatible `detail` value sets and several tools still lack any `detail` parameter. Some individual issues have been split into narrower reports, but the cross-tool contract remains fragmented.

## Evidence

- `src/mtdata/shared/schema.py` defines `CompactFullDetailLiteral`, `CompactStandardFullDetailLiteral`, `SummaryCompactFullDetailLiteral`, and `SummaryAliasCompactFullDetailLiteral`.
- `src/mtdata/core/trading/requests.py::TradePlaceRequest` uses an inline five-value detail literal with `preview`, `basic`, `full`, `compact`, and `summary`.
- `src/mtdata/core/labels.py::labels_triple_barrier` uses an inline literal that includes the legacy `summary_only` alias.
- `src/mtdata/core/regime/api.py::regime_detect` uses `SummaryCompactFullDetailLiteral`.
- `src/mtdata/core/data/requests.py::DataFetchTicksRequest` uses `format` for output shape rather than `detail`.
- `src/mtdata/core/symbols.py::symbols_list`, `src/mtdata/core/temporal.py::temporal_analyze`, `src/mtdata/core/causal.py::causal_discover_signals`, and `cointegration_test` do not expose `detail`.
- `src/mtdata/core/forecast.py::forecast_volatility_estimate` takes a request model without a `detail` field.

## Why this should not be fixed inline

This is a cross-cutting API design problem touching most tool families. Adding missing `detail` parameters, removing enum values, or renaming output-shape parameters changes public schemas and output contracts across MCP and CLI surfaces. The change needs a staged migration and per-tool behavior tests.

## Recommended approach

Define a canonical detail vocabulary and when tools are allowed to extend it. Separate output shape from verbosity for tools like `data_fetch_ticks` and `trade_place`. Migrate one domain at a time, deleting aliases only after compatibility decisions are made and tests document compact/full behavior for each tool.

## Scope

- Files:
  - `src/mtdata/shared/schema.py`
  - `src/mtdata/core/data/requests.py`
  - `src/mtdata/core/trading/requests.py`
  - `src/mtdata/core/labels.py`
  - `src/mtdata/core/regime/api.py`
  - `src/mtdata/core/causal.py`
  - `src/mtdata/core/temporal.py`
  - `src/mtdata/core/symbols.py`
  - `src/mtdata/core/forecast.py`
  - affected docs and CLI/schema tests
- Symbols:
  - Shared detail literal aliases
  - `TradePlaceRequest.detail`
  - `labels_triple_barrier.detail`
  - `DataFetchTicksRequest.format`
  - tools without `detail` listed above
- Tests:
  - Output-contract tests
  - Per-domain compact/full response tests
  - CLI/schema generation tests
  - Minimal-output tests
- Docs/config/CLI/API affected:
  - MCP schemas for many tools
  - generated CLI help
  - documentation describing detail values and examples

## Risks

- Existing clients may rely on tool-specific detail values like `standard`, `summary`, `preview`, or `basic`.
- Some tools need output-shape controls that should not be forced into a binary detail model.
- Adding `detail` to tools without output shaping can create false affordances unless response behavior is implemented at the same time.
- A one-shot migration would be hard to review and likely brittle.

## Verification plan

- Inventory generated schemas before migration.
- Add contract tests for each selected detail vocabulary and alias policy.
- Migrate one tool family per commit with focused tests.
- Run CLI/schema tests after each public signature change.
- Run full pytest after the final consolidation.

## Expected impact

- Estimated LOC reduction: medium after duplicate aliases and normalization paths are removed.
- Complexity reduction: high by making output verbosity predictable.
- Maintenance benefit: a single documented output-control contract across the toolset.
