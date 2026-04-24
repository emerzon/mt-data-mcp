# forecast_barrier_optimize interface and output shape

## Source report

- Original report: `code-reports/06-forecast_barrier_optimize-format-detail-override.md`
- Original report: `code-reports/07-barrier-optimize-overwhelming-output.md`
- Original report: `code-reports/forecast_barrier_optimize_excessive_params.md`

## Status

Deferred for larger refactor.

## Validated issue

`forecast_barrier_optimize` has a very large public request model and several overlapping output-shaping controls. Current code partially avoids overriding explicit `format`, `concise`, or `return_grid` values unless `detail` is explicitly provided, but the public contract still exposes multiple controls that affect one output axis.

## Evidence

- `src/mtdata/forecast/requests.py::ForecastBarrierOptimizeRequest` exposes many top-level fields for grid modes, volatility scaling, ratios, refinement, search profiles, and statistical robustness.
- The same request model exposes `return_grid`, `top_k`, `format`, `viable_only`, `concise`, and `detail`.
- `src/mtdata/forecast/use_cases.py::run_forecast_barrier_optimize` derives `format_value`, `concise_value`, and `return_grid_value` from `detail` when `detail` is explicitly set or when no legacy output-shaping fields are explicitly set.
- The use case forwards the full parameter set to the optimization implementation, preserving the large schema surface.
- Existing tests cover barrier optimization behavior across grid/statistical features, indicating changes need broad verification.

## Why this should not be fixed inline

This is a public API and schema redesign for one of the largest request models in the project. Moving parameters into nested configs, removing legacy output controls, or changing compact output shape would affect MCP clients, CLI help, tests, docs, and possibly downstream scripts.

## Recommended approach

Separate the work into two migrations. First, define a cleaner output contract where `detail` owns compact/standard/full behavior and grid visibility is explicit. Second, reduce the request surface by grouping advanced options into nested config fields or a documented `params` structure. Preserve compatibility only if a migration window is required.

## Scope

- Files:
  - `src/mtdata/forecast/requests.py`
  - `src/mtdata/forecast/use_cases.py`
  - `src/mtdata/forecast/barriers_optimization.py`
  - `src/mtdata/core/forecast.py`
  - barrier optimization tests and docs
- Symbols:
  - `ForecastBarrierOptimizeRequest`
  - `run_forecast_barrier_optimize`
  - `forecast_barrier_optimize`
  - output fields `best`, `results`, `actionability`, `no_action`, and grid metadata
- Tests:
  - Barrier optimization request validation tests
  - Output-shape tests for compact/standard/full
  - CLI/schema generation tests
  - Statistical robustness and grid-mode tests
- Docs/config/CLI/API affected:
  - MCP schema for `forecast_barrier_optimize`
  - generated CLI flags
  - barrier optimization documentation/examples

## Risks

- Existing users may depend on top-level advanced parameters.
- Existing users may depend on `format`, `concise`, or `return_grid` precedence.
- Compact output changes could hide fields used by scripts.
- Grouping params changes schema shape and CLI parsing behavior.

## Verification plan

- Add schema/precedence tests before changing the request model.
- Define compact/standard/full expected payloads with representative optimizer results.
- Migrate one output-control path at a time and run forecast/barrier tests.
- Run CLI/schema tests after request-model changes.
- Run full pytest after the final request-surface migration.

## Expected impact

- Estimated LOC reduction: medium after legacy controls are removed.
- Complexity reduction: high for users by shrinking the visible parameter set and clarifying output modes.
- Maintenance benefit: easier barrier optimization schema maintenance and less fragile output-shaping logic.
