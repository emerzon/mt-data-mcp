# Forecast dead detail parameters

## Source report

- Original report: `code-reports/forecast-detail-dead-parameters.md`

## Status

Deferred for larger refactor.

## Validated issue

Several forecast request models expose a `detail` parameter even though the corresponding tool implementations do not use it to shape response output. This creates a public schema affordance that is currently either only logged or not referenced by the runtime path.

## Evidence

- `src/mtdata/forecast/requests.py` defines `detail` on `ForecastGenerateRequest`, `ForecastBacktestRequest`, `StrategyBacktestRequest`, `ForecastBarrierProbRequest`, and `ForecastBarrierOptimizeRequest`.
- `src/mtdata/core/forecast.py::forecast_generate` does not pass `request.detail` into operation logging or response shaping.
- `src/mtdata/core/forecast.py::forecast_backtest_run` passes `detail=request.detail` only to operation metadata before returning the raw sanitized result.
- `src/mtdata/core/forecast.py::strategy_backtest` does not reference `request.detail`.
- `src/mtdata/core/forecast.py::forecast_barrier_prob` and `forecast_barrier_optimize` do not reference `request.detail` while invoking their use cases.

## Why this should not be fixed inline

The report spans multiple public request models and forecast tools. Removing the fields changes MCP schemas and generated CLI arguments, while implementing actual verbosity shaping requires per-tool response-contract decisions. A safe cleanup needs design agreement on which tools should keep verbosity control and which should drop it.

## Recommended approach

Audit each affected forecast tool separately. For tools where compact/full output is valuable, implement explicit response shaping and add tests that prove output changes by detail level. For tools with a fixed response contract, remove `detail` from the request model and update schema/CLI/docs tests accordingly.

## Scope

- Files:
  - `src/mtdata/forecast/requests.py`
  - `src/mtdata/core/forecast.py`
  - Forecast use-case modules that construct response payloads
  - Forecast tests under `tests/forecast/`
- Symbols:
  - `ForecastGenerateRequest`
  - `ForecastBacktestRequest`
  - `StrategyBacktestRequest`
  - `ForecastBarrierProbRequest`
  - `ForecastBarrierOptimizeRequest`
  - `forecast_generate`
  - `forecast_backtest_run`
  - `strategy_backtest`
  - `forecast_barrier_prob`
  - `forecast_barrier_optimize`
- Tests:
  - Forecast business logic tests for affected tools
  - CLI/schema tests for generated arguments
- Docs/config/CLI/API affected:
  - MCP tool schemas
  - generated CLI help
  - forecast examples mentioning verbosity, if present

## Risks

- Existing clients may pass `detail` even if it currently has no effect.
- Removing `detail` could break generated CLI usage or schema consumers.
- Implementing response shaping could accidentally hide fields that current clients read.

## Verification plan

- For every affected tool, add or update tests covering compact and full behavior or removal of the parameter.
- Run focused forecast tests for each changed tool.
- Run CLI/schema generation tests to verify the public interface changes intentionally.
- For response shaping, compare representative compact/full payloads to ensure no required fields disappear unexpectedly.

## Expected impact

- Estimated LOC reduction: small if parameters are removed; potentially neutral if real verbosity shaping is implemented.
- Complexity reduction: medium by eliminating false verbosity controls.
- Maintenance benefit: clearer forecast tool contracts and fewer misleading schema fields.
