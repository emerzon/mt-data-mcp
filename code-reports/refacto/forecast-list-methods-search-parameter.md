# forecast_list_methods search parameter

## Source report

- Original report: `code-reports/forecast_list_methods_search_dual_param.md`

## Status

Deferred for larger refactor.

## Validated issue

`forecast_list_methods` exposes both `search` and `search_term` parameters for the same filtering behavior. The implementation normalizes both values, rejects conflicting non-empty values, and then passes one effective value to the list-methods implementation.

## Evidence

- `src/mtdata/core/forecast.py` defines `forecast_list_methods(detail, limit, search, search_term)`.
- The same function computes `effective_search = search_term_value or search_value`.
- `tests/forecast/test_core_forecast_business_logic.py` covers `search_term` as an alias and checks that conflicting `search` and `search_term` values return an error.
- Other list-style tools such as `indicators_list` and `symbols_list` use a single `search_term` parameter.

## Why this should not be fixed inline

Removing `search` from the function signature would change the generated MCP schema and CLI surface for an existing public forecast tool. Keeping it as a hidden compatibility alias is not possible with the current plain function signature without introducing a separate request model or compatibility path, so the cleanup needs an explicit migration decision.

## Recommended approach

Decide whether `search` remains supported for a deprecation window. If compatibility is not required, remove the parameter and update tests/docs. If compatibility is required, migrate `forecast_list_methods` to a request model or another schema mechanism that exposes only `search_term` while accepting legacy `search` input internally.

## Scope

- Files:
  - `src/mtdata/core/forecast.py`
  - `tests/forecast/test_core_forecast_business_logic.py`
  - Forecast CLI/schema coverage tests, if any
- Symbols:
  - `forecast_list_methods`
  - `_forecast_list_methods_impl`
- Tests:
  - Forecast list-method filtering tests
  - CLI/tool schema generation tests
- Docs/config/CLI/API affected:
  - MCP tool schema for `forecast_list_methods`
  - generated CLI arguments for `forecast_list_methods`

## Risks

- Existing clients may still call `forecast_list_methods(search=...)`.
- Changing the signature may alter CLI generation and MCP schema snapshots.
- A compatibility implementation could add complexity if the project decides legacy `search` must remain accepted.

## Verification plan

- Add or update tests that assert only the intended public parameter appears in generated schema/help.
- Run `python -m pytest tests/forecast/test_core_forecast_business_logic.py`.
- Run relevant CLI/schema tests for generated tool arguments.
- Manually inspect `forecast_list_methods` help/schema output.

## Expected impact

- Estimated LOC reduction: small.
- Complexity reduction: low to medium by removing conflict-resolution logic.
- Maintenance benefit: simpler, consistent list-tool filtering semantics.
