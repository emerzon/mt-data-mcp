# Forecast method discovery contract

## Source report

- Original report: `code-reports/07-forecast_list-methods-vs-models-overlap.md`
- Related source report: `code-reports/18-forecast-list-methods-overwhelming.md`

## Status

Deferred for larger refactor.

## Validated issue

The report is partially stale but still identifies a real discovery-contract issue. `forecast_list_methods` now supports compact/full detail, descriptions, categories, search, limits, namespace metadata, support hints, and parameter counts. However, it still does not accept a `library` filter, and `forecast_list_library_models` still exposes library-scoped model names using a different primary key (`model`) from `forecast_list_methods` (`method`).

A related report's claim that `forecast_list_methods` is entirely flat is stale because compact output now includes `category_summary` and category metadata. Its remaining valid concerns are that unavailable methods are still included by default and that the tool does not provide a curated/recommended subset.

## Evidence

- `src/mtdata/core/forecast.py::forecast_list_methods` accepts `detail`, `limit`, `search`, and `search_term`, but no `library` parameter.
- `src/mtdata/core/forecast.py::_forecast_list_methods_impl` builds `category_summary`, includes descriptions in compact rows, and includes full metadata under `detail="full"`.
- The same implementation counts unavailable methods and includes them in compact rows by default, sorted after available methods within categories.
- `src/mtdata/core/forecast.py::forecast_list_library_models` remains a separate library-scoped discovery tool.
- The two tools still use different naming conventions for the primary row key: `method` versus `model`.

## Why this should not be fixed inline

Consolidating forecast discovery changes MCP schemas, generated CLI help, and public output contracts. A compatible migration needs a decision about whether to keep both tools, deprecate one, or make one tool a superset of the other. Standardizing `method`/`model` names also affects downstream consumers.

## Recommended approach

1. Decide on the canonical discovery tool and row key (`method` is likely preferred because `forecast_generate` accepts `method`).
2. Add a `library`/`category` filter to `forecast_list_methods` if it remains the primary tool.
3. Add an explicit `show_unavailable` or `availability` filter instead of changing unavailable-method visibility silently.
4. Add a curated/recommended view only after defining stable recommendation criteria.
5. Either deprecate `forecast_list_library_models` or make it a thin compatibility wrapper over the same discovery data.
6. Add migration tests for both tool outputs.

## Scope

- Files:
  - `src/mtdata/core/forecast.py`
  - Forecast method registry/snapshot helpers
  - Forecast CLI/schema tests
- Symbols:
  - `forecast_list_methods`
  - `_forecast_list_methods_impl`
  - `forecast_list_library_models`
  - forecast method snapshot metadata
- Tests:
  - Compact/full discovery output
  - Library/category filtering
  - Unavailable-method filtering
  - Recommended/curated view behavior if added
  - Compatibility behavior for `forecast_list_library_models`
- Docs/config/CLI/API affected:
  - MCP tool schemas
  - Generated CLI options
  - Forecast method documentation and examples

## Risks

- Breaking clients that parse `forecast_list_library_models.models[].model`.
- Confusing users if both tools remain but expose different metadata.
- Increasing output size if all library metadata is merged into one default response.

## Verification plan

1. Add tests for library/category filters and row-key compatibility.
2. Run forecast discovery tests and CLI schema tests.
3. Run full forecast test subset.
4. Run the full pytest suite before merging the public API change.

## Expected impact

- Estimated LOC reduction: medium if duplicate discovery tools are consolidated.
- Complexity reduction: medium.
- Maintenance benefit: high, by centralizing forecast method discovery semantics.
