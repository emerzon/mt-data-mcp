# News pagination and filters

## Source report

- Original report: `code-reports/news-missing-pagination.md`

## Status

Deferred for larger refactor.

## Validated issue

The safe subset was implemented inline: `news` now accepts an optional `limit` parameter to cap each returned news bucket without changing default output behavior.

The remaining issue is still valid: the unified `news` tool has no page, date-range, source, or category filters, while lower-level Finviz and calendar-style tools expose some of those controls.

## Evidence

- `src/mtdata/core/news.py::news` aggregates buckets from `fetch_unified_news`.
- `src/mtdata/services/unified_news.py::fetch_unified_news` currently accepts only `symbol`, so date/source/page filtering cannot be passed through cleanly yet.
- Finviz service functions support `limit` and `page`, but the unified news service also merges MT5 news, calendar events, impact headlines, and symbol-relevance buckets.

## Why this should not be fixed inline

Pagination and date/source filters must be defined across heterogeneous buckets. A single global page can have ambiguous meaning when the output contains `general_news`, `related_news`, `impact_news`, `upcoming_events`, and `recent_events`. Date filtering also requires consistent parsing and timezone behavior across headline and calendar-event sources.

## Recommended approach

1. Add filter support to `fetch_unified_news` before expanding the MCP tool surface further.
2. Define whether pagination applies per bucket or to a flattened combined feed.
3. Add date parsing shared with other tools that use `start`/`end` aliases.
4. Add source/category filtering only after source names and bucket taxonomy are documented.
5. Keep `limit` as the lightweight per-bucket cap.

## Scope

- Files:
  - `src/mtdata/core/news.py`
  - `src/mtdata/services/unified_news.py`
  - `tests/core/test_news_business_logic.py`
  - `tests/services/test_unified_news_business_logic.py`
- Symbols:
  - `news`
  - `normalize_news_output`
  - `fetch_unified_news`
  - Finviz and MT5 news adapters
- Tests:
  - Per-bucket limit behavior
  - Page semantics
  - Date range filtering
  - Source/category filtering
- Docs/config/CLI/API affected:
  - MCP schema and generated CLI help for `news`
  - News tool examples

## Risks

- Ambiguous pagination over multiple buckets.
- Inconsistent timezone handling for date filters.
- Filtering at the wrong layer can hide relevant symbol-specific events.

## Verification plan

1. Add service-level tests for date/source/page filters.
2. Add tool-level tests for new parameters and validation.
3. Run `python -m pytest tests\\core\\test_news_business_logic.py tests\\services\\test_unified_news_business_logic.py -q`.
4. Run the full pytest suite before merging the broader filter change.

## Expected impact

- Estimated LOC reduction: none; this is an API-control improvement.
- Complexity reduction: medium if filtering is centralized in the unified news service.
- Maintenance benefit: medium, by making response size and relevance easier to control.
