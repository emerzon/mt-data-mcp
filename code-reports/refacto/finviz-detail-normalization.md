# Finviz detail and field normalization

## Source report

- Original report: `code-reports/finviz-tools-missing-detail.md`
- Original report: `code-reports/11-finviz-no-detail-raw-fields.md`

## Status

Deferred for larger refactor.

## Validated issue

The Finviz MCP tools do not expose a `detail` parameter and largely return Finviz-shaped payloads. Some payloads have targeted normalization, but there is no consistent compact/full output contract or comprehensive snake_case field mapping across the domain.

## Evidence

- `src/mtdata/core/finviz.py` defines public tools such as `finviz_fundamentals`, `finviz_description`, `finviz_news`, `finviz_insider`, `finviz_ratings`, `finviz_peers`, `finviz_screen`, `finviz_market_news`, `finviz_insider_activity`, `finviz_forex`, `finviz_crypto`, `finviz_futures`, `finviz_calendar`, and `finviz_earnings`.
- These tool signatures do not include `detail`.
- `finviz_screen` has `limit`, `page`, and `view`, but those control result count/page and Finviz view, not compact/full verbosity.
- The module has partial normalization helpers such as `_normalize_finviz_news_payload`, showing normalization exists only for selected payload shapes.
- Raw upstream field names remain part of several outputs, especially fundamentals and screener data.

## Why this should not be fixed inline

Adding `detail` and normalizing fields across the whole Finviz domain changes public schemas and output keys for many tools at once. A safe implementation requires per-tool field mapping, compact-field selection, backward compatibility decisions for raw payloads, and broad tests.

## Recommended approach

Create a Finviz output contract with compact and full modes. For compact mode, select trader-relevant fields per tool and expose normalized snake_case keys. For full mode, preserve raw upstream fields under an explicit raw section or documented compatibility key. Migrate one Finviz tool group at a time.

## Scope

- Files:
  - `src/mtdata/core/finviz.py`
  - `src/mtdata/services/finviz/`
  - Finviz tests under `tests/`
  - Finviz documentation/examples
- Symbols:
  - All `finviz_*` MCP tools
  - Finviz payload normalization helpers
  - `_build_tool_contract_meta`
- Tests:
  - Per-tool compact/full response shape tests
  - Field-name normalization tests
  - Pagination and limit tests for list-like Finviz tools
  - CLI/schema generation tests
- Docs/config/CLI/API affected:
  - MCP schemas for Finviz tools
  - generated CLI help
  - examples showing Finviz output keys

## Risks

- Existing clients may parse raw Finviz field names.
- Upstream Finviz field names and formats may change.
- Compact field selection is product/domain-sensitive and needs explicit agreement.
- Parsing unit-suffixed strings into structured values can introduce conversion edge cases.

## Verification plan

- Define field maps and compact field sets per tool.
- Add tests with representative raw Finviz payloads for each normalized output.
- Run all Finviz-related tests and CLI/schema tests.
- Manually inspect representative compact and full outputs for fundamentals, screen, ratings, and news.

## Expected impact

- Estimated LOC reduction: neutral initially; later reduced caller-side complexity.
- Complexity reduction: medium for users, although implementation adds mapping logic.
- Maintenance benefit: consistent Finviz output contract and easier scripted consumption.
