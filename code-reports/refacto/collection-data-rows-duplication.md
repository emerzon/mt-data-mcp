# Collection data and rows duplication

## Source report

- Original report: `code-reports/03-symbols_list-duplicate-data-rows.md`
- Related source report: `code-reports/01-data-duplication-in-outputs.md`
- Related source report: `code-reports/05-collection-metadata-noise.md`

## Status

Deferred for larger refactor.

## Validated issue

Several collection-style tools return both legacy `data` and normalized `rows` fields containing the same tabular records. This increases payload size and can confuse users, but the duplication is currently an explicit compatibility behavior in the shared collection contract helper.

A broader follow-up report validates that the same compatibility pattern appears across multiple collection-style tools, including `data_fetch_candles` (`data`/`series`), symbols and market-scan outputs (`data`/`rows`), causal summaries (`data.items`/highlight subsets), and trade history (`items`/`normalized_items`).

Another related report validates that collection contract metadata (`collection_kind`, `collection_contract_version`) and per-tool diagnostics can add noise to compact outputs. Whether those fields should remain in compact mode is part of the same versioned output-contract decision.

## Evidence

- `src/mtdata/core/output_contract.py::attach_collection_contract` documents that it adds normalized collection fields while preserving legacy payload shape.
- `attach_collection_contract` uses `out.setdefault("rows", rows)` without removing existing `data`.
- `tests/core/test_output_contract.py::test_attach_collection_contract_adds_rows_without_replacing_legacy_data` asserts that both `data` and `rows` remain present.
- `src/mtdata/core/symbols.py::symbols_list`, `_list_symbol_groups`, and `_market_scan_table` pass `rows=result.get("data")` after building legacy tabular payloads.
- Other modules also call `attach_collection_contract`, so changing the helper would affect more than the symbols tools named in the source report.
- `data_fetch_candles` and other time-series outputs can expose both legacy `data` and normalized `series` keys through the same collection contract.
- Some related duplication, such as causal highlights and trade history normalized rows, is implemented outside `attach_collection_contract` but should be handled by the same output-contract migration policy.
- `src/mtdata/core/output_contract.py::apply_output_verbosity` strips some diagnostic keys in compact mode but does not strip collection contract keys or tool-specific row diagnostics.

## Why this should not be fixed inline

Removing either key changes a shared output contract and may break clients that still consume legacy `data` or normalized `rows`. The helper is used across multiple domains, so this requires an API migration rather than a symbols-only edit.

## Recommended approach

Define a versioned collection-output migration. Prefer `rows` as the canonical table field for collection.v2, then decide whether `data` should be removed entirely, omitted only in compact/detail modes, or retained behind a compatibility option. Update all `attach_collection_contract` call sites consistently instead of special-casing symbols.

For non-collection-contract duplication such as causal highlights and trade-history normalized rows, align the compact/full response policy with the same migration rather than special-casing each tool independently.

Also define which protocol metadata belongs in compact output. If collection metadata is needed for machine consumers, consider moving it under `meta` or retaining it only in a full/detail contract version.

## Scope

- Files:
  - `src/mtdata/core/output_contract.py`
  - `src/mtdata/core/symbols.py`
  - `src/mtdata/core/data/use_cases.py`
  - `src/mtdata/forecast/use_cases.py`
  - `tests/core/test_output_contract.py`
  - Symbols/data/forecast tests for collection payloads
- Symbols:
  - `attach_collection_contract`
  - `symbols_list`
  - `_list_symbol_groups`
  - `_market_scan_table`
- Tests:
  - Shared output contract tests
  - Symbols list and market scan tests
  - Compact/full verbosity tests for collection metadata
  - Any tests that assert `data` or `rows` presence
- Docs/config/CLI/API affected:
  - MCP responses for collection-style tools
  - CLI rendering for table outputs
  - documentation examples that show `data` or `rows`

## Risks

- Existing clients may read only `data`.
- Newer clients may read only `rows`.
- Removing duplication without versioning could silently break automation.
- Detail-mode gating could make output shape depend on verbosity in surprising ways.
- Removing metadata from compact mode could break generic clients that use collection fields for extraction.

## Verification plan

- Add contract tests for the chosen collection version and compatibility behavior.
- Run `python -m pytest tests/core/test_output_contract.py tests/utils/test_symbols_coverage.py tests/utils/test_symbols_market_scan_coverage.py`.
- Run focused tests for other modules that call `attach_collection_contract`.
- Inspect representative JSON and CLI output for collection tools.

## Expected impact

- Estimated LOC reduction: small to medium depending on migration strategy.
- Complexity reduction: medium by making collection payloads predictable.
- Maintenance benefit: smaller responses and a clearer output contract for list/table tools.
