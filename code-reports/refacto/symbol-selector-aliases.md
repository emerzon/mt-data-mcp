# Symbol selector aliases

## Source report

- Original report: `code-reports/symbol-vs-symbols-dual-param.md`
- Original report: `code-reports/causal_symbol_symbols_dual_param.md`
- Original report: `code-reports/03-symbol-vs-symbols-param-confusion.md`

## Status

Deferred for larger refactor.

## Validated issue

Several multi-symbol analysis and scanning tools expose both `symbol` and `symbols` parameters for the same selector concept. Current code no longer silently lets one conflicting value win: shared validation rejects conflicting aliases unless both resolve to the same selector set. The schema still exposes both parameters, which keeps the ambiguity visible to users.

## Evidence

- `src/mtdata/core/causal.py::causal_discover_signals`, `correlation_matrix`, and `cointegration_test` define both `symbols` and `symbol`.
- `src/mtdata/core/symbols.py::market_scan` defines both `symbols` and `symbol`.
- `src/mtdata/shared/parameter_contracts.py::normalize_symbol_selector_aliases` parses both aliases and rejects conflicting selector sets.
- `tests/forecast/test_causal_helpers.py` asserts that `symbol` populates the plural contract and that conflicting aliases are rejected.
- Causal tool docstrings describe `symbol` as a compatibility alias for `symbols`, but the generated MCP schema still presents both as independent optional fields.

## Why this should not be fixed inline

Removing either alias changes the public schema and generated CLI arguments for multiple tools. Keeping one hidden compatibility alias is not straightforward with current flat function signatures and would need a request-model or schema-generation migration.

## Recommended approach

Choose a single canonical selector parameter for multi-symbol tools. Prefer `symbols` for tools that accept comma-separated lists, while documenting single-symbol auto-expansion where applicable. If compatibility is required, migrate affected tools to request models or another schema mechanism that exposes only the canonical selector while accepting legacy aliases internally.

## Scope

- Files:
  - `src/mtdata/core/causal.py`
  - `src/mtdata/core/symbols.py`
  - `src/mtdata/shared/parameter_contracts.py`
  - `tests/forecast/test_causal_helpers.py`
  - causal and market-scan tests
  - CLI/schema tests for affected tools
- Symbols:
  - `normalize_symbol_selector_aliases`
  - `causal_discover_signals`
  - `correlation_matrix`
  - `cointegration_test`
  - `market_scan`
- Tests:
  - Symbol selector alias tests
  - Causal/correlation/cointegration tests
  - Market scan selector tests
  - CLI/schema tests for generated parameters
- Docs/config/CLI/API affected:
  - MCP schemas for affected tools
  - generated CLI help and positional argument behavior
  - docs/examples describing symbol selectors

## Risks

- Existing clients may call the compatibility alias.
- Single-symbol auto-expansion semantics differ by tool and must remain explicit.
- `group` is another selector path on some tools, so validation messages and precedence rules must remain clear.
- Request-model migration may alter generated CLI behavior.

## Verification plan

- Add schema/help tests proving only the canonical selector appears publicly.
- Preserve or explicitly reject legacy alias behavior according to the migration decision.
- Run causal helper tests, causal business logic tests, market scan tests, and relevant CLI coverage.
- Manually inspect generated schemas/help for affected tools.

## Expected impact

- Estimated LOC reduction: small after alias compatibility is removed.
- Complexity reduction: medium by eliminating duplicate selector parameters and conflict-resolution paths.
- Maintenance benefit: simpler multi-symbol tool contracts and clearer user-facing schemas.
