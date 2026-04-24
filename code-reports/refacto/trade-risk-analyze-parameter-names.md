# trade_risk_analyze parameter names

## Source report

- Original report: `code-reports/risk-analyze-parameter-naming.md`

## Status

Deferred for larger refactor.

## Validated issue

`trade_risk_analyze` still exposes `proposed_entry`, `proposed_sl`, and `proposed_tp` as its canonical request field names, while related trading tools expose the same concepts as `price`, `stop_loss`, and `take_profit`. The risk tool accepts `entry`, `stop_loss`, and `take_profit` as validation aliases, but MCP/tool schema generation is still driven by the model field names.

## Evidence

- `src/mtdata/core/trading/requests.py` defines `TradeRiskAnalyzeRequest.proposed_entry`, `proposed_sl`, and `proposed_tp`.
- `src/mtdata/core/trading/requests.py` defines `TradePlaceRequest.stop_loss` and `take_profit` with `sl`/`tp` aliases.
- `src/mtdata/core/trading/use_cases.py` reads `request.proposed_entry`, `request.proposed_sl`, and `request.proposed_tp` throughout the risk analysis path.
- `tests/trading/test_trading_risk_business_logic.py` and `tests/trading/test_trading_coverage.py` construct `TradeRiskAnalyzeRequest` with the current `proposed_*` names.

## Why this should not be fixed inline

Renaming the Pydantic fields would change the public tool schema and could affect MCP clients, CLI argument generation, tests, and downstream users that submit the current canonical names. A safe implementation needs an explicit compatibility and migration strategy rather than a small deletion-only cleanup.

## Recommended approach

Rename the request model fields to the trader-facing names that should appear in schema output, while preserving compatibility for existing `proposed_*` inputs through validation aliases if that compatibility is still required. Update internal use-case code to consume the new field names directly, then update tests and docs so only intentional legacy paths remain.

## Scope

- Files:
  - `src/mtdata/core/trading/requests.py`
  - `src/mtdata/core/trading/use_cases.py`
  - `src/mtdata/core/trading/risk.py`
  - `tests/trading/test_trading_risk_business_logic.py`
  - `tests/trading/test_trading_coverage.py`
- Symbols:
  - `TradeRiskAnalyzeRequest`
  - `run_trade_risk_analyze`
  - risk-analysis validation and guidance fields
- Tests:
  - Trading risk request validation and business logic tests
  - CLI/schema generation tests for `trade_risk_analyze`
- Docs/config/CLI/API affected:
  - MCP tool schema
  - generated CLI arguments
  - user-facing risk-analysis examples

## Risks

- Existing clients may still send `proposed_entry`, `proposed_sl`, or `proposed_tp`.
- Changing canonical field names may alter generated CLI options and schema snapshots.
- Internal guidance and error payloads currently mention `proposed_*`; changing them may affect tests or user workflows.

## Verification plan

- Add request-model tests proving both canonical and intended legacy input names work during migration.
- Run `python -m pytest tests/trading/test_trading_risk_business_logic.py tests/trading/test_trading_coverage.py`.
- Run relevant CLI/schema generation tests that cover `trade_risk_analyze`.
- Inspect generated schema or CLI help to confirm the canonical names are the intended trader-facing names.

## Expected impact

- Estimated LOC reduction: small after migration-only aliases can be removed.
- Complexity reduction: medium, because related trading tools would share parameter names for equivalent concepts.
- Maintenance benefit: lower support burden and fewer schema-level surprises for trading workflows.
