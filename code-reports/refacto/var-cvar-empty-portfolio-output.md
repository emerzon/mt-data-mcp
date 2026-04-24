# VaR/CVaR empty portfolio output

## Source report

- Original report: `code-reports/09-var-cvar-empty-position-bloat.md`

## Status

Deferred for larger refactor.

## Validated issue

`var_cvar_calculate` returns a full zero-filled risk summary plus empty arrays when there are no open positions. This is verbose for the no-action case where no VaR/CVaR calculation can be performed.

## Evidence

- `src/mtdata/core/trading/use_cases.py::run_trade_var_cvar_calculate` builds a `summary` containing method/configuration fields, zero observations, zero positions, zero symbols, zero notional, zero VaR/CVaR, and optional equity/currency when `positions_get()` returns no positions.
- The same branch returns empty `symbol_exposures`, `positions`, and `worst_observations` arrays.
- `tests/trading/test_trading_var_cvar_business_logic.py::test_run_trade_var_cvar_calculate_returns_no_action_when_no_open_positions` asserts the current zero-filled summary shape, so the verbose response is covered as current behavior.
- `TradeVarCvarRequest` in `src/mtdata/core/trading/requests.py` has no `detail` field that could preserve the existing verbose shape behind an explicit full-detail mode.

## Why this should not be fixed inline

Removing the zero-filled summary would change a tested public response shape. The report's suggested compatibility path depends on a `detail=full` option that does not currently exist for `TradeVarCvarRequest`, so a safe migration requires request-schema design and response-contract tests rather than a small deletion.

## Recommended approach

1. Add an explicit `detail` field to `TradeVarCvarRequest` if the project wants compact/full behavior for this tool.
2. Define compact no-action output as a small top-level payload with `message`, `no_action`, and optional `equity`/`currency`.
3. Keep the existing zero-filled `summary` only for `detail="full"` or for a documented compatibility period.
4. Apply the same policy to related no-position portfolio-risk output only after validating its consumers and tests.

## Scope

- Files:
  - `src/mtdata/core/trading/requests.py`
  - `src/mtdata/core/trading/use_cases.py`
  - `tests/trading/test_trading_var_cvar_business_logic.py`
- Symbols:
  - `TradeVarCvarRequest`
  - `run_trade_var_cvar_calculate`
  - no-position branch in VaR/CVaR calculation
- Tests:
  - No-position compact output
  - No-position full output
  - Existing open-position VaR/CVaR calculations
- Docs/config/CLI/API affected:
  - MCP request schema and generated CLI options if `detail` is added
  - Response examples for `var_cvar_calculate`

## Risks

- Breaking consumers that read zero values from `summary` even when `no_action` is true.
- Expanding the tool schema with a new `detail` parameter.
- Inconsistency with other trading tools if no-action compactness is handled differently.

## Verification plan

1. Add request-schema tests for `detail`, if introduced.
2. Add compact/full no-position output tests.
3. Run `python -m pytest tests\\trading\\test_trading_var_cvar_business_logic.py -q`.
4. Run related trading risk tests and the full pytest suite before merging.

## Expected impact

- Estimated LOC reduction: low to medium in compact outputs.
- Complexity reduction: low; the main benefit is reduced no-action output noise.
- Maintenance benefit: medium, because no-action response contracts become more intentional.
