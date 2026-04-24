# trade-prefixed VaR/CVaR tool name

## Source report

- Original report: `code-reports/var_cvar_calculate_missing_trade_prefix.md`

## Status

Deferred for larger refactor.

## Validated issue

`var_cvar_calculate` is implemented in the trading package and uses trading runtime infrastructure, but its public tool name does not follow the `trade_` prefix used by the rest of the trading tool family.

## Evidence

- `src/mtdata/core/trading/risk.py` defines `var_cvar_calculate(request: TradeVarCvarRequest)`.
- `src/mtdata/core/trading/risk.py` logs and executes the operation as `var_cvar_calculate`.
- `src/mtdata/core/trading/use_cases.py::run_trade_var_cvar_calculate` also logs operation names as `var_cvar_calculate`.
- `src/mtdata/core/trading/__init__.py` exports `var_cvar_calculate`.
- `README.md`, `docs/CLI.md`, and `tests/core/test_tool_registry.py` reference `var_cvar_calculate` as the public tool name.
- The request and use-case names already include `TradeVarCvarRequest` and `run_trade_var_cvar_calculate`, showing the domain mismatch is limited to the public tool name.

## Why this should not be fixed inline

Renaming the function/tool to `trade_var_cvar_calculate` changes the public MCP tool name and generated CLI command. Adding a deprecated alias would temporarily increase tool surface area rather than simplify it. This needs an explicit API migration decision.

## Recommended approach

Choose a migration policy for tool renames. If compatibility is not required, rename the tool and update docs/tests. If compatibility is required, introduce a time-boxed alias with clear deprecation messaging and a plan to remove `var_cvar_calculate` after downstream clients migrate.

## Scope

- Files:
  - `src/mtdata/core/trading/risk.py`
  - `src/mtdata/core/trading/use_cases.py`
  - `src/mtdata/core/trading/__init__.py`
  - `README.md`
  - `docs/CLI.md`
  - `tests/core/test_tool_registry.py`
  - Trading and CLI tests for tool discovery
- Symbols:
  - `var_cvar_calculate`
  - `trade_var_cvar_calculate`
  - `TradeVarCvarRequest`
  - `run_trade_var_cvar_calculate`
- Tests:
  - Tool registry tests
  - CLI discovery/help tests
  - Trading VaR/CVaR business logic tests
- Docs/config/CLI/API affected:
  - MCP tool name
  - generated CLI command
  - README and CLI docs

## Risks

- Existing MCP clients and scripts may call `var_cvar_calculate`.
- Generated CLI command names would change.
- Keeping both names during migration increases the active tool surface temporarily.

## Verification plan

- Run `python -m pytest tests/core/test_tool_registry.py tests/trading/test_trading_var_cvar_business_logic.py`.
- Run relevant CLI discovery tests that enumerate commands.
- Inspect `mtdata-cli --help` or generated command metadata for the intended public name.

## Expected impact

- Estimated LOC reduction: small after any migration alias is removed.
- Complexity reduction: low, mainly naming consistency.
- Maintenance benefit: clearer trading tool grouping and easier discovery.
