# trade_account_info verbosity and request shape

## Source report

- Original report: `code-reports/trade_account_info_dual_verbosity.md`
- Original report: `code-reports/trade-account-info-outlier.md`

## Status

Deferred for larger refactor.

## Validated issue

`trade_account_info` is an outlier among trading tools because it uses direct keyword parameters and exposes both a multi-value `detail` parameter and a `verbose` boolean. The `verbose` flag is merged into output-contract resolution, which can override compact/summary shaping.

## Evidence

- `src/mtdata/core/trading/account.py::trade_account_info` defines `detail: Literal["summary", "compact", "basic", "full"] = "full"` and `verbose: bool = False`.
- The implementation calls `resolve_output_contract(..., verbose=verbose, default_detail="full")`.
- `output_mode` is derived from `contract.shape_detail`, so `verbose=True` can change the effective output shape.
- Other trading tools such as `trade_get_open` use request models instead of flat keyword parameters.
- `tests/trading/test_trading_account_business_logic.py` covers summary/basic/full output behavior and execution diagnostics.
- `tests/trading/test_trading_coverage.py` and CLI coverage exercise current account-info behavior and discovery.

## Why this should not be fixed inline

Removing `verbose`, changing the default detail level, or switching to a Pydantic request model would change a public trading tool schema and generated CLI behavior. Folding verbose diagnostics into `detail="full"` also changes current interactions between `detail` and `verbose`, so it requires a migration decision and broader tests.

## Recommended approach

Decide the target account-info contract first. Prefer a single request model with one output-control field. If `verbose` is deprecated, fold all diagnostic fields into `detail="full"` and remove the boolean after updating docs, tests, and CLI/schema expectations. Consider changing the default only if existing script compatibility is explicitly not required.

## Scope

- Files:
  - `src/mtdata/core/trading/account.py`
  - `src/mtdata/core/trading/requests.py` if a request model is introduced
  - `tests/trading/test_trading_account_business_logic.py`
  - `tests/trading/test_trading_coverage.py`
  - CLI/schema tests covering trading tools
- Symbols:
  - `trade_account_info`
  - `_trade_account_payload_for_mode`
  - `resolve_output_contract`
  - potential `TradeAccountInfoRequest`
- Tests:
  - Account-info output mode tests
  - CLI/tool schema discovery tests
  - Trading tool registry tests
- Docs/config/CLI/API affected:
  - MCP schema for `trade_account_info`
  - generated CLI help and command examples
  - trading account documentation/examples, if present

## Risks

- Existing clients may rely on `verbose=True`.
- Existing clients may rely on the current `detail="full"` default.
- Request-model migration may alter generated schema shape.
- Changing diagnostic inclusion can break scripts that inspect execution readiness fields.

## Verification plan

- Add tests for the final target schema and output modes.
- Run `python -m pytest tests/trading/test_trading_account_business_logic.py tests/trading/test_trading_coverage.py`.
- Run relevant CLI coverage tests for `trade_account_info`.
- Inspect generated CLI help and MCP schema for the intended single output-control path.

## Expected impact

- Estimated LOC reduction: small after `verbose` and any migration glue are removed.
- Complexity reduction: medium by removing dual verbosity resolution for account info.
- Maintenance benefit: consistent trading-tool request shape and clearer account output modes.
