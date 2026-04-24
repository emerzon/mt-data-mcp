# Market ticker daily context

## Source report

- Original report: `code-reports/market-ticker-missing-fields.md`

## Status

Deferred for larger refactor.

## Validated issue

`market_ticker` lacked some basic ticker fields. The safe subset was implemented inline: compact output now keeps `last`, and the tool exposes `tick_volume` from `symbol_info_tick().volume` when available.

The remaining requested fields require broader design work: daily high/low naming, daily change/change percent calculation, and a plural ticker-board variant.

## Evidence

- `src/mtdata/core/market_depth.py` already reads `tick.last` but compact mode previously dropped `last`.
- `symbol_info_tick()` fixtures in `tests/trading/test_market_depth_business_logic.py` include `volume`, confirming the field is available in the current gateway shape.
- `market_ticker` already calls `symbol_info()` for digits/point/tick value, so bid/ask daily bound fields may be available from the same object, but the public output shape for those bounds is not currently defined.
- Change and change percent require an additional reference price source, such as prior close or prior tick, which is not part of the current tool contract.

## Why this should not be fixed inline

Daily bounds and change fields need semantics that are not obvious from the existing code. MT5 exposes bid and ask high/low fields separately, while the report asks for generic `daily_high` and `daily_low`; choosing bid, ask, midpoint, or last-trade semantics changes the meaning of the output. Daily change/change percent would require additional historical data access and error handling. A plural `market_tickers` tool changes the public API surface.

## Recommended approach

1. Define ticker field semantics explicitly:
   - Whether daily high/low should use bid, ask, midpoint, or last-trade data.
   - Whether change/change percent should compare to prior close, session open, or previous tick.
2. Add the required MT5 data access in one place with clear fallbacks and explicit unavailable-field behavior.
3. Consider a separate `market_tickers` design that accepts multiple symbols and returns a compact tabular payload.
4. Update docs and tests for compact and full output contracts.

## Scope

- Files:
  - `src/mtdata/core/market_depth.py`
  - `tests/trading/test_market_depth_business_logic.py`
  - CLI/output tests that mention `market_ticker`
- Symbols:
  - `market_ticker`
  - `_compact_market_ticker_payload`
  - MT5 gateway methods for tick, symbol, and historical rate access
- Tests:
  - Compact/full ticker output tests
  - Error handling for unavailable historical data
  - Optional multi-symbol ticker output tests
- Docs/config/CLI/API affected:
  - MCP tool schema and generated CLI help if a plural tool is added
  - User-facing examples for ticker output

## Risks

- Ambiguous daily high/low semantics could mislead users.
- Additional historical data calls may slow the lightweight ticker path.
- Adding a plural ticker tool expands the API surface and needs naming/compatibility review.
- Compact output field additions may affect consumers expecting a minimal field set.

## Verification plan

1. Add tests for daily high/low semantics using symbol_info bid/ask bounds.
2. Add tests for change/change percent with mocked historical prior-close data and unavailable-data behavior.
3. Add compact/full output contract tests for any new fields.
4. Run `python -m pytest tests\\trading\\test_market_depth_business_logic.py tests\\core\\test_cli_coverage.py -q`.
5. Run the full pytest suite before merging the broader ticker refactor.

## Expected impact

- Estimated LOC reduction: none; this is an output completeness and API clarity refactor.
- Complexity reduction: medium if ticker field semantics are centralized.
- Maintenance benefit: medium, especially if future ticker-board behavior reuses the same snapshot builder.
