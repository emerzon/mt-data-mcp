# SL/TP parameter unification

## Source report

- Original report: `code-reports/parameter-naming-inconsistency.md`

## Status

Deferred for larger refactor.

## Validated issue

Stop-loss and take-profit concepts use different parameter names and units across forecasting, labeling, optimization, and trading tools. Some aliasing has already been added in parts of the trading request layer, but the broader toolkit does not have one consistent naming and unit-conversion policy.

## Evidence

- Barrier utilities and forecast/label tools use names such as `tp_abs`, `sl_abs`, `tp_pct`, `sl_pct`, `tp_ticks`, and legacy `tp_pips`/`sl_pips`.
- Trading tools use execution-oriented names such as `stop_loss` and `take_profit`.
- `TradeRiskAnalyzeRequest` already supports aliases for related concepts, showing that Pydantic aliases are the preferred compatibility pattern for request models.
- Adding percentage/pip support to real trade execution tools is not just naming; it requires symbol precision, point/pip conventions, entry-side semantics, and validation to avoid unsafe orders.

## Why this should not be fixed inline

This affects public tool schemas and real trading behavior. A simple alias expansion could cause dangerous ambiguity, such as treating `0.5` as an absolute price in one tool and a percentage in another. Conversion from pips/percentages to absolute prices requires account/symbol context and must be validated carefully before order placement.

## Recommended approach

1. Define canonical SL/TP concepts and units:
   - absolute price,
   - percentage distance,
   - point/tick/pip distance.
2. Add explicit aliases with Pydantic `validation_alias` where the target semantics are identical.
3. Add conversion helpers for percentage/pip values only after defining direction, entry, symbol point, digits, and broker distance rules.
4. For execution tools, reject ambiguous values with clear errors instead of silently guessing units.
5. Document accepted aliases and units in every affected tool's help text.

## Scope

- Files:
  - `src/mtdata/core/trading/requests.py`
  - `src/mtdata/core/trading/use_cases.py`
  - `src/mtdata/forecast/use_cases.py`
  - `src/mtdata/utils/barriers.py`
  - Labeling/barrier request models and tests
- Symbols:
  - Trade request models for placement, modification, and risk analysis
  - Forecast barrier probability and optimization request paths
  - `labels_triple_barrier`
  - shared barrier normalization helpers
- Tests:
  - Alias acceptance tests
  - Ambiguous-unit rejection tests
  - Symbol-aware pip/percentage conversion tests
  - Real-order dry-run safety tests
- Docs/config/CLI/API affected:
  - MCP schemas and generated CLI flags
  - Trading safety documentation
  - Forecast/labeling examples

## Risks

- Misinterpreting SL/TP units in real trading tools can place unsafe orders.
- Adding many aliases can make generated schemas noisy.
- Silent conversion can hide user mistakes unless validation and messaging are strict.
- Conflicting aliases may be supplied together and need deterministic error handling.

## Verification plan

1. Add unit tests for normalization conflicts and aliases.
2. Add dry-run trade placement tests for absolute, percentage, and pip inputs if those units are supported.
3. Add forecast/labeling tests confirming backward-compatible legacy names.
4. Run trading, forecast barrier, and labeling test subsets.
5. Run the full pytest suite before merging.

## Expected impact

- Estimated LOC reduction: low; the main benefit is consistency and safer UX.
- Complexity reduction: medium if shared normalization replaces per-tool ad hoc names.
- Maintenance benefit: high because SL/TP semantics become explicit across analysis and execution tools.
