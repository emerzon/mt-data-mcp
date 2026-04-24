# CLI symbol positional consistency

## Source report

- Original report: `code-reports/positional-argument-inconsistency.md`

## Status

Deferred for larger refactor.

## Validated issue

CLI handling for `symbol` is inconsistent between direct function tools and request-model tools. Some commands support `symbol` as a first positional argument, while request-model-based trading tools such as `trade_risk_analyze` expose it only through named request fields.

## Evidence

- `src/mtdata/core/cli/api.py` contains custom positional-symbol handling for some commands, such as `forecast_generate`.
- `trade_risk_analyze` is implemented as a request-model MCP tool via `TradeRiskAnalyzeRequest`, so the dynamic CLI exposes model fields as options rather than adding a positional `symbol`.
- CLI examples currently document `trade_risk_analyze --symbol BTCUSD ...`, which confirms named-symbol usage is the supported behavior today.
- `trade_modify` is also called out as inconsistent, showing this is not isolated to a single command.

## Why this should not be fixed inline

Adding one special positional parser path for `trade_risk_analyze` would increase CLI inconsistency elsewhere. A robust fix should define how request-model tools expose one or more positional fields, how that interacts with existing named flags, and how conflicts are reported.

## Recommended approach

1. Define a CLI convention for request-model tools that have a primary `symbol` field.
2. Add a reusable dynamic-CLI mechanism for optional first positional `symbol` where safe.
3. Preserve `--symbol` as a named alias.
4. Add conflict validation when both positional symbol and `--symbol` are supplied with different values.
5. Apply the rule consistently to `trade_risk_analyze`, `trade_modify` if appropriate, and future request-model tools.

## Scope

- Files:
  - `src/mtdata/core/cli/api.py`
  - CLI parsing/runtime helpers
  - CLI tests
  - Trading request-model definitions
- Symbols:
  - `TradeRiskAnalyzeRequest`
  - `trade_risk_analyze`
  - dynamic CLI argument builder
  - command example generation
- Tests:
  - Positional and named symbol parsing
  - Conflict handling
  - Existing request-model CLI behavior
  - Help output/examples
- Docs/config/CLI/API affected:
  - CLI examples for trading tools
  - Generated help text

## Risks

- Ambiguity if a request model has multiple plausible positional fields.
- Breaking scripts if positional parsing consumes values intended for other options.
- Special-casing one tool would make the CLI harder to maintain.

## Verification plan

1. Add parser tests for `trade_risk_analyze EURUSD ...` and `trade_risk_analyze --symbol EURUSD ...`.
2. Add conflict tests for positional and named symbol disagreement.
3. Run CLI coverage tests.
4. Run the full pytest suite before merging the parser change.

## Expected impact

- Estimated LOC reduction: low.
- Complexity reduction: medium if implemented as a shared request-model positional convention.
- Maintenance benefit: medium, by making CLI symbol handling predictable.
