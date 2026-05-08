# Known Limitations and Documentation Gaps

This page collects practical caveats that are easy to miss when you are new to mtdata. The goal is to help you choose the right tool path and avoid surprises.

## Operational Limits

| Area | What to Know | Recommended Practice |
|------|--------------|----------------------|
| MT5 platform | The MetaTrader 5 Python package is Windows-only. | Run mtdata on Windows, or connect to a Windows host from another machine. |
| Live trading | `trade_*` commands operate on the MT5 account currently logged into the terminal. | Use a demo account first and preview supported actions with `--dry-run true`. |
| Web API coverage | The Web API exposes a focused subset of the CLI/MCP tools. | Use `mtdata-cli` or MCP for tools not listed in [WEB_API.md](WEB_API.md). |
| Market depth | `market_depth_fetch` is disabled unless explicitly enabled and requires broker DOM data. | Set `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` only when your broker supports it. |
| Options chains | Yahoo Finance options endpoints may reject unauthenticated requests. | Treat options-chain tools as best-effort external data; use `options_barrier_price` for local QuantLib pricing. |
| Forecast methods | Method availability depends on installed optional dependencies. Defaults can differ by method. | Check `forecast_list_methods --json` and set important `--params` explicitly. |
| Timestamps | MT5 broker server time, UTC, client-local time, and external provider time can differ. | Configure `MT5_SERVER_TZ` or `MT5_TIME_OFFSET_MINUTES` before analysis and keep timezone metadata with saved results. |

## Documentation Gaps to Close

- Per-tool response schemas with compact, standard, full, and `extras` examples.
- Per-forecast-method default parameter tables and reproducibility guidance.
- A unified timestamp policy covering MT5, Web API, CLI output, and external providers.
- A dedicated trading safety runbook for `trade_place`, `trade_modify`, `trade_close`, guardrails, and broker-specific behavior.
- Deployment guidance for running MCP/Web API as a persistent local service.

## If You Are Unsure

Start with read-only commands:

```bash
mtdata-cli symbols_list --limit 10
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 50 --json
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

Move to trading commands only after the account, symbol, timezone, and risk controls are clear.
