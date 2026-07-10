# Known Limitations and Documentation Gaps

This page collects practical caveats that are easy to miss when you are new to mtdata. The goal is to help you choose the right tool path and avoid surprises.

## Operational Limits

| Area | What to Know | Recommended Practice |
|------|--------------|----------------------|
| MT5 platform | The MetaTrader 5 Python package is Windows-only. | Run mtdata on Windows, or connect to a Windows host from another machine. |
| Live trading | `trade_*` commands operate on the MT5 account currently logged into the terminal. | Use a demo account first and preview supported actions with `--dry-run true`. |
| Web API coverage | The Web API exposes a focused subset of the CLI/MCP tools. | Use `mtdata-cli` or MCP for tools not listed in [WEB_API.md](WEB_API.md). |
| Market depth | `market_depth_fetch` is disabled unless explicitly enabled and requires broker DOM data. | Set `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` only when your broker supports it. |
| Options chains | Yahoo Finance options endpoints may reject unauthenticated requests. mtdata can retry Yahoo when Tradier is unavailable, but the fallback is still best-effort only. | Prefer Tradier via `MTDATA_OPTIONS_PROVIDER=tradier` and `MTDATA_OPTIONS_API_KEY`, then use `options_provider_status` to confirm the effective provider. Use `options_barrier_price` for local QuantLib pricing when live chains are unavailable. |
| Forecast methods | Method availability depends on installed optional dependencies. Defaults can differ by method. | Check `forecast_list_methods --json`, see [forecast/METHODS.md](forecast/METHODS.md), and set important `--params` explicitly. |
| Timestamps | MT5 broker server time, UTC, client-local time, and external provider time can differ. | Configure `MT5_SERVER_TZ` or `MT5_TIME_OFFSET_MINUTES` before analysis; see [TIMESTAMPS.md](TIMESTAMPS.md) and keep the `timezone` field with saved results. |

## Reference Coverage

Previously-tracked documentation gaps now have dedicated references:

- Response envelope with `detail`/`extras`, pagination, and error codes → [OUTPUT.md](OUTPUT.md).
- Per-forecast-method default parameters, libraries, and dependencies → [forecast/METHODS.md](forecast/METHODS.md).
- Unified timestamp policy covering MT5, client-local, and external providers → [TIMESTAMPS.md](TIMESTAMPS.md).
- Trading safety runbook for `trade_place`, `trade_modify`, `trade_close`, guardrails, and broker behavior → [TRADING_SAFETY.md](TRADING_SAFETY.md).
- Running the MCP server or Web API as a persistent local service → [DEPLOYMENT.md](DEPLOYMENT.md).

## If You Are Unsure

Start with read-only commands:

```bash
mtdata-cli symbols_list --limit 10
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 50 --json
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

Move to trading commands only after the account, symbol, timezone, and risk controls are clear.
