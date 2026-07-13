# Known limitations

Practical caveats that are easy to miss when you are new. Read this before deep integrations so you pick the right path and avoid surprises.

**Related:** [Setup](SETUP.md) · [Forecast methods](forecast/METHODS.md) · [Timestamps](TIMESTAMPS.md) · [Trading safety](TRADING_SAFETY.md)

## Operational limits

| Area | What to Know | Recommended Practice |
|------|--------------|----------------------|
| MT5 platform | The MetaTrader 5 Python package is Windows-only. | Run mtdata on Windows, or connect to a Windows host from another machine. |
| Live trading | `trade_*` commands operate on the MT5 account currently logged into the terminal. | Use a demo account first and preview supported actions with `--dry-run true`. |
| Web API coverage | The Web API exposes a focused subset of the CLI/MCP tools. | Use `mtdata-cli` or MCP for tools not listed in [WEB_API.md](WEB_API.md). |
| Market depth | `market_depth_fetch` is disabled unless explicitly enabled and requires broker DOM data. | Set `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` only when your broker supports it. |
| Options chains | Yahoo Finance options endpoints may reject unauthenticated requests. mtdata can retry Yahoo when Tradier is unavailable, but the fallback is still best-effort only. | Prefer Tradier via `MTDATA_OPTIONS_PROVIDER=tradier` and `MTDATA_OPTIONS_API_KEY`, then use `options_provider_status` to confirm the effective provider. Use `options_barrier_price` for local QuantLib pricing when live chains are unavailable. |
| Forecast methods | Method availability depends on installed optional dependencies. Defaults can differ by method. | Check `forecast_list_methods --json`, see [forecast/METHODS.md](forecast/METHODS.md), and set important `--params` explicitly. |
| Timestamps | MT5 broker server time, UTC, client-local time, and external provider time can differ. | Configure `MT5_SERVER_TZ` or `MT5_TIME_OFFSET_MINUTES` before analysis; see [TIMESTAMPS.md](TIMESTAMPS.md) and keep the `timezone` field with saved results. |

## Where the details live

| Topic | Doc |
|-------|-----|
| Response envelope, `detail` / `extras`, pagination | [OUTPUT.md](OUTPUT.md) |
| Forecast method defaults and dependencies | [forecast/METHODS.md](forecast/METHODS.md) |
| Timestamp policy | [TIMESTAMPS.md](TIMESTAMPS.md) |
| Trading dry-run, guardrails, broker behavior | [TRADING_SAFETY.md](TRADING_SAFETY.md) |
| Long-lived MCP / Web API service | [DEPLOYMENT.md](DEPLOYMENT.md) |

## If you are unsure

Stay on **read-only** commands until account, symbol, timezone, and risk controls are clear:

```bash
mtdata-cli symbols_list --limit 10
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 50 --json
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

Only then consider `trade_*` — on demo, with [TRADING_SAFETY.md](TRADING_SAFETY.md).
