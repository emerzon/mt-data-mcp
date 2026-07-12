# MT5-Native Advanced Analytics

These five read-only tools extend the basic tick, journal, backtest, risk, and
scanner workflows. They use data from the connected MT5 terminal only.

## Tick microstructure

`market_microstructure_analyze` measures spread distributions, quote-update
intensity, gaps, mid-price volatility, and liquidity-stress windows.

```bash
mtdata-cli market_microstructure_analyze EURUSD --minutes-back 60 --json
```

The result identifies the feed as `quote_only`, `trade_ticks`, or
`trade_volume`. Volume-impact metrics are omitted unless the broker supplies
enough non-zero real trade volume. Quote pressure is a proxy, not centralized
FX order flow.

## Execution quality

`trade_execution_quality` joins MT5 deal history to order history and nearby
ticks. It reports side-aware slippage, latency, partial fills, fees, and
post-fill markouts.

```bash
mtdata-cli trade_execution_quality --symbol EURUSD --minutes-back 43200 \
  --markout-seconds 1,5,30 --detail full --json
```

Positive slippage is worse for the trader; positive markout is favorable.
Unmatched or unbenchmarked fills are counted rather than silently discarded.

## Fixed-candidate chronological validation

`strategy_validate` evaluates predeclared built-in or forecast-threshold
candidates with anchored expanding chronological folds. Outcomes must finish
inside their test fold; prior calibration samples are horizon-purged and
embargo bars are excluded. Evidence uses block-bootstrap expectancy tests with
Holm correction and reports `positive`, `negative`, or `inconclusive`.

```bash
mtdata-cli strategy_validate EURUSD --timeframe H1 --lookback 3000 \
  --candidates '[{"id":"fast-cross","type":"builtin_strategy","strategy":"ema_cross","params":{"fast_period":10,"slow_period":30}}]' \
  --barrier '{"horizon":12,"tp_pct":0.5,"sl_pct":0.5}' --json
```

Candidate parameters are fixed before validation; this tool does not optimize
and validate on the same sample.

Same-bar TP/SL touches default to `sl_first` and are echoed in the result.

## Portfolio risk decomposition

`portfolio_risk_decompose` maps current MT5 positions into account-currency
filtered-historical scenarios. It returns multi-horizon VaR/Expected Shortfall,
component ES, concentration, prescribed stresses, and optional proposed-trade
incremental ES and margin.

```bash
mtdata-cli portfolio_risk_decompose --timeframe H1 --lookback 1000 \
  --horizon-bars 1,5 --confidence 0.95,0.99 --json
```

The default fails closed if a material position cannot be priced safely. Use
`--allow-partial true` only when an explicitly partial portfolio result is
acceptable.

The perfect-positive-correlation stress applies a common one-sigma factor to
horizon marginal volatilities. Opposing sensitivities therefore offset.

## Relative strength and breadth

`market_relative_strength` ranks a bounded MT5 universe with volatility-scaled,
factor-adjusted momentum across several horizons. It also reports breadth,
rank stability, live spread, and data-coverage exclusions.

```bash
mtdata-cli market_relative_strength --group "Forex\\Majors" --timeframe H1 \
  --horizons 5,20,60 --weights 0.2,0.3,0.5 --limit 10 --json
```

Use homogeneous symbol groups when possible. Instruments with substantially
different trading sessions can produce less comparable cross-sectional ranks.

## Data caveats

- Historical tick and candle availability is controlled by the broker and the
  terminal's local history.
- FX `tick_volume` is a broker tick count, not traded lots.
- `last` and `volume_real` are commonly zero for OTC instruments.
- DOM is not required by these tools and remains a separate, gated live
  snapshot through `market_depth_fetch`.
- Volume-impact estimates describe only the connected broker's tick feed,
  even when `volume_real` is present.
- The focused FastAPI/Web UI does not expose these tools in v1; use MCP or the
  dynamic CLI.
