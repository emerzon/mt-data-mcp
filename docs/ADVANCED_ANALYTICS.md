# MT5-native advanced analytics

**Audience:** User

Five **read-only** tools for when the basic chart is not enough: how tight the quote is, how your fills behaved, whether a rule held up out of sample, how portfolio risk splits, and which names led.

You do not need a second data vendor — these read the connected MetaTrader 5 terminal. Skip this page until [SAMPLE-TRADE.md](SAMPLE-TRADE.md) feels comfortable.

**Dense terms:** [Microstructure](GLOSSARY.md#microstructure) · [Execution quality](GLOSSARY.md#execution-quality) · [Relative strength](GLOSSARY.md#relative-strength) · [VaR / CVaR](GLOSSARY.md#var-value-at-risk) · [Spread](GLOSSARY.md#spread)

**Related:** [Trading risk](TRADING_RISK.md) · [CLI](CLI.md) · [Example workflow](EXAMPLE.md) · [Glossary](GLOSSARY.md)

## Tick microstructure

`market_microstructure_analyze` measures spread distributions, quote-update
intensity, gaps, mid-price volatility, and liquidity-stress windows.
Its compact spread summary marks locked, one-sided, and inverted latest quotes as
unsafe rather than treating them as unusually tight execution spreads.

```bash
mtdata-cli market_microstructure_analyze EURUSD --minutes-back 60 --json
```

The result identifies the feed as `quote_only`, `trade_ticks`, or
`trade_volume`. Volume-impact metrics are omitted unless the broker supplies
enough non-zero real trade volume. Quote pressure is a proxy, not centralized
FX order flow.

MT5 tick rows are complete snapshots. The analyzer uses the `flags` bitmask to
identify trade events, so a quote update that repeats the last price and volume
is not counted as another trade.

## Execution quality

`trade_execution_quality` joins MT5 deal history to order history and nearby
ticks. It reports side-aware slippage, latency, partial fills, fees, and
post-fill markouts.

```bash
mtdata-cli trade_execution_quality --symbol EURUSD --minutes-back 43200 \
  --markout-seconds 1,5,30 --detail full --json
```

Positive slippage is worse for the trader; positive markout is favorable. With
the default arrival-quote policy, the headline slippage distribution contains
market-order fills only. Pending fills are compared with their submitted order
price, while setup-to-fill price movement is reported separately as arrival
implementation shortfall. Unmatched or unbenchmarked fills are counted rather
than silently discarded.

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

Built-in `sma_cross` and `ema_cross` candidates enter only on fast/slow moving-
average cross events; `rsi_reversion` enters only when RSI crosses into an
oversold or overbought zone. Flat bars do not create periodic re-entries.
Each ranking exposes this contract in `signal_definition` (`cross_event`,
`zone_entry_event`, or `forecast_threshold_anchor`).

Forecast-threshold candidates execute at most the latest 200 eligible forecast
anchors to keep validation bounded. Their folds partition that computed signal
window rather than empty earlier history. Each candidate reports signal range,
requested/evaluated folds, skipped-fold reasons, and fold coverage; incomplete
coverage uses `evaluation_status=partial` and cannot receive a positive evidence
classification. `evaluation_status=complete` is reserved for candidates that
evaluate every requested fold.

The default `current_spread_proxy` is complete when its tick window contains a
valid priced spread; `cost_model.complete` describes cost availability, not
whether the source was fixed. The cost block retains the proxy source and
observation window. An insufficient forecast-threshold candidate reports the
required trade count, computed-anchor coverage, long/short/neutral counts, and
a reason distinguishing unavailable forecasts from an uncrossed threshold.

Same-bar TP/SL touches default to `sl_first` and are echoed in the result.

## Portfolio risk decomposition

`portfolio_risk_decompose` maps current MT5 positions into account-currency
filtered-historical scenarios. It returns multi-horizon VaR/Expected Shortfall,
component ES, concentration, prescribed stresses, and optional proposed-trade
incremental ES and margin.

When `proposed_trade` is supplied, its symbol is resolved against the broker
catalog and its volume is validated against that symbol's minimum, maximum, and
lot step before any scenarios run. Invalid requests return the constraints and
the nearest valid volume instead of modeling a trade the broker would reject.

```bash
mtdata-cli portfolio_risk_decompose --timeframe H1 --lookback 1000 \
  --horizon-bars 1,5 --confidence 0.95,0.99 --json
```

The default fails closed if a material position cannot be priced safely. Use
`--allow-partial true` only when an explicitly partial portfolio result is
acceptable. Fail-closed coverage includes both live sensitivity pricing and
the completed return history required by the scenario model. Partial results
list every omitted symbol and the omission stage in `data_quality`.

The perfect-positive-correlation stress applies a common one-sigma factor to
horizon marginal volatilities. Opposing sensitivities therefore offset.

## Relative strength and breadth

`market_relative_strength` ranks a bounded MT5 universe with volatility-scaled,
factor-adjusted momentum across several horizons. It also reports breadth,
rank stability, live spread, per-symbol bar/alignment windows, and data-coverage
exclusions. `limit` is a global output cap split between the strongest and
weakest tails; odd limits assign the extra row to leaders. Full detail exposes
the same bounded selection as `rankings`, not an unbounded universe dump.
Ranking membership is based on completed-bar history; a stale or closed-session
quote is retained as quality metadata unless an explicit spread filter cannot
be evaluated. If candidate latest-bar endpoints exceed one timeframe of
separation, the tool returns `status=incomparable` and withholds ranks and
breadth instead of publishing a misleading cross-section.

```bash
mtdata-cli market_relative_strength --group "Forex\\Majors" --timeframe H1 \
  --horizons 5,20,60 --weights 0.2,0.3,0.5 --limit 10 --json
```

Use homogeneous symbol groups when possible. Instruments with substantially
different trading sessions can produce less comparable cross-sectional ranks.
Omitting both `symbols` and `group` intentionally ranks the bounded visible
Market Watch universe, which may mix asset classes; use `--group` or explicit
symbols when that mixed-universe behavior is not desired.
Inspect `data_window.endpoint_alignment` before comparing mixed-session
instruments. Per-symbol windows are available in full-detail data-quality
diagnostics; compact and summary ranked rows expose concise quote/history status
fields when the endpoints are comparable.

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
