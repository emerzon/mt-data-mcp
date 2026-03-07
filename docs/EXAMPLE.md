# End-to-End Trading Workflow Example

**Related:**
- [SETUP.md](SETUP.md) — Installation and MT5 connection
- [CLI.md](CLI.md) — CLI usage patterns and output formats
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Guided workflow (start here if new)

This walkthrough shows a practical “research loop”: discover methods, fetch data, add context (indicators/denoise), generate forecasts, size risk (volatility + barriers), and validate with a quick backtest.

Assumptions:
- MetaTrader 5 terminal is running.
- You call tools via `mtdata-cli <command> ...`.
- Replace `EURUSD` and `H1` with your symbol/timeframe.

Tip: add `--json` for structured output.

---

## 1) Discover what’s available

Forecast methods + availability (optional libraries show up here):

```bash
mtdata-cli forecast_list_methods --json
```

Forecast model spaces:

```bash
mtdata-cli forecast_list_library_models native --json
mtdata-cli forecast_list_library_models statsforecast --json
mtdata-cli forecast_list_library_models sktime --json
mtdata-cli forecast_list_library_models pretrained --json
```

Indicators:

```bash
mtdata-cli indicators_list --limit 50
mtdata-cli indicators_describe rsi --json
```

---

## 2) Inspect symbol and fetch candles

```bash
mtdata-cli symbols_list --limit 20
mtdata-cli symbols_describe EURUSD --json
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 300 --json
```

Optional (liquidity snapshot):

```bash
mtdata-cli market_depth_fetch EURUSD --json
```

---

## 3) Prepare data: indicators + denoise

Compute a few indicators:

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)" \
  --json
```

Denoise before indicators (smoother inputs):

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14),ema(50)" \
  --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2,keep_original=true" \
  --json
```

Denoise after indicators (smoother signals):

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3,keep_original=true" \
  --json
```

---

## 4) Baseline price forecasts

Simple baseline (Theta):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model theta --json
```

Pattern-based “analog” forecast (nearest-neighbor style):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model analog --model-params "window_size=64 search_depth=5000 top_k=20 scale=zscore" \
  --json
```

Optional: foundation model (if available):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --model chronos2 --model-params "context_length=512" \
  --json
```

Optional: Monte Carlo simulation forecast (range of outcomes):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model mc_gbm --model-params "n_sims=3000 seed=7" \
  --json
```

---

## 5) Volatility forecast (risk sizing)

EWMA (fast default):

```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method ewma --params "lambda=0.94" --json
```

HAR-RV (uses intraday realized volatility):

```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22" --json
```

---

## 6) Barrier analytics (TP/SL planning)

Barrier probabilities (TP/SL odds within the horizon):

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
  --method mc --mc-method hmm_mc --tp-pct 0.5 --sl-pct 0.3 --params "n_sims=5000 seed=7" \
  --json
```

Barrier optimization (search a TP/SL grid):

```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style volatility --refine true \
  --tp-min 0.25 --tp-max 1.5 --tp-steps 7 \
  --sl-min 0.25 --sl-max 2.5 --sl-steps 9 \
  --params "n_sims=5000 seed=7" --json
```

Interpretation shortcuts:

- `edge` ≈ P(TP first) − P(SL first)
- `prob_resolve` ≈ 1 − P(no hit)
- Wider stops/targets may look “worse” on edge but can be better after accounting for time-to-hit and payoff ratio.

---

## 7) Regime detection (gatekeeper)

Change-points (BOCPD):

```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method bocpd --threshold 0.6 --output summary --lookback 300 --json
```

Regime labels (HMM):

```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method hmm --params "n_states=3" --output compact --lookback 300 --json
```

---

## 8) Quick backtest (sanity check)

Run a rolling-origin backtest to compare a few methods:

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --steps 50 --spacing 5 --methods "theta sf_autoarima analog" --json
```

---

## 9) Pattern detection (optional context)

Candlestick patterns:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode candlestick --limit 500 \
  --robust-only true --json
```

Classic chart patterns:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode classic --limit 800 --json
```

---

## 10) Combine signals into a decision (simple recipe)

A pragmatic decision checklist:

- Forecast direction: do multiple methods agree on the sign of the next `horizon` bars?
- Volatility: is projected volatility within your risk budget?
- Regime: are you trading in a regime your strategy is designed for?
- Barriers: does the TP/SL pair have acceptable `edge` / `prob_resolve`?
- Patterns: do detections support or contradict the idea (optional)?

---

## Appendix: Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## See Also

- [CLI.md](CLI.md) — Full command reference
- [FORECAST.md](FORECAST.md) — Forecasting methods guide
- [TEMPORAL.md](TEMPORAL.md) — Session and seasonal analysis
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — Barrier optimization deep dive
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Guided trade workflow

