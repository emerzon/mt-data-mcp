# End-to-End Trading Workflow Example

**Related documentation:**
- [README.md](../README.md) - Project overview
- [README.md](README.md) - Docs index
- [CLI.md](CLI.md) - CLI usage patterns
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) - Beginner guide
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) - Advanced playbook
- [FORECAST.md](FORECAST.md) - Forecasting overview
- [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md) - `forecast_generate`
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) - Volatility forecasting
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) - Barrier analytics deep dive
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) - Indicators
- [DENOISING.md](DENOISING.md) - Denoising and smoothing
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

This walkthrough shows a practical “research loop”: discover methods, fetch data, add context (indicators/denoise), generate forecasts, size risk (volatility + barriers), and validate with a quick backtest.

Assumptions:
- MetaTrader 5 terminal is running.
- You call tools via `python cli.py <command> ...`.
- Replace `EURUSD` and `H1` with your symbol/timeframe.

Tip: add `--format json` for structured output.

---

## 1) Discover what’s available

Forecast methods + availability (optional libraries show up here):

```bash
python cli.py forecast_list_methods --format json
```

Forecast model spaces:

```bash
python cli.py forecast_list_library_models native --format json
python cli.py forecast_list_library_models statsforecast --format json
python cli.py forecast_list_library_models sktime --format json
python cli.py forecast_list_library_models pretrained --format json
```

Indicators:

```bash
python cli.py indicators_list --limit 50
python cli.py indicators_describe rsi --format json
```

---

## 2) Inspect symbol and fetch candles

```bash
python cli.py symbols_list --limit 20
python cli.py symbols_describe EURUSD --format json
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 300 --format json
```

Optional (liquidity snapshot):

```bash
python cli.py market_depth_fetch EURUSD --format json
```

---

## 3) Prepare data: indicators + denoise

Compute a few indicators:

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)" \
  --format json
```

Denoise before indicators (smoother inputs):

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14),ema(50)" \
  --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2,keep_original=true" \
  --format json
```

Denoise after indicators (smoother signals):

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3,keep_original=true" \
  --format json
```

---

## 4) Baseline price forecasts

Simple baseline (Theta):

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model theta --format json
```

Pattern-based “analog” forecast (nearest-neighbor style):

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model analog --model-params "window_size=64 search_depth=5000 top_k=20 scale=zscore" \
  --format json
```

Optional: foundation model (if available):

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --model chronos2 --model-params "context_length=512" \
  --format json
```

Optional: Monte Carlo simulation forecast (range of outcomes):

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model mc_gbm --model-params "n_sims=3000 seed=7" \
  --format json
```

---

## 5) Volatility forecast (risk sizing)

EWMA (fast default):

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method ewma --params "lambda=0.94" --format json
```

HAR-RV (uses intraday realized volatility):

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22" --format json
```

---

## 6) Barrier analytics (TP/SL planning)

Barrier probabilities (TP/SL odds within the horizon):

```bash
python cli.py forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
  --method mc --mc-method hmm_mc --tp-pct 0.5 --sl-pct 0.3 --params "n_sims=5000 seed=7" \
  --format json
```

Barrier optimization (search a TP/SL grid):

```bash
python cli.py forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style volatility --refine true \
  --tp-min 0.25 --tp-max 1.5 --tp-steps 7 \
  --sl-min 0.25 --sl-max 2.5 --sl-steps 9 \
  --params "n_sims=5000 seed=7" --format json
```

Interpretation shortcuts:

- `edge` ≈ P(TP first) − P(SL first)
- `prob_resolve` ≈ 1 − P(no hit)
- Wider stops/targets may look “worse” on edge but can be better after accounting for time-to-hit and payoff ratio.

---

## 7) Regime detection (gatekeeper)

Change-points (BOCPD):

```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method bocpd --threshold 0.6 --output summary --lookback 300 --format json
```

Regime labels (HMM):

```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method hmm --params "n_states=3" --output compact --lookback 300 --format json
```

---

## 8) Quick backtest (sanity check)

Run a rolling-origin backtest to compare a few methods:

```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --steps 50 --spacing 5 --methods "theta sf_autoarima analog" --format json
```

---

## 9) Pattern detection (optional context)

Candlestick patterns:

```bash
python cli.py patterns_detect EURUSD --timeframe H1 --mode candlestick --limit 500 \
  --robust-only true --format json
```

Classic chart patterns:

```bash
python cli.py patterns_detect EURUSD --timeframe H1 --mode classic --limit 800 --format json
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
