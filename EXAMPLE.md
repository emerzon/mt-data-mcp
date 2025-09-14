# End-to-End Trading Workflow Example

This example walks through discovering capabilities, preparing data, generating price and volatility forecasts, running a quick backtest, performing advanced pattern search with dimensionality reduction, and combining signals into a trading decision. Every step includes a runnable CLI command and highlights advanced parameters.

Assumptions
- You have MetaTrader5 running and the server available via `python server.py`.
- You invoke tools via the dynamic CLI: `python cli.py <command> [...args]`.
- Replace `EURUSD` and `H1` with your symbol/timeframe as needed.

Tip: add `--format json` to print structured JSON instead of CSV summaries.

---

## 1) Discover Capabilities and Methods

See what features are available on your machine (frameworks, methods, reducers, backends):

```bash
python cli.py list_capabilities --format json

# Focus on specific sections
python cli.py list_capabilities --sections frameworks forecast dimred pattern_search --format json
```

Notes
- `frameworks` reports availability of optional libraries (NeuralForecast, StatsForecast, MLForecast, foundation models, ARCH, Torch/CUDA).
- `forecast_methods`, `volatility_methods`, `denoise_methods`, and `dimred_methods` enumerate methods, params, and availability.

---

## 2) Choose Symbol, Timeframe, and Inspect Basics

List visible symbols and describe one:

```bash
python cli.py list_symbols --limit 20
python cli.py describe_symbol EURUSD --format json
```

Fetch recent candles for sanity checks:

```bash
python cli.py fetch_candles EURUSD --timeframe H1 --limit 300 --format json
```

Optional: Market depth (liquidity snapshot):

```bash
python cli.py fetch_market_depth EURUSD --format json
```

---

## 3) Prepare Data: Denoise and Indicators

Denoise the close series before feature engineering or forecasting (EMA example):

```bash
python cli.py fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2,keep_original=false" --format json
```

Compute indicators with denoise (post-indicator smoothing):

```bash
python cli.py fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14),ema(50)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3,keep_original=true" \
  --format json
```

Notes
- `when=pre_ti` applies denoise before indicators, `post_ti` after.
- `keep_original=true` preserves original columns (adds `_dn` suffix by default).

---

## 4) Baseline Price Forecasts (Classical + Frameworks)

Quick baseline (Theta):

```bash
python cli.py forecast EURUSD --timeframe H1 --method theta --horizon 12 --format json
```

StatsForecast AutoARIMA (if installed):

```bash
python cli.py forecast EURUSD --timeframe H1 --method sf_autoarima --horizon 24 \
  --params "{\"seasonality\":24,\"stepwise\":true}" --format json
```

MLForecast + LightGBM (if installed):

```bash
python cli.py forecast EURUSD --timeframe H1 --method mlf_lightgbm --horizon 12 \
  --params "{\"lags\":[1,2,3,24],\"rolling_agg\":\"mean\",\"n_estimators\":400,\"learning_rate\":0.05}" \
  --format json
```

NeuralForecast NHITS (if installed):

```bash
python cli.py forecast EURUSD --timeframe H1 --method nhits --horizon 24 \
  --params "{\"max_epochs\":30,\"input_size\":256,\"batch_size\":64}" --format json
```

Foundation model (Chronos‑Bolt) with quantiles and device mapping:

```bash
python cli.py forecast EURUSD --timeframe H1 --method chronos_bolt --horizon 24 \
  --params "{\"model_name\":\"amazon/chronos-bolt-base\",\"context_length\":512,\"quantiles\":[0.1,0.5,0.9],\"device_map\":\"auto\",\"trust_remote_code\":true}" \
  --format json
```

Notes
- Many methods accept `denoise` and `indicators` too (see fetch_candles step), but keep it simple unless you need exogenous features.
- Foundation models can accept `quantization` (e.g., `int8`, `int4`) when supported by the backend.

---

## 5) Volatility Forecasts (Risk Sizing)

Direct estimators:

```bash
python cli.py forecast_volatility EURUSD --timeframe H1 --horizon 12 \
  --method ewma --params "halflife=60,lookback=1500" --format json
```

GARCH via `arch` (if installed):

```bash
python cli.py forecast_volatility EURUSD --timeframe H1 --horizon 12 \
  --method garch --params "fit_bars=2000,mean=Zero,dist=t" --format json
```

Proxy modeling (ARIMA over `log_r2`):

```bash
python cli.py forecast_volatility EURUSD --timeframe H1 --horizon 12 \
  --method arima --proxy log_r2 --params "p=1,d=0,q=1" --format json
```

Use volatility to set position size e.g., target risk per trade by scaling lot size inversely to forecasted sigma.

---

## 6) Quick Backtest (Sanity Check)

Run a rolling backtest to compare a few methods quickly:

```bash
python cli.py forecast_backtest EURUSD --timeframe H1 --horizon 12 \
  --steps 50 --spacing 5 \
  --methods "theta sf_autoarima nhits" --format json
```

Notes
- Metrics include MAE, RMSE, and directional accuracy. Use this to select a small set of candidate models.

---

## 7) Advanced Pattern Search (Similarity + Dimensionality Reduction)

Run a similarity search using many instruments, correlation filtering, ANN engine, DTW refinement, and UMAP dim‑red:

```bash
python cli.py pattern_search EURUSD --timeframe H1 \
  --window-size 30 --future-size 8 --top-k 100 \
  --max-symbols 50 --scale zscore --metric euclidean --engine hnsw \
  --min-symbol-correlation 0.3 --corr-lookback 1000 \
  --refine-k 400 --shape-metric dtw --allow-lag 5 \
  --dimred-method umap --dimred-params "n_components=8,n_neighbors=15,min_dist=0.1" \
  --cache-id eurusd_h1_w30_f8_umap --format json
```

Faster parametric DREAMS‑CNE dimensionality reduction (if installed):

```bash
python cli.py pattern_search EURUSD --timeframe H1 \
  --window-size 30 --future-size 8 --top-k 80 \
  --max-symbols 25 --scale minmax --metric cosine --engine ckdtree \
  --refine-k 300 --shape-metric ncc --allow-lag 5 \
  --dimred-method dreams_cne_fast --dimred-params "n_components=2" \
  --cache-id eurusd_h1_w30_f8_cne --format json
```

Notes
- `cache_id` persists the index to disk for reuse; pair with `cache_dir` to control location.
- Transform‑capable reducers (PCA, KPCA, UMAP, Isomap, parametric DREAMS‑CNE) support querying; t‑SNE/Laplacian do not.
- Use `scale=zscore` + `metric=cosine` to emphasize shape over level.
- Try `time_scale_span=0.1` to allow multi‑scale coarse retrieval across ±10% time stretch with re‑ranking.

---

## 8) Combine Signals into a Decision

A simple decision recipe:
- Go long if all three align:
  - Pattern search: `prob_gain >= 0.60` and `distance_weighted_avg_pct_change > 0.0`.
  - Price forecast median/mean: positive for your selected method(s) over horizon.
  - Volatility forecast: below your risk threshold (e.g., projected 1% daily sigma).
- Go short if signs flip symmetrically.
- Else: no trade or smaller position.

Example: Fetch all inputs programmatically or via sequential CLI + scripting. For illustration, here are the key fields to check in JSON:

- From `pattern_search` (with `--format json --compact false` recommended):
  - `prob_gain`, `distance_weighted_avg_pct_change`, `avg_pct_change`, `forecast_confidence`.
- From `forecast`:
  - `forecast_price` or `forecast_return`, optional `forecast_quantiles`.
- From `forecast_volatility`:
  - `forecast_sigma` (per‑bar sigma), or horizon sigma if modeled that way.

---

## 9) Execute and Monitor

Check market depth to validate liquidity and current spread:

```bash
python cli.py fetch_market_depth EURUSD --format json
```

Monitor outcome after entering a trade (not covered by tools; depends on your execution stack). Consider scheduling periodic refresh of forecasts and similarity signals.

---

## 10) Reproducibility and Caching

- Use `cache_id` and optionally `cache_dir` in `pattern_search` to persist/load indexes.
- Pin framework versions in your environment; record params in a run log (see CLI stdout and JSON fields).
- For foundation models, specify `revision` to pin model weights and `trust_remote_code` only when you trust the repo.

---

## Appendix: Troubleshooting

- No methods listed? Run `python cli.py list_capabilities --sections frameworks` to check availability flags and install optional packages.
- t‑SNE/Laplacian reducers throw on query? They don’t support transforming new samples; use PCA/KPCA/UMAP/parametric DREAMS‑CNE instead.
- Pattern search returns few matches? Increase `max_bars_per_symbol` and/or `max_symbols`; reduce `min_symbol_correlation`.
- Forecast errors about history depth? Increase candle `--limit` in your historical fetch or relax model complexity.
- ARCH/GARCH fits are slow? Reduce `fit_bars` or use EWMA/rolling estimators for faster volatility proxies.

---

Happy trading! Tune thresholds and horizons to your instrument’s regime and always validate with backtests.

