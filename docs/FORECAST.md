# Forecasting and Pattern-Based Signals

This document covers the forecasting features and pattern-based similarity search, their parameters, outputs, and practical guidance. Use it as a reference when building trading signals.

## Approaches

- Statistical forecasting (`forecast` tool): classical models (naive, theta, Holt-Winters, ARIMA, etc.) that project the next `horizon` bars from a single series.
- Pattern-based similarity (`pattern_*` tools): finds historical windows similar to the most recent window and aggregates their subsequent moves into a signal.

Both approaches can be combined: use pattern-based signals as features or validation for forecasts, or vice versa.

---

## Statistical Forecasts (`forecast`)

Generates point forecasts for the next `horizon` bars. Optionally returns confidence bounds for supported methods.

Key parameters
- `symbol`: instrument, e.g. `EURUSD`.
- `timeframe`: bar timeframe, e.g. `H1`.
- `method`: one of `naive`, `drift`, `seasonal_naive`, `theta`, `fourier_ols`, `ses`, `holt`, `holt_winters_add`, `holt_winters_mul`, `arima`, `sarima`, `ensemble`.
- `horizon`: how many future bars to predict.
- `indicators`: optional technical indicators to include before forecasting.
- `denoise`: optional denoising applied pre/post indicators.
- `target`: `price` or `return` (if supported).

CLI examples
- `python cli.py forecast EURUSD --timeframe H1 --method theta --horizon 12 --format json`
- `python cli.py forecast EURUSD --timeframe H4 --method holt_winters_add --horizon 8`

Outputs
- `forecast_price` and/or `forecast_return` arrays where applicable.
- Optional `lower_price`, `upper_price` with `ci_alpha` for confidence intervals.
- Metadata echoing parameters and any denoise/options used.

Tips
- Use `seasonal_naive`/`holt_winters_*` on clearly seasonal intraday datasets.
- `theta` and `fourier_ols` are strong general-purpose baselines.
- For ARIMA/SARIMA, ensure enough history; consider denoising to stabilize.

### Framework Integrations

The server can integrate with several forecasting frameworks. These are optional; install the packages you need. Methods become available automatically and are listed by `list_forecast_methods`.

#### StatsForecast (classical, fast)

Install: `pip install statsforecast`

Methods:
- `sf_autoarima` — AutoARIMA with seasonal support
- `sf_theta` — Theta model
- `sf_autoets` — Automatic ETS (exponential smoothing)
- `sf_seasonalnaive` — Seasonal Naive baseline

Examples:
```bash
python cli.py forecast EURUSD --timeframe H1 --method sf_autoarima --horizon 12 --params "{\"seasonality\":24,\"stepwise\":true}"
python cli.py forecast EURUSD --timeframe H1 --method sf_autoets   --horizon 12 --params "{\"seasonality\":24}"
```

Notes:
- StatsForecast is optimized with NumPy/Numba and is generally fast and robust.

#### MLForecast (tree/GBM models over lag features)

Install:
- RandomForest: `pip install mlforecast scikit-learn`
- LightGBM: `pip install mlforecast lightgbm`

Methods:
- `mlf_rf` — sklearn RandomForestRegressor on lag + rolling features
- `mlf_lightgbm` — LightGBM regressor on lag + rolling features

Examples:
```bash
python cli.py forecast EURUSD --timeframe H1 --method mlf_rf         --horizon 12 --params "{\"lags\":[1,2,3,24],\"rolling_agg\":\"mean\",\"n_estimators\":300}"
python cli.py forecast EURUSD --timeframe H1 --method mlf_lightgbm  --horizon 12 --params "{\"lags\":[1,2,3,24],\"n_estimators\":400,\"learning_rate\":0.05,\"num_leaves\":63}"
```

Notes:
- Add exogenous features by computing indicators via `fetch_candles --indicators ...` and wiring them into a custom training flow if needed.

#### NeuralForecast (deep learning)

Install: `pip install neuralforecast[torch]`

Methods:
- `nhits`, `nbeatsx`, `tft`, `patchtst`

Examples:
```bash
python cli.py forecast EURUSD --timeframe H1 --method nhits   --horizon 12 --params "{\"max_epochs\":30,\"input_size\":256}"
python cli.py forecast EURUSD --timeframe H1 --method nbeatsx --horizon 12 --params "{\"max_epochs\":20,\"model_params\":{\"stack_types\":[\"trend\",\"seasonality\"]}}"
```

Notes:
- Defaults are conservative to keep training time reasonable. Increase `max_epochs` and tune `input_size` for better accuracy.
- Panel (multi‑symbol) training can be added; current adapter uses single‑series.

#### Foundation Models (one‑shot inference)

Install: `pip install transformers torch accelerate`.
Optional native libs: `pip install chronos-forecasting` (Chronos), `pip install timesfm` (TimesFM, if available).

Methods:
- `chronos_bolt` — Amazon Chronos‑Bolt (native or via Transformers)
- `timesfm` — Google TimesFM (native or via Transformers)
- `lag_llama` — Lag‑Llama (via Transformers)

Shared params:
- `model_name`: HF repo id (e.g., `amazon/chronos-bolt-base`, `google/timesfm-1.0-200m`)
- `context_length`: tail window size to feed
- `device` or `device_map`: compute placement
- `quantization`: `int8` or `int4` (best‑effort, requires compatible backend)
- `quantiles`: e.g., `[0.05, 0.5, 0.95]` to request prediction quantiles
- `revision`: HF branch/tag/commit
- `trust_remote_code`: allow custom code from the repo

Examples:
```bash
# Chronos‑Bolt with quantiles
python cli.py forecast EURUSD --timeframe H1 --method chronos_bolt --horizon 12 \
  --params "{\"model_name\":\"amazon/chronos-bolt-base\",\"context_length\":512,\"quantiles\":[0.05,0.5,0.95]}"

# TimesFM default model, 8‑bit load
python cli.py forecast EURUSD --timeframe H1 --method timesfm --horizon 12 \
  --params "{\"context_length\":512,\"quantization\":\"int8\"}"

# Lag‑Llama via Transformers
python cli.py forecast EURUSD --timeframe H1 --method lag_llama --horizon 12 \
  --params "{\"model_name\":\"time-series-foundation-models/Lag-Llama\",\"context_length\":512,\"device\":\"cuda:0\"}"
```

Notes:
- Transformers auto‑downloads models and caches them. For private/gated repos, login or set `HUGGINGFACE_HUB_TOKEN`.
- Quantile outputs are returned under `forecast_quantiles` and are transformed to price space for `target=price`.
- Some repos may require `trust_remote_code: true`.

---

### Rolling Backtests

Run rolling‑origin evaluations with the integrated `forecast_backtest` tool.

Parameters:
- `steps`: number of historical anchors
- `spacing`: bars between anchors
- `methods`: optional list; if omitted, a sensible subset is chosen based on availability

Example:
```bash
python cli.py forecast_backtest EURUSD --timeframe H1 --horizon 12 --steps 5 --spacing 24 \
  --methods theta sf_autoarima mlf_rf
```

Outputs include per‑method averages (MAE, RMSE, directional accuracy) and per‑anchor details.

---

## Pattern-Based Similarity Search

Builds an index of sliding windows across one or more instruments and searches for the nearest historical patterns to the most recent window. Aggregates their forward moves to produce a signal.

Tools
<!-- pattern_prepare_index removed: pattern_search now builds/loads indexes on demand -->
<!-- pattern_search_recent removed; use pattern_search (builds/loads index automatically). -->
- `pattern_search(...)`: unified search; reuses cached index or builds using selected or visible symbols. Also supports shape re‑ranking.

Key parameters (shared)
- `timeframe`: e.g., `H1`.
- `window_size`: length of the lookback window to match (e.g., 20 bars).
- `future_size`: how many bars ahead to evaluate in matches (e.g., 5 bars).
- `denoise`: optional; applied to `close` before windowing.
- `scale`: `minmax` (default), `zscore`, or `none` per-window scaling.
- `metric`: `euclidean` (default), `cosine`, or `correlation` (implemented via vector normalization).
- `pca_components`: optional dimensionality reduction; requires scikit-learn.
- `engine`: `ckdtree` (default exact) or `hnsw` (optional ANN via `hnswlib`).
- `lookback`: limit of bars per instrument used to build the index (caps history scanned). When omitted, defaults to the existing `max_bars_per_symbol` behavior.
- `refine_k`: retrieve this many nearest candidates initially (e.g., 100–300) and then re‑rank down to `top_k` using a shape metric.
- `shape_metric`: optional second‑pass re‑ranker; `ncc` (normalized cross‑correlation with optional lag), `affine` (fit scale/offset then compare residual RMSE), `dtw`, `softdtw`, or `none`.
- `allow_lag`: maximum bars of left/right shift when using `ncc` to maximize correlation (e.g., 3–10).
- `time_scale_span`: multi‑scale tolerance around the anchor window length. A value `s` generates scales `[max(0.05, 1−s), 1.0, 1+s]` (e.g., `0.1` → `0.9, 1.0, 1.1`). Candidates are resampled to the anchor length for final shape comparison.

Indexing-specific
- `symbols` / `max_symbols`: explicitly choose instruments or use all visible (capped).
- `max_bars_per_symbol`: history cap per instrument; index and responses reflect actual bars used.

Search-specific
- `top_k`: how many nearest neighbors to aggregate for the signal.
- `include_values`: if true, include raw series for each match (omitted by default for compactness).
- `min_symbol_correlation`: when searching across instruments, filter out matches whose log-return correlation with the anchor (over `corr_lookback` bars) falls below this threshold (e.g., `0.3`).
- `corr_lookback`: number of bars for correlation computation (default `1000`).

Outputs (key fields)
- `forecast_type`: `gain` or `loss` based on majority direction of matched futures.
- `forecast_confidence`: majority confidence (e.g., `0.68` if 68% of matches imply gain).
- `prob_gain`: same as above for clarity.
- `avg_change`, `median_change`, `std_change`: absolute future move stats over `future_size` bars.
- `avg_pct_change`, `median_pct_change`, `std_pct_change`: percentage move stats.
- `per_bar_avg_change`, `per_bar_avg_pct_change`: normalized per bar.
- `distance_weighted_avg_change`, `distance_weighted_avg_pct_change`: weighted by 1/distance for stronger influence of closer matches.
- `matches`: top matches with summary fields per match: `symbol`, `distance`, `start_date`, `end_date`, `todays_value`, `future_value`, `change`, `pct_change` (+ `values` if requested).
- `n_matches`, `n_candidates`: how many matches remained after correlation filtering vs initially retrieved.
- `engine`, `scale`, `metric`, `pca_components` and history diagnostics: `max_bars_per_symbol`, `bars_per_symbol`, `windows_per_symbol`.
- `anchor_end_epoch`, `anchor_end_time`: time of the last closed bar in the anchor window (useful for aligning forecast timestamps).
 - `refine_k`, `shape_metric`, `allow_lag`: effective settings used for the retrieval + refinement process.

CLI examples
- Recent search on one symbol:
  - `python cli.py call pattern_search EURUSD --timeframe H1 --window-size 20 --top-k 50 --future-size 5`
- Search across many symbols with correlation filter and ANN:
  - `python cli.py call pattern_search EURUSD --timeframe H1 --window-size 30 --top-k 100 --future-size 8 --max-symbols 50 --engine hnsw --min-symbol-correlation 0.3`
<!-- Prebuild example removed; use pattern_search with cache_id to persist/load. -->

Tuning guidelines
- Window and future sizes: common pairs are `(20, 5)`, `(30, 8)`, `(50, 10)`; larger windows can generalize better but reduce sample counts.
- Scaling and metric: `minmax + euclidean` is robust; try `zscore + cosine` for shape emphasis.
- PCA: use if `window_size` is large (>100) or to denoise; start with 10–30 components.
- top_k: 25–200 commonly; too low → noisy, too high → dilute. Use distance-weighted averages.
- Cross-instrument matches: enable `min_symbol_correlation` (e.g., `0.2–0.5`) when using many instruments.
- Engine: `ckdtree` is default, exact, and fast up to ~100k–300k windows. Use `hnsw` for large indices; `pip install hnswlib`.
- History: increase `max_bars_per_symbol` to grow sample size; monitor `windows_per_symbol` in responses.
- Overlap control: same-symbol windows that overlap the anchor window are skipped; additionally, selected same-symbol matches are required to be non-overlapping with each other to avoid duplication.
- Re‑ranking: set `refine_k` to 5–10× `top_k` and use `shape_metric=ncc` with `allow_lag=3..10` to improve shape fidelity without heavy compute.
- DTW/Soft‑DTW: for variable‑tempo patterns, use `shape_metric=dtw` or `softdtw` (more flexible but slower). Keep `refine_k` modest (≤200) and consider using `time_scale_span` so coarse retrieval accounts for length variability.
 - Amplitude/offset differences: use `shape_metric=affine` to align candidate amplitude/offset to the anchor before scoring (robust when the shape is similar but regimes differ in level/volatility).
 - Multi‑scale: use `time_scale_span` (e.g., `0.1`) to tolerate modest time stretching/shrinking; results across scales are resampled and compared on the anchor length.

Operational tips
- Caching: pattern_search supports in-memory cache and optional disk persistence using `cache_id`/`cache_dir`.
- Denoising: modest smoothing before windowing can improve match stability; keep consistent between index and query.
- Stability: consider restricting to the same instrument unless you explicitly want cross-instrument transfer.
 - Non-overlap: by design, same-symbol matches do not overlap; if you need looser or stricter spacing, adjust `window_size` (or extend the API to expose a spacing parameter).

Limitations
- Similarity does not imply causation; signals can drift in regime changes.
- ANN (HNSW) is approximate; tune `ef_search` if recall is critical.
- Correlation filter is symmetric linear correlation of log-returns; for cointegration or nonlinear relations, consider extending.

---

## Volatility Forecasts (`forecast_volatility`)

Estimate volatility over the next `horizon` bars using direct estimators (EWMA, range-based), GARCH-family models, or general forecasters on a volatility proxy. Use the dedicated `forecast_volatility` tool.

Key parameters
- `symbol`, `timeframe`, `horizon`.
- `method` (direct): `ewma`, `parkinson`, `gk` (Garman–Klass), `rs` (Rogers–Satchell), `yang_zhang`, `rolling_std`, `garch`, `egarch`, `gjr_garch`.
- `method` (general, require proxy): `arima`, `sarima`, `ets`, `theta`.
- `proxy` (required for general): `squared_return` | `abs_return` | `log_r2`.
- `params` (by method):
  - `ewma`: `lookback` (bars), `halflife` or `lambda_` (RiskMetrics). If both provided, `halflife` is used.
  - `parkinson`/`gk`/`rs`: `window` (bars) for moving average of per‑bar variance.
  - `garch`: `fit_bars` (history to fit), `mean` (`Zero`|`Constant`), `dist` (`normal`|`t`). Requires `arch` package.
- `denoise`: optional pre‑processing of price columns (OHLC for range estimators; close for EWMA/GARCH).
- `as_of`: compute using data up to a specific time.

Outputs
- `sigma_bar_return`: current per‑bar return volatility (stdev of returns).
- `sigma_annual_return`: annualized per‑bar volatility (scaled by sqrt(bars_per_year)).
- `horizon_sigma_return`: horizon return volatility over `horizon` bars (stdev of sum of returns).
- `horizon_sigma_annual`: annualized horizon volatility.
- Diagnostics: `bars_used`, `params_used`, `last_close`, `as_of`.

CLI examples
- EWMA with halflife:
  - `python cli.py forecast_volatility EURUSD --timeframe H1 --horizon 12 --method ewma --params "halflife=60,lookback=1500"`
- Parkinson range estimator:
  - `python cli.py forecast_volatility EURUSD --timeframe H1 --horizon 12 --method parkinson --params "window=20"`
- GARCH(1,1) with t‑dist:
  - `python cli.py forecast_volatility EURUSD --timeframe H1 --horizon 12 --method garch --params "fit_bars=2000,mean=Zero,dist=t"`
- ARIMA on log-variance proxy:
  - `python cli.py forecast_volatility EURUSD --timeframe H1 --horizon 12 --method arima --proxy log_r2 --params "p=1,d=0,q=1"`

Notes
- Range estimators need `open/high/low/close`; EWMA/GARCH require only `close`.
- Annualization uses timeframe‑based bars per year. For intraday, this approximates 365*24 hours.
- For robust returns, the server uses log differences; GARCH operates on percent returns.
