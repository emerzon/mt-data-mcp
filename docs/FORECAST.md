# Forecasting and Pattern-Based Signals

This document covers the forecasting features and pattern-based similarity search, their parameters, outputs, and practical guidance. Use it as a reference when building trading signals.

**Related Documentation:**
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) - Deep dive into barrier analytics methods and algorithms
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) - Step-by-step trading workflow examples
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) - Advanced playbook with regimes, HAR-RV, conformal intervals
- [SKTIME.md](SKTIME.md) - Sktime adapter documentation
- [COMMON_ERRORS.md](COMMON_ERRORS.md) - Troubleshooting common errors
- [EXAMPLE.md](EXAMPLE.md) - Complete end-to-end workflow

## Approaches

- Statistical forecasting (`forecast_generate` tool): classical models (naive, theta, Holt-Winters, ARIMA, etc.) that project the next `horizon` bars from a single series.
- Monte Carlo simulation (`forecast_generate` with mc_gbm/hmm_mc): distributional forecasts with confidence bands, suitable for risk sizing and TP/SL planning.
- Pattern-based similarity (`pattern_*` tools): finds historical windows similar to the most recent window and aggregates their subsequent moves into a signal.

Both approaches can be combined: use pattern-based signals as features or validation for forecasts, or vice versa.

---

## Statistical Forecasts (`forecast_generate`)

Generates point forecasts for the next `horizon` bars. Optionally returns confidence bounds for supported methods.

Key parameters
- `symbol`: instrument, e.g. `EURUSD`.
- `timeframe`: bar timeframe, e.g. `H1`.
- `library`: forecast library, e.g. `native`, `statsforecast`, `sktime`, `pretrained`, `mlforecast`.
- `model`: model name within the library (e.g., `theta`, `AutoARIMA`, `ThetaForecaster`).
- `model_params`: model configuration (JSON or key=value pairs).
- `horizon`: how many future bars to predict.
- `indicators`: optional technical indicators to include before forecasting.
- `denoise`: optional denoising applied pre/post indicators.
- `quantity`: `price`, `return`, or `volatility` (if supported).
 - `target_spec` (optional): define a custom target series and aggregation.
   - `base`: a column name present in the data (e.g., `close`, `RSI_14`, `EMA_50`) or alias `typical|hl2|ohlc4|ha_close`.
   - `indicators`: compute indicators first if `base` references them (e.g., `"rsi(14),ema(50)"`).
   - `transform`: `none|return|log_return|diff|pct_change` (optional `k` for step size; default 1).
   - `horizon_agg`: `last|mean|sum|slope|max|min|range|vol` with optional `normalize`: `none|per_bar|pct`.
   - `classification`: `sign|threshold` (with `threshold`), to label the aggregated target.
 - `features`: optional block to include multivariate/exogenous inputs and DR:
   - `include`: `ohlcv` (default) or list of column names; `close` is excluded to avoid leakage.
   - `indicators`: e.g., `"rsi(14),ema(50),macd(12,26,9)"` to add TI columns before selection.
   - `dimred_method` / `dimred_params`: apply DR across feature columns (e.g., PCA/KPCA/UMAP) before passing as exog.
   - Note: In this release, exogenous features are consumed by SARIMAX (`arima`/`sarima`). Other adapters continue to run univariate.

CLI examples
- `python cli.py forecast_generate EURUSD --timeframe H1 --model theta --horizon 12 --format json`
- `python cli.py forecast_generate EURUSD --timeframe H4 --model holt_winters_add --horizon 8`

Monte Carlo / HMM examples

```bash
# GBM Monte Carlo with 2000 simulations
python cli.py forecast_generate EURUSD --timeframe H1 --model mc_gbm --horizon 12 --model-params "n_sims=2000 seed=7" --format json

# Regime-aware HMM Monte Carlo (3 states)
python cli.py forecast_generate EURUSD --timeframe H1 --model hmm_mc --horizon 12 --model-params "n_states=3 n_sims=3000 seed=7" --format json
```

Outputs
- `forecast_price` and/or `forecast_return` arrays where applicable.
- Optional `lower_price`, `upper_price` with `ci_alpha` for confidence intervals.
- For `mc_gbm` and `hmm_mc`, bands come from simulation quantiles.
- Metadata echoing parameters and any denoise/options used.

Tips
- Use `seasonal_naive`/`holt_winters_*` on clearly seasonal intraday datasets.
- `theta` and `fourier_ols` are strong general-purpose baselines.
- For ARIMA/SARIMA, ensure enough history; consider denoising to stabilize.
- `mc_gbm` is fast and stable; `hmm_mc` adapts to regimes (e.g., trend vs. range) and often provides better sizing signals.

---

## Barrier Analytics (Monte Carlo)

Two tools turn MC/HMM paths into actionable risk metrics for TP/SL planning.

### barrier_hit_probabilities

Estimate probability of hitting TP before SL within a horizon, plus time-to-hit stats.

Inputs:
- `tp_abs`/`sl_abs` (absolute prices) OR `tp_pct`/`sl_pct` (percent points, 0.5 => 0.5%) OR `tp_pips`/`sl_pips` (approx pip = 10×point for 5/3-digit FX)
- `method`: `mc_gbm`, `mc_gbm_bb`, `hmm_mc`, `garch`, `bootstrap`, `heston`, `jump_diffusion`, or `auto` (auto returns `method_used` and `auto_reason`)
- `garch` requires the `arch` package

Example:
```bash
python cli.py forecast_barrier_hit_probabilities --symbol EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --tp_pct 0.5 --sl_pct 0.3 --params "n_sims=5000 seed=7" --format json
```

Outputs:
- `prob_tp_first`, `prob_sl_first`, `prob_no_hit`, `edge` (difference)
- `method_used`/`auto_reason` when `method=auto`
- `tp_hit_prob_by_t`, `sl_hit_prob_by_t` (cumulative hit curves)
- `time_to_tp_bars/seconds`, `time_to_sl_bars/seconds` (mean/median)

Auto method selection (simple):
- Short history -> `mc_gbm_bb` for short horizons (<=12), otherwise `mc_gbm`.
- Heavy tails/jumps -> `jump_diffusion`.
- Regime shift in volatility -> `hmm_mc`.
- Volatility clustering -> `garch` (if available) or `heston`.
- Non-normal skew without jumps -> `bootstrap`.

### barrier_optimize

Search a TP/SL grid (percent or pips) to maximize an objective.

Inputs:
- `mode`: `pct` or `pips`
- `tp_min/max/steps`, `sl_min/max/steps`
- `method`: `mc_gbm`, `mc_gbm_bb`, `hmm_mc`, `garch`, `bootstrap`, `heston`, `jump_diffusion`, or `auto`
- `objective`: `edge` (default), `prob_tp_first`, `prob_resolve`, `kelly`, `kelly_cond`, `ev`, `ev_cond`, `ev_per_bar`, `profit_factor`, `min_loss_prob`, or `utility`
- Constraints: `min_prob_win`, `max_prob_no_hit`, `max_median_time` (bars)
- `garch` requires the `arch` package

Objectives (plain language):
- `edge`: maximize win probability minus loss probability.
- `prob_tp_first`: maximize win rate.
- `prob_resolve`: avoid trades that never hit TP/SL.
- `kelly`: maximize long-run growth (raw).
- `kelly_cond`: same, but only on resolved trades.
- `ev`: maximize average payoff using TP/SL sizes.
- `ev_cond`: average payoff on resolved trades only.
- `ev_per_bar`: prefer faster trades (EV per bar).
- `profit_factor`: maximize win payout vs loss payout.
- `min_loss_prob`: minimize chance of loss.
- `utility`: risk-averse growth (log utility).

Constraints (plain language):
- `min_prob_win`: only keep candidates with enough win rate.
- `max_prob_no_hit`: drop candidates that often do not resolve.
- `max_median_time`: drop candidates that take too long to resolve (bars).

Example:
```bash
python cli.py forecast_barrier_optimize --symbol EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --tp_min 0.25 --tp_max 1.5 --tp_steps 7 \
  --sl_min 0.25 --sl_max 2.5 --sl_steps 9 --params "n_sims=5000 seed=7" --format json
```

Optional flags worth knowing:
- `--grid-style {fixed,volatility,ratio,preset}` switches between the classic fixed grid and the adaptive modes.
- `--preset` activates one of the built-in percent presets (`scalp`, `intraday`, `swing`, `position`).
- `--vol-window`, `--vol-min-mult`, `--vol-max-mult` control the volatility-driven grid span; `--vol-sl-extra` widens stops relative to targets.
- `--ratio-min`, `--ratio-max`, `--ratio-steps` generate TP/SL pairs by reward/risk ratios.
- `--refine` with `--refine-radius` and `--refine-steps` adds a focused zoom around the best coarse result.
- `--min-prob-win`, `--max-prob-no-hit`, `--max-median-time` filter candidates before ranking.

Example with volatility scaling and refinement:
```bash
python cli.py forecast_barrier_optimize --symbol EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style volatility --refine true --refine-radius 0.35 \
  --params "n_sims=5000 seed=7 vol_window=360 vol_min_mult=0.5 vol_max_mult=4.0" --format json
```

Output highlights:
- `best`: grid point with metrics (edge, kelly, EV, conditional metrics, resolve probability, hit probabilities, resolve time stats)
- `grid`: all grid evaluations

Usage notes:
- Use `edge` to bias toward consistent advantage; `kelly` when you also care about payoff ratio.
- Recalibrate per symbol/timeframe and monitor stability via rolling backtests.

---

## Regime & Change‑Point Detection

Adapt strategies to market structure by detecting breaks and labeling regimes.

### BOCPD (Bayesian Online Change‑Point Detection)

Probabilistic, online change‑point detection for Gaussian data. We apply it to log‑returns by default.

CLI:
```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1000 --method bocpd --threshold 0.6 --format json
```

Outputs:
- `cp_prob`: change‑point probability per step (P(run_length=0 | data))
- `change_points`: list with time and probability where `cp_prob >= threshold`

Use cases:
- Pause trading near high `cp_prob` spikes; retrain/resample models after breaks
- Reset risk and widen bands following a structural change

Params:
- `hazard_lambda` (default 250): average run length between changes (higher → fewer breaks)
- `max_run_length`: cap to control runtime/memory (defaults to min(1000, N))

### HMM‑Lite Regimes (Gaussian Mixture)

Fast soft labeling of regimes via a Gaussian mixture over returns (no transition fit); returns per‑step state probabilities and a hard assignment.

CLI:
```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1000 --method hmm --params "n_states=3" --format json
```

Outputs:
- `state`: most likely regime index per step
- `state_probabilities`: per‑state probabilities per step

### Markov‑Switching AR (statsmodels)

Regime‑switching AR via `statsmodels` when available; provides smoothed regime probabilities with AR dynamics.

CLI:
```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1200 --method ms_ar --params "k_regimes=2 order=1" --format json
```

Outputs:
- `state`: most probable regime per step
- `state_probabilities`: smoothed marginal probabilities per state

Notes:
- Prefer BOCPD for change detection; HMM/MS‑AR for ongoing regime labeling to switch playbooks (trend/range, low/high vol).

### Framework Integrations

The server can integrate with several forecasting frameworks. These are optional; install the packages you need. Methods become available automatically and are listed under `forecast_methods` in `list_capabilities`.

#### StatsForecast (classical, fast)

Install: `pip install statsforecast`

Models:
- `sf_autoarima` — AutoARIMA with seasonal support
- `sf_theta` — Theta model
- `sf_autoets` — Automatic ETS (exponential smoothing)
- `sf_seasonalnaive` — Seasonal Naive baseline

Examples (with optional exogenous):
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --model sf_autoarima --horizon 12 --model-params "{\"seasonality\":24,\"stepwise\":true}"
python cli.py forecast_generate EURUSD --timeframe H1 --model sf_autoets   --horizon 12 --model-params "{\"seasonality\":24}"
# with exogenous features (if provided via --features), the adapter uses X_df and X_future internally
```

Notes:
- StatsForecast is optimized with NumPy/Numba and is generally fast and robust.

#### MLForecast (tree/GBM models over lag features)

Install:
- RandomForest: `pip install mlforecast scikit-learn`
- LightGBM: `pip install mlforecast lightgbm`

Models:
- `mlf_rf` — sklearn RandomForestRegressor on lag + rolling features
- `mlf_lightgbm` — LightGBM regressor on lag + rolling features

Examples (exogenous supported when provided via --features):
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --model mlf_rf         --horizon 12 --model-params "{\"lags\":[1,2,3,24],\"rolling_agg\":\"mean\",\"n_estimators\":300}"
python cli.py forecast_generate EURUSD --timeframe H1 --model mlf_lightgbm  --horizon 12 --model-params "{\"lags\":[1,2,3,24],\"n_estimators\":400,\"learning_rate\":0.05,\"num_leaves\":63}"
# features provided via --features are merged into training and future frames
```

Notes:
- Add exogenous features by computing indicators via `data_fetch_candles --indicators ...` and wiring them into a custom training flow if needed.

#### NeuralForecast (deep learning)

Install: `pip install neuralforecast[torch]`

Models:
- `nhits`, `nbeatsx`, `tft`, `patchtst`

Examples (past/future covariates supported when provided via --features):
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --model nhits   --horizon 12 --model-params "{\"max_epochs\":30,\"input_size\":256}"
python cli.py forecast_generate EURUSD --timeframe H1 --model nbeatsx --horizon 12 --model-params "{\"max_epochs\":20,\"model_params\":{\"stack_types\":[\"trend\",\"seasonality\"]}}"
# --features future_covariates=... (e.g., fourier:24,hour,dow) are passed to model predict horizon
```

Notes:
- Defaults are conservative to keep training time reasonable. Increase `max_epochs` and tune `input_size` for better accuracy.
- Panel (multi‑symbol) training can be added; current adapter uses single‑series.

#### Custom Target Examples

- Predict RSI(14) path and return its mean over the horizon:
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --model sarima --horizon 24 \
  --target-spec "{base:'RSI_14',indicators:'rsi(14)',transform:'none',horizon_agg:'mean'}" --format json
```

- Predict EMA(50) changes with slope aggregation (normalized per bar):
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --model theta --horizon 24 \
  --target-spec "{base:'EMA_50',indicators:'ema(50)',transform:'diff',horizon_agg:'slope',normalize:'per_bar'}" \
  --format json
```

#### Multivariate / Exogenous example (SARIMAX)

Pass OHLCV+TIs as exogenous regressors with PCA(8) across feature columns:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --model sarima --horizon 24 \
  --model-params "{\"p\":1,\"d\":0,\"q\":1,\"seasonality\":24,\"P\":1,\"D\":0,\"Q\":0}" \
  --features "include=ohlcv,indicators=rsi(14),macd(12,26,9),dimred_method=pca,dimred_params={n_components:8}" \
  --format json
```

#### Foundation Models (one‑shot inference)

Install per model:
- Chronos‑Bolt: `pip install chronos-forecasting torch`
- TimesFM: `pip install timesfm torch`
- Lag‑Llama: `pip install lag-llama gluonts torch` (optionally `huggingface_hub` for auto ckpt download)

Models:
- `chronos_bolt` — Amazon Chronos‑Bolt (native Chronos)
- `timesfm` — Google TimesFM (native)
- `lag_llama` — Lag‑Llama (native estimator)
- `moirai` — Salesforce Moirai via uni2ts (one‑shot)
- `gt_deepar` — GluonTS DeepAR (quick train)
- `gt_sfeedforward` — GluonTS SimpleFeedForward (quick train)
- `gt_prophet` — GluonTS Prophet wrapper (install `prophet`)
- `gt_tft` — GluonTS Temporal Fusion Transformer (quick train, torch)
- `gt_wavenet` — GluonTS WaveNet (quick train, torch)
- `gt_deepnpts` — GluonTS DeepNPTS (quick train, torch)
- `gt_mqf2` — GluonTS MQF2 (quick train, torch; quantiles)
- `gt_npts` — GluonTS NPTS (non-parametric)

Shared model_params:
- `context_length`: tail window size to feed
- `quantiles`: e.g., `[0.05, 0.5, 0.95]` to request prediction quantiles (if supported)

Examples:
```bash
# Chronos‑Bolt with quantiles
python cli.py forecast_generate EURUSD --timeframe H1 --model chronos_bolt --horizon 12 \
  --model-params "{\"context_length\":512,\"quantiles\":[0.05,0.5,0.95]}"

# TimesFM default model
python cli.py forecast_generate EURUSD --timeframe H1 --model timesfm --horizon 12 \
  --model-params "{\"context_length\":512}"

# Lag‑Llama native estimator (auto‑download default ckpt)
python cli.py forecast_generate EURUSD --timeframe H1 --model lag_llama --horizon 12 \
  --model-params "{\"context_length\":512}"

# GluonTS DeepAR with 5 epochs on series
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_deepar --horizon 12 \
  --model-params "{\"context_length\":64,\"train_epochs\":5}"

# GluonTS SimpleFeedForward
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_sfeedforward --horizon 12 \
  --model-params "{\"context_length\":64,\"train_epochs\":5}"

# GluonTS Prophet wrapper
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_prophet --horizon 12 \
  --model-params "{\"prophet_params\":{\"seasonality_mode\":\"additive\"}}"

# GluonTS Temporal Fusion Transformer (PyTorch)
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_tft --horizon 12 \
  --model-params "{\"context_length\":128,\"train_epochs\":5}"

# GluonTS WaveNet (PyTorch)
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_wavenet --horizon 12 \
  --model-params "{\"context_length\":128,\"train_epochs\":5}"

# GluonTS DeepNPTS (PyTorch)
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_deepnpts --horizon 12 \
  --model-params "{\"context_length\":128,\"train_epochs\":5}"

# GluonTS MQF2 (PyTorch, quantiles)
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_mqf2 --horizon 12 \
  --model-params "{\"context_length\":128,\"train_epochs\":5,\"quantiles\":[0.05,0.5,0.95]}"

# GluonTS NPTS (non-parametric)
python cli.py forecast_generate EURUSD --timeframe H1 --model gt_npts --horizon 12 \
  --model-params "{\"season_length\":1,\"kernel\":\"parzen\",\"window_size\":128}"
```

# Moirai via uni2ts

Install: `pip install uni2ts torch`

Example:
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --model moirai --horizon 12 \
  --model-params "{\"context_length\":512,\"variant\":\"1.0-R-small\"}"
```
Notes:
- The adapter uses uni2ts.get_timeseries_model(variant) for one‑shot inference.
- Set model_params.quantiles to request quantile outputs when supported by the variant.

Notes:
- Lag‑Llama loads a pre‑trained `.ckpt` (specify via `ckpt_path` or allow auto‑download with `huggingface_hub`).
- Quantile outputs are returned under `forecast_quantiles` and are transformed to price space for `target=price`.

GluonTS installation notes:
- Install Torch-enabled GluonTS: `pip install "gluonts[torch]" torch`
- MQF2 requires cpflows: `pip install cpflows`
- Prophet wrapper requires prophet: `pip install prophet`
- If you see `No module named 'gluonts.torch.trainer'`, upgrade GluonTS: `pip install --upgrade "gluonts[torch]"`.

---

### Rolling Backtests

Run rolling‑origin evaluations with the integrated `forecast_backtest_run` tool.

Parameters:
- `steps`: number of historical anchors
- `spacing`: bars between anchors
- `methods`: optional list; if omitted, a sensible subset is chosen based on availability

Example:
```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --steps 5 --spacing 24 \
  --methods theta sf_autoarima mlf_rf
```

Outputs include per‑method averages (MAE, RMSE, directional accuracy) and per‑anchor details.

---

## Conformal Prediction (Valid Intervals)

Wrap any base method with valid finite‑sample prediction intervals via split‑conformal calibration.

CLI:
```bash
python cli.py forecast_conformal_intervals EURUSD --timeframe H1 --method fourier_ols --horizon 12 --steps 25 --spacing 10 --alpha 0.1 --format json
```

How it works:
- Runs a rolling backtest to collect per‑step residuals |ŷ_t+i − y_t+i| over `steps` anchors.
- Uses the (1−α) quantile of residuals per horizon step to build bands around the point forecast.
- Coverage holds under exchangeability; intervals adapt to difficulty (wider when residuals larger).

Notes:
- You can denoise during calibration and prediction for consistency.
- For multi‑step coverage guarantees, per‑step quantiles are reported (`conformal.per_step_q`).

---

## Triple‑Barrier Labeling

Produce +1/−1/0 labels by simulating TP/SL barrier hits on historical paths up to a fixed horizon.

CLI:
```bash
python cli.py labels_triple_barrier EURUSD --timeframe H1 --limit 1500 --horizon 12 --tp_pct 0.5 --sl_pct 0.3 --label-on high_low --format json
```

Outputs:
- `entries`: entry times
- `labels`: +1 (TP first), −1 (SL first), 0 (no hit by horizon)
- `holding_bars`: bars until label is decided

Use cases:
- Train classifiers/regressors on features (including regimes) to predict label or holding time.
- Evaluate signal quality and data leakage by shifting features appropriately.

---

## Closed‑Form Barrier Probability (GBM)

Fast approximation for single‑barrier hit probability within a horizon under GBM.

CLI:
```bash
python cli.py forecast_barrier_closed_form EURUSD --timeframe H1 --horizon 12 --direction up --barrier 1.1000 --format json
```

Notes:
- Estimates μ and σ from recent log‑returns if not provided.
- Use as a sanity check against Monte Carlo outputs or as a speed
- Returns both `mu_annual` (GBM drift used in the formula) and `log_drift_annual` (raw log-return drift before the 0.5·sigma² adjustment).
up in optimizers.

---

## Pattern-Based Similarity Search

Builds an index of sliding windows across one or more instruments and searches for the nearest historical patterns to the most recent window. Aggregates their forward moves to produce a signal.

Tools
<!-- pattern_prepare_index removed: pattern_search now builds/loads indexes on demand -->
<!-- pattern_search_recent removed; use pattern_search (builds/loads index automatically). -->
- `pattern_search(...)`: unified search; reuses cached index or builds using selected or visible symbols. Also supports shape re‑ranking and flexible dimensionality reduction.

Key parameters (shared)
- `timeframe`: e.g., `H1`.
- `window_size`: length of the lookback window to match (e.g., 20 bars).
- `future_size`: how many bars ahead to evaluate in matches (e.g., 5 bars).
- `denoise`: optional; applied to `close` before windowing.
- `scale`: `minmax` (default), `zscore`, or `none` per-window scaling.
- `metric`: `euclidean` (default), `cosine`, or `correlation` (implemented via vector normalization).
- Dimensionality reduction:
  - `pca_components`: backward‑compatible shortcut for PCA dimensionality reduction (requires scikit‑learn).
  - `dimred_method`: choose among `none` (default), `pca`, `svd`, `spca`, `kpca`, `umap`, `isomap`, `laplacian`, `diffusion`, `tsne`, `dreams_cne`, `dreams_cne_fast` (and `lda` for supervised use only). Advanced placeholders: `deep_diffusion_maps`, `dreams`, `pcc` (require plugins).
  - `dimred_params`: dict of method‑specific params. Examples:
    - PCA: `{n_components: 8}`
    - SparsePCA: `{n_components: 8, alpha: 1.0}`
    - KernelPCA: `{n_components: 8, kernel: 'rbf', gamma: 0.1}`
    - SVD: `{n_components: 16}`
    - UMAP: `{n_components: 8, n_neighbors: 15, min_dist: 0.1}`
    - Isomap: `{n_components: 8, n_neighbors: 10}`
    - Laplacian: `{n_components: 8, n_neighbors: 10}`
    - Diffusion: `{n_components: 8, alpha: 0.5, epsilon: auto, k: auto}`
    - t‑SNE: `{n_components: 2, perplexity: 30, learning_rate: 200, n_iter: 1000}`
    - DREAMS‑CNE: `{n_components: 2, k: 15, negative_samples: 500, n_epochs: 250, batch_size: 4096, learning_rate: 0.001, parametric: true, regularizer: true, reg_lambda: 0.0005, reg_scaling: 'norm'}`
    - DREAMS‑CNE‑FAST: `{n_components: 2, k: 10, negative_samples: 200, n_epochs: 60, batch_size: 1024, learning_rate: 0.005, parametric: true}`
  - Notes:
    - sklearn t‑SNE and Spectral Embedding (laplacian) do not support transforming new samples; they can embed the index at build time, but cannot embed the query. Use `pca`, `kpca`, or `umap` for seamless querying.
    - LDA is supervised and requires class labels. It is not applicable to unsupervised pattern search.
    - Diffusion Maps require `pydiffmap` and support out‑of‑sample transform via Nyström in most implementations.
    - DREAMS‑CNE requires installation from source (`berenslab/DREAMS-CNE`); enable `parametric: true` to support transforming new queries. Training can be slow on large indices; use `dreams_cne_fast` for quicker runs.
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
- `engine`, `scale`, `metric`, `pca_components`/`dimred_method`+`dimred_params` and history diagnostics: `max_bars_per_symbol`, `bars_per_symbol`, `windows_per_symbol`.
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

## Volatility Forecasts (`forecast_volatility_estimate`)

Estimate volatility over the next `horizon` bars using direct estimators (EWMA, range-based), GARCH-family models, or general forecasters on a volatility proxy. Use the dedicated `forecast_volatility_estimate` tool.

Key parameters
- `symbol`, `timeframe`, `horizon`.
- `method` (direct): `ewma`, `parkinson`, `gk` (Garman–Klass), `rs` (Rogers–Satchell), `yang_zhang`, `rolling_std`, `har_rv`, `garch`, `egarch`, `gjr_garch`.
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
  - `python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma --params "halflife=60,lookback=1500"`
- Parkinson range estimator:
  - `python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method parkinson --params "window=20"`
- GARCH(1,1) with t‑dist:
  - `python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method garch --params "fit_bars=2000,mean=Zero,dist=t"`
- ARIMA on log-variance proxy:
  - `python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method arima --proxy log_r2 --params "p=1,d=0,q=1"`
- HAR‑RV from M5 realized variance:
  - `python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22"`

Notes
- Range estimators need `open/high/low/close`; EWMA/GARCH require only `close`.
- HAR‑RV computes daily realized variance from intraday returns and uses a daily HAR model; it maps RV to the requested timeframe via sqrt‑of‑time.
- Annualization uses timeframe‑based bars per year. For intraday, this approximates 365*24 hours.
- For robust returns, the server uses log differences; GARCH operates on percent returns.

