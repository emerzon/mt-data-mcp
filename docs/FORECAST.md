# Forecasting Guide

This guide covers tools for predicting future market behavior: price direction, volatility, and uncertainty.

**Related:**
- [GLOSSARY.md](GLOSSARY.md) — Definitions of terms
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation
- [forecast/UNCERTAINTY.md](forecast/UNCERTAINTY.md) — Confidence intervals

---

## Key Concepts

### Horizon
How far ahead the forecast predicts, measured in bars.

**Example:** `--horizon 12` with H1 timeframe = predict the next 12 hours.

### Lookback
How much historical data the model uses. More data can improve accuracy but increases computation time.

### Confidence vs. Reality
Forecasts are estimates, not guarantees. Always:
1. Use confidence intervals to understand uncertainty
2. Validate with backtests before trading
3. Size positions based on volatility, not point forecasts

---

## Price Forecasting (`forecast_generate`)

### Basic Usage

```bash
# Theta method (fast, reliable baseline)
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta

# With JSON output for programmatic use
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

### Choosing a Method

Run this to see available methods:
```bash
mtdata-cli forecast_list_methods
```

Availability is environment-dependent:
- On the supported Python 3.14 install path, GluonTS `gt_*` methods are unsupported.
- NeuralForecast methods (`nhits`, `tft`, `patchtst`, `nbeatsx`) require a separate manual install: `pip install neuralforecast torch`.
- `forecast_list_methods --json` is the source of truth for what your current environment can actually run.

| Category | Models | When to Use |
|----------|--------|-------------|
| **Classical** | `theta`, `naive`, `drift`, `ses`, `holt`, `arima` | Fast baselines, short horizons |
| **Seasonal** | `seasonal_naive`, `ets`, `holt_winters_add`, `holt_winters_mul`, `sarima`, `fourier_ols` | Data with recurring patterns |
| **Statistical** | `sf_autoarima`, `sf_autoets`, `sf_autotheta` | Auto-tuning, medium horizons |
| **ML-Based** | `mlf_lightgbm`, `mlf_rf` | Non-linear patterns, feature engineering |
| **Neural** | `nhits`, `tft`, `patchtst`, `nbeatsx` | Deep learning, long horizons; manual `neuralforecast` install required |
| **Foundation** | `chronos2`, `chronos_bolt`, `timesfm`, `lag_llama` | Pretrained models (optional deps) |
| **GluonTS** | `gt_deepar`, `gt_tft`, `gt_wavenet`, `gt_prophet`, `gt_sfeedforward`, `gt_deepnpts`, `gt_mqf2`, `gt_npts` | Documented for completeness, but unsupported on the project's Python 3.14 runtime |
| **Simulation** | `mc_gbm`, `hmm_mc` | Risk sizing, barrier analysis |
| **Ensemble** | `ensemble` | Combine multiple models |

**Ensemble note:** `ensemble` supports advanced modes (`average`, `bma`, `stacking`). See [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md) for parameters and examples.

### Classical Models

**Theta Method** — Decomposes trend and curvature. Robust baseline.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

**ARIMA** — Models autocorrelation in the data.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method arima \
  --params "p=2 d=1 q=2"
```

**ETS** — Exponential smoothing with optional trend/seasonality.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --method ets \
  --params "seasonality=24"
```

### Foundation Models

Pre-trained deep learning models that work without tuning.

**Chronos 2** — Amazon's foundation model for time series.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --method chronos2
```

*Requires: `pip install chronos-forecasting torch`*

**Chronos-Bolt** — Faster Chronos variant.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --method chronos_bolt
```

**TimesFM** — Google's foundation model (TimesFM 2.x adapter).
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --method timesfm
```

*Requires: `pip install -e .[forecast-timesfm]`.*

**Lag-Llama** — Foundation model via GluonTS.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --method lag_llama --params '{"ckpt_path":"C:/path/to/lag-llama.ckpt"}'
```

*Requires a manual nonstandard setup (`gluonts[torch]` plus Lag-Llama) and is not part of the supported Python 3.14 environment in this project.*

### Monte Carlo Simulation

Generates thousands of possible future paths instead of a single forecast.

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method mc_gbm --params "n_sims=2000 seed=42"
```

**Output includes:**
- Point forecast (median of simulations)
- Percentile bands (5th, 25th, 75th, 95th)
- Useful for risk sizing and barrier analysis

### Analog Forecasting

Finds historical windows similar to the current pattern and averages what happened next.

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method analog --params "window_size=64 top_k=20"
```

**Parameters:**
- `window_size`: Pattern length to match (default: 64 bars)
- `search_depth`: How far back to search (default: 5000 bars)
- `top_k`: Number of similar patterns to average (default: 20)
- `metric`: Distance metric (euclidean, dtw, cosine)

---

## Model Libraries

mtdata supports multiple forecasting libraries. Use `--library` to select:

| Library | Description | Example Models |
|---------|-------------|----------------|
| `native` | Built-in implementations | theta, naive, mc_gbm, analog |
| `statsforecast` | Nixtla's fast statistical models | AutoARIMA, AutoETS, Theta |
| `sktime` | Scikit-learn style time series | Various forecasters |
| `mlforecast` | ML models with lag features | LightGBM, RandomForest |
| `pretrained` | Foundation models | Chronos, Chronos-Bolt, TimesFM, Lag-Llama (manual/nonstandard setup only) |

**List models in a library:**
```bash
mtdata-cli forecast_list_library_models native
mtdata-cli forecast_list_library_models statsforecast
mtdata-cli forecast_list_library_models pretrained
```

---

## Backtesting

Validate forecast accuracy with rolling-origin backtests.

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --methods "theta sf_autoarima analog" --steps 20 --spacing 10
```

**Parameters:**
- `--steps`: Number of historical test points
- `--spacing`: Bars between test points
- `--methods`: Space-separated list of models to compare

**Output includes:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Directional accuracy

See **[BACKTESTING.md](forecast/BACKTESTING.md)** for complete guide including parameter optimization.

---

## Adding Features

### Technical Indicators

Add indicators as input features:
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta \
  --features "include=close,volume"
```

### Denoising

Smooth data before forecasting:
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta \
  --denoise '{"method":"ema","params":{"alpha":0.2}}'
```

See [DENOISING.md](DENOISING.md) for available filters.

---

## Submodule Documentation

- **[BACKTESTING.md](forecast/BACKTESTING.md)** — Rolling backtests and parameter optimization
- **[FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md)** — Detailed `forecast_generate` reference
- **[VOLATILITY.md](forecast/VOLATILITY.md)** — Volatility forecasting methods
- **[REGIMES.md](forecast/REGIMES.md)** — Regime and change-point detection
- **[UNCERTAINTY.md](forecast/UNCERTAINTY.md)** — Confidence and conformal intervals
- **[PATTERN_SEARCH.md](forecast/PATTERN_SEARCH.md)** — Pattern detection and analog search

## Parameter Optimization

Two tools are available for automated hyperparameter tuning:

### Genetic Algorithm (`forecast_tune_genetic`)

Evolutionary search through parameter space. Good for discrete/mixed search spaces.

```bash
mtdata-cli forecast_tune_genetic EURUSD --method theta --horizon 12 \
  --metric avg_rmse --mode min --population 20 --generations 10
```

See [BACKTESTING.md](forecast/BACKTESTING.md) for full parameters and examples.

### Optuna (`forecast_tune_optuna`)

Bayesian optimization with TPE, CMA-ES, or random sampling. Supports early stopping (pruning), parallel trials, and persistent study storage.

```bash
mtdata-cli forecast_tune_optuna EURUSD --method theta --horizon 12 \
  --metric avg_rmse --mode min --n-trials 40 --sampler tpe --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--method` | `theta` | Forecast method to optimize |
| `--n-trials` | 40 | Number of optimization trials |
| `--sampler` | `tpe` | Sampling algorithm: `tpe`, `random`, `cmaes` |
| `--pruner` | `median` | Early stopping: `median`, `hyperband`, `percentile`, `none` |
| `--timeout` | (none) | Max wall-clock seconds |
| `--n-jobs` | 1 | Parallel trial workers |
| `--study-name` | (auto) | Name for resumable study |
| `--storage` | (none) | DB URL for persistence (e.g., `sqlite:///study.db`) |
| `--seed` | 42 | Random seed |

*Requires: `pip install optuna`*

---

## Background Training & Model Store

Heavyweight methods (neural / foundation models, large `mlforecast` runs) can take minutes to fit. mtdata exposes a small task-and-cache layer so those fits happen once and are reused.

```bash
# Kick off a background training job
mtdata-cli forecast_train EURUSD --timeframe H1 --method nhits --horizon 24

# Returns: {"task_id": "...", "status": "queued", ...}

# Poll progress
mtdata-cli forecast_task_status --task-id <task_id> --json
mtdata-cli forecast_task_list --json

# Cancel if needed
mtdata-cli forecast_task_cancel --task-id <task_id>
```

Once the task completes the model is persisted on disk and any later `forecast_generate` call with the same `(method, symbol, timeframe, params)` key reuses it without re-fitting.

```bash
mtdata-cli forecast_models_list --json
mtdata-cli forecast_models_delete --model-id "nhits/EURUSD-H1/abc123"
```

Configuration (see [ENV_VARS.md](ENV_VARS.md#async-training--model-store)):

- `MTDATA_TRAIN_WORKERS` — size of the background training thread pool (default `4`).
- `MTDATA_HEAVY_LIMIT` — concurrent heavyweight (neural / foundation) jobs (default `1`).
- `MTDATA_MODEL_STORE` — root directory for cached models (default `~/.mtdata/models`).
- `MTDATA_MODEL_TTL_DAYS` — cache expiry in days (default `7`).

`forecast_generate` will also auto-train in the background when called with `async_mode=true` and the requested method is heavy / moderate; the response includes a `task_id` you can poll with `forecast_task_status`. Without `async_mode`, `forecast_generate` blocks until the fit completes (and still caches the result for next time).

---

## Quick Reference

| Task | Command |
|------|---------|
| List methods | `mtdata-cli forecast_list_methods` |
| Basic forecast | `mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta` |
| Foundation method | `mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2 --horizon 24` |
| Monte Carlo | `mtdata-cli forecast_generate EURUSD --method mc_gbm --params "n_sims=2000"` |
| Backtest | `mtdata-cli forecast_backtest_run EURUSD --methods "theta analog" --steps 20` |
| Conformal intervals | `mtdata-cli forecast_conformal_intervals EURUSD --method theta --horizon 12` |
| Tune (genetic) | `mtdata-cli forecast_tune_genetic EURUSD --method theta --metric avg_rmse` |
| Tune (Optuna) | `mtdata-cli forecast_tune_optuna EURUSD --method theta --metric avg_rmse --n-trials 40` |

---

## See Also

- [CLI.md](CLI.md) — Full command reference
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — Barrier optimization
- [TEMPORAL.md](TEMPORAL.md) — Seasonal analysis
- [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) — QuantLib pricing tools

