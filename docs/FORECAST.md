# Forecasting guide

Predict **where price might go**, **how wide the range is**, and **how much it might move** — then validate before you act. mtdata covers classical baselines, ML, foundation models, simulation methods, and async training with a model cache.

Forecasts are **estimates**, not guarantees. Pair point forecasts with uncertainty, barriers, and backtests.

**Dense terms:** [Horizon](GLOSSARY.md#horizon) · [Lookback](GLOSSARY.md#lookback) · [Theta](GLOSSARY.md#theta-method) · [ARIMA](GLOSSARY.md#arima-autoregressive-integrated-moving-average) · [Monte Carlo](GLOSSARY.md#monte-carlo-simulation) · [Foundation models](GLOSSARY.md#chronos--foundation-models) · [Conformal](GLOSSARY.md#conformal-intervals) · [MAE / RMSE](GLOSSARY.md#mae-mean-absolute-error)

**Related:** [Glossary](GLOSSARY.md) · [Methods reference](forecast/METHODS.md) · [Volatility](forecast/VOLATILITY.md) · [Uncertainty](forecast/UNCERTAINTY.md) · [Barriers](BARRIER_FUNCTIONS.md)

---

## Key concepts

| Term | Meaning |
|------|---------|
| **Horizon** | How far ahead (in bars). `--horizon 12` on H1 ≈ next 12 hours. |
| **Lookback** | How much history the model sees. More can help, but costs time. |
| **Quantity** | Often price, returns, or other targets depending on method and flags. |

**Confidence vs reality**

1. Read intervals, not only the midline
2. Validate with [backtests](forecast/BACKTESTING.md) before trading ideas
3. Size risk from **volatility**, not a single point forecast

---

## Price forecasting (`forecast_generate`)

### Basic usage

```bash
# Fast, reliable baseline
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta

# Structured output for scripts / agents
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

### Choosing a method

```bash
mtdata-cli forecast_list_methods
mtdata-cli forecast_list_methods --json   # source of truth for *your* install
```

Availability depends on extras you installed:

- Supported foundation options on the Python 3.14 path include Chronos, Chronos-Bolt, and TimesFM (TimesFM via opt-in extra).
- NeuralForecast methods (`nhits`, `tft`, `patchtst`, `nbeatsx`) need a manual `neuralforecast` + `torch` setup. On Windows Python 3.14 they do not resolve because `ray` (a NeuralForecast dependency) has no Windows cp314 wheels.
- Always trust `forecast_list_methods --json` over static docs for what runs locally.

Full per-method keys, defaults, and dependencies: [forecast/METHODS.md](forecast/METHODS.md).

| Category | Models | When to Use |
|----------|--------|-------------|
| **Classical** | `theta`, `naive`, `drift`, `ses`, `holt`, `arima` | Fast baselines, short horizons |
| **Seasonal** | `seasonal_naive`, `ets`, `holt_winters_add`, `holt_winters_mul`, `sarima`, `fourier_ols` | Data with recurring patterns |
| **Statistical** | `sf_autoarima`, `sf_autoets`, `sf_autotheta` | Auto-tuning, medium horizons |
| **ML-Based** | `mlf_lightgbm`, `mlf_rf` | Non-linear patterns, feature engineering |
| **Neural** | `nhits`, `tft`, `patchtst`, `nbeatsx` | Deep learning, long horizons; manual `neuralforecast` install required |
| **Foundation** | `chronos2`, `chronos_bolt`, `timesfm` | Pretrained models (optional deps) |
| **Simulation** | `mc_gbm`, `hmm_mc` | Risk sizing, barrier analysis |
| **Ensemble** | `ensemble` | Combine multiple models |

**Ensemble note:** `ensemble` supports advanced modes (`average`, `bma`, `stacking`). See [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md) for parameters and examples.

---

## Recommended workflow

Treat forecast tools as **stages**, each answering one question:

| Stage | Question | Tool |
|-------|----------|------|
| 1. Discover | Which methods are available here? | `forecast_list_methods` |
| 2. Forecast | What is the point forecast? | `forecast_generate` |
| 3. Uncertainty | How wide is the plausible range? | `forecast_conformal_intervals` |
| 4. Trade levels | What TP/SL levels fit this horizon? | `forecast_barrier_optimize` |
| 5. Probability check | How likely is a specific TP/SL pair? | `forecast_barrier_prob` |
| 6. Validation | Did this method work historically? | `forecast_backtest_run` |
| 7. Tuning | Can parameters improve validation metrics? | `forecast_tune_optuna` or `forecast_tune_genetic` |

Keep `--symbol`, `--timeframe`, `--horizon`, and `--method` aligned across stages unless you are deliberately comparing alternatives.

```bash
mtdata-cli forecast_list_methods
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
mtdata-cli forecast_conformal_intervals EURUSD --timeframe H1 --horizon 12 --method theta
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --direction long
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --methods theta --steps 20 --spacing 12
```

### Reproducibility notes

Defaults vary by method and can change over time. For any result you want to compare later, make the run **self-describing**:

- Save the exact command, including `--symbol`, `--timeframe`, `--horizon`, `--lookback`, `--method`, `--library`, and `--params`.
- Prefer `--json` for stored results so downstream scripts do not depend on text formatting.
- Set important method parameters explicitly instead of relying on implicit defaults.
- Use `forecast_list_methods --json` to confirm which methods are available in the current environment.

Example:

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --method arima --lookback 500 \
  --params "p=2 d=1 q=2" --json
```

---

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

GPU-backed forecast calls run in a short-lived child process by default
(`MTDATA_FORECAST_PROCESS_ISOLATION=gpu`). This lets the child exit after
inference so CUDA context memory is returned instead of staying reserved by a
long MCP server session. Set `MTDATA_FORECAST_PROCESS_ISOLATION=all` to isolate
every forecast tool call, or `off` to keep the previous in-process behavior.

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
| `pretrained` | Foundation models | Chronos, Chronos-Bolt, TimesFM |

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

Three tools are available for automated tuning and configuration search:

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

### Configuration Search (`forecast_optimize_hints`)

Broader than single-method tuning: `forecast_optimize_hints` runs a genetic search across **timeframes, methods, and method-specific parameters at once**, returning the top-N configurations ranked by a composite trading-fitness score (it falls back to forecast error and directional accuracy when trade metrics are unavailable). Use it to answer *"which timeframe/method/params should I even start from?"* before drilling in with `forecast_tune_genetic` / `forecast_tune_optuna`.

```bash
mtdata-cli forecast_optimize_hints EURUSD --timeframes H1 H4 D1 \
  --methods theta ets arima --horizon 12 --top-n 5 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timeframes` | `H1 H4 D1 W1` | Timeframes to search (space- or comma-separated) |
| `--methods` | (all eligible) | Methods to search; omit to let the search pick |
| `--horizon` | 12 | Bars forecast after each backtest anchor |
| `--steps` | 5 | Rolling-origin backtest anchors per candidate |
| `--population` / `--generations` | 8 / 5 | Genetic search population and generation counts |
| `--fitness-metric` | `composite` | Objective; `composite` blends trading metrics with accuracy |
| `--top-n` | 5 | Number of ranked configurations to return |

---

## Background Training & Model Store

Heavyweight methods (neural / foundation models, large `mlforecast` runs) can take minutes to fit. mtdata exposes a small task-and-cache layer so those fits happen once and are reused.

```bash
# Kick off a background training job
mtdata-cli forecast_train EURUSD --timeframe H1 --method nhits --horizon 24

# Returns: {"task_id": "...", "status": "queued", ...}

# Poll progress
mtdata-cli forecast_task_status --task-id <task_id> --json
mtdata-cli forecast_task_wait --task-id <task_id> --timeout-seconds 120 --json
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
- `MTDATA_FORECAST_JOBS_DB` — durable SQLite task registry (default `~/.mtdata/forecast/jobs.sqlite`).
- `MTDATA_TRAIN_TIMEOUT_*_SECONDS` — per-category training timeouts for `instant`, `fast`, `moderate`, and `heavy` methods.
- `MTDATA_FORECAST_HEARTBEAT_SECONDS`, `MTDATA_FORECAST_CANCEL_GRACE_SECONDS`, `MTDATA_FORECAST_SWEEPER_SECONDS` — task liveness, cancellation, and cleanup tuning.
- `MTDATA_MODEL_STORE` — root directory for cached models (default `~/.mtdata/models`).
- `MTDATA_MODEL_TTL_DAYS` — cache idle expiry in days since last use (default `7`); this is not a maximum model age.

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
