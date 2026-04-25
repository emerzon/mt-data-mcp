# `forecast_generate` Reference

Detailed reference for the `forecast_generate` command, which produces price forecasts for the next N bars.

**Related:**
- [../FORECAST.md](../FORECAST.md) — Forecasting overview
- [../TECHNICAL_INDICATORS.md](../TECHNICAL_INDICATORS.md) — Indicators as features
- [../DENOISING.md](../DENOISING.md) — Preprocessing
- [../BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — Using forecasts for TP/SL

---

## Basic Usage

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

**Output:**
```
forecast[12]{time,forecast}:
    "2026-01-01 18:00",1.17569
    "2026-01-01 19:00",1.17570
    ...
```

---

## Parameters

### Required
| Parameter | Description |
|-----------|-------------|
| `symbol` | Trading symbol (positional argument) |

### Method Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--library` | `native` | Method library: native, statsforecast, sktime, mlforecast, pretrained |
| `--method` | `theta` | Method name within the library |
| `--params` | — | Method-specific parameters (JSON or `key=value`) |

### Window
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timeframe` | `H1` | Candle timeframe |
| `--horizon` | 5 | Bars to forecast |
| `--lookback` | auto | Historical bars to use |
| `--as-of` | now | Reference time (for backtesting) |

### Target
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--quantity` | `price` | What to forecast: price, return, volatility |

### Uncertainty
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ci-alpha` | 0.05 | Confidence interval alpha (0.05 = 95% CI) |

### Pipeline
| Parameter | Description |
|-----------|-------------|
| `--denoise` | Denoising method (ema, kalman, etc.) |
| `--features` | Feature specification |
| `--dimred-method` | Dimensionality reduction method |
| `--dimred-params` | Dimred parameters |

---

## Quantity (`--quantity`)

`forecast_generate` can model different target quantities:

- `price` (default): forecasts the future **close price** (output includes `forecast_price`).
- `return`: forecasts **log returns** (`ln(close_t / close_{t-1})`). Output includes `forecast_return` and a reconstructed `forecast_price` path when possible.
- `volatility`: routes to the volatility forecasters (same family as `forecast_volatility_estimate`). When using `--quantity volatility`, set `--method` to a volatility method (e.g., `ewma`, `garch`).

Examples:
```bash
# Price forecast (default)
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity price

# Return forecast (log returns) + reconstructed price path
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity return

# Volatility forecast (recommended alternative: use forecast_volatility_estimate)
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity volatility --method ewma
```

---

## Dimensionality Reduction (`--dimred-method`)

Dimensionality reduction (dimred) compresses the feature matrix when you provide many inputs (for example via `--features`). This is most useful for ML-style methods that consume multiple features.

Supported dimred methods in the forecasting pipeline:
- `pca` — Principal Component Analysis (`n_components`)
- `tsne` — t-SNE (`n_components` is typically 2 or 3)
- `selectkbest` — keep top-K features (`k`)

Examples:
```bash
mtdata-cli forecast_generate EURUSD --horizon 12 --method mlf_lightgbm \
  --features '{"include":["close","volume"]}' \
  --dimred-method pca --dimred-params "n_components=5"
```

Tip: the Web UI exposes a broader method list via `GET /api/dimred/methods` (for example: `svd` (TruncatedSVD), `umap`, `isomap`), depending on what is installed (see [../WEB_API.md](../WEB_API.md)).

---

## Model Libraries

### Native (`--library native`)

Built-in implementations with minimal dependencies.

```bash
mtdata-cli forecast_generate EURUSD --library native --method theta
mtdata-cli forecast_generate EURUSD --library native --method arima
mtdata-cli forecast_generate EURUSD --library native --method mc_gbm
mtdata-cli forecast_generate EURUSD --library native --method analog
```

**Available models:**
```bash
mtdata-cli forecast_list_library_models native
```

### StatsForecast (`--library statsforecast`)

Fast statistical models from Nixtla.

```bash
mtdata-cli forecast_generate EURUSD --library statsforecast --method AutoARIMA
mtdata-cli forecast_generate EURUSD --library statsforecast --method AutoETS
```

**Requires:** `pip install statsforecast`

**Note:** Use capitalized class names (AutoARIMA, not autoarima). Or use native wrappers (`sf_autoarima`).

### Pretrained (`--library pretrained`)

Foundation models pre-trained on large time series datasets.

On the supported Python 3.14 install path:
- `chronos2` and `chronos_bolt` are part of the package-index install path
- `timesfm` remains a Git-backed extra
- `lag_llama` is documented for completeness, but not part of the supported environment

```bash
mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2    
mtdata-cli forecast_generate EURUSD --library pretrained --method chronos_bolt
mtdata-cli forecast_generate EURUSD --library pretrained --method timesfm
mtdata-cli forecast_generate EURUSD --library pretrained --method lag_llama \
  --params '{"ckpt_path":"C:/path/to/lag-llama.ckpt"}'
```

Tip: `mtdata-cli forecast_list_library_models pretrained` shows requirements for your current environment.

**Dependencies (by model):**
- `chronos2` / `chronos_bolt`: `chronos-forecasting`, `torch`
- `timesfm`: `timesfm`, `torch` (install with `pip install -e .[forecast-timesfm]`)
- `lag_llama`: `lag-llama`, `gluonts[torch]`, `torch` (manual/nonstandard setup only; unsupported on the project's Python 3.14 runtime)

**Parameters:**
- Common: `context_length`, `quantiles`
- Chronos: `model_name`, `device_map`
- TimesFM: `device`, `model_class`
- Lag-Llama: `ckpt_path` (or `hf_repo`/`hf_filename` for auto-download), `num_samples`, `device`, `freq`

### sktime (`--library sktime`)

Scikit-learn style time series forecasters.

```bash
mtdata-cli forecast_generate EURUSD --library sktime --method ThetaForecaster
mtdata-cli forecast_generate EURUSD --library sktime --method NaiveForecaster \
  --params "strategy=last sp=24"
```

**Requires:** `pip install sktime`

### MLForecast (`--library mlforecast`)

Machine learning models with lag features.

```bash
mtdata-cli forecast_generate EURUSD --library mlforecast --method LGBMRegressor
```

**Requires:** `pip install mlforecast lightgbm`

---

## Common Models

### Classical

| Model | Description | Example Params |
|-------|-------------|----------------|
| `theta` | Theta decomposition | — |
| `naive` | Last value repeated | — |
| `ses` | Simple exponential smoothing | `alpha=0.3` |
| `holt` | Double exponential smoothing | `damped=true` |
| `arima` | ARIMA(p,d,q) | `p=2 d=1 q=2` |
| `sarima` | Seasonal ARIMA | `seasonality=24` |

### Simulation

| Model | Description | Example Params |
|-------|-------------|----------------|
| `mc_gbm` | Monte Carlo GBM | `n_sims=2000 seed=42` |
| `hmm_mc` | HMM-based Monte Carlo | `n_states=2 n_sims=1000` |

### Pattern-Based

| Model | Description | Example Params |
|-------|-------------|----------------|
| `analog` | Historical pattern matching | `window_size=64 top_k=20` |
| `ensemble` | Combine multiple methods | `{"methods":["theta","naive"],"mode":"bma"}` |

### Foundation

| Model | Description | Example Params |
|-------|-------------|----------------|
| `chronos2` | Amazon Chronos-II | `context_length=512` |
| `chronos_bolt` | Fast Chronos variant | `context_length=256` |
| `timesfm` | TimesFM (foundation model adapter) | `context_length=512` |
| `lag_llama` | Lag-Llama via GluonTS (manual/nonstandard setup only) | `context_length=32 num_samples=100` |

---

## Examples

### Basic Forecast
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

### With Confidence Intervals
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method arima --ci-alpha 0.1 --json
```

### Monte Carlo Simulation
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method mc_gbm --params "n_sims=3000 seed=7"
```

### Foundation Model
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --method chronos2 --params "context_length=512"
```

### Analog Forecasting
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method analog --params "window_size=64 search_depth=5000 top_k=20"
```

### With Denoising
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method theta --denoise ema
```

### Ensemble

`ensemble` combines multiple base methods. Common `--params` keys:
- `methods` (list): component methods to run
- `mode` (str): `average`, `bma`, or `stacking`
- `weights` (list): manual weights (only used when `mode=average`)
- `cv_points` (int): walk-forward anchors used for `bma`/`stacking` weighting
- `method_params` (dict): per-method parameter overrides
- `expose_components` (bool): include component forecasts in the JSON output

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method ensemble --params '{"methods":["theta","naive","arima"],"mode":"average"}'

# Bayesian model averaging (weights inferred from walk-forward CV)
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method ensemble --params '{"methods":["theta","naive","fourier_ols"],"mode":"bma","cv_points":12}'
```

---

## Output Fields

| Field | Description |
|-------|-------------|
| `forecast` | Array of predicted values |
| `time` | Timestamps for each forecast point |
| `lower` | Lower confidence bound (if `--ci-alpha`) |
| `upper` | Upper confidence bound (if `--ci-alpha`) |
| `trend` | Detected trend direction (if available) |
| `method` | Method used |
| `params_used` | Actual parameters applied |

---

## Quick Reference

| Task | Command |
|------|---------|
| List methods | `mtdata-cli forecast_list_methods` |
| List library models | `mtdata-cli forecast_list_library_models native` |
| Basic forecast | `mtdata-cli forecast_generate EURUSD --method theta --horizon 12` |
| With CI | `mtdata-cli forecast_generate EURUSD --method theta --ci-alpha 0.1` |
| Foundation method | `mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2` |
| JSON output | `mtdata-cli forecast_generate EURUSD --method theta --json` |

---

## See Also

- [../FORECAST.md](../FORECAST.md) — Overview
- [../DENOISING.md](../DENOISING.md) — Preprocessing
- [VOLATILITY.md](VOLATILITY.md) — Volatility forecasting
- [UNCERTAINTY.md](UNCERTAINTY.md) — Confidence intervals

