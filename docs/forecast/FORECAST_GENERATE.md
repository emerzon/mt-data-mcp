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
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta
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

### Model Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--library` | `native` | Model library: native, statsforecast, sktime, mlforecast, pretrained |
| `--model` | `theta` | Model name within the library |
| `--model-params` | — | Model-specific parameters (JSON or `key=value`) |

### Window
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timeframe` | `H1` | Candle timeframe |
| `--horizon` | 12 | Bars to forecast |
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
- `volatility`: routes to the volatility forecasters (same family as `forecast_volatility_estimate`). When using `--quantity volatility`, set `--model` to a volatility method (e.g., `ewma`, `garch`).

Examples:
```bash
# Price forecast (default)
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity price

# Return forecast (log returns) + reconstructed price path
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity return

# Volatility forecast (recommended alternative: use forecast_volatility_estimate)
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity volatility --model ewma
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
python cli.py forecast_generate EURUSD --horizon 12 --model mlf_lightgbm \
  --features '{"include":["close","volume"]}' \
  --dimred-method pca --dimred-params "n_components=5"
```

Tip: the Web UI exposes a broader method list via `GET /api/dimred/methods` (for example: `svd` (TruncatedSVD), `umap`, `isomap`), depending on what is installed (see [../WEB_API.md](../WEB_API.md)).

---

## Model Libraries

### Native (`--library native`)

Built-in implementations with minimal dependencies.

```bash
python cli.py forecast_generate EURUSD --library native --model theta
python cli.py forecast_generate EURUSD --library native --model arima
python cli.py forecast_generate EURUSD --library native --model mc_gbm
python cli.py forecast_generate EURUSD --library native --model analog
```

**Available models:**
```bash
python cli.py forecast_list_library_models native
```

### StatsForecast (`--library statsforecast`)

Fast statistical models from Nixtla.

```bash
python cli.py forecast_generate EURUSD --library statsforecast --model AutoARIMA
python cli.py forecast_generate EURUSD --library statsforecast --model AutoETS
```

**Requires:** `pip install statsforecast`

**Note:** Use capitalized class names (AutoARIMA, not autoarima). Or use native wrappers (`sf_autoarima`).

### Pretrained (`--library pretrained`)

Foundation models pre-trained on large time series datasets.

```bash
python cli.py forecast_generate EURUSD --library pretrained --model chronos2    
python cli.py forecast_generate EURUSD --library pretrained --model chronos_bolt
python cli.py forecast_generate EURUSD --library pretrained --model timesfm
python cli.py forecast_generate EURUSD --library pretrained --model lag_llama \
  --model-params '{"ckpt_path":"C:/path/to/lag-llama.ckpt"}'
```

Tip: `python cli.py forecast_list_library_models pretrained` shows requirements for your current environment.

**Dependencies (by model):**
- `chronos2` / `chronos_bolt`: `chronos-forecasting`, `torch`
- `timesfm`: `timesfm`, `torch` (TimesFM is installed from Git in `requirements.txt`)
- `lag_llama`: `lag-llama`, `gluonts[torch]`, `torch` (may not be installable on all Python versions due to upstream pins)

**Parameters:**
- Common: `context_length`, `quantiles`
- Chronos: `model_name`, `device_map`
- TimesFM: `device`, `model_class`
- Lag-Llama: `ckpt_path` (or `hf_repo`/`hf_filename` for auto-download), `num_samples`, `device`, `freq`

### sktime (`--library sktime`)

Scikit-learn style time series forecasters.

```bash
python cli.py forecast_generate EURUSD --library sktime --model ThetaForecaster
python cli.py forecast_generate EURUSD --library sktime --model NaiveForecaster \
  --model-params "strategy=last sp=24"
```

**Requires:** `pip install sktime`

### MLForecast (`--library mlforecast`)

Machine learning models with lag features.

```bash
python cli.py forecast_generate EURUSD --library mlforecast --model LGBMRegressor
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
| `lag_llama` | Lag-Llama via GluonTS | `context_length=32 num_samples=100` |

---

## Examples

### Basic Forecast
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta
```

### With Confidence Intervals
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model arima --ci-alpha 0.1 --format json
```

### Monte Carlo Simulation
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model mc_gbm --model-params "n_sims=3000 seed=7"
```

### Foundation Model
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --model chronos2 --model-params "context_length=512"
```

### Analog Forecasting
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model analog --model-params "window_size=64 search_depth=5000 top_k=20"
```

### With Denoising
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model theta --denoise ema
```

### Ensemble

`ensemble` combines multiple base methods. Common `--model-params` keys:
- `methods` (list): component methods to run
- `mode` (str): `average`, `bma`, or `stacking`
- `weights` (list): manual weights (only used when `mode=average`)
- `cv_points` (int): walk-forward anchors used for `bma`/`stacking` weighting
- `method_params` (dict): per-method parameter overrides
- `expose_components` (bool): include component forecasts in the JSON output

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model ensemble --model-params '{"methods":["theta","naive","arima"],"mode":"average"}'

# Bayesian model averaging (weights inferred from walk-forward CV)
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model ensemble --model-params '{"methods":["theta","naive","fourier_ols"],"mode":"bma","cv_points":12}'
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
| List methods | `python cli.py forecast_list_methods` |
| List library models | `python cli.py forecast_list_library_models native` |
| Basic forecast | `python cli.py forecast_generate EURUSD --model theta --horizon 12` |
| With CI | `python cli.py forecast_generate EURUSD --model theta --ci-alpha 0.1` |
| Foundation model | `python cli.py forecast_generate EURUSD --library pretrained --model chronos2` |
| JSON output | `python cli.py forecast_generate EURUSD --model theta --format json` |

---

## See Also

- [../FORECAST.md](../FORECAST.md) — Overview
- [../DENOISING.md](../DENOISING.md) — Preprocessing
- [VOLATILITY.md](VOLATILITY.md) — Volatility forecasting
- [UNCERTAINTY.md](UNCERTAINTY.md) — Confidence intervals
