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
```

**Requires:** `pip install chronos-forecasting torch`

**Parameters:**
- `context_length`: How many bars to feed (default: auto)
- `device_map`: Device to use (auto, cpu, cuda)

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
| `ensemble` | Combine multiple methods | `methods=theta,naive,arima` |

### Foundation

| Model | Description | Example Params |
|-------|-------------|----------------|
| `chronos2` | Amazon Chronos-II | `context_length=512` |
| `chronos_bolt` | Fast Chronos variant | `context_length=256` |

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
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model ensemble --model-params "methods=theta,naive,arima"
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
