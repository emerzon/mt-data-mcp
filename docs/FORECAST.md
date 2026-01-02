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
# Theta model (fast, reliable baseline)
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta

# With JSON output for programmatic use
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta --format json
```

### Choosing a Model

Run this to see available models:
```bash
python cli.py forecast_list_methods
```

| Category | Models | When to Use |
|----------|--------|-------------|
| **Classical** | `theta`, `naive`, `ses`, `holt`, `arima` | Fast baselines, short horizons |
| **Seasonal** | `seasonal_naive`, `ets`, `holt_winters_add` | Data with recurring patterns |
| **Statistical** | `sf_autoarima`, `sf_autoets` | Auto-tuning, medium horizons |
| **ML-Based** | `mlf_lightgbm`, `mlf_rf` | Non-linear patterns, feature engineering |
| **Foundation** | `chronos2`, `chronos_bolt` | State-of-the-art, no tuning required |
| **Simulation** | `mc_gbm`, `hmm_mc` | Risk sizing, barrier analysis |
| **Ensemble** | `ensemble` | Combine multiple models |

### Classical Models

**Theta Method** — Decomposes trend and curvature. Robust baseline.
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta
```

**ARIMA** — Models autocorrelation in the data.
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model arima \
  --model-params "p=2 d=1 q=2"
```

**ETS** — Exponential smoothing with optional trend/seasonality.
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 --model ets \
  --model-params "seasonality=24"
```

### Foundation Models

Pre-trained deep learning models that work without tuning.

**Chronos 2** — Amazon's foundation model for time series.
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --model chronos2
```

*Requires: `pip install chronos-forecasting torch`*

### Monte Carlo Simulation

Generates thousands of possible future paths instead of a single forecast.

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model mc_gbm --model-params "n_sims=2000 seed=42"
```

**Output includes:**
- Point forecast (median of simulations)
- Percentile bands (5th, 25th, 75th, 95th)
- Useful for risk sizing and barrier analysis

### Analog Forecasting

Finds historical windows similar to the current pattern and averages what happened next.

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model analog --model-params "window_size=64 top_k=20"
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
| `pretrained` | Foundation models | Chronos, Chronos-Bolt |

**List models in a library:**
```bash
python cli.py forecast_list_library_models native
python cli.py forecast_list_library_models statsforecast
python cli.py forecast_list_library_models pretrained
```

---

## Backtesting

Validate forecast accuracy with rolling-origin backtests.

```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
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
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta \
  --features "include=close,volume"
```

### Denoising

Smooth data before forecasting:
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta \
  --denoise ema --model-params "alpha=0.2"
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

---

## Quick Reference

| Task | Command |
|------|---------|
| List methods | `python cli.py forecast_list_methods` |
| Basic forecast | `python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta` |
| Foundation model | `python cli.py forecast_generate EURUSD --library pretrained --model chronos2 --horizon 24` |
| Monte Carlo | `python cli.py forecast_generate EURUSD --model mc_gbm --model-params "n_sims=2000"` |
| Backtest | `python cli.py forecast_backtest_run EURUSD --methods "theta analog" --steps 20` |
| Conformal intervals | `python cli.py forecast_conformal_intervals EURUSD --method theta --horizon 12` |
