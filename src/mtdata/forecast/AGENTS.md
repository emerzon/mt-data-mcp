# forecast/ — Forecasting Engine

Price forecasting pipeline: preprocessing → method selection → execution → post-processing. 34 files across `forecast/` and `forecast/methods/`.

## FILE MAP

### Engine Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| `forecast_engine.py` | 735 | Main orchestrator: prep → run → post-process |
| `forecast_preprocessing.py` | 641 | Data cleaning, normalization, feature extraction |
| `forecast_validation.py` | — | Input validation and sanity checks |
| `forecast_methods.py` | — | Method dispatch helpers |
| `forecast.py` | — | High-level forecast convenience functions |
| `use_cases.py` | 856 | Forecast use case orchestration |
| `requests.py` | — | Forecast request models |

### Method Registry

| File | Purpose |
|------|---------|
| `interface.py` | `ForecastMethod` ABC + `ForecastResult` dataclass |
| `forecast_registry.py` | Maps method names → implementations |
| `registry.py` | Additional registry helpers |

### Barrier Analysis

| File | Lines | Purpose |
|------|-------|---------|
| `barriers.py` | — | Barrier tool entry points |
| `barriers_shared.py` | 501 | Shared barrier types and helpers |
| `barriers_probabilities.py` | — | TP/SL hit probability calculation |
| `barriers_optimization.py` | 1504 | Barrier level optimization |

### Specialized Engines

| File | Lines | Purpose |
|------|-------|---------|
| `volatility.py` | 1258 | Volatility estimation (GARCH, realized, etc.) |
| `monte_carlo.py` | 720 | Monte Carlo price simulation |
| `backtest.py` | — | Rolling forecast backtesting |
| `tune.py` | 829 | Hyperparameter tuning (Optuna) |
| `quantlib_tools.py` | — | QuantLib barrier pricing, Heston calibration |

### Shared

| File | Purpose |
|------|---------|
| `common.py` | Time normalization utilities (**DO NOT re-normalize**) |
| `exceptions.py` | Forecast-specific exceptions |
| `target_builder.py` | Forecast target construction |

### methods/ — Model Implementations

| File | Lines | Models | Dep Group |
|------|-------|--------|-----------|
| `classical.py` | — | Theta, naive, seasonal naive | core |
| `ets_arima.py` | 527 | ETS, ARIMA, auto-ARIMA | forecast-classical |
| `statsforecast.py` | — | StatsForecast wrappers | forecast-classical |
| `sktime.py` | — | sktime model wrappers | forecast-classical |
| `mlforecast.py` | — | LightGBM via mlforecast | forecast-classical |
| `neural.py` | — | Neural network models | forecast-foundation |
| `pretrained.py` | 817 | Chronos, TimesFM | forecast-foundation |
| `pretrained_helpers.py` | — | Pretrained model utilities | forecast-foundation |
| `gluonts_extra.py` | 669 | GluonTS/Lag-Llama (excluded on 3.14) | — |
| `analog.py` | — | Analog/pattern-matching forecast | core |
| `monte_carlo.py` | — | MC-specific forecast method | core |

## HOW TO ADD A FORECAST METHOD

1. Create class in `methods/` implementing `ForecastMethod` ABC from `interface.py`
2. Implement `name`, `category`, `required_packages`, `forecast()` method
3. Register in `forecast_registry.py` mapping name → class
4. If new dependency needed, add to appropriate group in `pyproject.toml`

## ANTI-PATTERNS

- **Never** re-normalize already-normalized time data — see `common.py` comment.
- **Never** import all methods eagerly — some have heavy deps (torch, QuantLib). Use lazy imports.
- Methods in `forecast-foundation` group require GPU-capable torch — guard with try/except on import.
