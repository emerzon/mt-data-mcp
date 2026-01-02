# Backtesting Guide

Backtesting validates forecast accuracy by testing models on historical data. This guide covers rolling-origin backtests, performance metrics, and parameter optimization.

**Related:**
- [GLOSSARY.md](../GLOSSARY.md) — Definitions of MAE, RMSE, Sharpe ratio, etc.
- [FORECAST.md](../FORECAST.md) — Forecasting methods
- [FORECAST_GENERATE.md](FORECAST_GENERATE.md) — Detailed forecast options
- [VOLATILITY.md](VOLATILITY.md) — Volatility forecasting

---

## Key Concepts

### What is Backtesting?

Backtesting answers: *"How well would this forecast method have performed on past data?"*

Instead of testing on the same data used for training (overfitting), backtesting:
1. Picks historical "anchor" points
2. At each anchor, generates a forecast using only data available at that time
3. Compares the forecast to what actually happened
4. Aggregates error metrics across all test points

### Rolling-Origin Backtest

The standard backtesting approach in mtdata:

```
Timeline: [----history----][forecast horizon]
                          ^
                       anchor
```

**Parameters:**
- **steps**: Number of anchor points to test
- **spacing**: Bars between anchor points  
- **horizon**: How far ahead each forecast predicts

**Example:** `steps=20, spacing=10, horizon=12` creates 20 test points, each 10 bars apart, each forecasting 12 bars ahead.

---

## Quick Start

### Compare Forecasting Methods

```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --methods "theta sf_autoarima analog" --steps 20 --spacing 10
```

### Single Method with Custom Parameters

```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --methods theta --params "seasonality=24" --steps 30
```

### Volatility Backtest

```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --quantity volatility --methods "ewma parkinson garch" --steps 20
```

---

## Command Reference

```bash
python cli.py forecast_backtest_run <SYMBOL> [OPTIONS]
```

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Trading symbol (e.g., EURUSD) |
| `--timeframe` | H1 | Candle timeframe |
| `--horizon` | 12 | Bars to forecast at each anchor |
| `--steps` | 5 | Number of test anchors |
| `--spacing` | 20 | Bars between anchors |
| `--methods` | auto | Space or comma-separated method names |

### Method Parameters

| Parameter | Description |
|-----------|-------------|
| `--params` | Parameters applied to all methods (JSON or `k=v`) |
| `--params-per-method` | Per-method parameters: `{"theta": {"seasonality": 24}}` |

**Example with per-method params:**
```bash
python cli.py forecast_backtest_run EURUSD --horizon 12 \
  --methods "theta arima" \
  --params-per-method '{"theta": {"seasonality": 24}, "arima": {"p": 2, "d": 1, "q": 2}}'
```

### Quantity and Target

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--quantity` | `price`, `return`, `volatility` | What to forecast |
| `--target` | `price`, `return` | Target series for comparison |

**Examples:**
```bash
# Forecast returns instead of prices
python cli.py forecast_backtest_run EURUSD --quantity return --target return

# Backtest volatility methods
python cli.py forecast_backtest_run EURUSD --quantity volatility --methods "ewma garch"
```

### Trade Simulation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--slippage-bps` | 0.0 | Transaction cost in basis points (1 bp = 0.01%) |
| `--trade-threshold` | 0.0 | Minimum expected return to trigger a trade |

**Example with trading costs:**
```bash
# Simulate 2 bps slippage per side (4 bps round-trip)
python cli.py forecast_backtest_run EURUSD --horizon 12 --methods theta \
  --slippage-bps 2 --trade-threshold 0.0005
```

### Preprocessing Options

| Parameter | Description |
|-----------|-------------|
| `--denoise` | Denoising method (e.g., `ema`, `kalman`) |
| `--denoise-params` | Denoising parameters |
| `--features` | Feature engineering spec |
| `--dimred-method` | Dimensionality reduction (e.g., `pca`) |
| `--dimred-params` | Dim reduction parameters |

**Example with denoising:**
```bash
python cli.py forecast_backtest_run EURUSD --horizon 12 --methods theta \
  --denoise ema --denoise-params "alpha=0.2"
```

---

## Understanding Output

### Aggregate Metrics

```json
{
  "results": {
    "theta": {
      "success": true,
      "avg_mae": 0.00142,
      "avg_rmse": 0.00186,
      "avg_directional_accuracy": 0.583,
      "win_rate": 0.625,
      "successful_tests": 20,
      "num_tests": 20
    }
  }
}
```

| Metric | Description | Good Value |
|--------|-------------|------------|
| `avg_mae` | Mean Absolute Error (average) | Lower is better |
| `avg_rmse` | Root Mean Squared Error (average) | Lower is better |
| `avg_directional_accuracy` | % of correct direction predictions | > 0.55 |
| `win_rate` | % of profitable trades | > 0.50 |
| `successful_tests` | Tests that completed without error | = num_tests |

### Trading Performance Metrics

When `slippage-bps` or `trade-threshold` is set:

```json
{
  "metrics": {
    "avg_return_per_trade": 0.00082,
    "win_rate": 0.625,
    "sharpe_ratio": 1.45,
    "max_drawdown": 0.034,
    "calmar_ratio": 2.12,
    "cumulative_return": 0.0164,
    "annual_return": 0.087,
    "num_trades": 20,
    "trades_per_year": 365
  }
}
```

| Metric | Description | Good Value |
|--------|-------------|------------|
| `sharpe_ratio` | Risk-adjusted return | > 1.0 |
| `max_drawdown` | Largest peak-to-trough decline | < 0.10 (10%) |
| `calmar_ratio` | Annual return / max drawdown | > 1.0 |
| `cumulative_return` | Total return over test period | > 0 |
| `win_rate` | Fraction of profitable trades | > 0.50 |

### Per-Anchor Details

Add `--format json` to see individual test results:

```json
{
  "details": [
    {
      "anchor": "2025-12-15 14:00",
      "success": true,
      "mae": 0.00128,
      "rmse": 0.00165,
      "directional_accuracy": 0.636,
      "forecast": [1.0542, 1.0545, ...],
      "actual": [1.0540, 1.0548, ...],
      "entry_price": 1.0538,
      "exit_price": 1.0552,
      "expected_return": 0.00094,
      "position": "long",
      "trade_return": 0.00133
    }
  ]
}
```

---

## Method Comparison

### Default Methods

If `--methods` is not specified, the backtest uses available classical methods:
- `naive`, `drift`, `seasonal_naive`, `theta`, `fourier_ols`
- Plus `sf_autoarima`, `sf_theta` if statsforecast is installed

### Comparing Categories

**Fast baselines:**
```bash
python cli.py forecast_backtest_run EURUSD --horizon 12 \
  --methods "naive drift theta seasonal_naive" --steps 30
```

**Statistical models:**
```bash
python cli.py forecast_backtest_run EURUSD --horizon 12 \
  --methods "sf_autoarima sf_autoets sf_theta" --steps 30
```

**ML models:**
```bash
python cli.py forecast_backtest_run EURUSD --horizon 12 \
  --methods "mlf_lightgbm mlf_rf" --steps 20
```

**Foundation models:**
```bash
python cli.py forecast_backtest_run EURUSD --horizon 24 \
  --methods "chronos2 chronos_bolt" --steps 15
```

---

## Parameter Optimization

### Genetic Search (`forecast_tune_genetic`)

Automatically find optimal parameters for a forecasting method:

```bash
python cli.py forecast_tune_genetic EURUSD --timeframe H1 --method theta \
  --horizon 12 --steps 20 --spacing 10 \
  --metric avg_rmse --mode min \
  --population 20 --generations 10
```

### Genetic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--method` | (required) | Method to optimize |
| `--metric` | `avg_rmse` | Metric to optimize |
| `--mode` | `min` | `min` to minimize, `max` to maximize |
| `--population` | 20 | Population size per generation |
| `--generations` | 10 | Number of generations |
| `--crossover-rate` | 0.7 | Probability of crossover |
| `--mutation-rate` | 0.2 | Probability of mutation |
| `--seed` | None | Random seed for reproducibility |

### Available Metrics

| Metric | Mode | Description |
|--------|------|-------------|
| `avg_mae` | min | Minimize mean absolute error |
| `avg_rmse` | min | Minimize root mean squared error |
| `avg_directional_accuracy` | max | Maximize direction accuracy |
| `win_rate` | max | Maximize profitable trades |
| `sharpe_ratio` | max | Maximize risk-adjusted return |
| `calmar_ratio` | max | Maximize return/drawdown ratio |

### Custom Search Space

Define which parameters to search:

```bash
python cli.py forecast_tune_genetic EURUSD --method theta \
  --search-space '{"seasonality": {"type": "int", "min": 12, "max": 48}}'
```

**Search space format:**
```json
{
  "param_name": {
    "type": "int" | "float" | "categorical",
    "min": 0,
    "max": 100,
    "log": false,          // For float: use log scale
    "choices": [...]       // For categorical
  }
}
```

### Default Search Spaces

Each method has sensible defaults. Examples:

| Method | Parameters Searched |
|--------|-------------------|
| `theta` | seasonality (8-72) |
| `arima` | p (0-3), d (0-2), q (0-3) |
| `fourier_ols` | m (8-96), K (1-6), trend (true/false) |
| `sf_autoarima` | seasonality, stepwise, d, D |
| `mlf_lightgbm` | n_estimators, learning_rate, num_leaves, max_depth |

---

## Practical Examples

### Example 1: Find Best Method for Scalping

```bash
# Short horizon, tight spacing
python cli.py forecast_backtest_run EURUSD --timeframe M5 --horizon 6 \
  --methods "naive theta fourier_ols sf_autoarima" \
  --steps 50 --spacing 12 \
  --slippage-bps 1 --trade-threshold 0.0003
```

**What to look for:**
- Highest `win_rate` with positive `avg_trade_return`
- Low `max_drawdown`
- `sharpe_ratio` > 1.0

### Example 2: Optimize Theta for Swing Trading

```bash
# Step 1: Find optimal seasonality
python cli.py forecast_tune_genetic EURUSD --timeframe H4 --method theta \
  --horizon 48 --steps 30 --spacing 24 \
  --metric sharpe_ratio --mode max \
  --population 30 --generations 15

# Step 2: Backtest with optimal params
python cli.py forecast_backtest_run EURUSD --timeframe H4 --horizon 48 \
  --methods theta --params "seasonality=36" \
  --steps 50 --slippage-bps 2
```

### Example 3: Compare Volatility Methods

```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --quantity volatility \
  --methods "ewma parkinson garch har_rv" \
  --steps 30 --spacing 24
```

**Output interpretation:**
- `forecast_sigma`: Predicted volatility
- `realized_sigma`: Actual volatility that occurred
- `mae`: Error between forecast and realized

### Example 4: Robust Testing with Denoising

```bash
# Test if denoising improves accuracy
python cli.py forecast_backtest_run EURUSD --horizon 12 --methods theta \
  --steps 30 --denoise ema --denoise-params "alpha=0.3"

# Compare to non-denoised
python cli.py forecast_backtest_run EURUSD --horizon 12 --methods theta \
  --steps 30
```

### Example 5: Walk-Forward Optimization

Simulate real-world model updates:

```bash
# Period 1: Optimize on first 6 months
python cli.py forecast_tune_genetic EURUSD --method theta --horizon 12 \
  --steps 50 --spacing 24 --metric avg_rmse

# Record best params, then test on next 3 months with those params
python cli.py forecast_backtest_run EURUSD --horizon 12 --methods theta \
  --params "seasonality=24" --steps 30 --spacing 24

# Repeat: re-optimize, test out-of-sample
```

---

## Interpreting Results

### Good Results Checklist

✅ `avg_rmse` is small relative to price volatility  
✅ `avg_directional_accuracy` > 0.55 (better than random)  
✅ `win_rate` > 0.50 with positive `avg_trade_return`  
✅ `sharpe_ratio` > 1.0  
✅ `max_drawdown` < 10-15%  
✅ Results consistent across different `spacing` values  

### Warning Signs

⚠️ Very high accuracy on backtests but poor live results → overfitting  
⚠️ `successful_tests` << `num_tests` → method fails frequently  
⚠️ `avg_rmse` much larger than `avg_mae` → outlier errors  
⚠️ `max_drawdown` > 20% → high risk  
⚠️ Results vary wildly with small parameter changes → unstable  

### Avoiding Overfitting

1. **Use enough test points:** `steps` ≥ 20 for statistical significance
2. **Test across timeframes:** Method should work on H1, H4, D1
3. **Test across symbols:** Don't optimize for a single pair
4. **Out-of-sample validation:** Reserve recent data for final test
5. **Realistic costs:** Include `slippage-bps` and `trade-threshold`

---

## Performance Tips

### Speed Optimization

1. **Reduce steps for initial screening:**
   ```bash
   --steps 10 --spacing 30  # Quick check
   --steps 50 --spacing 10  # Full validation
   ```

2. **Use fast methods first:**
   - `naive`, `theta`, `seasonal_naive` are instant
   - `sf_autoarima`, `chronos2` are slower

3. **Limit genetic search:**
   ```bash
   --population 15 --generations 8  # Quick
   --population 30 --generations 20 # Thorough
   ```

### Parallelization

Run multiple backtests in parallel (different terminals):

```bash
# Terminal 1
python cli.py forecast_backtest_run EURUSD --methods theta --steps 30

# Terminal 2
python cli.py forecast_backtest_run GBPUSD --methods theta --steps 30
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Compare methods | `python cli.py forecast_backtest_run EURUSD --methods "theta arima analog" --steps 20` |
| With trading costs | `--slippage-bps 2 --trade-threshold 0.0005` |
| Volatility backtest | `--quantity volatility --methods "ewma garch"` |
| With denoising | `--denoise ema --denoise-params "alpha=0.2"` |
| Optimize params | `python cli.py forecast_tune_genetic EURUSD --method theta --metric avg_rmse` |
| JSON output | `--format json` |

---

## See Also

- [GLOSSARY.md](../GLOSSARY.md) — MAE, RMSE, Sharpe ratio definitions
- [FORECAST.md](../FORECAST.md) — Forecasting methods overview
- [FORECAST_GENERATE.md](FORECAST_GENERATE.md) — Forecast generation options
- [DENOISING.md](../DENOISING.md) — Preprocessing options
