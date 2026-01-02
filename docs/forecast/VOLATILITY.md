# Volatility Forecasting

Volatility measures how much price typically moves. It's essential for:
- Setting realistic stop-loss and take-profit distances
- Sizing positions (risk more when volatility is low)
- Understanding barrier hit probabilities

**Related:**
- [GLOSSARY.md](../GLOSSARY.md) — Definitions of volatility terms
- [FORECAST.md](../FORECAST.md) — Price forecasting
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — Using volatility for TP/SL sizing

---

## Quick Start

```bash
# EWMA volatility (fast, reliable)
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma

# With custom smoothing
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method ewma --params "lambda=0.94"
```

---

## Understanding the Output

```
success: true
  symbol: EURUSD
  timeframe: H1
  method: ewma
  horizon: 12
  sigma_bar_return: 0.00062     # Per-bar volatility (as return)
  sigma_annual_return: 0.058    # Annualized volatility
  horizon_sigma_return: 0.002145  # Expected volatility over horizon
```

**Interpretation:**
- `sigma_bar_return: 0.00062` → Expect ~0.06% moves per hour (1 standard deviation)
- `horizon_sigma_return: 0.002145` → Over 12 hours, expect ~0.21% total range (1 σ)
- For EURUSD at 1.1750, 0.21% ≈ 25 pips

**Rule of thumb:** Set stop-loss at 1.5-2x horizon volatility to avoid getting stopped by noise.

---

## Methods

### Fast Estimators

Use recent data to estimate current volatility. Best for quick calculations.

| Method | Description | When to Use |
|--------|-------------|-------------|
| `ewma` | Exponentially weighted MA | General purpose, fast |
| `rolling_std` | Simple rolling standard deviation | Quick baseline |
| `parkinson` | Uses high/low range (more efficient) | When H/L data is reliable |
| `gk` | Garman-Klass (uses OHLC) | More efficient than close-to-close |
| `rs` | Rogers-Satchell | Accounts for drift |
| `yang_zhang` | Combines overnight and intraday | Most efficient range-based |

**EWMA Example:**
```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method ewma --params "lambda=0.94"
```

**Parkinson Example:**
```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method parkinson
```

---

### GARCH Family

Models volatility clustering—the tendency for high-volatility periods to follow high-volatility periods.

| Method | Description |
|--------|-------------|
| `garch` | Standard GARCH(1,1) |
| `egarch` | Exponential GARCH (asymmetric) |
| `gjr_garch` | GJR-GARCH (leverage effect) |
| `figarch` | Long-memory GARCH |

**GARCH Example:**
```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method garch
```

**When to use:** When volatility clusters are visible (big moves follow big moves). GARCH is slower but more accurate for regime-switching markets.

---

### Realized Volatility

Uses high-frequency data to compute more accurate volatility estimates.

| Method | Description |
|--------|-------------|
| `realized_kernel` | Kernel-based realized volatility |
| `har_rv` | HAR-RV model (daily/weekly/monthly components) |

**HAR-RV Example:**
```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method har_rv --params "rv_timeframe=M5,days=150"
```

**Parameters:**
- `rv_timeframe`: Timeframe for computing realized variance (M1, M5, M15)
- `days`: Historical days for HAR regression
- `window_w`: Weekly window (default: 5)
- `window_m`: Monthly window (default: 22)

**When to use:** When you need the most accurate volatility forecasts and have access to intraday data.

---

### Volatility Proxies

Forecast a volatility proxy (like squared returns) using any forecasting method.

**Example:**
```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method theta --proxy squared_return
```

**Proxies available:**
- `squared_return`: (close/prev_close - 1)²
- `abs_return`: |close/prev_close - 1|
- `range`: (high - low) / close

---

## Practical Applications

### Setting Stop-Loss Distance

Use volatility to set stops that won't be hit by normal noise:

```bash
# Get hourly volatility
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 1 --method ewma
# Output: sigma_bar_return: 0.00062 (0.062%)

# For EURUSD at 1.1750:
# 1σ = 1.1750 × 0.00062 = 0.00073 (7.3 pips)
# Recommended SL: 2σ = 14.6 pips minimum
```

### Position Sizing

Size positions inversely to volatility:

```bash
# High volatility → smaller position
# Low volatility → larger position

# Example: Risk $100 per trade
# sigma = 0.002 (0.2%)
# SL distance = 2 × 0.002 = 0.4%
# Position size = $100 / 0.4% = $25,000 notional
```

### Barrier Optimization

Use volatility-scaled barriers instead of fixed percentages:

```bash
# Let the optimizer scale barriers to current volatility
python cli.py forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --grid-style volatility --vol-window 250
```

---

## Comparison of Methods

| Method | Speed | Accuracy | Data Needed |
|--------|-------|----------|-------------|
| `ewma` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Close prices |
| `parkinson` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High/Low |
| `yang_zhang` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OHLC |
| `garch` | ⭐⭐ | ⭐⭐⭐⭐ | Close prices |
| `har_rv` | ⭐⭐ | ⭐⭐⭐⭐⭐ | Intraday data |

**Recommendations:**
- **Quick checks:** Use `ewma` or `parkinson`
- **Trading decisions:** Use `yang_zhang` or `garch`
- **Research/backtesting:** Use `har_rv`

---

## Quick Reference

| Task | Command |
|------|---------|
| EWMA volatility | `python cli.py forecast_volatility_estimate EURUSD --method ewma` |
| Parkinson (H/L) | `python cli.py forecast_volatility_estimate EURUSD --method parkinson` |
| GARCH | `python cli.py forecast_volatility_estimate EURUSD --method garch` |
| HAR-RV | `python cli.py forecast_volatility_estimate EURUSD --method har_rv --params "rv_timeframe=M5"` |

---

## See Also

- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
- [FORECAST.md](../FORECAST.md) — Price forecasting
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL probability analysis
- [REGIMES.md](REGIMES.md) — Regime detection
