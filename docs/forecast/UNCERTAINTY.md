# Uncertainty & Confidence Intervals

Forecasts are estimates, not guarantees. Understanding uncertainty helps you size positions appropriately and avoid overconfidence.

**Related:**
- [FORECAST.md](../FORECAST.md) — Price forecasting
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL probability analysis
- [GLOSSARY.md](../GLOSSARY.md) — Term definitions

---

## Types of Uncertainty

### Model Confidence Intervals
Intervals derived from the forecasting model's assumptions (e.g., normal distribution of errors).

**Limitation:** Financial data often violates these assumptions (fat tails, regime changes), so intervals may be too narrow.

### Conformal Intervals
Intervals calibrated from historical forecast errors. No distributional assumptions—just empirical coverage.

**Advantage:** More realistic bounds based on actual performance.

---

## Model Confidence Intervals

Request intervals with `--ci-alpha`:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model theta --ci-alpha 0.1 --format json
```

**Parameters:**
- `--ci-alpha 0.1` → 90% confidence interval
- `--ci-alpha 0.05` → 95% confidence interval

**Output includes:**
```json
{
  "forecast": [...],
  "lower": [...],
  "upper": [...]
}
```

**Interpretation:**
- "If model assumptions hold, the true value will fall between `lower` and `upper` about 90% (or 95%) of the time."

**Caution:** Financial markets have fat tails. Model CIs often underestimate extreme moves.

---

## Conformal Intervals (`forecast_conformal_intervals`)

Conformal prediction calibrates intervals from rolling backtest residuals, making no distributional assumptions.

### How It Works
1. Run a rolling-origin backtest on historical data
2. Collect actual forecast errors at each horizon step
3. Use error quantiles to set interval width for new forecasts

### Usage

```bash
python cli.py forecast_conformal_intervals EURUSD --timeframe H1 \
  --method theta --horizon 12 --steps 25 --spacing 10 --alpha 0.1 --format json
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Forecasting method | theta |
| `--horizon` | Forecast horizon | 12 |
| `--steps` | Number of backtest points | 25 |
| `--spacing` | Bars between backtest points | 10 |
| `--alpha` | Miscoverage rate (0.1 = 90% interval) | 0.1 |

### Output

```json
{
  "forecast": [1.1755, 1.1756, ...],
  "lower_price": [1.1740, 1.1738, ...],
  "upper_price": [1.1770, 1.1774, ...],
  "conformal_residual_quantiles": [0.0005, 0.0008, ...]
}
```

**Interpretation:**
- `lower_price` / `upper_price`: Empirically calibrated bounds
- "Based on the last 25 forecasts, the actual price stayed within these bounds ~90% of the time."

### When to Use
- When you don't trust model-based intervals
- For trading decisions where reliability matters
- When historical data shows fat tails or regime changes

---

## Triple-Barrier Labeling (`labels_triple_barrier`)

A different approach to uncertainty: instead of predicting *where* price goes, label *what happened* historically.

### Concept

For each historical bar, ask: "Within the next N bars, did price hit the take-profit level, stop-loss level, or neither?"

**Labels:**
- `+1` (Win): TP hit first
- `-1` (Loss): SL hit first
- `0` (Neutral): Neither hit within horizon

### Usage

```bash
python cli.py labels_triple_barrier EURUSD --timeframe H1 --horizon 12 \
  --tp-pct 0.5 --sl-pct 0.3 --output compact --format json
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `--horizon` | Maximum bars to wait |
| `--tp-pct` | Take-profit distance (% of price) |
| `--sl-pct` | Stop-loss distance (% of price) |
| `--tp-pips` / `--sl-pips` | Alternative: distance in pips |

### Output

```json
{
  "entries": ["2025-12-18 17:00", "2025-12-18 18:00", ...],
  "labels": [1, -1, 0, 0, 1, ...],
  "holding_bars": [5, 3, 12, 12, 8, ...],
  "summary": {
    "counts": {"pos": 45, "neg": 32, "neut": 123}
  }
}
```

**Interpretation:**
- Label distribution shows historical win/loss rates for these barrier levels
- Use this to evaluate signal quality or train ML models

---

## Practical Applications

### Conservative Position Sizing

Use conformal intervals instead of model CIs:

```bash
# Get conformal intervals
python cli.py forecast_conformal_intervals EURUSD --horizon 12 --alpha 0.1

# Use lower_price as stop-loss floor
# Size position so max loss (if lower_price is hit) is within risk budget
```

### Validating Signal Quality

Use triple-barrier labels to evaluate entry signals:

```bash
# Label historical entry points
python cli.py labels_triple_barrier EURUSD --horizon 12 --tp-pct 0.5 --sl-pct 0.3

# Check win rate: counts.pos / (counts.pos + counts.neg)
# If win rate < 50%, signal needs improvement
```

### Comparing Forecast Methods

Backtest with conformal intervals to compare reliability:

```bash
# Method A
python cli.py forecast_conformal_intervals EURUSD --method theta --horizon 12 --steps 50

# Method B
python cli.py forecast_conformal_intervals EURUSD --method sf_autoarima --horizon 12 --steps 50

# Compare interval widths—narrower = more precise (if coverage is similar)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Model CI (90%) | `python cli.py forecast_generate EURUSD --model theta --ci-alpha 0.1` |
| Conformal intervals | `python cli.py forecast_conformal_intervals EURUSD --method theta --horizon 12` |
| Triple-barrier labels | `python cli.py labels_triple_barrier EURUSD --horizon 12 --tp-pct 0.5 --sl-pct 0.3` |

---

## See Also

- [FORECAST.md](../FORECAST.md) — Price forecasting
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL probability analysis
- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
