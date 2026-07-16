# Uncertainty and confidence intervals

A point forecast without a range invites overconfidence. This page covers **model intervals** and **conformal intervals** so you can size risk and set levels with eyes open.

**Dense terms:** [Confidence interval](../GLOSSARY.md#confidence-interval) · [Conformal intervals](../GLOSSARY.md#conformal-intervals) · [CI alpha](../GLOSSARY.md#ci-alpha) · [Horizon](../GLOSSARY.md#horizon)

**Related:** [Forecasting](../FORECAST.md) · [Barriers](../BARRIER_FUNCTIONS.md) · [Volatility](VOLATILITY.md) · [Glossary](../GLOSSARY.md)

---

## Types of uncertainty

### Model confidence intervals
Intervals from the model’s assumptions (for example normal errors).

**Limitation:** Markets often have fat tails and regime shifts, so these bands can be **too narrow**.

### Conformal intervals
Bands calibrated from **historical forecast errors** — empirical coverage without strong distributional assumptions.

**Advantage:** More realistic bounds based on actual performance.

---

## Model Confidence Intervals

Request intervals with `--ci-alpha`:

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method analog --ci-alpha 0.1 --json
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
mtdata-cli forecast_conformal_intervals EURUSD --timeframe H1 \
  --method theta --horizon 12 --steps 50 --spacing 20 --ci-alpha 0.1 --json
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Forecasting method | theta |
| `--horizon` | Forecast horizon | 12 |
| `--steps` | Number of calibration anchors (default 50 for stabler quantiles) | 50 |
| `--spacing` | Bars between calibration anchors | 20 |
| `--ci-alpha` | Miscoverage rate (0.1 = 90% interval) | 0.1 |

> When `--steps > 1`, `--spacing` must be `>= --horizon` so calibration windows do not overlap; otherwise the request is rejected.

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
- "Based on the last 50 forecasts, the actual price stayed within these bounds ~90% of the time."

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

In `high_low` mode, `same_bar_policy` resolves a bar that touches both
barriers. The default is conservatively `sl_first`; `tp_first` and `neutral`
are explicit alternatives.

### Usage

```bash
mtdata-cli labels_triple_barrier EURUSD --timeframe H1 --horizon 12 \
  --tp-pct 0.5 --sl-pct 0.3 --json
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
mtdata-cli forecast_conformal_intervals EURUSD --horizon 12 --ci-alpha 0.1

# Use lower_price as stop-loss floor
# Size position so max loss (if lower_price is hit) is within risk budget
```

### Validating Signal Quality

Use triple-barrier labels to evaluate entry signals:

```bash
# Label historical entry points
mtdata-cli labels_triple_barrier EURUSD --horizon 12 --tp-pct 0.5 --sl-pct 0.3

# Check win rate: counts.pos / (counts.pos + counts.neg)
# If win rate < 50%, signal needs improvement
```

### Comparing Forecast Methods

Backtest with conformal intervals to compare reliability:

```bash
# Method A
mtdata-cli forecast_conformal_intervals EURUSD --method theta --horizon 12 --steps 50

# Method B
mtdata-cli forecast_conformal_intervals EURUSD --method sf_autoarima --horizon 12 --steps 50

# Compare interval widths—narrower = more precise (if coverage is similar)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Method CI (90%) | `mtdata-cli forecast_generate EURUSD --method analog --ci-alpha 0.1` |
| Conformal intervals | `mtdata-cli forecast_conformal_intervals EURUSD --method theta --horizon 12` |
| Triple-barrier labels | `mtdata-cli labels_triple_barrier EURUSD --horizon 12 --tp-pct 0.5 --sl-pct 0.3` |

---

## See Also

- [FORECAST.md](../FORECAST.md) — Price forecasting
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL probability analysis
- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
