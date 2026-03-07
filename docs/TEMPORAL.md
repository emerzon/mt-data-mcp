# Temporal Analysis

The `temporal_analyze` command computes grouped statistics (returns, volatility, volume) by time period—day of week, hour of day, or calendar month. Use it to discover session effects, optimal trading windows, and seasonal patterns.

**Related:**
- [CLI.md](CLI.md) — Command usage and output formats
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation

---

## Quick Start

```bash
# Average returns by day of week
python cli.py temporal_analyze EURUSD --group-by dow --json

# Volatility by hour of day
python cli.py temporal_analyze EURUSD --group-by hour --limit 2000 --json

# Monthly seasonality
python cli.py temporal_analyze EURUSD --timeframe D1 --group-by month --limit 1000 --json
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Trading symbol |
| `--timeframe` | `H1` | Candle timeframe |
| `--limit` | 1000 | Number of bars to analyze |
| `--start` | (optional) | Start date (ISO or flexible format) |
| `--end` | (optional) | End date (ISO or flexible format) |
| `--group-by` | `dow` | Grouping: `dow` (day of week), `hour`, `month`, `all` |
| `--day-of-week` | (optional) | Filter to a specific day (0–6 or name, e.g., `Mon`, `Friday`) |
| `--month` | (optional) | Filter to a specific month (1–12 or name, e.g., `Jan`, `September`) |
| `--time-range` | (optional) | Filter by time window `HH:MM-HH:MM` (wraps midnight, e.g., `22:00-02:00`) |
| `--return-mode` | `pct` | Return calculation: `pct` (percentage) or `log` (logarithmic) |

---

## Grouping Modes

### Day of Week (`--group-by dow`)

Shows performance by weekday. Useful for detecting day-of-week effects.

```bash
python cli.py temporal_analyze EURUSD --group-by dow --limit 2000 --json
```

**Example output (simplified):**
```
group  bars  avg_return  volatility  win_rate  avg_volume
Mon     400   -0.012%     0.065%      48.2%     1250
Tue     400    0.008%     0.071%      51.0%     1380
Wed     400    0.015%     0.078%      52.5%     1420
Thu     400   -0.003%     0.074%      49.8%     1350
Fri     400    0.005%     0.062%      50.5%     1100
```

### Hour of Day (`--group-by hour`)

Shows performance by hour. Reveals session activity patterns.

```bash
python cli.py temporal_analyze EURUSD --group-by hour --limit 5000 --json
```

Use `--time-range` to focus on a specific session:
```bash
# London session hours
python cli.py temporal_analyze EURUSD --group-by hour --time-range "08:00-16:00" --json

# Asian session (wraps midnight)
python cli.py temporal_analyze EURUSD --group-by hour --time-range "22:00-07:00" --json
```

### Calendar Month (`--group-by month`)

Shows seasonal effects across months. Best with daily data and a long history.

```bash
python cli.py temporal_analyze EURUSD --timeframe D1 --group-by month --limit 2000 --json
```

### Overall Summary (`--group-by all`)

Single aggregate summary across all bars.

```bash
python cli.py temporal_analyze EURUSD --group-by all --json
```

---

## Output Fields

Each group includes these statistics:

| Field | Description |
|-------|-------------|
| `group` | Group label (e.g., `Mon`, `14:00`, `Jan`) |
| `group_key` | Numeric group identifier |
| `bars` | Number of bars in group |
| `returns` | Count of return observations |
| `avg_return` | Average return (%) |
| `median_return` | Median return (%) |
| `volatility` | Standard deviation of returns |
| `avg_abs_return` | Average absolute return |
| `win_rate` | Percentage of bars with positive return |
| `avg_range` | Average high-low range |
| `avg_range_pct` | Average range as percentage of close |
| `avg_volume` | Average volume (real or tick) |

Top-level response also includes `overall` (summary statistics) and `volume_source` (whether `real_volume` or `tick_volume` was used).

---

## Filtering

Combine grouping with filters to drill down:

```bash
# Only Mondays, grouped by hour
python cli.py temporal_analyze EURUSD --group-by hour --day-of-week Mon --json

# Only January, grouped by day of week
python cli.py temporal_analyze EURUSD --timeframe D1 --group-by dow --month Jan --limit 2000 --json

# London session hours, grouped by day of week
python cli.py temporal_analyze EURUSD --group-by dow --time-range "08:00-16:00" --json
```

---

## Practical Applications

### Find Best Trading Days
```bash
python cli.py temporal_analyze EURUSD --group-by dow --limit 5000 --json
# Look for days with highest win_rate and positive avg_return
```

### Find Active Trading Hours
```bash
python cli.py temporal_analyze EURUSD --group-by hour --limit 5000 --json
# Look for hours with highest avg_range_pct and avg_volume
```

### Seasonal Patterns
```bash
python cli.py temporal_analyze SPX500 --timeframe D1 --group-by month --limit 3000 --json
# Compare monthly avg_return and volatility
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Day-of-week stats | `python cli.py temporal_analyze EURUSD --group-by dow` |
| Hourly stats | `python cli.py temporal_analyze EURUSD --group-by hour` |
| Monthly seasonality | `python cli.py temporal_analyze EURUSD --timeframe D1 --group-by month` |
| Filter to Mondays | `python cli.py temporal_analyze EURUSD --group-by hour --day-of-week Mon` |
| London session only | `python cli.py temporal_analyze EURUSD --group-by hour --time-range "08:00-16:00"` |

---

## See Also

- [CLI.md](CLI.md) — Command usage
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
