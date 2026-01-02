# Pattern Detection & Similarity Search

This document covers two related capabilities:
1. **Pattern Detection:** Identify known candlestick and chart patterns
2. **Analog Forecasting:** Find historical windows similar to the current market

**Related:**
- [../FORECAST.md](../FORECAST.md) — Price forecasting
- [../TECHNICAL_INDICATORS.md](../TECHNICAL_INDICATORS.md) — Technical analysis context
- [../GLOSSARY.md](../GLOSSARY.md) — Term definitions

---

## Pattern Detection (`patterns_detect`)

Identifies specific visual patterns that traders use for entry/exit signals.

### Candlestick Patterns

Single or multi-bar patterns with historical significance.

```bash
python cli.py patterns_detect EURUSD --timeframe H1 --mode candlestick --limit 200
```

**Output:**
```
data[29]{time,pattern}:
    "2025-12-18 12:00",Bullish INSIDE
    "2025-12-19 05:00",Bearish ENGULFING
    "2025-12-22 10:00",Bearish ENGULFING
    ...
```

**Filter to robust patterns only:**
```bash
python cli.py patterns_detect EURUSD --mode candlestick --robust-only true
```

**Common patterns detected:**
| Pattern | Meaning |
|---------|---------|
| **Engulfing** | Current candle completely covers previous (reversal) |
| **Doji** | Open ≈ Close (indecision) |
| **Hammer/Hanging Man** | Small body, long lower wick |
| **Inside** | Current bar inside previous bar's range |
| **Harami** | Small body inside previous large body |
| **Morning/Evening Star** | Three-bar reversal pattern |

### Classic Chart Patterns

Larger geometric patterns formed over multiple bars.

```bash
python cli.py patterns_detect EURUSD --timeframe H1 --mode classic --limit 500
```

**Patterns detected:**
| Pattern | Description |
|---------|-------------|
| **Head and Shoulders** | Three peaks, middle highest (bearish reversal) |
| **Inverse H&S** | Three troughs, middle lowest (bullish reversal) |
| **Double Top/Bottom** | Two peaks/troughs at similar level |
| **Triangle** | Converging trendlines (breakout setup) |
| **Wedge** | Rising or falling wedge |
| **Rectangle** | Horizontal consolidation |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `candlestick` | Pattern type: candlestick, classic, elliott |
| `--limit` | 500 | Bars to analyze |
| `--robust-only` | false | Only return high-confidence patterns |
| `--whitelist` | — | Comma-separated list of specific patterns |
| `--min-strength` | — | Minimum pattern strength score |

### Filtering Patterns

**By name:**
```bash
python cli.py patterns_detect EURUSD --mode candlestick \
  --whitelist "ENGULFING,HAMMER,DOJI"
```

**By confidence:**
```bash
python cli.py patterns_detect EURUSD --mode candlestick --robust-only true
```

---

## Analog Forecasting

Finds historical windows that "look like" the current market and uses them to predict what happens next.

### Concept

"History doesn't repeat, but it rhymes."

1. Take the last N bars (the "query window")
2. Search through historical data for similar patterns
3. Look at what happened after those patterns
4. Average/aggregate those future moves into a forecast

### Basic Usage

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model analog --model-params "window_size=64 top_k=20"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 64 | Length of pattern to match |
| `search_depth` | 5000 | How far back to search |
| `top_k` | 20 | Number of similar patterns to use |
| `metric` | euclidean | Distance metric |
| `scale` | zscore | Normalization: zscore, minmax, none |
| `refine_metric` | none | Refinement: dtw, softdtw, affine, ncc |
| `search_engine` | ckdtree | Search algorithm |

### Scaling Options

| Scale | Description | When to Use |
|-------|-------------|-------------|
| `zscore` | Standardize to mean=0, std=1 | Default, handles varying volatility |
| `minmax` | Scale to [0,1] | When range matters more than volatility |
| `none` | No scaling | When absolute levels matter |

### Distance Metrics

| Metric | Description |
|--------|-------------|
| `euclidean` | Standard L2 distance (fast) |
| `cosine` | Cosine similarity (shape-focused) |
| `dtw` | Dynamic Time Warping (handles warping) |
| `softdtw` | Differentiable DTW |
| `ncc` | Normalized cross-correlation |

**Example with refinement:**
```bash
python cli.py forecast_generate EURUSD --horizon 12 \
  --model analog --model-params "window_size=64 metric=euclidean refine_metric=dtw"
```

This first finds candidates with fast Euclidean distance, then refines ranking using DTW.

### Search Engines

| Engine | Description |
|--------|-------------|
| `ckdtree` | Scipy KD-tree (default, fast) |
| `hnsw` | Approximate nearest neighbor (scalable) |
| `matrix_profile` | STUMPY-based (specialized for time series) |
| `mass` | Mueen's MASS algorithm |

---

## Practical Applications

### Pattern-Based Entry Filter

Use pattern detection as a confirmation signal:

```bash
# Check for reversal patterns at support
python cli.py patterns_detect EURUSD --mode candlestick --robust-only true

# If bullish pattern detected at support level → consider long entry
```

### Analog-Based Targets

Use analog forecasts to set price targets:

```bash
# Find similar historical patterns
python cli.py forecast_generate EURUSD --model analog \
  --model-params "window_size=64 top_k=20" --format json

# Use forecast percentiles for TP levels
```

### Combining with Technical Analysis

```bash
# Get patterns and indicators together
python cli.py data_fetch_candles EURUSD --limit 200 \
  --indicators "ema(20),rsi(14)"

# Then check patterns
python cli.py patterns_detect EURUSD --mode candlestick --robust-only true

# Look for pattern + indicator confluence
```

---

## Interpreting Results

### Pattern Detection

```
data[5]{time,pattern}:
    "2025-12-19 05:00",Bearish ENGULFING
    "2025-12-22 10:00",Bearish ENGULFING
```

**Interpretation:**
- Pattern occurred at specific times
- "Bearish" suggests potential downward move
- Combine with other analysis (support/resistance, indicators)

### Analog Forecast

```json
{
  "forecast": [1.1755, 1.1758, 1.1762, ...],
  "lower": [1.1740, 1.1738, ...],
  "upper": [1.1770, 1.1778, ...],
  "analogs_found": 20
}
```

**Interpretation:**
- `forecast`: Median of analog outcomes
- `lower`/`upper`: Spread of analog outcomes
- Wide spread = diverse outcomes in similar historical patterns

---

## Quick Reference

| Task | Command |
|------|---------|
| Candlestick patterns | `python cli.py patterns_detect EURUSD --mode candlestick` |
| Robust patterns only | `python cli.py patterns_detect EURUSD --mode candlestick --robust-only true` |
| Chart patterns | `python cli.py patterns_detect EURUSD --mode classic` |
| Analog forecast | `python cli.py forecast_generate EURUSD --model analog --model-params "window_size=64 top_k=20"` |
| Analog with DTW | `python cli.py forecast_generate EURUSD --model analog --model-params "refine_metric=dtw"` |

---

## See Also

- [../FORECAST.md](../FORECAST.md) — Price forecasting overview
- [../TECHNICAL_INDICATORS.md](../TECHNICAL_INDICATORS.md) — Technical indicators
- [../GLOSSARY.md](../GLOSSARY.md) — Term definitions
