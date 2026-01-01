# Patterns & Similarity Methods

**Related documentation:**
- [../FORECAST.md](../FORECAST.md) - Forecasting overview
- [../TECHNICAL_INDICATORS.md](../TECHNICAL_INDICATORS.md) - Indicators as additional context

This project supports two different “pattern” ideas:

1. **Pattern detection**: identify known chart/candlestick patterns (human-readable labels).
2. **Similarity forecasting (analogs)**: find historical windows that “look like now” and aggregate what happened next.

## 1) Pattern detection (`patterns_detect`)

Candlestick patterns (example):

```bash
python cli.py patterns_detect EURUSD --timeframe H1 --mode candlestick --limit 500 \
  --robust-only true --format json
```

Classic chart patterns (example):

```bash
python cli.py patterns_detect EURUSD --timeframe H1 --mode classic --limit 800 --format json
```

Tips:

- Use `--whitelist` to focus on specific patterns.
- Use `--min-strength` / `--robust-only true` to reduce noisy detections.

## 2) Similarity forecasting (“analog” method)

Analog forecasting is like “historical nearest neighbors” for time series:

- take the last `window_size` bars
- search back through history to find similar windows
- look at what happened next after those windows
- average/aggregate those future moves into a forecast

Example:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model analog \
  --model-params "window_size=64 search_depth=5000 top_k=20 scale=zscore metric=euclidean" \
  --format json
```

To see the full parameter list for `analog` (and available search backends), run:

```bash
python cli.py forecast_list_methods --format json
```

