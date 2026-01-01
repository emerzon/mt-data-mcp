# Volatility Forecasting (`forecast_volatility_estimate`)

**Related documentation:**
- [../FORECAST.md](../FORECAST.md) - Forecasting overview
- [../BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) - Using volatility to size TP/SL
- [../SAMPLE-TRADE.md](../SAMPLE-TRADE.md) - A practical workflow that uses volatility

Volatility is a measure of “how much price typically moves”. It’s one of the most practical inputs for trading because it helps you size:

- stop-loss / take-profit distances (so they’re not unrealistically tight)
- position size (so risk stays bounded when markets get wild)

## Quick start (EWMA)

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method ewma --params "lambda=0.94" --format json
```

## Method families (plain language)

- **Fast, robust estimators**: use recent bars to estimate “current” volatility.
  - Examples: `ewma`, `rolling_std`, `parkinson`, `gk`, `rs`, `yang_zhang`
- **Realized volatility models**: try to be more accurate by using intraday structure.
  - Examples: `realized_kernel`, `har_rv`
- **GARCH family**: explicitly models volatility clustering (quiet periods followed by stormy ones).
  - Examples: `garch`, `egarch`, `gjr_garch`, `figarch` (+ `_t` variants)
- **Forecast-on-proxy**: build a volatility proxy series (like squared returns) and forecast it using a simple forecaster.
  - Examples: `arima`, `sarima`, `ets`, `theta` with `--proxy ...`

## Examples

### Parkinson (high/low based)

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method parkinson --format json
```

### HAR-RV (realized volatility over multiple windows)

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22" --format json
```

### GARCH (volatility clustering)

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method garch --format json
```

### Forecast a volatility proxy (squared returns) with Theta

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method theta --proxy squared_return --format json
```

