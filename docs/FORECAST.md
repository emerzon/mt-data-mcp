# Forecasting & Pattern-Based Signals

This page is the entry point for forecasting-related features: price forecasts, volatility estimates, regimes/change-points, uncertainty (conformal intervals), and pattern-based methods.

**Related documentation:**
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) - Indicators as context/features
- [DENOISING.md](DENOISING.md) - Smoothing inputs/outputs
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) - TP/SL hit probabilities and barrier optimization
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) - Beginner workflow
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) - Advanced workflow
- [EXAMPLE.md](EXAMPLE.md) - End-to-end walkthrough

## Concepts (plain language)

- **Price forecast**: a best-guess path for the next `horizon` candles.
- **Volatility forecast**: a best-guess “typical move size” over the next `horizon` candles (often more useful than the exact price).
- **Regimes**: market phases like “trending/quiet” vs “choppy/volatile”.
- **Conformal intervals**: forecast bands calibrated from backtests (fewer assumptions than model-based confidence intervals).
- **Barrier analytics**: “what is the probability TP hits before SL within N bars?” (risk sizing and trade planning).

## Where to start (quick)

1) See what forecasting methods are available:

```bash
python cli.py forecast_list_methods --format json
```

2) Generate a simple baseline forecast:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model theta --format json
```

3) Estimate near-term volatility:

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method ewma --params "lambda=0.94" --format json
```

## Documentation map (submodules)

- Price forecasts: [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md)
- Volatility: [forecast/VOLATILITY.md](forecast/VOLATILITY.md)
- Regimes & change-points: [forecast/REGIMES.md](forecast/REGIMES.md)
- Uncertainty + labels: [forecast/UNCERTAINTY.md](forecast/UNCERTAINTY.md)
- Pattern detection + analogs: [forecast/PATTERN_SEARCH.md](forecast/PATTERN_SEARCH.md)

