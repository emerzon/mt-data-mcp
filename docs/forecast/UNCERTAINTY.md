# Uncertainty & Labels

**Related documentation:**
- [../FORECAST.md](../FORECAST.md) - Forecasting overview
- [../BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) - Probabilistic TP/SL outcomes

Forecasts are more useful when they include uncertainty. This project supports two common ideas:

- **Model confidence intervals** (when the model can produce them)
- **Conformal intervals** (calibrated from backtests; fewer assumptions)

## 1) Model confidence intervals (`forecast_generate --ci-alpha`)

Some models can return bands around the point forecast. Example:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model theta --ci-alpha 0.1 --format json
```

Interpretation (plain language):

- A 90% interval is saying “if the model assumptions hold, outcomes should land inside this band about 9 times out of 10”.
- It does not guarantee coverage in real markets.

## 2) Conformalized intervals (`forecast_conformal_intervals`)

Conformal prediction wraps a base forecasting method and uses rolling-origin residuals to build empirically calibrated bands.

Example:

```bash
python cli.py forecast_conformal_intervals EURUSD --timeframe H1 \
  --method theta --horizon 12 --steps 25 --spacing 10 --alpha 0.1 --format json
```

How it works (high level):

1. Run a small rolling backtest (“walk forward”).
2. Collect the forecast errors for each horizon step.
3. Use an error quantile to widen the future forecast into an interval.

## 3) Triple-barrier labeling (`labels_triple_barrier`)

Triple-barrier labeling is a way to turn future price paths into simple outcomes:

- hit TP first
- hit SL first
- no hit within the horizon

This is useful for:

- evaluating a signal (“did it work?”)
- building supervised ML datasets (labels)

Example (percent barriers):

```bash
python cli.py labels_triple_barrier EURUSD --timeframe H1 --horizon 12 \
  --tp-pct 0.5 --sl-pct 0.3 --output compact --format json
```

