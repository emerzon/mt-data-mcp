# Regime & Change-Point Detection (`regime_detect`)

**Related documentation:**
- [../FORECAST.md](../FORECAST.md) - Forecasting overview
- [../SAMPLE-TRADE-ADVANCED.md](../SAMPLE-TRADE-ADVANCED.md) - Uses regimes as a “gatekeeper”

Markets are not stationary: the same strategy can work in one environment (e.g., trending + low volatility) and fail in another (e.g., choppy + high volatility). Regime detection tries to label these phases and/or detect sudden changes.

## Quick start (change-point detection)

```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method bocpd --threshold 0.6 --output summary --lookback 300 --format json
```

## Methods (plain language)

- `bocpd`: change-point detection (“did the process just change?”). Useful as an alert.
- `hmm`: hidden Markov model (“what regime are we in?”). Useful for ongoing labeling.
- `ms_ar`: Markov-switching autoregression (statsmodels). Similar goal, different model family.

## Example: HMM regimes

```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method hmm --params "n_states=3" --output compact --lookback 300 --format json
```

Practical use:

- Use the most likely `state` as a coarse regime label.
- Use `state_probabilities` to avoid overreacting when confidence is low.

