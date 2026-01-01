# Denoising & Smoothing

**Related documentation:**
- [CLI.md](CLI.md) - How to run commands
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) - Indicators you may denoise
- [FORECAST.md](FORECAST.md) - Forecasting pipelines that can use denoising

“Denoising” means applying a smoothing filter to a price series (or to indicator series) to reduce short-term randomness and make structure easier to see or model.

## Why denoise?

In real market data, the “true” trend/mean you care about is often hidden under:

- Microstructure noise (spreads, quote jitter)
- One-off spikes (news candles, data glitches)
- Choppy ranges where direction is unclear

Denoising can help:

- Make indicators less twitchy (fewer false “crossovers”)
- Stabilize models that assume smoother dynamics
- Reduce sensitivity to outliers when building signals

## Where denoising is used

Denoising is available in several tools, typically via `--denoise` and `--denoise-params`.

Most commonly:

- `data_fetch_candles ... --denoise ...`
- `patterns_detect ... --denoise ...`
- `report_generate ... --denoise ...`

## Pre vs post indicators

When denoising is applied relative to technical indicators:

- `when=pre_ti`: denoise the raw price series first, then compute indicators on the smoothed series.
- `when=post_ti`: compute indicators first, then denoise the indicator columns.

Rule of thumb:

- Use `pre_ti` when you want smoother inputs (trend estimation).
- Use `post_ti` when you want to keep the raw price intact but smooth an indicator output (signal stabilization).

## Avoiding look-ahead bias (important for backtests)

Some filters can be run in a “zero-phase” way (they look both backward and forward to smooth a point). That can look great on charts, but it is not usable in a causal trading system.

Rule of thumb:

- For live trading / backtesting: use causal filters (only past data).
- For visualization / exploration: zero-phase filters can be acceptable.

## Common parameters

Most denoise specs support these concepts:

- `columns`: which columns to denoise (default is usually `close`)
- `when`: `pre_ti` or `post_ti`
- `keep_original`: keep original columns and add a suffixed copy (e.g., `_dn`)

## Examples

### 1) Denoise close before indicators (EMA)

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14),ema(50)" \
  --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2,keep_original=true" \
  --format json
```

### 2) Denoise an indicator after it’s computed (smooth RSI)

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3,keep_original=true" \
  --format json
```

### 3) Use a robust filter to remove spikes (median / Hampel)

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 2000 \
  --denoise median --denoise-params "window=7,columns=close,keep_original=true" \
  --format json
```

## Denoising methods (high level)

Different methods solve different “kinds of noise”:

- Moving averages (`ema`, `sma`): general-purpose smoothing.
- Robust rolling filters (`median`, `hampel`): reduce spikes/outliers.
- Frequency filters (`lowpass_fft`, `butterworth`): remove high-frequency jitter.
- Trend extractors (`hp`, `l1_trend`, `tv`): keep slow trend, reduce fast noise.
- Adaptive/state-space (`kalman`, `lms`, `rls`): adjust smoothing as conditions change.
- Decomposition (`stl`, `ssa`, `vmd`, `wavelet_packet`): split into components and reconstruct smoother parts.

If you’re unsure, start with `ema` and `median`, then try `kalman` or `tv` for more structure-preserving smoothing.

