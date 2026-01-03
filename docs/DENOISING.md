# Denoising & Smoothing

Denoising removes random fluctuations ("noise") from price data to reveal the underlying trend ("signal").

**Related:**
- [CLI.md](CLI.md) — Command usage
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) — Indicators to denoise
- [FORECAST.md](FORECAST.md) — Using denoising in forecasts
- [GLOSSARY.md](GLOSSARY.md) — Term definitions

---

## Why Denoise?

Market data contains:
- **Signal:** The true underlying trend or pattern
- **Noise:** Random fluctuations from microstructure, spreads, and short-term volatility

Denoising helps:
- Reduce false indicator crossovers
- Clarify trend direction
- Improve model stability
- Remove outliers and spikes

**Trade-off:** More smoothing = clearer trend but more lag (delay in detecting changes).

---

## Quick Start

**Smooth closing prices:**
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 \
  --denoise ema --denoise-params "alpha=0.2"
```

**Remove spikes:**
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 \
  --denoise median --denoise-params "window=5"
```

## Dependencies

Some denoising methods require optional packages:

- `statsmodels`: used by several statistical filters (for example HP/STL)
- `PyWavelets`: required for wavelet denoising (`wavelet`)
- `vmdpy`, `EMD-signal`: required for some decomposition-based methods

Tip: `GET /api/denoise/methods` (see [WEB_API.md](WEB_API.md)) reports availability and required packages for the current environment.

---

## When to Apply: Pre vs Post Indicators

### Pre-Indicator (`when=pre_ti`)
Apply denoising to raw price, then calculate indicators on smoothed data.

**Use when:** You want smoother inputs for trend estimation.

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2"
```

### Post-Indicator (`when=post_ti`)
Calculate indicators on raw data, then smooth the indicator output.

**Use when:** You want to keep raw price intact but reduce indicator noise.

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3"
```

---

## Denoising Methods

### Moving Averages

General-purpose smoothing.

| Method | Description | Parameters |
|--------|-------------|------------|
| `ema` | Exponential Moving Average | `alpha` (0.1-0.5) |
| `sma` | Simple Moving Average | `window` |

**Example:**
```bash
--denoise ema --denoise-params "alpha=0.2"
```

### Robust Filters

Remove outliers and spikes without excessive smoothing.

| Method | Description | Parameters |
|--------|-------------|------------|
| `median` | Median filter | `window` |
| `hampel` | Hampel identifier | `window`, `threshold` |

**Example (spike removal):**
```bash
--denoise median --denoise-params "window=5"
```

### Frequency Filters

Separate high-frequency noise from low-frequency trend.

| Method | Description | Parameters |
|--------|-------------|------------|
| `lowpass_fft` | FFT low-pass filter | `cutoff_ratio` |
| `butterworth` | Butterworth filter | `order`, `cutoff` |

**Example:**
```bash
--denoise lowpass_fft --denoise-params "cutoff_ratio=0.1"
```

### Trend Extractors

Isolate the slow-moving trend component.

| Method | Description | Parameters |
|--------|-------------|------------|
| `hp` | Hodrick-Prescott filter | `lambda` |
| `l1_trend` | L1 trend filter | `lambda` |
| `tv` | Total variation denoising | `lambda` |

**Example:**
```bash
--denoise hp --denoise-params "lambda=1600"
```

### Adaptive Filters

Automatically adjust smoothing based on data.

| Method | Description | Parameters |
|--------|-------------|------------|
| `kalman` | Kalman filter | `transition_cov`, `observation_cov` |
| `lms` | Least Mean Squares | `mu`, `order` |
| `rls` | Recursive Least Squares | `delta`, `order` |

**Example:**
```bash
--denoise kalman --denoise-params "transition_cov=0.01"
```

### Decomposition Methods

Split into components and reconstruct smoother parts.

| Method | Description | Parameters |
|--------|-------------|------------|
| `stl` | Seasonal-Trend decomposition | `period` |
| `ssa` | Singular Spectrum Analysis | `window` |
| `vmd` | Variational Mode Decomposition | `K`, `alpha` |
| `wavelet` | Wavelet denoising | `wavelet`, `level` |

**Example:**
```bash
--denoise wavelet --denoise-params "wavelet=db4,level=3"
```

---

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `columns` | Which columns to denoise | `close` |
| `when` | `pre_ti` or `post_ti` | `pre_ti` |
| `keep_original` | Keep original column (adds `_dn` suffix) | `false` |
| `alpha` | Smoothing factor (EMA) | 0.1 |
| `window` | Window size (filters) | 5 |

---

## Avoiding Look-Ahead Bias

**Critical for backtesting:** Some filters use future data to smooth each point (zero-phase filtering). This looks great on charts but creates unrealistic results.

**Causal filters** (use only past data):
- `ema`, `sma`, `kalman`, `lms`, `rls`

**Non-causal filters** (use past and future):
- `lowpass_fft`, `butterworth`, `hp`, `wavelet` (default mode)

**Recommendation:** Use causal filters for backtesting and live trading.

---

## Method Selection Guide

| Noise Type | Recommended Method |
|------------|-------------------|
| General high-frequency noise | `ema`, `sma` |
| Spikes/outliers | `median`, `hampel` |
| Microstructure noise | `kalman` |
| Seasonal patterns | `stl` |
| Unknown/complex | Start with `ema`, try `kalman` |

---

## Examples

### Smooth Closing Prices
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 \
  --denoise ema --denoise-params "alpha=0.2,keep_original=true"
```

### Remove Price Spikes
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 \
  --denoise hampel --denoise-params "window=7,threshold=3"
```

### Smooth RSI Output
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3"
```

### Kalman Filter (Adaptive)
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 \
  --denoise kalman --denoise-params "transition_cov=0.01"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Basic EMA smoothing | `--denoise ema --denoise-params "alpha=0.2"` |
| Spike removal | `--denoise median --denoise-params "window=5"` |
| Adaptive filter | `--denoise kalman` |
| Keep original column | `--denoise-params "keep_original=true"` |
| Post-indicator smoothing | `--denoise-params "when=post_ti"` |

---

## See Also

- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) — Indicators to denoise
- [FORECAST.md](FORECAST.md) — Using denoising in forecasts
