# Technical Indicators

Technical indicators transform OHLCV data into signals for trend, momentum, volatility, and volume analysis.

**Related:**
- [CLI.md](CLI.md) — How to run commands
- [DENOISING.md](DENOISING.md) — Smoothing indicator outputs
- [FORECAST.md](FORECAST.md) — Using indicators as forecast features
- [GLOSSARY.md](GLOSSARY.md) — Term definitions

---

## Quick Start

**List available indicators:**
```bash
python cli.py indicators_list --limit 20
```

**Filter by category:**
```bash
python cli.py indicators_list --category momentum
python cli.py indicators_list --category trend
python cli.py indicators_list --category volatility
```

**Get indicator details:**
```bash
python cli.py indicators_describe rsi --format json
python cli.py indicators_describe macd --format json
```

---

## Using Indicators

### With Candle Data

Add indicators directly when fetching candles:
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 \
  --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)"
```

**Syntax:**
- `indicator_name(param1,param2,...)` — With parameters
- `indicator_name` — Uses defaults
- Comma-separated list for multiple indicators

### Output Columns

Indicators add new columns to the output:
```
time,open,high,low,close,volume,EMA_20,EMA_50,RSI_14,MACD_12_26_9,MACDh_12_26_9,MACDs_12_26_9
```

Column naming convention: `INDICATOR_PARAM1_PARAM2`

---

## Indicator Categories

### Trend / Overlap

Show direction and dynamic support/resistance levels.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `ema` | Exponential Moving Average | `ema(20)` |
| `sma` | Simple Moving Average | `sma(50)` |
| `dema` | Double EMA | `dema(20)` |
| `tema` | Triple EMA | `tema(20)` |
| `wma` | Weighted Moving Average | `wma(20)` |
| `kama` | Kaufman Adaptive MA | `kama(10,2,30)` |
| `vwap` | Volume Weighted Avg Price | `vwap` |
| `bbands` | Bollinger Bands | `bbands(20,2)` |
| `kc` | Keltner Channels | `kc(20,2)` |

**Usage example:**
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 100 \
  --indicators "ema(20),ema(50),bbands(20,2)"
```

**Interpretation:**
- Price above EMA → Bullish bias
- Price between Bollinger Bands → Normal volatility
- Price touches upper/lower band → Potential reversal zone

### Momentum

Measure speed and strength of price changes.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `rsi` | Relative Strength Index | `rsi(14)` |
| `macd` | Moving Average Convergence Divergence | `macd(12,26,9)` |
| `stoch` | Stochastic Oscillator | `stoch(14,3,3)` |
| `cci` | Commodity Channel Index | `cci(20)` |
| `willr` | Williams %R | `willr(14)` |
| `roc` | Rate of Change | `roc(10)` |
| `mom` | Momentum | `mom(10)` |
| `ao` | Awesome Oscillator | `ao` |

**RSI interpretation:**
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- RSI = 50: Neutral

**MACD interpretation:**
- MACD crosses above Signal: Bullish momentum
- MACD crosses below Signal: Bearish momentum
- Histogram expanding: Momentum strengthening

### Volatility

Measure price movement magnitude.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `atr` | Average True Range | `atr(14)` |
| `natr` | Normalized ATR | `natr(14)` |
| `bbands` | Bollinger Bands width | `bbands(20,2)` |
| `kc` | Keltner Channels | `kc(20,2)` |
| `donchian` | Donchian Channels | `donchian(20)` |

**ATR usage:**
- Set stop-loss: `SL = Entry ± (2 × ATR)`
- Compare volatility across timeframes
- Position sizing: Smaller size when ATR is high

### Volume

Analyze trading activity and participation.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `obv` | On-Balance Volume | `obv` |
| `ad` | Accumulation/Distribution | `ad` |
| `adosc` | Chaikin A/D Oscillator | `adosc` |
| `mfi` | Money Flow Index | `mfi(14)` |
| `vwap` | Volume Weighted Avg Price | `vwap` |

**Note:** Volume indicators are most useful for instruments with reliable volume data (equities, futures). Forex volume is typically indicative only.

---

## Denoising Indicators

Smooth noisy indicator outputs to reduce false signals:

**Smooth RSI after calculation:**
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3"
```

**Smooth price before calculating indicators:**
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2"
```

See [DENOISING.md](DENOISING.md) for more options.

---

## Common Indicator Combinations

### Trend + Momentum
```bash
--indicators "ema(20),ema(50),rsi(14)"
```
- EMA crossover for trend direction
- RSI for entry timing (buy oversold in uptrend)

### Trend + Volatility
```bash
--indicators "ema(20),bbands(20,2),atr(14)"
```
- EMA for trend
- Bollinger Bands for volatility context
- ATR for stop-loss sizing

### Full Suite
```bash
--indicators "ema(20),ema(50),rsi(14),macd(12,26,9),atr(14)"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| List indicators | `python cli.py indicators_list` |
| Momentum indicators | `python cli.py indicators_list --category momentum` |
| Indicator details | `python cli.py indicators_describe rsi` |
| Fetch with indicators | `python cli.py data_fetch_candles EURUSD --indicators "ema(20),rsi(14)"` |

---

## See Also

- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [DENOISING.md](DENOISING.md) — Smoothing techniques
- [FORECAST.md](FORECAST.md) — Using indicators in forecasts
