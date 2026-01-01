# Technical Indicators

**Related documentation:**
- [CLI.md](CLI.md) - How to run commands
- [DENOISING.md](DENOISING.md) - Smoothing inputs/outputs
- [FORECAST.md](FORECAST.md) - Forecasting pipelines that can use indicators

Technical indicators are transformations of raw OHLCV price data (open/high/low/close/volume) that try to summarize market behavior in a more “readable” form: trend, momentum, volatility, and volume/flow.

Indicators do not predict the future by themselves. They are best treated as *features* or *context* that you combine with risk management and validation (backtests, paper trading, etc.).

## Discover available indicators

List indicators (default limit):

```bash
python cli.py indicators_list
```

Filter by category or search term:

```bash
python cli.py indicators_list --category momentum --limit 50
python cli.py indicators_list --search-term "rsi"
```

Get details (parameters + description):

```bash
python cli.py indicators_describe rsi --format json
python cli.py indicators_describe macd --format json
```

## Using indicators with candles

Most workflows start by fetching candles and adding indicators in one step:

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 \
  --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)" \
  --format json
```

Notes:

- The indicator string is a comma-separated list.
- Parameters use parentheses: `rsi(14)`, `macd(12,26,9)`.
- A shorthand like `EMA20` may work for some indicators (the trailing number is treated as a length).
- Output column names come from the underlying indicator implementation (often like `EMA_20`, `RSI_14`, `MACD_12_26_9`).

## Common indicator families (plain language)

- **Trend / “overlap”** (moving averages, bands): help you see direction and dynamic support/resistance.
  - Example: EMA(20), EMA(50)
- **Momentum** (speed of price changes): helps answer “is the move getting stronger or weaker?”
  - Example: RSI(14), MACD(12,26,9)
- **Volatility** (how much the price moves): helps size stops/targets and compare regimes.
  - Example: ATR, Bollinger Bands width
- **Volume / flow**: helps evaluate participation and confirm moves (if volume is meaningful for the instrument).

## Indicators + denoising (recommended pattern)

If indicators are too noisy, denoise either the input (pre-indicator) or the indicator outputs (post-indicator). See [DENOISING.md](DENOISING.md) for details.

Example: compute RSI then smooth it:

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1500 \
  --indicators "rsi(14)" \
  --denoise ema --denoise-params "columns=RSI_14,when=post_ti,alpha=0.3,keep_original=true" \
  --format json
```

