# Glossary

This glossary explains technical terms used in mtdata with simple language and real-world trading examples.

---

## Core Concepts

### Time Series
A sequence of data points indexed by time. In trading, this is typically OHLCV data (Open, High, Low, Close, Volume) at regular intervals.

**Example:** 500 hourly candles of EURUSD form a time series.

### Horizon
How many bars into the future a forecast predicts.

**Example:** `--horizon 12` with H1 timeframe means "predict the next 12 hours."

### Lookback
How many historical bars a model uses to make its prediction.

**Example:** `--lookback 500` tells the model to learn from the last 500 candles.

---

## Forecasting Methods

### Theta Method
A simple, robust forecasting technique that decomposes a time series into trend and seasonality components.

**When to use:** As a baseline for short-to-medium horizons. Often surprisingly accurate.

**Example:**
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta
```

### ARIMA (AutoRegressive Integrated Moving Average)
A classical statistical model that predicts future values based on:
- **AR (AutoRegressive):** Past values of the series
- **I (Integrated):** Differencing to make data stationary
- **MA (Moving Average):** Past forecast errors

**When to use:** Short-term forecasting when data shows clear autocorrelation patterns.

**Interpretation:** ARIMA works best on *stationary* data (constant mean/variance). Raw prices aren't stationary, but returns usually are.

### ETS (Error, Trend, Seasonality)
Exponential smoothing that weighs recent observations more heavily than older ones.

**When to use:** Data with clear seasonal patterns (e.g., higher volatility during market opens).

### Monte Carlo Simulation
Generates thousands of possible future price paths by randomly sampling from historical return distributions.

**When to use:** When you need a *range* of outcomes rather than a single prediction. Essential for risk sizing and barrier analysis.

**Example:**
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --model mc_gbm --model-params "n_sims=2000"
```

**Interpretation:** Instead of one forecast line, you get percentile bands showing where price might land.

### Chronos
A foundation model for time series (like GPT for text). Pre-trained on millions of time series, then applied to your data.

**When to use:** When you want state-of-the-art accuracy without tuning. Requires `chronos` and `torch` packages.

**Example:**
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library pretrained --model chronos2
```

---

## Uncertainty & Intervals

### Confidence Interval
A range around the forecast indicating uncertainty. A 95% CI means "if model assumptions hold, the true value will fall within this range 95% of the time."

**Caution:** Financial data often violates model assumptions (fat tails, regime changes), so intervals may be too narrow.

### Conformal Intervals
A distribution-free method to create prediction intervals. Instead of assuming a bell curve, it uses historical forecast errors to calibrate the band width.

**How it works:**
1. Run a small rolling backtest
2. Collect actual errors at each horizon step
3. Use error quantiles to set band width

**When to use:** When you don't trust model-based intervals. Conformal intervals are empirically calibrated.

**Example:**
```bash
python cli.py forecast_conformal_intervals EURUSD --timeframe H1 \
  --method theta --horizon 12 --steps 25 --alpha 0.1
```

---

## Regime Detection

### Regime
The current "behavior mode" of the market. Different regimes require different strategies.

**Common regimes:**
| Regime | Characteristics | Strategy Implication |
|--------|-----------------|---------------------|
| **Low Volatility / Ranging** | Price oscillates in a narrow band | Mean reversion works well |
| **Trending** | Price moves directionally | Trend-following works well |
| **High Volatility / Crisis** | Large, unpredictable swings | Reduce position sizes |

### Hidden Markov Model (HMM)
An algorithm that assumes the market switches between hidden "states" (regimes). It estimates which state generated the current data based on observed returns and volatility.

**Output:** State ID (0, 1, 2...) and confidence probability.

**Important:** The model doesn't know "Bull" or "Bear." You must interpret what each state means by examining its mean return and volatility.

**Example:**
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"
```

**Interpretation:**
- State 0: Low volatility (σ = 0.0003) → Ranging market
- State 1: High volatility (σ = 0.0008) → Trending or volatile market

### Change-Point Detection (BOCPD)
Bayesian Online Change Point Detection. Answers: "Did the market's underlying behavior just change?"

**Output:** Probability (0-1) that each bar represents a regime shift.

**When to use:**
- If probability > 0.5, the previous pattern may be breaking down
- Useful for detecting breakouts or structural changes

**Example:**
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method bocpd --threshold 0.5
```

---

## Volatility

### Volatility
A measure of how much price typically moves. Higher volatility = larger price swings.

**Types:**
| Type | Definition |
|------|------------|
| **Realized** | How much price *did* move (historical) |
| **Forecasted** | How much we expect it *will* move |
| **Implied** | Market's expectation (from options prices) |

### EWMA (Exponentially Weighted Moving Average)
A volatility estimator that gives more weight to recent observations.

**Parameter:** `lambda` (0.94 is standard). Higher = slower adaptation; lower = faster reaction to new data.

**Example:**
```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 \
  --horizon 12 --method ewma --params "lambda=0.94"
```

**Interpretation:** If output is `sigma_bar_return: 0.0006`, expect hourly returns to have ~0.06% standard deviation.

### GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
A model that captures *volatility clustering*—the tendency for high-volatility periods to follow high-volatility periods.

**When to use:** When you observe that big moves cluster together. Common in equity and FX markets.

### HAR-RV (Heterogeneous AutoRegressive Realized Volatility)
Uses realized volatility at multiple time scales (daily, weekly, monthly) to forecast future volatility.

**When to use:** When you have access to high-frequency intraday data and want more accurate volatility forecasts.

---

## Barrier Analysis

### Barrier
A price level that triggers an event when touched:
- **Take Profit (TP):** Close position at profit target
- **Stop Loss (SL):** Close position to limit loss

### Triple-Barrier Labeling
A method to label historical data based on which barrier was hit first:
- **+1 (Win):** TP hit first
- **-1 (Loss):** SL hit first
- **0 (Neutral):** Neither hit within horizon

**Use case:** Creating labels for machine learning models.

### Edge
The probability advantage of a trade setup.

**Formula:** `Edge = P(TP first) - P(SL first)`

**Interpretation:**
- Edge > 0: Setup has positive expectation
- Edge = 0.15: 15% more likely to win than lose
- Doesn't account for reward/risk ratio

### Kelly Criterion
Optimal position sizing based on win probability and payoff ratio.

**Interpretation:** Tells you what fraction of capital to risk for maximum long-term growth. Use a fraction (e.g., 0.25 × Kelly) in practice to reduce volatility.

---

## Signal Processing

### Denoising
Removing random fluctuations ("noise") to reveal the underlying trend ("signal").

**Trade-off:** More smoothing = clearer trend but more lag (delay).

**Common methods:**
| Method | Best For |
|--------|----------|
| `ema` | General smoothing |
| `median` | Spike removal |
| `kalman` | Adaptive filtering |
| `wavelet` | Frequency-based separation |

**Example:**
```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 \
  --denoise ema --denoise-params "alpha=0.2"
```

### Stationarity
A statistical property where mean and variance don't change over time. Most forecasting models assume stationarity.

**Problem:** Raw prices are *not* stationary (they trend up or down).
**Solution:** Use returns (percent change) instead, which are typically stationary.

---

## Technical Indicators

### Moving Average
The average price over the last N bars. Smooths out short-term fluctuations.

- **SMA:** Simple Moving Average (equal weights)
- **EMA:** Exponential Moving Average (recent bars weighted more)

### RSI (Relative Strength Index)
A momentum oscillator measuring speed and change of price movements. Range: 0-100.

**Interpretation:**
- RSI > 70: Potentially overbought (consider selling)
- RSI < 30: Potentially oversold (consider buying)
- RSI = 50: Neutral

### MACD (Moving Average Convergence Divergence)
Shows relationship between two moving averages.

**Components:**
- MACD line: Fast EMA - Slow EMA
- Signal line: EMA of MACD line
- Histogram: MACD - Signal

**Interpretation:** When MACD crosses above Signal, bullish momentum. Below, bearish momentum.

### ATR (Average True Range)
Measures volatility by averaging the true range (high-low including gaps) over N bars.

**Use case:** Setting stop-loss distance. A common rule is `SL = 2 × ATR`.

---

## Pattern Recognition

### Candlestick Patterns
Visual patterns formed by one or more candles that historically precede certain price moves.

**Examples:**
- **Engulfing:** Large candle completely covers previous candle (reversal signal)
- **Doji:** Open ≈ Close (indecision)
- **Hammer:** Small body with long lower wick (potential bottom)

### Chart Patterns
Larger-scale geometric shapes formed over multiple candles.

**Examples:**
- **Head and Shoulders:** Three peaks, middle highest (reversal)
- **Double Top/Bottom:** Two peaks/troughs at similar level
- **Triangle:** Converging trendlines (breakout imminent)

---

## Data & Execution

### OHLCV
Standard candle data format:
- **O**pen: First price of the period
- **H**igh: Highest price
- **L**ow: Lowest price
- **C**lose: Last price
- **V**olume: Trading activity

### Market Depth (DOM)
Order book showing pending buy/sell orders at various price levels.

### Slippage
Difference between expected execution price and actual fill price. Occurs due to market movement or insufficient liquidity.

### Pip
Smallest price increment for a currency pair. For most pairs: 0.0001. For JPY pairs: 0.01.

---

## See Also

- [Forecasting Guide](FORECAST.md)
- [Regime Detection](forecast/REGIMES.md)
- [Barrier Analysis](BARRIER_FUNCTIONS.md)
- [Technical Indicators](TECHNICAL_INDICATORS.md)
