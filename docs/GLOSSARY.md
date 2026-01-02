# Glossary

This glossary explains technical terms used in mtdata with simple language and real-world trading examples.

---

## Basic Trading Concepts (Start Here)

If you're new to trading, skim this section first. It's enough to follow the examples in this repo.

### Long vs Short
- **Long**: you profit if price goes **up** (buy now, sell later).
- **Short**: you profit if price goes **down** (sell now, buy back later).

### TP/SL (Take Profit / Stop Loss)
- **Take Profit (TP)**: close a position when price reaches your profit target.
- **Stop Loss (SL)**: close a position to cap the loss if price moves against you.

### Bid/Ask and Spread
- **Ask**: the price you pay to buy.
- **Bid**: the price you receive when you sell.
- **Spread**: `ask - bid` (a cost you pay to enter/exit). See also: [Spread](#spread).

### Pip (and Pipette)
- A **pip** is a standard unit of FX price movement (often `0.0001`, or `0.01` for JPY pairs). See also: [Pip](#pip).
- Some brokers quote an extra digit (“pipettes”): `1 pip = 10 pipettes`.

### Timeframe and Candles (Bars)
- A **timeframe** (e.g., `M5`, `H1`, `D1`) controls how data is grouped.
- A **candle/bar** summarizes that period (open/high/low/close and volume).

### Lot Size
Forex position sizes are often expressed in **lots** (see: [Lot Size](#lot-size)).

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

---

## Barrier Optimization Objectives

When optimizing TP/SL levels, you must choose what to maximize. Each objective answers a different trading question.

### Edge
**What it measures:** The raw probability advantage of winning vs losing.

**Formula:** `Edge = P(TP first) - P(SL first)`

**Example:** Edge = 0.20 means you win 20% more often than you lose.

**When to use:**
- General-purpose objective for consistent advantage
- When you want high win rates regardless of payoff size
- Good starting point for most strategies

**Limitation:** Ignores how much you win/lose. A setup with 60% wins but tiny TP and large SL may have positive edge but negative EV.

---

### Kelly Criterion
**What it measures:** The optimal fraction of your capital to bet for maximum long-term growth.

**Formula:** `Kelly = P(win) - P(loss) / (TP/SL)`

**Example:** Kelly = 0.25 means bet 25% of capital per trade for maximum growth.

**When to use:**
- Position sizing decisions
- When you want to grow capital as fast as possible
- Long-term systematic trading

**Limitation:** Full Kelly is aggressive and leads to large drawdowns. Most traders use "fractional Kelly" (e.g., 0.25 × Kelly).

**Interpretation of output:**
- `kelly`: Raw Kelly fraction based on all paths
- `kelly_cond`: Kelly fraction based only on paths that resolved (TP or SL hit)

---

### EV (Expected Value)
**What it measures:** The average profit per trade, accounting for both probability and payoff size.

**Formula:** `EV = P(win) × TP - P(loss) × SL`

**Example:** EV = 0.15 means you expect to gain 0.15% per trade on average.

**When to use:**
- When payoff asymmetry matters (small wins, big losses or vice versa)
- Comparing setups with different TP/SL ratios
- Maximizing total profit over many trades

**Variants:**
- `ev`: Based on all paths (includes no-hit scenarios as 0)
- `ev_cond`: Based only on resolved paths (ignores trades that never hit TP or SL)

---

### EV Per Bar
**What it measures:** Expected value normalized by how long the trade takes.

**Formula:** `EV_per_bar = EV / mean_time_to_resolution`

**Example:** Setup A has EV=0.30 but takes 20 bars; Setup B has EV=0.15 but takes 5 bars. Setup B has higher EV per bar (0.03 vs 0.015).

**When to use:**
- When capital turnover matters (you want fast trades to reinvest)
- Comparing setups across different timeframes
- Scalping and high-frequency strategies

**Limitation:** Favors fast trades, which may have higher transaction costs.

---

### Prob Resolve
**What it measures:** The probability that the trade closes (hits either TP or SL) within the horizon.

**Formula:** `Prob_resolve = P(TP first) + P(SL first) = 1 - P(no hit)`

**Example:** Prob_resolve = 0.85 means 85% of trades close before the time limit.

**When to use:**
- When you want trades to actually complete
- Avoiding setups where most trades expire without hitting targets
- Day trading where you must close by end of session

**Limitation:** High resolve probability doesn't mean profitable—could resolve at a loss.

---

### Profit Factor
**What it measures:** Ratio of expected gains to expected losses.

**Formula:** `Profit_factor = (P(win) × TP) / (P(loss) × SL)`

**Example:** Profit factor = 2.0 means you expect $2 in wins for every $1 in losses.

**When to use:**
- When you want a risk/reward focused metric
- Comparing profitability of different strategies
- Common in backtesting reports

**Interpretation:**
- Profit factor > 1: Profitable
- Profit factor > 2: Very good
- Profit factor < 1: Losing money

---

### Min Loss Prob
**What it measures:** Minimizes the probability of losing (SL hit first).

**Formula:** Minimize `P(SL first)`

**Example:** Setup with min_loss_prob = 0.15 means only 15% chance of hitting stop loss.

**When to use:**
- Capital preservation is priority
- Conservative trading styles
- When you can't afford drawdowns (e.g., trading client money)

**Limitation:** May result in very small TP targets or trades that rarely resolve.

---

### Utility (Log Utility)
**What it measures:** Risk-adjusted expected value using logarithmic utility (diminishing returns on gains).

**Formula:** `Utility = P(win) × log(1 + TP) + P(loss) × log(1 - SL)`

**Example:** Log utility penalizes large losses more than it rewards equivalent gains.

**When to use:**
- Risk-averse traders
- When you want to avoid ruin (log utility naturally avoids bets that could wipe you out)
- Academic/theoretical contexts

**Interpretation:** Higher utility = better risk-adjusted outcome. Negative utility means the trade is worse than doing nothing.

---

## Objective Selection Guide

| Your Priority | Use Objective | Why |
|---------------|---------------|-----|
| Win rate | `edge` | Maximizes probability advantage |
| Total profit | `ev` | Accounts for both probability and payoff |
| Fast trades | `ev_per_bar` | Optimizes profit per unit time |
| Position sizing | `kelly` | Tells you how much to bet |
| Avoid losses | `min_loss_prob` | Minimizes chance of stop-loss hit |
| Trade completion | `prob_resolve` | Ensures trades actually close |
| Risk-adjusted | `utility` | Penalizes large losses |
| Gross profit ratio | `profit_factor` | Gains vs losses ratio |

---

## Backtesting Metrics

### MAE (Mean Absolute Error)
**What it measures:** Average size of forecast errors, ignoring direction.

**Formula:** `MAE = average(|actual - predicted|)`

**Example:** MAE = 0.0015 means forecasts are off by 0.15% on average.

**Interpretation:**
- Lower is better
- Easy to interpret (same units as forecast)
- Doesn't penalize large errors more than small ones

---

### RMSE (Root Mean Squared Error)
**What it measures:** Average size of forecast errors, penalizing large errors more.

**Formula:** `RMSE = sqrt(average((actual - predicted)²))`

**Example:** RMSE = 0.0020 with MAE = 0.0015 suggests some large outlier errors.

**Interpretation:**
- Lower is better
- Always ≥ MAE (equality means all errors are same size)
- RMSE much larger than MAE → outliers are a problem

---

### Directional Accuracy
**What it measures:** How often the forecast predicts the correct direction (up/down).

**Example:** Directional accuracy = 0.58 means 58% of forecasts got the direction right.

**Interpretation:**
- 0.50 = random guessing
- 0.55+ is generally useful for trading
- High directional accuracy with high MAE: right direction, wrong magnitude

---

## Monte Carlo Simulation Terms

### n_sims (Number of Simulations)
How many random price paths to generate. More simulations = more stable results but slower.

**Guidelines:**
- Quick checks: 1,000
- Normal use: 2,000-5,000
- Publication/final decisions: 10,000+

---

### seed
Random number generator seed. Set this for reproducible results.

**Example:** `--params "seed=42"` ensures the same random paths each time.

---

### Drift (μ)
The expected average return per period. In GBM simulations, drift represents the trend component.

---

### Diffusion (σ)
The volatility component of price movement. Represents random fluctuation around the drift.

---

## Probability Terms

### prob_tp_first
Probability that take-profit is hit before stop-loss.

**Example:** prob_tp_first = 0.62 means 62% of simulated paths hit TP first.

---

### prob_sl_first
Probability that stop-loss is hit before take-profit.

**Example:** prob_sl_first = 0.28 means 28% of simulated paths hit SL first.

---

### prob_no_hit
Probability that neither TP nor SL is hit within the horizon.

**Example:** prob_no_hit = 0.10 means 10% of paths expire without hitting either barrier.

**Important:** High prob_no_hit means your barriers may be too far or horizon too short.

---

### prob_resolve
Probability that the trade completes (hits TP or SL).

**Formula:** prob_resolve = prob_tp_first + prob_sl_first = 1 - prob_no_hit

---

## Grid Optimization Terms

### grid_style
How to generate the TP/SL combinations to test:

| Style | Description | Use When |
|-------|-------------|----------|
| `fixed` | Regular grid from min to max | You know the range to search |
| `volatility` | Scales barriers to current volatility | Adaptive to market conditions |
| `ratio` | Varies risk/reward ratios | You want specific RR profiles |
| `preset` | Pre-configured for trading styles | Quick setup for scalp/swing/position |

---

### refine
Two-stage optimization: coarse search, then fine search around the best result.

**Example:** `--refine true --refine-radius 0.3` searches ±30% around the best initial candidate.

---

### top_k
Number of best candidates to return from optimization.

**Example:** `--top-k 5` returns the top 5 TP/SL combinations instead of just the best.

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

**Example:** You place a market buy at 1.1750, but get filled at 1.1752. Slippage = 2 pips.

**Why it matters:** Slippage reduces profits and increases losses. Account for it when backtesting.

### Pip
Smallest price increment for a currency pair. For most pairs: 0.0001. For JPY pairs: 0.01.

**Example:** EURUSD moves from 1.1750 to 1.1775 = 25 pips.

**Note:** Some brokers show "pipettes" (5 decimal places), where 1 pip = 10 pipettes.

### Spread
Difference between bid (sell) and ask (buy) price. This is a transaction cost.

**Example:** Bid 1.1748, Ask 1.1750 → Spread = 2 pips.

**Impact:** You start every trade 1 spread behind. Tight spreads are crucial for scalping.

### Lot Size
Standard position size in forex:
- **Standard lot:** 100,000 units
- **Mini lot:** 10,000 units
- **Micro lot:** 1,000 units

**Example:** 0.1 lots of EURUSD = 10,000 EUR notional.

### Risk/Reward Ratio (RR)
Ratio of potential profit to potential loss.

**Formula:** RR = TP distance / SL distance

**Example:** TP = 50 pips, SL = 25 pips → RR = 2.0 (you risk 25 to make 50).

**Rule of thumb:** Higher RR means you can be wrong more often and still profit.

### Drawdown
Peak-to-trough decline in account equity.

**Example:** Account peaks at $10,000, drops to $8,500 → Drawdown = 15%.

**Maximum drawdown:** Largest historical drawdown. Key risk metric.

---

## See Also

- [Forecasting Guide](FORECAST.md)
- [Regime Detection](forecast/REGIMES.md)
- [Barrier Analysis](BARRIER_FUNCTIONS.md)
- [Technical Indicators](TECHNICAL_INDICATORS.md)
