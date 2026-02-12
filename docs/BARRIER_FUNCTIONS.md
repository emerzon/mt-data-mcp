# Barrier Functions - Comprehensive Guide

## Overview

**Related:**
- [GLOSSARY.md](GLOSSARY.md) — TP/SL, pips, and spread
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Practical workflow example
- [CLI.md](CLI.md) — CLI usage and output formats

Barrier functions are essential tools for risk management in trading. They help answer the critical question: *"What's the probability that my take-profit will be hit before my stop-loss within a given time horizon?"*

This document provides a deep dive into the barrier analytics available in mtdata, covering the underlying algorithms, when to use each method, and real-world trading scenarios.

---

## Quick Start (Simple Usage First)

### 1) Probability for one TP/SL pair

Percent barriers are expressed in percent (e.g., `--tp-pct 0.40` means **0.40%**, not 40%):
```bash
python cli.py forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
  --method mc --mc-method mc_gbm --direction long --tp-pct 0.40 --sl-pct 0.60 --format json
```

Look for `prob_tp_first`, `prob_sl_first`, `prob_no_hit`, and `edge` in the output.

### 2) Search for “good” TP/SL levels

```bash
python cli.py forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style volatility --objective edge --format json
```

---

## Core Concepts

### What is a Barrier?

A barrier is a price level that, when touched, triggers an event:
- **Take-Profit (TP)**: Profit-taking level
- **Stop-Loss (SL)**: Risk-limiting level

### The Barrier Problem

Given:
- Current price: `S₀`
- TP level: `B_tp`
- SL level: `B_sl`
- Time horizon: `T` (number of bars)
- Trade direction: long/short

We want to compute:
- `P(TP first)`: Probability TP is hit before SL
- `P(SL first)`: Probability SL is hit before TP
- `P(no hit)`: Probability neither is hit by time T
- `E[time to hit]`: Expected time until resolution

---

## Available Methods & Algorithms

### 1. Monte Carlo GBM (`mc_gbm`)

**Algorithm**: Geometric Brownian Motion simulation

**Mathematical Model**:
```
S_t = S₀ * exp((μ - 0.5σ²)t + σW_t)
```
where:
- `μ` = drift (mean log-return)
- `σ` = volatility (std of log-returns)
- `W_t` = Wiener process (Brownian motion)

**How it works**:
1. Calibrate μ and σ from historical log-returns
2. Generate N random paths using the formula above
3. Count paths that hit TP first vs SL first

**Strengths**:
- Fast and stable
- Minimal assumptions
- Works with limited history

**Weaknesses**:
- Assumes constant volatility
- Ignores fat tails
- No regime awareness

**When to use**:
- Short horizons (< 20 bars)
- Stable market conditions
- Quick screening of many TP/SL combinations
- Baseline for comparison

**Real-life example**:
> **Scenario**: EURUSD scalp on 5-minute chart, horizon = 12 bars (1 hour)
>
> **Why GBM**: Short timeframe, relatively stable intraday volatility, need fast results for live trading
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob EURUSD --timeframe M5 --horizon 12 \
>   --method mc --mc-method mc_gbm --tp-pct 0.2 --sl-pct 0.15
> ```

---

### 2. Brownian Bridge GBM (`mc_gbm_bb`)

**Algorithm**: GBM with Brownian Bridge correction

**Mathematical Model**:
The Brownian Bridge is a Brownian motion conditioned to return to a specific value at time T. For barrier hitting, it provides more accurate probabilities for short horizons by accounting for the path between endpoints.

**Key insight**: If we know the price at time T (from simulation), the probability of hitting a barrier between t and t+Δt is:
```
P(hit | S_t, S_T) = exp(-2 * (B - S_t)(B - S_T) / (σ²Δt))
```

**How it works**:
1. Generate GBM paths as normal
2. Apply bridge correction between each time step
3. Detect hits that occur "between" discrete simulation points

**Strengths**:
- More accurate for short horizons
- Captures intra-barrier hits
- Better resolution for tight TP/SL

**Weaknesses**:
- Slightly slower than plain GBM
- Requires more history for stable σ

**When to use**:
- Very short horizons (< 10 bars)
- Tight TP/SL (near current price)
- Scalping strategies
- When precision matters more than speed

**Real-life example**:
> **Scenario**: Crypto scalping on 1-minute BTC chart, TP = 0.1%, SL = 0.08%, horizon = 6 bars
>
> **Why Bridge**: Tiny barriers, very short horizon - need to capture hits that occur between 1-minute marks
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob BTCUSD --timeframe M1 --horizon 6 \
>   --method mc --mc-method mc_gbm_bb --tp-pct 0.1 --sl-pct 0.08
> ```

---

### 3. HMM Monte Carlo (`hmm_mc`)

**Algorithm**: Hidden Markov Model with regime-switching

**Mathematical Model**:
```
State s_t ∈ {1, 2, ..., K} (hidden)
Return r_t ~ N(μ_{s_t}, σ_{s_t})
Transition: P(s_t = j | s_{t-1} = i) = A_{ij}
```

**How it works**:
1. Fit Gaussian mixture to log-returns (EM algorithm)
2. Estimate transition matrix from soft assignments
3. Simulate state paths via Markov chain
4. Sample returns conditional on current state
5. Build price paths

**Strengths**:
- Captures volatility regimes (low/high vol)
- Adapts to changing market conditions
- More realistic than constant-volatility models
- Handles trending vs ranging markets

**Weaknesses**:
- Requires more history (200+ bars)
- Computationally intensive
- Needs careful initialization

**When to use**:
- Medium to long horizons (10-100 bars)
- Clearly regime-switching markets
- Swing/position trading
- When volatility clustering is evident

**Real-life example**:
> **Scenario**: GBPUSD swing trade on 4H chart, horizon = 48 bars (8 days)
>
> **Why HMM**: Markets show clear regimes - trending days vs consolidation periods
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob GBPUSD --timeframe H4 --horizon 48 \
>   --method mc --mc-method hmm_mc --tp-pct 1.5 --sl-pct 1.0 \
>   --params "n_states=2 n_sims=5000"
> ```

---

### 4. GARCH Monte Carlo (`garch`)

**Algorithm**: Generalized Autoregressive Conditional Heteroskedasticity

**Mathematical Model**:
```
r_t = μ + ε_t
ε_t = σ_t * z_t, z_t ~ N(0,1)
σ_t² = ω + α * ε_{t-1}² + β * σ_{t-1}²
```

**How it works**:
1. Fit GARCH(1,1) to scaled returns
2. Forecast volatility path conditional on history
3. Simulate returns using time-varying σ_t
4. Build price paths

**Strengths**:
- Explicit volatility modeling
- Captures volatility clustering
- Widely studied and validated
- Good for risk forecasting

**Weaknesses**:
- Requires `arch` package
- Needs substantial history (100+ bars)
- Assumes normal innovations
- Sensitive to parameter initialization

**When to use**:
- Volatility clustering is strong (high autocorrelation in squared returns)
- Risk management focus
- Options pricing or volatility trading
- When you need volatility forecasts, not just prices

**Real-life example**:
> **Scenario**: SPY options hedge on daily chart, horizon = 20 days
>
> **Why GARCH**: Volatility clustering is well-documented in equity indices; need accurate vol for options
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob SPY --timeframe D1 --horizon 20 \
>   --method mc --mc-method garch --tp-pct 3.0 --sl-pct 2.0 \
>   --params "p=1 q=1"
> ```

---

### 5. Bootstrap (`bootstrap`)

**Algorithm**: Circular block bootstrap

**How it works**:
1. Compute historical returns
2. Sample contiguous blocks of returns (size = `block_size`)
3. Concatenate blocks to create synthetic paths
4. No parametric assumptions

**Strengths**:
- Non-parametric (no distribution assumptions)
- Preserves autocorrelation structure
- Captures fat tails and skew
- Works with any return distribution

**Weaknesses**:
- Limited to historical patterns
- Cannot extrapolate beyond observed volatility
- Requires long history for good coverage

**When to use**:
- Non-normal returns with complex patterns
- When you suspect model misspecification
- Validation of parametric models
- Markets with structural breaks

**Real-life example**:
> **Scenario**: Emerging market FX (e.g., USDTRY), horizon = 24 bars
>
> **Why Bootstrap**: Returns have fat tails, skew, and jumps that parametric models miss
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob USDTRY --timeframe H1 --horizon 24 \
>   --method mc --mc-method bootstrap --tp-pct 2.0 --sl-pct 1.5 \
>   --params "block_size=5"
> ```

---

### 6. Heston Model (`heston`)

**Algorithm**: Stochastic volatility (Heston 1993)

**Mathematical Model**:
```
dS_t = μS_t dt + √v_t S_t dW_t^1
dv_t = κ(θ - v_t) dt + ξ√v_t dW_t^2
dW_t^1 dW_t^2 = ρ dt
```

**Parameters**:
- `κ`: mean reversion speed
- `θ`: long-term variance
- `ξ`: volatility of volatility
- `ρ`: correlation between price and vol
- `v₀`: initial variance

**Strengths**:
- Captures leverage effects (ρ)
- Stochastic volatility
- Closed-form option pricing available
- More realistic than GBM

**Weaknesses**:
- Complex parameter estimation
- Requires many assumptions
- Computationally expensive
- Hard to calibrate

**When to use**:
- Equity options trading
- When leverage effect is important
- Long-dated derivatives
- Academic/research contexts

**Real-life example**:
> **Scenario**: AAPL options strategy, horizon = 60 days
>
> **Why Heston**: Need stochastic vol + leverage effect for accurate option pricing
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob AAPL --timeframe D1 --horizon 60 \
>   --method mc --mc-method heston --tp-pct 10.0 --sl-pct 5.0 \
>   --params "kappa=2.0 theta=0.04 xi=0.3 rho=-0.5"
> ```

---

### 7. Jump Diffusion (`jump_diffusion`)

**Algorithm**: Merton jump-diffusion

**Mathematical Model**:
```
dS_t = (μ - λk)S_t dt + σS_t dW_t + S_t dJ_t
J_t: compound Poisson process with log-normal jumps
```

**Parameters**:
- `jump_lambda`: jump intensity (frequency)
- `jump_mu`: mean jump size
- `jump_sigma`: jump size volatility
- `jump_threshold`: outlier threshold for calibration

**Strengths**:
- Captures sudden price jumps
- Models tail events
- Better for crisis periods
- Accounts for news events

**Weaknesses**:
- Difficult to calibrate reliably
- Many parameters
- Overfits easily
- Requires long history

**When to use**:
- Earnings announcements
- Major news events
- Cryptocurrency markets
- Crisis periods

**Real-life example**:
> **Scenario**: Trading around FOMC announcement, horizon = 4 hours
>
> **Why Jump Diffusion**: Expecting discrete price jumps from news, not continuous movement
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob EURUSD --timeframe M15 --horizon 16 \
>   --method mc --mc-method jump_diffusion --tp-pct 0.5 --sl-pct 0.3 \
>   --params "jump_lambda=0.5 jump_mu=0.001 jump_sigma=0.002"
> ```

---

### 8. Automatic Selection (`auto`)

**Algorithm**: Heuristic method selector based on data characteristics

**Decision Tree**:
```
Note: `garch` requires the `arch` package; auto falls back to `heston` if it is not available.
1. Insufficient history (< 10 bars or < 30 returns)?
   → mc_gbm_bb if horizon ≤ 12, else mc_gbm

2. Crypto or short timeframe (≤ 15 min)?
   → If heavy tails (kurtosis > 2, jump_ratio > 4): jump_diffusion
   → Otherwise continue

3. Heavy tails (kurtosis > 3.5, jump_ratio > 5)?
   → jump_diffusion

4. Volatility regime change (vol_ratio > 1.6 or < 0.65)?
   → hmm_mc (if enough history)

5. Strong volatility clustering (vol_corr > 0.3, r2 size ≥ 150)?
   → garch (if available) or heston

6. Moderate volatility clustering (vol_corr > 0.15)?
   → heston

7. Non-normal but no clear patterns (skew > 0.7, not jumpy)?
   → bootstrap

8. Short horizon and mild tails?
   → mc_gbm_bb

9. Default:
   → mc_gbm
```

**Strengths**:
- No need to manually select method
- Adapts to data characteristics
- Prevents obvious mistakes

**Weaknesses**:
- Black box (but reports reason)
- May not be optimal for all cases
- Still needs sufficient history

**When to use**:
- Initial exploration
- Production automation
- When unsure about market characteristics
- Multi-asset screening

**Real-life example**:
> **Scenario**: Screening 20 currency pairs for trading opportunities
>
> **Why Auto**: Don't want to manually analyze each pair's characteristics
>
> **Command**:
> ```bash
> python cli.py forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
>   --method auto --tp-pct 0.5 --sl-pct 0.3
>
> # Returns: method_used: "hmm_mc", auto_reason: "auto: regime shift (volatility change)"
> ```

---

## Optimization: Finding Optimal TP/SL

### Grid Search (`forecast_barrier_optimize`)

Searches combinations of TP/SL to maximize an objective function.

**Grid Styles**:

#### 1. Fixed Grid (`grid_style=fixed`)
```
TP: [tp_min, tp_max] with tp_steps points
SL: [sl_min, sl_max] with sl_steps points
Total combinations: tp_steps × sl_steps
```

**Use case**: General purpose, known reasonable ranges

**Example**:
```bash
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct \
  --tp_min 0.25 --tp_max 1.5 --tp_steps 7 \
  --sl_min 0.25 --sl_max 2.5 --sl_steps 9
```

---

#### 2. Volatility Grid (`grid_style=volatility`)
Scales TP/SL based on recent volatility

**Algorithm**:
```
vol_per_bar = std(returns[-vol_window:])
vol_horizon = vol_per_bar * sqrt(horizon)
vol_pct = vol_horizon * 100

tp_start = max(vol_floor_pct, vol_pct * vol_min_mult)
tp_end = vol_pct * vol_max_mult
sl_start = max(vol_floor_pct, vol_pct * vol_min_mult * 0.8)
sl_end = sl_start * vol_sl_multiplier
```

**Use case**: Adapts to current market volatility

**Example**:
```bash
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style volatility \
  --params "vol_window=250 vol_min_mult=0.5 vol_max_mult=4.0 vol_sl_multiplier=1.8"
```

---

#### 3. Ratio Grid (`grid_style=ratio`)
Fixes risk/reward ratio

**Algorithm**:
```
For each SL in [sl_min, sl_max]:
  For each ratio in [ratio_min, ratio_max]:
    TP = SL * ratio
```

**Use case**: When you want specific risk/reward profiles

**Example**:
```bash
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style ratio \
  --ratio_min 1.5 --ratio_max 3.0 --ratio_steps 5 \
  --sl_min 0.3 --sl_max 1.0 --sl_steps 5
```

---

#### 4. Preset Grid (`grid_style=preset`)
Pre-configured ranges for trading styles

**Presets**:
- `scalp`: TP 0.08-0.60%, SL 0.20-1.20%
- `intraday`: TP 0.25-1.50%, SL 0.25-2.50%
- `swing`: TP 0.60-3.50%, SL 0.50-4.50%
- `position`: TP 1.00-8.00%, SL 0.75-6.00%

**Use case**: Quick setup for standard trading styles

**Example**:
```bash
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style preset \
  --preset swing
```

---

### Objectives

Choose what to optimize. Each objective answers a different trading question:

#### Quick Reference Table

| Objective | Formula | Best For |
|-----------|---------|----------|
| `edge` | `P(win) - P(loss)` | General purpose, consistent win rate |
| `prob_tp_first` | `P(win)` | Maximize win rate only |
| `prob_resolve` | `1 - P(no hit)` | Ensure trades complete |
| `kelly` | `P(win) - P(loss)/RR` | Position sizing |
| `kelly_cond` | Kelly on resolved trades only | Position sizing (ignoring timeouts) |
| `ev` | `P(win)*TP - P(loss)*SL` | Maximize profit per trade |
| `ev_cond` | EV on resolved trades only | Profit per trade (ignoring timeouts) |
| `ev_per_bar` | `EV / mean_resolve_time` | Fast trades, capital turnover |
| `profit_factor` | `(P(win)*TP) / (P(loss)*SL)` | Risk/reward ratio focus |
| `min_loss_prob` | Minimize `P(loss)` | Capital preservation |
| `utility` | `P(win)*log(1+TP) + P(loss)*log(1-SL)` | Risk-averse trading |

#### Detailed Descriptions

**`edge`** — *"How often do I win vs lose?"*
- Measures raw probability advantage
- Edge = 0.20 means you win 20% more often than you lose
- **Use when:** You want consistent winners, regardless of payoff size
- **Limitation:** Ignores how much you win/lose

**`ev` (Expected Value)** — *"What's my average profit per trade?"*
- Accounts for both probability AND payoff size
- EV = 0.15% means you expect to gain 0.15% per trade on average
- **Use when:** Payoff asymmetry matters (e.g., small wins, big losses)
- **Limitation:** Doesn't account for trade duration

**`ev_per_bar`** — *"What's my profit per unit of time?"*
- Normalizes EV by how long trades take
- Favors fast trades over slow ones with same total EV
- **Use when:** Capital turnover matters (reinvesting profits)
- **Limitation:** May favor trades with higher transaction costs

**`kelly`** — *"How much should I bet?"*
- Optimal fraction of capital for maximum long-term growth
- Kelly = 0.25 means bet 25% of capital (but use fractional Kelly in practice)
- **Use when:** Position sizing decisions
- **Limitation:** Full Kelly leads to large drawdowns; use 0.25× Kelly

**`prob_resolve`** — *"Will my trade actually close?"*
- Probability that TP or SL is hit within the horizon
- High value means trades complete; low value means many expire
- **Use when:** You need trades to close (e.g., day trading)
- **Limitation:** High resolve doesn't mean profitable

**`profit_factor`** — *"What's my gains-to-losses ratio?"*
- Profit factor > 1 means profitable; > 2 is very good
- Common metric in backtesting reports
- **Use when:** Comparing strategies by risk/reward
- **Limitation:** Doesn't account for trade frequency

**`min_loss_prob`** — *"How do I avoid losses?"*
- Minimizes probability of stop-loss being hit
- **Use when:** Capital preservation is priority
- **Limitation:** May result in tiny TP or trades that never resolve

**`utility`** — *"What's my risk-adjusted outcome?"*
- Logarithmic utility penalizes large losses more than it rewards equivalent gains
- Naturally avoids bets that could wipe you out
- **Use when:** Risk-averse trading, avoiding ruin
- **Limitation:** More theoretical than practical

Notes:
- `_cond` variants (e.g., `ev_cond`, `kelly_cond`) calculate metrics only on trades that resolved (hit TP or SL), ignoring timeouts.
- `ev_per_bar` uses mean resolution time (`t_hit_resolve_mean`) when available.

---

### Two-Stage Refinement

**Problem**: Coarse grid may miss optimal values

**Solution**:
1. Run coarse grid search
2. Zoom around best result
3. Run fine grid in zoomed region

**Parameters**:
- `refine=true`: Enable refinement
- `refine_radius=0.3`: Search within ±30% of best
- `refine_steps=5`: Points per dimension in refinement

**Example**:
```bash
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct \
  --tp_min 0.25 --tp_max 1.5 --tp_steps 5 \
  --sl_min 0.25 --sl_max 2.5 --sl_steps 5 \
  --refine true --refine_radius 0.35 --refine_steps 7
```

---

### Constraints

Filter candidates before ranking:

| Constraint | Description | Example |
|------------|-------------|---------|
| `min_prob_win` | Minimum win probability | `0.5` (50%) |
| `max_prob_no_hit` | Maximum no-hit probability | `0.2` (20%) |
| `max_median_time` | Maximum resolution time (bars) | `10` |

**Example**:
```bash
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct \
  --tp_min 0.5 --tp_max 2.0 --tp_steps 5 \
  --sl_min 0.5 --sl_max 2.0 --sl_steps 5 \
  --min_prob_win 0.55 --max_prob_no_hit 0.15 --max_median_time 8
```

---

## Real-Life Trading Scenarios

### Scenario 1: Scalping EURUSD

**Context**: 5-minute chart, quick trades, tight spreads

**Requirements**:
- Fast execution
- High win rate
- Short holding time

**Method Selection**:
```bash
# Use auto or mc_gbm_bb for precision
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe M5 --horizon 6 \
  --method mc_gbm_bb --mode pct --grid-style preset \
  --preset scalp \
  --objective kelly_cond \
  --refine true --refine_radius 0.25
```

**Interpretation**:
- Look for `kelly_cond > 0` (positive edge after costs)
- `prob_resolve > 0.7` (most trades close out)
- `t_hit_resolve_median < 4` (fast resolution)

---

### Scenario 2: Swing Trading Gold

**Context**: 4H chart, multi-day holds, volatility spikes

**Requirements**:
- Handle regime changes
- Account for volatility clustering
- Wider stops

**Method Selection**:
```bash
# HMM captures regimes, volatility grid adapts to current vol
python cli.py forecast_barrier_optimize \
  XAUUSD --timeframe H4 --horizon 60 \
  --method hmm_mc --mode pct --grid-style volatility \
  --params "vol_window=300 vol_min_mult=0.8 vol_max_mult=3.0" \
  --objective edge --min_prob_win 0.5
```

**Interpretation**:
- Look for `edge > 0.1` (10% advantage)
- Check `prob_no_hit` (avoid if too high)
- Verify `time_to_tp_seconds.median` aligns with swing timeframe

---

### Scenario 3: Post-Earnings Options

**Context**: AAPL daily chart, jump risk from earnings

**Requirements**:
- Model jumps
- Short horizon (1-5 days)
- High volatility expected

**Method Selection**:
```bash
# Jump diffusion captures earnings moves
python cli.py forecast_barrier_optimize \
  AAPL --timeframe D1 --horizon 5 \
  --method jump_diffusion --mode pct \
  --tp_min 3.0 --tp_max 10.0 --tp_steps 5 \
  --sl_min 2.0 --sl_max 6.0 --sl_steps 5 \
  --params "jump_lambda=0.3" \
  --objective ev
```

**Interpretation**:
- Focus on `ev` (expected value)
- Check `prob_tie` (should be low)
- Verify jump parameters via `model_summary`

---

### Scenario 4: Range-Bound Market

**Context**: EURUSD in tight range, mean reversion

**Requirements**:
- High probability of no hit
- Small barriers
- Quick resolution if hit

**Method Selection**:
```bash
# Bootstrap preserves range characteristics
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method bootstrap --mode pct \
  --tp_min 0.2 --tp_max 0.6 --tp_steps 5 \
  --sl_min 0.2 --sl_max 0.6 --sl_steps 5 \
  --objective prob_resolve \
  --max_prob_no_hit 0.3
```

**Interpretation**:
- High `prob_resolve` (trades close out)
- Low `prob_no_hit`
- Balanced `prob_win` and `prob_loss`

---

### Scenario 5: Multi-Asset Screening

**Context**: Finding best setups across 20 pairs

**Requirements**:
- Automated selection
- Consistent comparison
- Fast processing

**Method Selection**:
```bash
# Auto method adapts to each pair
for pair in EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF; do
  python cli.py forecast_barrier_optimize \
    $pair --timeframe H1 --horizon 12 \
    --method auto --mode pct --grid-style volatility \
    --objective edge --output summary --top_k 1
done
```

**Interpretation**:
- Rank pairs by `edge` from `best` output
- Check `auto_reason` to understand method choice
- Filter by `prob_resolve > 0.6`

---

## Output Interpretation

### Single Barrier Query

```json
{
  "success": true,
  "method": "hmm_mc",
  "prob_tp_first": 0.62,
  "prob_sl_first": 0.28,
  "prob_no_hit": 0.10,
  "edge": 0.34,
  "time_to_tp_bars": {"mean": 6.2, "median": 5.0},
  "time_to_sl_bars": {"mean": 4.8, "median": 4.0}
}
```

**Interpretation**:
- `prob_tp_first = 0.62`: 62% chance TP hits first
- `prob_sl_first = 0.28`: 28% chance SL hits first
- `prob_no_hit = 0.10`: 10% chance neither hits
- `edge = 0.34`: 34% advantage (TP prob - SL prob)
- **Decision**: Good trade if edge > 0 and you have edge after costs

---

### Optimization Output

```json
{
  "best": {
    "tp": 0.85,
    "sl": 0.62,
    "rr": 1.37,
    "prob_win": 0.58,
    "prob_loss": 0.32,
    "prob_no_hit": 0.10,
    "prob_resolve": 0.90,
    "edge": 0.26,
    "kelly": 0.34,
    "kelly_cond": 0.47,
    "ev": 0.22,
    "ev_cond": 0.32,
    "ev_per_bar": 0.03,
    "profit_factor": 2.45,
    "utility": 0.18,
    "t_hit_resolve_mean": 6.8,
    "t_hit_resolve_median": 6.0
  },
  "auto_reason": "auto: regime shift (volatility change)"
}
```

**Interpretation**:
- **TP/SL**: 0.85% / 0.62% (RR = 1.37)
- **Win prob**: 58% (conditional on resolution: 58/(58+32) = 64%)
- **Edge**: 26% raw (TP prob − SL prob)
- **Resolve prob**: 90% of paths hit TP/SL within the horizon
- **Kelly**: 34% (raw), 47% conditional on resolution
- **EV**: 0.22% per trade; **EV per bar**: 0.03% based on mean resolve time
- **Profit factor**: 2.45 (wins/losses ratio)
- **Utility**: 0.18 (risk‑averse log utility)
- **Resolve time**: mean 6.8 bars, median 6 bars
- **Method used**: HMM (detected regime shift)

**Decision**: This is a good setup - positive edge, reasonable RR, fast resolution.

---

## Common Pitfalls & Solutions

### Pitfall 1: Insufficient History

**Problem**: Method fails or gives unreliable results

**Solution**:
```bash
# Check history length first
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500

# If insufficient, use mc_gbm (or mc_gbm_bb for short horizons)
python cli.py forecast_barrier_prob EURUSD --timeframe H1 --horizon 10 \
  --method mc --mc-method mc_gbm --tp-pct 0.2 --sl-pct 0.15
```

---

### Pitfall 2: Overfitting to Recent Volatility

**Problem**: Optimizer picks TP/SL that won't generalize

**Solution**:
```bash
# Use longer vol_window for stable estimates
--params "vol_window=500"

# Or bootstrap (non-parametric)
--method bootstrap

# Validate with rolling backtest
```

---

### Pitfall 3: Ignoring No-Hit Probability

**Problem**: High edge but many trades never close

**Solution**:
```bash
# Add constraint
--max_prob_no_hit 0.2

# Or optimize for resolve probability
--objective prob_resolve
```

---

### Pitfall 4: Wrong Method for Market Regime

**Problem**: Using GBM in highly regime-switching market

**Solution**:
```bash
# Use auto to select appropriate method
--method auto

# Or manually check for regimes
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=3"
```

---

### Pitfall 5: Not Accounting for Slippage/Spread

**Problem**: Paper trading results don't match reality

**Solution**:
- Reduce TP by spread/2
- Increase SL by spread/2
- Or use `tp_pips`/`sl_pips` which accounts for pip size

```bash
# Example: 2 pip spread on EURUSD
python cli.py forecast_barrier_prob \
  EURUSD --timeframe M5 --horizon 12 \
  --method mc --mc-method hmm_mc --tp_pips 20 --sl_pips 15  # RR = 1.33 after spread
```

---

## Advanced Tips

### 1. Multi-Horizon Analysis

Run multiple horizons to understand time dynamics:

```bash
for H in 6 12 24 48; do
  echo "Horizon $H bars:"
  python cli.py forecast_barrier_prob \
    EURUSD --timeframe H1 --horizon $H \
    --method mc --mc-method hmm_mc --tp-pct 0.5 --sl-pct 0.3 \
    --format json | jq '{horizon: .horizon, edge: .edge, prob_resolve: (.prob_tp_first + .prob_sl_first)}'
done
```

---

### 2. Regime-Conditional Analysis

Detect regimes first, then optimize per regime:

```bash
# Detect current regime
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"

# If high-vol regime, use wider barriers
# If low-vol regime, use tighter barriers
```

---

### 3. Denoising for Cleaner Signals

```bash
# Apply denoising before simulation
python cli.py forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style preset --preset swing \
  --denoise lowpass_fft --denoise-params "cutoff_ratio=0.1"
```

---

### 4. Parameter Sensitivity

Test stability of results:

```bash
# Vary n_sims
for N in 1000 2000 5000 10000; do
  python cli.py forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
    --method mc --mc-method hmm_mc --tp-pct 0.5 --sl-pct 0.3 --params "n_sims=$N" \
    --format json | jq '.prob_tp_first'
done
```

---

### 5. Cross-Validation

Compare methods on same data:

```bash
for METHOD in mc_gbm hmm_mc bootstrap; do
  echo "Method: $METHOD"
  python cli.py forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
    --method mc --mc-method $METHOD --tp-pct 0.5 --sl-pct 0.3 \
    --format json | jq '{method: .method, edge: .edge, prob_resolve: (.prob_tp_first + .prob_sl_first)}'
done
```

---

## Performance Considerations

### Speed vs Accuracy Trade-off

| Method | Speed | Accuracy | History Needed |
|--------|-------|----------|----------------|
| mc_gbm | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Minimal |
| mc_gbm_bb | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Minimal |
| hmm_mc | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 200+ bars |
| garch | ⭐⭐ | ⭐⭐⭐⭐ | 100+ bars |
| bootstrap | ⭐⭐⭐ | ⭐⭐⭐⭐ | 400+ bars |
| heston | ⭐ | ⭐⭐⭐⭐ | 200+ bars |
| jump_diffusion | ⭐ | ⭐⭐⭐⭐ | 300+ bars |

### Optimization Speed

- **Fixed grid**: `O(S × T × S)` where S=sl_steps, T=tp_steps
- **Volatility grid**: Similar, but adaptive ranges
- **Refinement**: Adds ~2× computation
- **n_sims**: Linear scaling (2× sims = 2× time)

**Tips for speed**:
1. Start with `mc_gbm` for screening
2. Use `output=summary` to limit grid output
3. Set `top_k` to limit evaluations
4. Reduce `n_sims` for initial runs (1000), increase for final (5000+)

---

## Summary Decision Tree

```
Start with: --method auto --horizon [your_horizon]

If horizon < 10:
  → Consider mc_gbm_bb for precision

If horizon > 50:
  → Consider hmm_mc or garch for regimes

If crypto/jumpy:
  → Consider jump_diffusion

If non-normal/fat tails:
  → Consider bootstrap

If optimizing:
  → Use grid-style volatility for adaptive ranges
  → Use refine=true for precision
  → Set constraints (min_prob_win, max_prob_no_hit)

Validate:
  → Check prob_no_hit < 0.2
  → Check edge > 0
  → Check median time fits your style
  → Run cross-validation across methods
```

---

## Quick Reference

### Most Common Commands

**Quick check**:
```bash
python cli.py forecast_barrier_prob \
  EURUSD --timeframe H1 --horizon 12 \
  --method auto --tp-pct 0.5 --sl-pct 0.3
```

**Find optimal**:
```bash
python cli.py forecast_barrier_optimize \
  EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style volatility \
  --refine true --objective edge
```

**Closed-form check**:
```bash
python cli.py forecast_barrier_prob \
  EURUSD --timeframe H1 --horizon 12 \
  --method closed_form --direction long --barrier 1.1000
```

---

## Further Reading

- [FORECAST.md](FORECAST.md) - General forecasting documentation
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) - Example trading workflows
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Troubleshooting

---

*Last updated: 2026-01-01*
