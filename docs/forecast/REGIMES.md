# Regime Detection

Regime detection identifies the current market "behavior mode" (trending, ranging, volatile). Different strategies work in different regimes.

**Related:**
- [GLOSSARY.md](../GLOSSARY.md) — Definitions of HMM, BOCPD, etc.
- [FORECAST.md](../FORECAST.md) — Price forecasting
- [SAMPLE-TRADE-ADVANCED.md](../SAMPLE-TRADE-ADVANCED.md) — Using regimes as trade filters

---

## Why Regimes Matter

Markets cycle through distinct behavioral phases:

| Regime | Characteristics | Strategy Implication |
|--------|-----------------|---------------------|
| **Low Volatility / Ranging** | Price oscillates in a band | Mean reversion works; trend-following fails |
| **Trending** | Directional momentum | Trend-following works; mean reversion fails |
| **High Volatility / Crisis** | Large unpredictable swings | Reduce size or stay out |

A strategy that profits in one regime may lose in another. Regime detection helps you:
1. **Filter trades:** Only enter when conditions match your strategy
2. **Adjust sizing:** Reduce risk in unfavorable regimes
3. **Detect breakouts:** Identify when the market is transitioning

---

## Methods

### 1. Hidden Markov Model (HMM)

**What it does:** Classifies each bar into one of N hidden "states" based on return and volatility patterns.

**How it works:**
1. Assumes the market switches between N underlying states
2. Each state has characteristic mean return and volatility
3. Uses observed data to estimate which state is currently active

**Example:**
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"
```

**Output:**
```
summary:
  last_state: 0
  state_shares:
    0: 0.623
    1: 0.377
  state_sigma:
    0: 0.000303   # Low volatility state
    1: 0.000739   # High volatility state
```

**Interpretation:**
- **State 0:** Low volatility (σ = 0.0003) — ranging/quiet market
- **State 1:** High volatility (σ = 0.0007) — trending/active market
- Currently in State 0 (low volatility)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_states` | 2 | Number of regimes to detect |

**When to use:**
- Ongoing regime classification
- Filtering strategies by market type
- Multi-day to multi-week analysis

---

### 2. Change-Point Detection (BOCPD)

**What it does:** Detects moments when the market's statistical properties changed.

**How it works:**
- Bayesian Online Change Point Detection
- Estimates probability that each bar marks a regime transition
- Doesn't classify regimes—just detects when changes occur

**Example:**
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method bocpd --threshold 0.5 --output summary
```

**Output:**
```
summary:
  last_cp_prob: 0.471
  max_cp_prob: 0.471
  change_points_count: 0
```

**Interpretation:**
- `last_cp_prob: 0.471` — 47% probability that a regime change just occurred
- Below threshold (0.5), so not flagged as a change point
- Higher probabilities indicate likely structural breaks

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Probability cutoff to flag a change point |
| `lookback` | 300 | Historical bars to analyze |

**When to use:**
- Detecting breakouts
- Alerting on market structure changes
- Invalidating stale forecasts

---

### 3. Markov-Switching AR (MS-AR)

**What it does:** Combines HMM with autoregressive modeling. Each regime has its own AR parameters.

**Example:**
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method ms_ar --params "k_regimes=2 order=1"
```

**When to use:**
- When regime changes affect both mean and autocorrelation structure
- Academic research contexts

---

## Practical Strategies

### Strategy 1: Regime Filter

Only enter trend-following trades when HMM detects high-volatility state:

```bash
# Check current regime
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"

# If state_sigma shows current state is high-volatility:
#   → Enable trend-following entries
# If current state is low-volatility:
#   → Disable trend-following or switch to mean reversion
```

### Strategy 2: Change-Point Alert

Monitor for regime transitions and reduce exposure when detected:

```bash
# Check for recent change points
python cli.py regime_detect EURUSD --timeframe H1 --method bocpd --threshold 0.6

# If last_cp_prob > 0.6:
#   → Tighten stops
#   → Reduce position size
#   → Consider closing positions
```

### Strategy 3: Regime-Conditional Barrier Analysis

Run barrier optimization separately for each regime:

```bash
# In low-volatility regime: tighter barriers
python cli.py forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --tp-min 0.15 --tp-max 0.5 --sl-min 0.1 --sl-max 0.4

# In high-volatility regime: wider barriers
python cli.py forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --tp-min 0.5 --tp-max 2.0 --sl-min 0.3 --sl-max 1.5
```

---

## Output Formats

### Summary Output (Default)
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --output summary
```
Shows aggregate statistics: current state, state distributions, volatility per state.

### Compact Output
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --output compact
```
Shows regime segments: start time, end time, duration, state ID.

### Full Output
```bash
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --output full
```
Shows per-bar state assignments and probabilities.

---

## Quick Reference

| Task | Command |
|------|---------|
| Classify regimes (2 states) | `python cli.py regime_detect EURUSD --method hmm --params "n_states=2"` |
| Classify regimes (3 states) | `python cli.py regime_detect EURUSD --method hmm --params "n_states=3"` |
| Detect change points | `python cli.py regime_detect EURUSD --method bocpd --threshold 0.5` |
| Markov-switching AR | `python cli.py regime_detect EURUSD --method ms_ar --params "k_regimes=2"` |

---

## See Also

- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
- [FORECAST.md](../FORECAST.md) — Price forecasting
- [VOLATILITY.md](VOLATILITY.md) — Volatility estimation
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL analysis
