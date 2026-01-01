## A Sample Forecast Analysis Guide

**Related Documentation:**
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) - Advanced version with regimes, HAR-RV, conformal intervals, MC barriers
- [FORECAST.md](FORECAST.md) - Detailed forecasting methods and parameters
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) - Deep dive into barrier analytics
- [EXAMPLE.md](EXAMPLE.md) - Complete end-to-end workflow
- [COMMON_ERRORS.md](COMMON_ERRORS.md) - Troubleshooting

Looking for a more advanced, risk‑aware version? See [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) (regimes, HAR‑RV, conformal intervals, Monte‑Carlo barrier optimization, and execution controls).

Below is a step‑by‑step walk‑through of the analysis that produced the expert report, targeting a short term trading in EURUSD.
Each step shows **what tool was used**, **why that particular input was chosen**, and **what the output tells us**.  
The language is kept simple so anyone with a basic interest in trading can follow the logic.

---

### 1. Pull the most recent price data (candles)

| Tool | Call | Why we used it |
|------|------|----------------|
| **`data_fetch_candles`** (H1) | `symbol=EURUSD`, `timeframe=H1`, `limit=200`, `indicators=EMA(20), EMA(50), RSI(14), MACD(12,26,9)` | <ul><li>**H1** = one‑hour bars – the natural granularity for short‑term (intraday) analysis.</li><li>200 bars give us roughly the last 8‑9 days of data, enough to see the current trend.</li><li>Adding **EMA‑20** and **EMA‑50** lets us see whether price is above or below short‑ and medium‑term moving averages (a quick trend check).</li><li>**RSI** (Relative Strength Index) shows if the market is “over‑bought” (>70) or “over‑sold” (<30).</li><li>**MACD** (Moving‑Average Convergence Divergence) tells us whether momentum is positive or negative.</li></ul> |
| **Result** | A table with columns `time,open,high,low,close,EMA_20,EMA_50,RSI_14,MACD,MACDh,MACDs`. The last few rows (the most recent hours) show: <br>‑ Price ≈ 1.1776 <br>‑ EMA‑20 ≈ 1.1754, EMA‑50 ≈ 1.1739 <br>‑ RSI ≈ 65 <br>‑ MACD line > signal (positive) | **Interpretation** <br>‑ Price is **above both EMAs** → bullish alignment. <br>‑ RSI is in the “strength” zone but not over‑bought. <br>‑ MACD histogram is near zero, meaning momentum is still positive but weakening – a possible short‑term pause. |

---

### 2. Get the daily price range for pivot‑point calculation

| Tool | Call | Why we used it |
|------|------|----------------|
| **`data_fetch_candles`** (D1) | `symbol=EURUSD`, `timeframe=D1`, `limit=30`, `ohlcv=ohlc` | <ul><li>Pivot points are traditionally calculated from the **previous day’s** high, low and close.</li><li>30 daily bars give us a recent history to confirm that the most recent day is representative.</li></ul> |
| **Result** | A table with the last 30 daily bars (open‑high‑low‑close). The most recent day (15 Sep) shows: <br>‑ High = 1.17745, Low = 1.16569, Close = 1.17608 | This daily H‑L‑C will be fed into the next step. |

---

### 3. Compute classic pivot‑point levels

| Tool | Call | Why we used it |
|------|------|----------------|
| **`pivot_compute_points`** | `symbol=EURUSD`, `timeframe=D1` | <ul><li>Each run returns classic, Fibonacci, Camarilla, Woodie, and DeMark tables so you can compare support/resistance ladders.</li><li>Every method lists **support (S1, S2, …)** and **resistance (R1, R2, …)** tiers that traders monitor.</li></ul> |
| **Result** | JSON with: <br>‑ Pivot (PP) = 1.17505 <br>‑ R1 = 1.17848 <br>‑ S1 = 1.17264 <br>‑ R2, S2, R3, S3 also provided. | **Interpretation** <br>‑ Current price (≈ 1.1776) sits **just below R1** and **above the pivot** – a classic “test‑and‑break” situation. <br>‑ If price falls, S1 (1.17264) is the first support; if it breaks above R1, the next target is R2 (≈ 1.1809). |

---

### 4. Estimate near‑future volatility

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_volatility_estimate`** | `symbol=EURUSD`, `timeframe=H1`, `horizon=12`, `method=ewma`, `params={lambda:0.94}` | <ul><li>**EWMA** (Exponentially Weighted Moving Average) gives a quick, robust estimate of recent volatility.</li><li>`lambda=0.94` is the standard smoothing factor used in many risk‑models (e.g., RiskMetrics).</li><li>`horizon=12` means we want the volatility for the next 12 hourly bars (≈ ½ day).</li></ul> |
| **Result** | <ul><li>Hourly σ (standard deviation) ≈ 0.000593 → **≈ 5.9 pips** per hour.</li><li>12‑hour σ ≈ 0.002055 → **≈ 20 pips** (≈ 0.20 %).</li></ul> | **Interpretation** <br>‑ Over the next half‑day we can expect the price to wander about **± 20 pips** (1 σ). <br>‑ This helps us size stops and targets so they are realistic relative to normal market moves. |

---

### 5. Forecast the price path for the next 12 hours

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_generate`** | `symbol=EURUSD`, `timeframe=H1`, `method=theta`, `horizon=12`, `quantity=price`, `target=price` | <ul><li>The **Theta** method is a fast, reliable forecasting model that works well on short‑term series.</li><li>We ask for a **price forecast** (not returns) for the next 12 hourly bars.</li></ul> |
| **Result** | JSON with: <br>‑ Forecasted price for each of the next 12 hours (≈ 1.17528 → 1.17543). <br>‑ 95 % confidence interval (lower ≈ 1.1717, upper ≈ 1.1789). <br>‑ Trend flag = **up**. | **Interpretation** <br>‑ The model expects a **small pull‑back** toward the pivot (1.1750) before the up‑trend resumes. <br>‑ The confidence band comfortably contains the pivot and the first resistance, confirming the “test‑and‑bounce” picture. |

---

### 6. Find the statistically‑optimal TP/SL (Take‑Profit / Stop‑Loss) levels

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_barrier_optimize`** | `symbol=EURUSD`, `timeframe=H1`, `horizon=12`, `method=hmm_mc`, `mode=pct`, `grid_style=volatility`, `refine=true`, `tp_min=0.25`, `tp_max=1.5`, `tp_steps=7`, `sl_min=0.25`, `sl_max=2.5`, `sl_steps=9`, `objective=edge` | <ul><li>**Monte-Carlo barrier analysis** simulates many possible price paths (here using a **Gaussian HMM** - a regime-switching model that captures changing volatility).</li><li>The volatility-scaled grid spans compact scalps (~0.25%) through swing targets (~1.5%) and lets stops widen automatically (up to ~2.5 multiples), so both defensive and aggressive reward/risk profiles are explored without hand-tuning.</li><li>With `refine=true` the optimizer performs a second, tighter sweep around the initial best combo.</li><li>The **objective "edge"** = *P(TP first) - P(SL first)*, i.e., the net probability of a winning trade.</li></ul> |
| **Result** | JSON with a 7x9 grid (63 combos). Each candidate includes TP/SL hit probabilities, resolve probability, edge, Kelly, EV (plus conditional and per-bar variants), and median time-to-hit stats. | **Interpretation** <br>- Use `objective=edge` for setups where TP-first odds are the priority; switch to `kelly`, `ev`, `ev_cond`, or `ev_per_bar` when payoff asymmetry or time-to-resolution matters more. <br>- Wider stops now show up in the default grid, so you can choose conservative ratios without redefining the search space. |

---

### 7. Putting it all together – Trade ideas

| Step | How the previous outputs shaped the idea |
|------|------------------------------------------|
| **Current market picture** (Step 1 & 3) | Price is above the 20‑EMA, below R1, and near the daily pivot → likely to **pull back** to the pivot before trying to break R1. |
| **Volatility check** (Step 4) | 12‑hour σ ≈ 20 pips → a 0.20 % TP (≈ 23 pips) is roughly **one‑sigma** away, a realistic target; a 1 % SL (≈ 118 pips) is far beyond normal moves, making the stop unlikely to be hit. |
| **Forecast** (Step 5) | The Theta forecast expects the price to settle around **1.1753**, i.e., near the pivot, confirming a short‑term pull‑back. |
| **Barrier optimisation** (Step 6) | Quantifies the **edge** of each TP/SL pair, giving us the **statistically‑best** setups (0.20 %/1 % and 0.40 %/0.80 %). |
| **Resulting trade plan** | • **Primary long**: Enter near the pivot (≈ 1.1750), TP = 0.20 % (≈ 1.1785), SL = 1.00 % (≈ 1.1658). <br>• **Secondary long** (more balanced): TP = 0.40 % (≈ 1.1795), SL = 0.80 % (≈ 1.1680). <br>• **Short‑term counter‑trend**: If price cleanly closes above R1, consider a short with TP back to the pivot and a modest SL. |

---

## TL;DR – The “Why” in Plain English

1. **Grab recent price data** (hourly candles) and add a few simple indicators (moving averages, RSI, MACD) to see the short‑term trend and momentum.  
2. **Pull the previous day’s high/low/close** to calculate classic support/resistance levels (pivot points).  
3. **Estimate how much the price normally wiggles** over the next half‑day (EWMA volatility).  
4. **Ask a forecasting model** what price it expects in the next 12 hours – it suggests a modest pull‑back toward the pivot.  
5. **Run a Monte‑Carlo simulation** that tries many possible TP/SL combos and tells us which pair gives the highest statistical edge.  
6. **Combine everything**: the trend, the pivot, the volatility, the forecast, and the edge‑analysis to craft concrete trade setups with clear entry, target, and stop levels.

By following these steps you move from raw price numbers to **data‑driven trade ideas** that are backed by both technical analysis and statistical probability. This is the same logical chain that underlies the expert report you received.

