## Advanced Forecast‑to‑Trade Playbook (EURUSD, H1)

**Related Documentation:**
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) - Basic workflow (start here if new)
- [FORECAST.md](FORECAST.md) - Detailed forecasting methods
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) - Barrier analytics deep dive
- [EXAMPLE.md](EXAMPLE.md) - Complete end-to-end workflow
- [COMMON_ERRORS.md](COMMON_ERRORS.md) - Troubleshooting

This guide extends the basic workflow with regime filters, conformal intervals, realized‑volatility (HAR‑RV), Monte‑Carlo barrier analytics, and disciplined risk/execution controls. It is designed to be modular: run each block, inspect outputs, and gate the next step by thresholds you calibrate via backtests.

Assumptions
- Horizon: 12 H1 bars (≈ half‑day)
- Symbol/TF: EURUSD/H1
- All commands are runnable via `python cli.py <tool> ... --format json`

---

### 0) Safety & Hygiene

- Skip high‑impact news windows (e.g., ±60 minutes of CPI/NFP/FOMC).
- Enforce daily loss cap (e.g., 1–2× average daily VaR) and per‑trade risk cap (e.g., 0.25–1.0% equity).
- Consider minimum spread/liquidity filter (e.g., spread < 1.5× median) before entry.

---

### 1) Regime & Break Detection (Gatekeeper)

Detect structural breaks and label regimes so you avoid trading through hostile phases.

1.1 BOCPD change‑points (returns)

```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method bocpd --threshold 0.6 --output summary --lookback 300 --format json
```

- Gate: if `max(cp_prob[-24:]) >= 0.6` → stand down or reduce size; retrain/recalibrate models.

1.2 HMM‑lite regimes (returns)

```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method hmm --params "n_states=3" --output compact --lookback 300 --format json
```

- Derive a simple regime tag: {trend‑lowvol, trend‑highvol, range} from `state` and `state_probabilities`.
- Gate: trade only when regime in {trend‑lowvol, trend‑midvol}; reduce risk in range/highvol.

Optional: MS‑AR(1) (statsmodels)
```bash
python cli.py regime_detect EURUSD --timeframe H1 --limit 1500 \
  --method ms_ar --params "k_regimes=2 order=1" --output summary --format json
```

---

### 2) Realized Volatility & Risk Budget (HAR‑RV)

Estimate daily realized variance from intraday returns, then map to H1.

```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22" --format json
```

- Extract `sigma_bar_return` (per‑bar σ) and `horizon_sigma_return` (k‑bar σ).
- Risk budget: set per‑trade risk ≤ min(0.7× daily VaR, fixed cap). Use σ to set realistic TP/SL and lot size.

---

### 3) Denoise + Quick Technical Context

Pull data with light denoising and a few TIs for situational awareness (no heavy feature stacks in this flow).

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 300 \
  --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)" \
  --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2,keep_original=true" --format json
```

- Context: price vs EMA(20/50), RSI near extremes, MACD momentum slope.
- Gate: prefer longs if price > EMA(20)>EMA(50) and regime=trend.

---

### 4) Forecast with Valid Intervals (Conformal)

Calibrate per‑step residual quantiles via rolling backtest; then get point + conformal bands.

```bash
python cli.py forecast_conformal_intervals EURUSD --timeframe H1 --method fourier_ols \
  --horizon 12 --steps 25 --spacing 10 --alpha 0.1 --format json
```

- Use `lower_price`/`upper_price` (conformal), not model CIs, for entry gating and sizing.
- Gate: only take longs if `conformal.lower_price[h-1] > pivot` and point forecast trend=up.

---

### 5) Barrier Analytics (MC + Closed‑Form)

5.1 Optimize TP/SL grid with HMM MC paths

```bash
python cli.py forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --grid-style volatility --refine true --refine-radius 0.35 \
  --tp_min 0.25 --tp_max 1.5 --tp_steps 7 --sl_min 0.25 --sl_max 2.5 --sl_steps 9 \
  --params "n_sims=5000 seed=7" --top-k 5 --return-grid false --output summary --format json
```

- Choose a combo by objective (edge/kelly/ev/ev_cond/ev_per_bar/prob_resolve/profit_factor/min_loss_prob/utility) subject to constraints:
  - Use `min_prob_win`, `max_prob_no_hit`, and `max_median_time` (bars) to enforce hit-rate and timing limits.

5.2 TP/SL odds for the chosen combo

```bash
python cli.py forecast_barrier_hit_probabilities EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --tp_pct 0.4 --sl_pct 0.8 --params "n_sims=5000 seed=7" --format json
```

5.3 Closed‑form GBM sanity check (fast)

```bash
python cli.py forecast_barrier_closed_form EURUSD --timeframe H1 --horizon 12 \
  --direction up --barrier 1.1795 --format json
```

- Flag discrepancies (e.g., MC>>GBM) to reduce size or re‑check calibration.

---

### 6) Labeling & Threshold Calibration (Optional but Recommended)

Use triple‑barrier labels offline for signal evaluation and meta‑models.

```bash
python cli.py labels_triple_barrier EURUSD --timeframe H1 --limit 2000 \
  --horizon 12 --tp_pct 0.4 --sl_pct 0.8 --label-on high_low \
  --output summary --lookback 300 --format json
```

- Compute in‑sample precision/recall for your entry rules; adjust thresholds (edge, cp_prob, RSI, EMA alignment) to reach desired trade quality.

---

### 7) Position Sizing & Execution

- Position size (conservative Kelly/VaR):
  - Kelly_cap = 0.25 × Kelly (from optimizer) or ≤ 1.0% equity, whichever is smaller.
  - VaR sizing: risk_to_SL ≤ risk_budget, where risk_to_SL uses spread‑adjusted SL and conformal lower.
- Stops & targets:
  - TP/SL from optimizer; time stop at horizon if neither is hit.
  - Optional partial TP at 0.5×TP; move stop to breakeven.
- Costs: subtract spread/commission from TP; inflate SL by typical slippage.

---

### 8) Backtest & Walk‑Forward Checks

1) Rolling backtest for chosen forecast method(s)

```bash
python cli.py forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --steps 50 --spacing 5 --methods "theta fourier_ols" --format json
```

2) Stress‑test entry thresholds
- Sweep edge minima (0.05→0.15), cp_prob caps, regime sets, and confirm Sharpe/win rate stability.

---

### 9) Trade Plans (Examples)

Plan A – Breakout with pullback filter
- Gates: regime in {trend}, cp_prob<0.6, price>EMA(20)>EMA(50), conformal.lower>pivot.
- Targets: TP=0.40%, SL=0.80% (from optimizer); time stop at 12 bars.
- Size: min(VaR_budget, 0.25×Kelly) with HAR‑RV sigma.

Plan B – Mean‑reversion in range regime (reduced size)
- Gates: regime=range & low cp_prob; RSI>70 near R1 or <30 near S1.
- Smaller size; tighter SL and 0.20–0.30% TP; MC must show positive edge.

---

### 10) Monitoring & Drift Handling

- Update BOCPD and HMM twice per session; stand down on cp spikes.
- Refresh HAR‑RV daily (intraday M5 RV aggregation).
- Re‑calibrate conformal residuals weekly; re‑grid barrier optimizer monthly.
- Track realized vs. forecast errors; trigger re‑training on degradation.

---

## TL;DR – Advanced Flow

1) Filter: regime & cp_prob gates.  
2) Calibrate risk with HAR‑RV; set budget.  
3) Denoise + quick TI context (EMA/RSI/MACD).  
4) Forecast with conformal intervals; gate by bands.  
5) Optimize TP/SL via MC; sanity‑check with GBM closed‑form.  
6) (Optional) Label history with triple‑barrier; tune thresholds.  
7) Execute with VaR/Kelly‑capped sizing, time stop, and costs.  
8) Walk‑forward checks; adjust thresholds and monitoring cadence.
