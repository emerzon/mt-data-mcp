# BTCUSD CLI E2E Findings

Test date: March 1, 2026

Environment used:
- Python 3.12.3
- Repo-declared supported runtime: Python 3.14
- Connected MT5 server: `ICMarketsSC-Demo`
- Terminal AutoTrading state: disabled at the client terminal during the test

## Status after fixes

Closed:
- `forecast_generate` now accepts the legacy internal `method=...` call path again, so report generation no longer fails on that mismatch.
- `trade_account_info` now exposes terminal readiness fields, and `trade_place` now blocks earlier with a structured preflight payload when AutoTrading or related execution gates are off.
- `trade_history --symbol BTCUSD` now post-filters returned rows by symbol, and a live recheck returned only `BTCUSD` rows for both `orders` and `deals`.
- `data_fetch_candles --format json` now returns numeric JSON values instead of stringified numbers.
- `report_generate` now captures model warnings into diagnostics instead of leaking raw warning lines into the main output.

Clarified:
- The `python cli.py --help` failure on Python 3.12 is not a product compatibility target for this repo because the project is intentionally Python 3.14-only. The remaining issue there is only that the unsupported-runtime failure mode is a raw traceback instead of a cleaner version error.

Still open:
- `patterns_detect` default output is still too noisy for fast discretionary use.
- `regime_detect` default output is still too verbose for execution-oriented decisions.
- Pivot/session timezone context is still more implicit than it should be.
- `trade_risk_analyze` still does not explain risk overshoot from lot-step rounding prominently enough.
- Positional-versus-flag UX and some modify/close diagnostics are still less intuitive than they should be.

## Workflow I attempted

I approached this as a discretionary trader trying to move from market study to execution through `cli.py`.

Study flow:
- `symbols_describe BTCUSD`
- `data_fetch_candles BTCUSD --timeframe H1 --limit 200 --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)"`
- `pivot_compute_points BTCUSD --timeframe D1`
- `forecast_volatility_estimate BTCUSD --timeframe H1 --horizon 12 --method ewma --params "lambda=0.94"`
- `regime_detect BTCUSD --timeframe H1 --method hmm --params "n_states=3"`
- `patterns_detect BTCUSD --timeframe H1 --mode candlestick --robust-only true`
- `report_generate BTCUSD --template advanced --timeframe H1 --output markdown`
- `forecast_barrier_prob BTCUSD --timeframe H1 --horizon 12 --method mc --mc-method hmm_mc --direction long --tp-pct 2 --sl-pct 1`
- `trade_risk_analyze --symbol BTCUSD --desired-risk-pct 1 --proposed-entry 66317 --proposed-sl 65000 --proposed-tp 69000`

Execution flow attempted:
- `trade_account_info`
- `trade_get_open`
- `trade_get_pending`
- `trade_place BTCUSD --volume 0.01 --order-type BUY_LIMIT --price 62000 --stop-loss 60000 --take-profit 68000 --expiration 30m --comment BTC_E2E_TEST`
- `trade_history --history-kind orders --symbol BTCUSD --limit 5`
- `trade_history --history-kind deals --symbol BTCUSD --limit 5`
- `trade_modify 999999999 --price 61000`
- `trade_close --close-kind pending --ticket 999999999`

## Market-study takeaways

The analysis stack was usable enough to build a view on BTCUSD:
- Hourly BTCUSD was roughly range-to-soft-bearish at test time, with price near the H1 EMAs and neutral RSI.
- D1 pivots were materially below the market after the prior session dislocation, which is plausible but needs better contextual explanation.
- H1 EWMA horizon volatility for 12 bars was about 2.93 percent.
- Barrier probabilities for a 2 percent TP versus 1 percent SL were unfavorable on the long side: `prob_tp_first=0.183`, `prob_sl_first=0.4865`, `prob_no_hit=0.3305`, `edge=-0.3035`.
- Position sizing for a 1 percent-risk BTCUSD idea produced `0.02` lots, but actual risk rounded to `1.08 percent`.

## Findings

### 1. Running `cli.py` on Python 3.12 fails with an opaque unsupported-runtime error

When I ran `python cli.py --help`, the CLI crashed before argument parsing with:

`TypeError: ArgumentParser.__init__() got an unexpected keyword argument 'suggest_on_error'`

My reaction as a user:
- I would assume the tool is broken, not that I am on an unsupported runtime.
- After confirmation, this is an environment mismatch rather than a repo compatibility bug, because the project is Python 3.14-only by design.
- The remaining UX gap is that the failure looks like a low-level `argparse` problem instead of a clear "Python 3.14 required" message.

### 2. `forecast_generate` is broken at the CLI layer

When I tried `forecast_generate BTCUSD --timeframe H1 --horizon 12 --library native --model theta`, the tool failed with:

`TypeError: forecast_generate() got an unexpected keyword argument 'method'`

My reaction as a user:
- This is the flagship forecasting command, so I expect it to be reliable.
- The CLI appears to be passing stale parameter names to the underlying forecast tool.
- I cannot trust the forecast workflow if the advertised top-level command fails on a vanilla case.

### 3. `trade_account_info` is not enough to determine whether execution is actually available

`trade_account_info` returned:
- `trade_allowed: true`
- `trade_expert: true`

But `trade_place` failed with:
- `retcode: 10027`
- `comment: "AutoTrading disabled by client"`

My reaction as a user:
- I thought the account was trade-ready because the account info looked permissive.
- In practice, the terminal-level AutoTrading gate was off, and I only learned that after a failed order submission.
- I want a preflight command, or I want `trade_account_info` to surface terminal readiness, demo/live state, and AutoTrading state explicitly.

### 4. `trade_place` reaches MT5, but execution readiness is discovered too late

The pending order request itself was well-formed and the returned payload included the exact MT5 request, which is useful.

My reaction as a user:
- I like that the failed response includes the request body.
- I do not like learning about terminal execution readiness only after submitting the order.
- I want a pre-submit readiness check that catches AutoTrading disabled, symbol trading disabled, and similar blockers before I fire an order.

### 5. `trade_history --symbol BTCUSD` returned non-BTC rows

I ran:
- `trade_history --history-kind orders --symbol BTCUSD --limit 5`
- `trade_history --history-kind deals --symbol BTCUSD --limit 5`

Both returned an `XAUUSD` row in the result set.

My reaction as a user:
- This is the most serious data-integrity issue I hit in the trading workflow.
- If I filter for `BTCUSD`, I should never see `XAUUSD`.
- I cannot safely audit trade history or performance if symbol filtering is not trustworthy.

### 6. JSON typing is inconsistent across tools

`data_fetch_candles --format json` returned candle values like `"close": "66317"` as strings.

But:
- `pivot_compute_points` returned numeric values as JSON numbers.
- `forecast_volatility_estimate` returned numeric values as JSON numbers.
- `forecast_barrier_prob` returned numeric values as JSON numbers.

My reaction as a user:
- I expect `--format json` to be structurally consistent and machine-friendly.
- Having candle prices and indicator values serialized as strings forces extra cleanup in any downstream analysis.
- This inconsistency makes automation brittle.

### 7. `patterns_detect` is too noisy for discretionary decision-making

`patterns_detect BTCUSD --timeframe H1 --mode candlestick --robust-only true` returned 218 pattern rows over 1000 candles.

My reaction as a user:
- The output is exhaustive, but not prioritized.
- I need ranking, recency emphasis, confidence, or at least a compact summary of the last few actionable patterns.
- As it stands, the signal-to-noise ratio is too low for a fast trading workflow.

### 8. `regime_detect` is informative but operationally overwhelming

The HMM regime output returned a very long list of regime segments with start, end, bars, regime, and confidence.

My reaction as a user:
- The raw detail is useful for research, but not for execution.
- I need a top summary first: current regime, prior regime, transition recency, volatility ordering, and maybe a confidence score.
- The current default output is too verbose for a trader trying to make a decision quickly.

### 9. `report_generate` is useful, but warnings leak into the user-facing output

The advanced report was one of the best outputs in the whole flow, but it printed a raw `sklearn` `ConvergenceWarning` before the report body.

My reaction as a user:
- I like the report content because it condenses multiple tools into one narrative.
- I do not want model-library warnings mixed into trader-facing output.
- Warnings should be captured, summarized, and labeled as diagnostics instead of dumped inline.

### 10. The report is helpful, but some context is still too implicit

The report showed:
- D1 pivots for `2026-02-27 16:00 -> 2026-02-28 16:00`
- mixed EMA context
- negative barrier edge
- backtest ranking

My reaction as a user:
- The report is close to what I want for a discretionary desk note.
- I still want clearer explanation of broker-session day boundaries and timezone assumptions, especially for pivot periods.
- Without that context, a cross-check against charting platforms is harder than it should be.

### 11. `trade_risk_analyze` is useful, but the rounded sizing result can exceed the requested risk

I asked for 1 percent risk and received:
- `suggested_volume: 0.02`
- `risk_pct: 1.08`

My reaction as a user:
- I understand this is likely due to volume-step rounding.
- I want the tool to say that explicitly and tell me whether the result is rounded up or down.
- For risk-sensitive execution, overshooting the requested risk should be called out prominently.

### 12. Optional parameters becoming flags versus positionals is not intuitive

I initially tried:
- `trade_risk_analyze BTCUSD ...`
- `trade_modify --ticket 999999999 ...`

The first failed because `symbol` is optional in the function signature, so the CLI expects `--symbol`.
The second failed because `ticket` is required and therefore positional.

My reaction as a user:
- The positional-versus-flag behavior follows internal signature rules, not trader intuition.
- I would prefer stronger help examples and more consistent UX for high-value trading commands.
- In the current form, I have to learn parser implementation details to use the commands correctly.

### 13. `trade_modify` and `trade_close` error messages are acceptable but still not ideal

`trade_modify 999999999 --price 61000` returned:
- `Pending order 999999999 not found. Note: price/expiration only apply to pending orders.`

`trade_close --close-kind pending --ticket 999999999` returned:
- `Pending order 999999999 not found`

My reaction as a user:
- The modify message is better because it explains the inference rule.
- I would still prefer the command to tell me whether it also checked open positions, pending orders, or both.

## What worked best

The strongest tools in this flow were:
- `symbols_describe`
- `forecast_volatility_estimate`
- `forecast_barrier_prob`
- `report_generate`
- `trade_place` failure payload formatting

These gave me useful information quickly and in mostly actionable form.

## What blocked true end-to-end completion

I could not complete an actual place-modify-cancel order lifecycle because the MT5 client terminal had AutoTrading disabled.

That blocker is partly environmental, but from a product perspective the CLI should surface it earlier and more clearly.
