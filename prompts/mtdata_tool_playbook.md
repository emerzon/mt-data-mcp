# mtdata Tool Playbook

This is the shared operating reference for the short-term trading profiles in
this directory. It describes the live mtdata registry as surveyed on
2026-07-10: 87 catalog entries, of which 86 are enabled by default and
`market_depth_fetch` is conditional.

The registry, not this file, remains the source of truth. At session boot call
`tools_list(detail="full", include_related=true, limit=200)`. If its inventory
differs from this document, do not guess at new or removed interfaces. Use the
runtime schema and report the mismatch before using the affected tool.

## Operating Principles

- A tool result is evidence, not permission to trade. Account safety, live
  tradability, price freshness, execution cost, structural invalidation, and
  quantified risk must all pass independently.
- Never claim that a strategy, forecast, pattern, or backtest guarantees a
  profit. Short-horizon evidence is market- and sample-dependent.
- Never trade from a forecast, pattern label, regime label, denoised series,
  simplified series, report summary, news headline, or indicator alone.
- Use raw completed candles for structure and a fresh bid/ask for execution.
  Buy entries use ask-side geometry; sell entries use bid-side geometry.
- Default to `detail="compact"`. Request `standard` or `full` only when the
  extra fields can change the current decision.
- Do not run optimizers, model training, broad scans, or large reports in a hot
  execution loop. Cache their conclusions and invalidate them on a new regime,
  material event, or broker-day boundary.
- Never simplify data used for risk, volatility, statistics, forecasting, or
  backtesting. Denoising used in a live workflow must be causal and compared
  with the raw series.

## Side-Effect Labels

| Label | Meaning |
|---|---|
| `R` | Read-only and normally bounded. |
| `R-heavy` | Read-only but potentially slow or computationally expensive. |
| `R-blocking` | Read-only but waits before returning. |
| `S` | Changes local task or model-store state, not broker exposure. |
| `L` | Can change live broker orders or positions. Its `dry_run` default is false. |
| `G` | Conditional tool; availability depends on configuration or provider support. |

## Common Result Parser

Apply this parser after every call, before interpreting tool-specific fields.

1. **Envelope:** if `success` is false or a non-empty `error` exists, stop using
   the payload as market evidence. Record `error_code`, `request_id`,
   `operation`, `remediation`, and `related_tools`. Retry once only when the
   failure is a corrected payload or a clearly transient read failure.
2. **Partial results:** if `partial_failure` is true, inspect `failed_sections`
   or nested errors. Never infer a missing section from the sections that did
   succeed.
3. **Freshness:** inspect `as_of`, `retrieved_at`, quote time,
   `data_age_seconds`, `data_stale`, `freshness`, and runtime timezone metadata.
   Stale quotes block new risk. Compare event times in one explicit timezone.
4. **Collections:** prefer the collection identified by `row_key` or
   `canonical_source`. Otherwise inspect, in order, `items`, `rows`, `data`,
   `series`, `groups`, or a documented tool-specific collection. Respect
   `count`, `total_count`, pagination, and empty collections.
5. **Units:** read `units`, `digits`, `point`, `trade_tick_size`, currency, and
   percentage/fraction labels before comparing numbers. A value of `0.55` is
   not interchangeable with `55` unless the contract says so.
6. **Quality:** treat `warning`, `warnings`, `sample_status`, `sample_quality`,
   `metrics_reliability`, `degraded`, incomplete risk, missing optional
   dependencies, and fallback data sources as explicit uncertainty.
7. **Probabilities:** use payoff-weighted, cost-adjusted expected value. Do not
   require `P(TP) > P(SL)` when payoff sizes differ, and do not ignore no-hit or
   unresolved probability.
8. **Execution:** a successful preview has `dry_run=true`,
   `actionability="preview_only"`, or `would_send_order=false`. A successful
   live response is still provisional until open positions, pending orders,
   and when necessary history confirm the resulting broker state.

## Shared Live-Risk Contract

The supplied profiles use these moderate defaults unless the user explicitly
lowers them:

- Maximum risk per new entry: 0.50% of current equity.
- Maximum combined risk for all profiles on one symbol: 1.00% of equity.
- Broker-day realized plus floating loss gate: 2.00% of day-start equity.
- Maximum quantified open risk across the account: 3.00% of equity.
- Pending orders count at their all-fill stop risk, not at zero risk.
- `fixed_fraction` is the default sizing method. Kelly sizing is unavailable
  until `trade_journal_analyze` has at least 30 comparable realized exits and
  returns a reliable positive edge; even then use half-Kelly or less and retain
  the limits above.
- Stop loss and take profit are required on new market orders. No martingale,
  uncapped grid, revenge trade, or add justified only by unrealized loss.
- After two consecutive losing campaigns for one profile and symbol, block new
  risk for at least 60 minutes and until a new primary-timeframe bar closes.

Magic-number coexistence assumes MT5 preserves independently ticketed
positions, as on a hedging account. On a netting account, same-symbol orders can
merge into one net position and magic ownership is not a safe mutation
boundary. If netting behavior is configured or observed, allow only one
risk-adding profile to own a symbol; the account-wide supervisor may still
protect or reduce the resulting net position.

Before every risk-increasing action: refresh account and exposure state, check
market status and quote freshness, account for spread and slippage, size with
`trade_risk_analyze`, preview with `dry_run=true`, refresh the quote, then send
the same protected payload with a stable `idempotency_key`. Make at most one
risk-increasing action in a cycle. Verify immediately afterward.

Example preview payload:

```json
{
  "symbol": "EURUSD",
  "volume": 0.01,
  "order_type": "BUY",
  "stop_loss": 1.0800,
  "take_profit": 1.0900,
  "magic": 71001,
  "dry_run": true,
  "require_sl_tp": true,
  "auto_close_on_sl_tp_fail": true,
  "idempotency_key": "scalp-eurusd-20260710-001"
}
```

Do not reuse an idempotency key for a different payload. The key is an
in-process safeguard, not broker-side idempotency and not durable across a
restart.

## Asset Context Routing

- Always start with `news(symbol=SYMBOL)`; it unifies general, calendar, and
  symbol-relevant context when providers are available.
- US equities: add `finviz_news`, `finviz_calendar(calendar="earnings")`, and
  `finviz_fundamentals` only when company-specific context can alter the hold.
  Use options tools only after `options_provider_status` passes.
- FX: add `finviz_calendar(calendar="economic")` and `finviz_forex` when
  currency or macro context is missing from `news`.
- Futures, metals, and indices: add `finviz_futures` and the economic calendar.
- Crypto: add `finviz_crypto`; treat exchange availability as distinct from
  the broker symbol's tradability and spread.
- Finviz is a context provider, not the executable quote source. Provider
  timestamps and symbol mappings can differ from MT5.

## Tool Catalog

### Research

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `causal_discover_signals` | R-heavy | Testing whether lagged returns from a bounded basket add predictive information. | Read ordered cause/effect rows, best lag, p-value, samples, transform, and alpha. Granger evidence is predictive association, not structural causality; require out-of-sample confirmation. |
| `cointegration_test` | R-heavy | Researching stable relative-value pairs or a Johansen basket. | For Engle-Granger read orientation, p-value, hedge/spread diagnostics, overlap, and significance. For Johansen read rank tests and vectors. Cointegration can break; it is not a live entry by itself. |
| `confluence_levels` | R | Several independently derived level families may overlap. | Read reference price, scored zones, source-family count, zone bounds, distance, and warnings. Treat a zone as an area, never an exact executable price. |
| `correlation_matrix` | R-heavy | Measuring portfolio concentration or discovering hedge candidates. | Read pair correlation, sample count, transform, aligned window, and optional matrix. Correlation is not cointegration, causation, or a stable hedge ratio. |
| `cross_correlation` | R-heavy | Confirming lead/lag or a proposed proxy hedge between exactly two symbols. | Read best lag, signed correlation, overlap, bootstrap confidence interval, and lag convention. Positive best lag means the first symbol leads the second. Reject unstable signs or intervals crossing zero. |
| `labels_triple_barrier` | R-heavy | Evaluating historical TP/SL outcomes for a fixed horizon and geometry. | Read historical labels, barrier outcome counts, direction, horizon, and no-hit cases. Labels use future paths and therefore must never become a live feature for the same bar. |
| `market_snapshot` | R | Obtaining a fast orientation before targeted calls. | Read `snapshot`, `as_of`, `quote_as_of`, `assembled_at`, `partial_failure`, and failed sections. Patterns are marked information-only. The bundle does not replace broker constraints, exposure, news, or final quote checks. |
| `news` | R/G | Session boot, scheduled refresh, abnormal moves, and before new risk. | Read provider buckets, event times, relevance, importance, symbol mapping, and warnings. Headlines change the event gate, not direction by themselves. |
| `outliers_detect` | R | Suspected bad bars, event shocks, or unstable model input. | Read flagged timestamps, fields, robust scores, method, threshold, and sample. Determine whether each outlier is a data issue or a real event before excluding anything. |
| `seasonality_detect` | R-heavy | Proposing repeatable periods for research or forecast configuration. | Read ranked periods, autocorrelation/spectral evidence, cycles observed, and sample size. Confirm on separate history; a dominant period is exploratory. |
| `stationarity_test` | R-heavy | Choosing a modeling transform or assessing a spread. | Read ADF/KPSS/PP results, null hypotheses, p-values, availability, and combined conclusion. `mixed` or `inconclusive` is not stationary. |
| `tools_list` | R | Boot discovery, schema drift checks, or finding related tools. | Read tool names, categories, required/optional parameters, enabled state, pagination, and modules. This is the canonical inventory. |
| `volatility_term_structure` | R | Comparing current realized volatility with its historical distribution. | Read each horizon's current value, cone percentiles, annualization flag, lookback, and warnings. Match horizon and units to the intended holding period. |
| `volume_profile_levels` | R | Mapping POC, value area, and high/low-volume structure. | Read POC, VAH, VAL, nodes, source, volume type, bucket settings, and reference price. `m1_bars` and FX tick volume are approximations, not exchange order flow. |

### Data Access

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `data_fetch_candles` | R | Reading raw completed structure or a deliberately small indicator pack. | Read OHLCV rows, timestamps, indicator columns, row counts, quality/freshness, incomplete-bar status, spread inclusion, denoise metadata, and simplification metadata. Exclude the forming candle from close-confirmed rules. |
| `data_fetch_ticks` | R | Spread baselines, recent quote activity, tight-stop checks, or execution timing. | Read tick rows or summary statistics, bid/ask/last, spread distribution, flags, source window, and returned count. A small tick sample is not market depth. |
| `wait_event` | R-blocking | No immediate action is justified and the next bar or watched state should trigger refresh. | Read event type, trigger reason, elapsed time, timeout, watched conditions, and symbol/timeframe. A timeout means refresh, not permission to trade. |

### Method Discovery

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `denoise_describe` | R | Before using one denoise method. | Read parameters, causality support, dependencies, defaults, and warnings. Live workflows permit causal settings only. |
| `denoise_list_methods` | R | Discovering installed filters and causal capability. | Read method availability, family, causality, optional dependencies, and automatic parameters. Availability does not establish trading value. |
| `indicators_describe` | R | Confirming syntax, output columns, warmup, and interpretation for one indicator. | Read parameter contract, category, output names, and documentation. Fetch enough warmup bars and avoid redundant indicators. |
| `indicators_list` | R | Finding a small candidate indicator set by style or category. | Read names, categories, availability, descriptions, pagination, and filters. Discovery is not strategy selection or validation. |

### Market and External Context

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `finviz_calendar` | R/G | Checking economic, earnings, or dividend events. | Read event type, date/time, impact, country/currency/symbol, estimate/actual fields, pagination, and provider warnings. Normalize provider time before applying a blackout. |
| `finviz_crypto` | R/G | Adding broad crypto performance context. | Read ranked rows, performance horizons, quote age, and provider status. Do not substitute it for the MT5 symbol quote. |
| `finviz_description` | R/G | Confirming a US company's business exposure. | Read description and symbol mapping. This is slow-changing context, not an entry signal. |
| `finviz_earnings` | R/G | Fast review of upcoming US earnings. | Read company, symbol, date/session, estimates when present, page, and freshness. Confirm symbol relevance and timing. |
| `finviz_filters_list` | R/G | Building a valid Finviz screen. | Read filter names, accepted values, pagination, and search results. Use returned values exactly. |
| `finviz_forex` | R/G | Broad currency-pair performance context. | Read pair rows and performance fields. Resolve naming differences before relating a row to an MT5 symbol. |
| `finviz_fundamentals` | R/G | A US equity hold depends on valuation, quality, growth, or balance-sheet context. | Read requested category/fields, values, missing fields, and provider timestamp. Fundamentals do not time an intraday entry. |
| `finviz_futures` | R/G | Broad futures, metals, commodity, or index context. | Read contract rows and performance/change fields. Provider instruments may not match the broker CFD exactly. |
| `finviz_insider` | R/G | Reviewing company-specific reported insider transactions. | Read transaction date, filing date, insider role, side, value/shares, pagination, and freshness. Filing latency prevents using it as a micro trigger. |
| `finviz_insider_activity` | R/G | Surveying market-wide reported insider activity. | Read option/filter, rows, dates, sides, values, and pagination. Use as background only. |
| `finviz_market_news` | R/G | Unified `news` is thin or a broad market narrative needs a provider fallback. | Read headline rows, source, publication time, link, news/blog type, and page. Deduplicate and check age. |
| `finviz_news` | R/G | A US equity needs ticker-specific headline context. | Read ticker mapping, headline, publication time, source, and pagination. Confirm material claims elsewhere when execution depends on them. |
| `finviz_peers` | R/G | Comparing a US equity with direct peers or finding hedge candidates. | Read peer symbols, count, and mapping. Peer status does not imply high or stable return correlation. |
| `finviz_ratings` | R/G | Recent analyst actions may explain a move or overnight risk. | Read action, firm, rating/target fields, date, and extras. Ratings are contextual and can be stale. |
| `finviz_screen` | R-heavy/G | Researching a bounded US equity candidate universe. | Read filters, view, ordered rows, page, and provider limits. Screening on current data followed by historical testing can introduce selection bias. |
| `market_scan` | R-heavy | Filtering a specified MT5 universe on price, spread, volume, RSI, or SMA state. | Read flat rows, ranking field/order, filters, offsets, timeframe, and failed symbols. Recheck any candidate with symbol-specific tools. |
| `market_status` | R | Session boot, reopen/close boundaries, or before new risk. | With a symbol read status, reason, `is_tradable`, `can_open_new_positions`, trade mode, and tick freshness. Region-only status is context, not symbol tradability. |
| `market_ticker` | R | Every executable-price decision and post-action verification. | Read bid, ask, mid/last, spread and cost fields, digits, quote time, age, stale flag, and freshness. This is the executable reference, subject to slippage. |
| `market_depth_fetch` | G | DOM can change market-versus-pending execution and the environment flag plus broker support are confirmed. | Read enabled/support state, bids, asks, sizes, spread, and subscription errors. It is disabled by default and `market_ticker` is the fallback. |

### Forecasting and Backtesting

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `forecast_backtest_run` | R-heavy | Selecting or revalidating forecast methods on rolling historical anchors. | Read per-method errors, RMSE/MAE, directional accuracy, trading metrics, steps, spacing, costs, sample notices, and ranking. Use time-ordered results and reject tiny samples. |
| `forecast_barrier_optimize` | R-heavy | Searching TP/SL geometry after direction and invalidation are independently known. | Read selected objective, best and candidate rows, TP/SL units, hit/no-hit probabilities, EV, edge, Kelly, time-to-hit, viability/tradability, costs, and warnings. Optimization can overfit; validate nearby candidates. |
| `forecast_barrier_prob` | R-heavy | Evaluating one already-defined TP/SL pair. | Read TP-first, SL-first, tie/no-hit probabilities, resolution, EV/edge when supplied, method, simulations, and horizon. Match barrier side and units to direction. |
| `forecast_conformal_intervals` | R-heavy | Calibrated uncertainty width can change size, stop, or abstention. | Read point path, lower/upper bands, alpha, calibration anchors, residual coverage, and warnings. Wide or poorly calibrated bands reduce conviction. |
| `forecast_generate` | R-heavy | A validated method provides secondary directional or path evidence. | Read method/library, quantity, horizon, forecast values, intervals, training/model metadata, warnings, and source window. Do not compare a return forecast directly with a price target. |
| `forecast_list_library_models` | R | Discovering installed model names in one forecast library. | Read model names, library, availability/reason, pagination, and capability metadata. Choose only enabled models. |
| `forecast_list_methods` | R | Session research or recovery from an unsupported method. | Read method, library, availability, cost/capabilities, CI/training support, profile, pagination, and parameter docs at full detail. |
| `forecast_models_cleanup` | S | Operator-approved model-store maintenance outside trading hours. | Read `dry_run`, matched model IDs, age/method filters, deleted/skipped counts, and errors. Keep `dry_run=true` unless deletion was explicitly requested. |
| `forecast_models_delete` | S | An operator explicitly names a stored model to delete. | Read requested model ID and deletion result. Never delete a model as an automatic response to a poor trade. |
| `forecast_models_list` | R | Finding reusable trained models and their metadata. | Read model IDs, methods, data scope, age, expiration/TTL, and availability. Confirm symbol/timeframe/target compatibility. |
| `forecast_optimize_hints` | R-heavy | Offline search across methods/timeframes/features with a fixed research budget. | Read fitness definition, evaluated candidates, top settings, seed, time limit, failures, and search space. Do not run in the live loop or treat the winner as out-of-sample evidence. |
| `forecast_task_cancel` | S | Cancelling one known background task. | Read task ID, prior/current status, cancellation acceptance, and errors. Verify with task status. |
| `forecast_task_cancel_all` | S | Operator-approved bulk task cancellation. | Read filters, `dry_run`, matched IDs, cancelled/skipped counts, and errors. Preview first. |
| `forecast_task_list` | R | Inspecting active and recent forecast jobs. | Read task IDs, state, method, scope, progress, timestamps, pagination, and failures. |
| `forecast_task_status` | R | Polling one task without blocking. | Read state, progress, result/model ID, timestamps, cancellation, and error details. Terminal failure is not a forecast. |
| `forecast_task_wait` | R-blocking | A completed training result is required and bounded waiting is acceptable. | Read task state, timeout status, result/model ID, progress, and errors. A timeout leaves the task unresolved. |
| `forecast_train` | S | Explicitly training a reusable model outside the hot loop. | Read accepted task ID, method, data scope, horizon, and initial state. Follow with status/wait and model listing. |
| `forecast_tune_genetic` | R-heavy | Offline parameter search with a prespecified metric and search space. | Read seed, metric/mode, generations, candidate scores, failures, and best params. Require untouched walk-forward validation afterward. |
| `forecast_tune_optuna` | R-heavy/G | Offline Optuna tuning when the dependency is available. | Read study/trials, metric/mode, best params/value, failures, timeout, and dependency status. Multiple trials increase selection bias. |
| `forecast_volatility_estimate` | R-heavy | Volatility-scaled stops, targets, holding horizon, or size need a forward estimate. | Read per-bar and horizon sigma, price/return units, method, proxy, annualization, source window, and warnings. Volatility estimates range, not direction. |
| `strategy_backtest` | R-heavy | Testing one built-in SMA-cross, EMA-cross, or RSI-reversion proxy with costs. | Read summary, `num_trades`, sample status, net/gross return, metrics, last signal, slippage, parameters, and full-detail trades. It validates only the exact built-in rule, not a richer live profile. |

### Options

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `options_barrier_price` | R/G | Locally valuing a specified barrier option under QuantLib assumptions. | Read NPV, inputs, barrier/option type, calendar, maturity basis, engine/model, and errors. It is theoretical valuation, not a live chain quote or spot direction signal. |
| `options_chain` | R/G | Equity/index option positioning or implied volatility can alter event risk. | Read contracts, expiration, call/put, strike, bid/ask, volume, open interest, IV, provider, and freshness. Reject crossed/stale/illiquid rows. |
| `options_expirations` | R/G | Discovering provider-supported expirations before chain queries. | Read expiration list, provider, symbol mapping, and freshness. Use one returned date exactly. |
| `options_heston_calibrate` | R-heavy/G | Research requires a stochastic-volatility fit and the chain is liquid enough. | Read fitted parameters, objective/error, contracts used/rejected, valuation date, and warnings. A poor calibration is unusable. |
| `options_provider_status` | R/G | Before any live option-chain request. | Read configured/effective provider, readiness, authentication, fallback, and diagnostics. Do not call chain/calibration when readiness fails. |

### Patterns and Regimes

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `patterns_detect` | R-heavy | A recent pattern near independently mapped structure can refine timing or invalidation. | Read mode/engine, highlights or detections, completion, timestamp, bias, strength, calibration, warnings, and `is_signal`/usage. Prefer completed robust patterns; treat all as secondary evidence. |
| `regime_detect` | R-heavy | Choosing between trend, range, and cooldown logic or detecting a change point. | Read method, current label/state, confidence/probability, durations/segments, change points, smoothing, sample, and warnings. Method labels are not interchangeable; confirm with raw structure. |

### Levels and Temporal Analysis

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `pivot_compute_points` | R | Session pivot context is relevant. | Read method, source completed bar, PP and support/resistance ladder, timeframe, and timestamp. Pivots are formula levels, not demonstrated reaction zones. |
| `support_resistance_levels` | R | Defining location, invalidation, targets, and breakout boundaries. | Read reference price, support/resistance zone bounds, touches, reaction statistics, score, distance, source window, volume weighting, and overlap warnings. Use zone envelopes, not only center values. |
| `temporal_analyze` | R-heavy | Session, hour, weekday, or month affects liquidity or expected behavior. | Read bucket counts, average return, win rate, volatility, range, volume, timezone/session mapping, and minimum samples. Require adequate observations and separate confirmation. |

### Reports

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `report_generate` | R-heavy | A human-readable orientation or handoff is needed outside the hot loop. | Read included/failed sections, generated content, template, timeframe/horizon, warnings, and data timestamps. A report aggregates underlying tools and is not independent confluence. |

### Symbols and Discovery

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `symbols_describe` | R | Session boot, before first risk, and after broker rejection or contract change. | Read exact broker symbol, category/group, digits, point, tick size/value, contract size, volume min/max/step, stop/freeze levels, filling/order/trade modes, currencies, and selection/tradability. |
| `symbols_list` | R | Resolving broker symbol names, groups, categories, or currency filters. | Read rows/groups, exact names, descriptions, visibility, pagination, and filters. Do not invent suffixes. |
| `symbols_top_markets` | R-heavy | Fast watchlist orientation by spread, volume, or price change. | Read rank field/order, rows, universe, timeframe, category/group, and count. It is a candidate list, not a signal. |

### Trading and Account Risk

| Tool | Mode | Call when | Parse and use |
|---|---:|---|---|
| `trade_account_info` | R | Boot and every risk-changing cycle. | At full detail read equity, balance, floating PnL, margin/free/level, leverage, live/demo state, terminal state, `execution_ready`, strict readiness, and hard/soft blockers. Hard blockers prohibit new risk. |
| `trade_close` | L | Cancelling one pending ticket, partially reducing, or closing a position. | Preview first. Read resolved target/kind, volume, operation, and would-send state. Live results require ticket/state verification; bulk close requires `close_all=true` and live confirmation. |
| `trade_get_open` | R | Boot, every cycle, and after any broker action. | Read position items, ticket, symbol, side/type, magic, volume, entry/current price, SL/TP, PnL, times, count, and filters. Missing SL means undefined risk, not zero risk. |
| `trade_get_pending` | R | Boot, every cycle, and after any broker action. | Read order items, ticket, symbol, type/side, magic, volume, trigger price, SL/TP, expiration, times, and count. Include all-fill risk in budgets. |
| `trade_history` | R | Reconciling fills, disappearance, rejection, partial close, or order lifecycle. | Deals expose deal/order/position tickets, fill time, side, volume, price, PnL, fee/swap/comment fields. Orders expose placed/done lifecycle. Use pagination and an explicit time window. |
| `trade_journal_analyze` | R | Daily loss calculation and sufficiently sampled performance review. | Read summary counts, net PnL, win rate, average win/loss, profit factor, sample quality/warning, and symbol/side breakdowns by detail. Small samples cannot support Kelly inputs. |
| `trade_modify` | L | Changing protection on one open position or changing price/protection/expiration on one pending order. | Preview first. Read resolved operation/ticket, applied values, validation, would-send state, retcode/comment, and no-change handling. Never widen currency risk without offsetting volume reduction. |
| `trade_place` | L | A fully validated and sized market or pending entry, including a supervisor hedge. | Preview first. Read validation, normalized order type, volume, price, protection, guardrails, actionability, and broker retcode/order/deal IDs. Live success is provisional until state verification. |
| `trade_risk_analyze` | R | Every proposed entry, add, hedge, or risk-widening redesign and periodic portfolio review. | Read `portfolio_risk`, total risk percent/currency, overall status, missing stops/calculation failures, `position_sizing.status`, suggested volume, compliance, trade evaluation, and `risk_alert`. Incomplete or unlimited risk blocks additions. |
| `trade_session_context` | R | Fast boot/cycle state for one symbol. | Read state, account summary, quote, open/pending sections, trade readiness, blockers, quote quality, tradability, partial failures, and other-position counts. It does not replace full symbol specs or detailed risk analysis. |
| `trade_stress_test` | R | Testing current positions under explicit deterministic shocks. | Read each shocked item, total PnL impact, equity before/after, impact percent, evaluated/unshocked counts, and scenario definition. It does not model gaps, spread expansion, or changing correlations. |
| `trade_var_cvar_calculate` | R-heavy | Material, correlated, or hedged account exposure needs a tail-risk estimate. | Read VaR, CVaR, confidence, method, one-bar holding timeframe, observations, exposures, and warnings. It is distribution-dependent and not a maximum-loss guarantee. |

## Strategy Research Standard

Before enabling a profile for a symbol and broker day:

1. Run its specified `strategy_backtest` proxy with a nonzero slippage value
   that includes the current round-trip spread estimate.
2. Require at least 30 trades, positive net return, profit factor greater than
   1.10, and usable drawdown/risk metrics. Otherwise the profile remains in
   observe-only mode for that symbol.
3. Do not tune the proxy on the same window used for approval. A favorable
   proxy does not prove the richer live rule.
4. If a forecast affects the decision, select it using rolling-origin
   `forecast_backtest_run`; do not choose it because the current forecast agrees
   with the desired trade.
5. Revalidate after a material regime shift, cost change, contract change, or
   broker-day boundary.

The design is informed by empirical work on
[market intraday momentum](https://www.sciencedirect.com/science/article/pii/S0304405X18301351),
[short-term reversal as liquidity provision](https://academic.oup.com/rfs/article-pdf/25/7/2005/24431763/hhs066.pdf),
and [volatility-managed portfolios](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513).
Those findings do not establish universal profitability. Research on
[intraday transaction costs](https://www.sciencedirect.com/science/article/pii/S1544612316300587),
[data snooping](https://eprints.lse.ac.uk/119144/1/dp303.pdf), and
[backtest overfitting](https://escholarship.org/uc/item/4w1110bb) motivates the
cost and sample gates above. The
[SEC day-trading risk notice](https://www.sec.gov/about/reports-publications/investorpubsdaytipshtm)
is the baseline risk warning for leveraged short-horizon operation.

## Failure and Recovery

- A malformed read payload may be corrected and retried once. A malformed
  payload supplies no trading evidence.
- On a trade error or ambiguous result, do not resend immediately. Refresh open
  positions, pending orders, account readiness, symbol constraints, and quote;
  then use history if the broker outcome is still unclear.
- Retry a broker action at most once, with the same idempotency key, and only
  after identifying a correctable cause. Do not retry into a widening spread,
  stale quote, event blackout, hard blocker, or invalid geometry.
- When risk cannot be quantified, treat it as unlimited. Protect or reduce it
  before considering new exposure.
- If the registry, output schema, or provider response contradicts this guide,
  follow the runtime contract conservatively and log the discrepancy for human
  review.
