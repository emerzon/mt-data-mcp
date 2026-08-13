# Web UI tool coverage inventory

**Audience:** Contributor

Every backend MCP/bootstrap tool classified for the SPA. User tour:
[WEBUI.md](WEBUI.md).

| Tool | Category | Surface | Frontend path | Confirm | Notes |
|---|---|---|---|---|---|
| `causal_discover_signals` | research | generic_runner | tools-runner/generic | no |  |
| `cointegration_test` | research | generic_runner | tools-runner/generic | no |  |
| `confluence_levels` | research | generic_runner | tools-runner/generic | no |  |
| `correlation_matrix` | research | generic_runner | tools-runner/generic | no |  |
| `cross_correlation` | research | generic_runner | tools-runner/generic | no |  |
| `data_fetch_candles` | data | dedicated_ui | chart-workspace/history | no |  |
| `data_fetch_ticks` | data | generic_runner | tools-runner/generic | no |  |
| `denoise_describe` | methods | dedicated_ui | chart-workspace/denoise-modal | no |  |
| `denoise_list_methods` | methods | dedicated_ui | chart-workspace/denoise-modal | no |  |
| `finviz_calendar` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_crypto` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_description` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_earnings` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_filters_list` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_forex` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_fundamentals` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_futures` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_insider` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_insider_activity` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_market_news` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_news` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_peers` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_ratings` | market | generic_runner | tools-runner/generic | no |  |
| `finviz_screen` | market | generic_runner | tools-runner/generic | no |  |
| `forecast_backtest_run` | forecast | dedicated_ui | forecast-panel/backtest | no |  |
| `forecast_barrier_optimize` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_barrier_prob` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_conformal_intervals` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_generate` | forecast | dedicated_ui | forecast-panel/price | no |  |
| `forecast_list_library_models` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_list_methods` | forecast | dedicated_ui | forecast-panel/methods | no |  |
| `forecast_models_cleanup` | forecast | generic_runner | tools-runner/generic | yes | mutation gate |
| `forecast_models_delete` | forecast | generic_runner | tools-runner/generic | yes | mutation gate |
| `forecast_models_list` | forecast | dedicated_ui | forecast-panel/models-browser | no |  |
| `forecast_optimize_hints` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_task_cancel` | forecast | generic_runner | tools-runner/generic | yes | mutation gate |
| `forecast_task_cancel_all` | forecast | generic_runner | tools-runner/generic | yes | mutation gate |
| `forecast_task_list` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_task_status` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_task_wait` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_train` | forecast | generic_runner | tools-runner/generic | no |  |
| `forecast_tune_genetic` | forecast | intentional_omit | Long-running optimization has no HTTP progress or cancellation contract. Run it through CLI or MCP instead. | no |  |
| `forecast_tune_optuna` | forecast | intentional_omit | Long-running optimization has no HTTP progress or cancellation contract. Run it through CLI or MCP instead. | no |  |
| `forecast_volatility_estimate` | forecast | dedicated_ui | forecast-panel/volatility | no |  |
| `indicators_describe` | methods | generic_runner | tools-runner/generic | no |  |
| `indicators_list` | methods | generic_runner | tools-runner/generic | no |  |
| `labels_triple_barrier` | research | generic_runner | tools-runner/generic | no |  |
| `market_depth_fetch` | market | generic_runner | tools-runner/generic | no | gated by `MTDATA_ENABLE_MARKET_DEPTH_FETCH`; disabled |
| `market_microstructure_analyze` | research | generic_runner | tools-runner/generic | no |  |
| `market_relative_strength` | research | generic_runner | tools-runner/generic | no |  |
| `market_scan` | market | generic_runner | tools-runner/generic | no |  |
| `market_snapshot` | research | generic_runner | tools-runner/generic | no |  |
| `market_status` | market | generic_runner | tools-runner/generic | no |  |
| `market_ticker` | market | dedicated_ui | chart-workspace/live-quotes | no |  |
| `news` | research | generic_runner | tools-runner/generic | no |  |
| `options_barrier_price` | options | generic_runner | tools-runner/generic | no |  |
| `options_chain` | options | generic_runner | tools-runner/generic | no |  |
| `options_expirations` | options | generic_runner | tools-runner/generic | no |  |
| `options_heston_calibrate` | options | generic_runner | tools-runner/generic | no |  |
| `options_provider_status` | options | generic_runner | tools-runner/generic | no |  |
| `outliers_detect` | research | generic_runner | tools-runner/generic | no |  |
| `patterns_detect` | pattern_regime | generic_runner | tools-runner/generic | no |  |
| `pivot_compute_points` | analysis | dedicated_ui | chart-workspace/pivot-overlay | no |  |
| `portfolio_risk_decompose` | research | generic_runner | tools-runner/generic | no |  |
| `regime_detect` | pattern_regime | generic_runner | tools-runner/generic | no |  |
| `report_generate` | report | generic_runner | tools-runner/generic | no |  |
| `seasonality_detect` | research | generic_runner | tools-runner/generic | no |  |
| `stationarity_test` | research | generic_runner | tools-runner/generic | no |  |
| `strategy_backtest` | forecast | generic_runner | tools-runner/generic | no |  |
| `strategy_validate` | forecast | generic_runner | tools-runner/generic | no |  |
| `support_resistance_levels` | analysis | dedicated_ui | chart-workspace/sr-overlay | no |  |
| `symbols_describe` | symbols | generic_runner | tools-runner/generic | no |  |
| `symbols_list` | symbols | generic_runner | tools-runner/generic | no |  |
| `symbols_top_markets` | symbols | generic_runner | tools-runner/generic | no |  |
| `temporal_analyze` | analysis | generic_runner | tools-runner/generic | no |  |
| `tools_list` | research | dedicated_ui | tools-runner/catalog | no |  |
| `trade_account_info` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_close` | trading | generic_runner | tools-runner/generic | yes | mutation gate |
| `trade_execution_quality` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_get_open` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_get_pending` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_history` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_journal_analyze` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_modify` | trading | generic_runner | tools-runner/generic | yes | mutation gate |
| `trade_place` | trading | generic_runner | tools-runner/generic | yes | mutation gate |
| `trade_risk_analyze` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_session_context` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_stress_test` | trading | generic_runner | tools-runner/generic | no |  |
| `trade_var_cvar_calculate` | trading | generic_runner | tools-runner/generic | no |  |
| `volatility_term_structure` | research | generic_runner | tools-runner/generic | no |  |
| `volume_profile_levels` | research | generic_runner | tools-runner/generic | no |  |
| `wait_event` | data | generic_runner | tools-runner/generic | no |  |

## Surface meanings

- **dedicated_ui** — primary path is a specialized chart/research control; also runnable via Tools runner.
- **generic_runner** — discoverable and invocable from the SPA Tools runner (schema-driven form).
- **intentional_omit** — visible with a rationale but not invocable from the SPA. Long-running tuning remains available through CLI/MCP; mutations use confirm gates instead.

**Total tools in inventory:** 92 (includes env-gated market_depth_fetch even when disabled).

Source of truth for runtime catalog: `GET /api/v1/tools` (bootstraps MCP tools).
Classification helpers: `mtdata.core.web_api_tools`.
