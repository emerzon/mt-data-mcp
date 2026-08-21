"""Forecast use-case orchestration package."""

from __future__ import annotations

import logging

from mtdata.forecast.use_cases.backtest import (
    _BACKTEST_METRICS_REASON_NOTES,
    _compact_backtest_result,
    run_forecast_backtest,
    run_strategy_backtest,
)
from mtdata.forecast.use_cases.barriers import (
    run_forecast_barrier_optimize,
    run_forecast_barrier_prob,
)
from mtdata.forecast.use_cases.compact import (
    _FORECAST_DIRECTION_MIN_THRESHOLD_PCT,
    _annotate_barrier_prob_context,
    _apply_barrier_prob_detail,
    _apply_forecast_generate_detail,
    _forecast_anchor_freshness,
    _forecast_generate_volatility_rows,
    _normalize_forecast_time_fields,
    _round_barrier_prob_payload,
    _symbol_price_currency,
)
from mtdata.forecast.use_cases.generate import (
    _DEFAULT_VOLATILITY_PROXY,
    _MIN_CONFORMAL_CALIBRATION_POINTS,
    _PRETRAINED_FORECAST_METHODS,
    _VOLATILITY_PROXY_METHODS,
    _apply_conformal_intervals_detail,
    _resolve_stored_model_execution_alias,
    run_forecast_conformal_intervals,
    run_forecast_generate,
    run_forecast_volatility_estimate,
)
from mtdata.forecast.use_cases.sktime_index import (
    _SKTIME_INDEX_SCHEMA_VERSION,
    _discover_sktime_forecasters,
    _load_sktime_forecaster_index,
    _normalize_forecaster_name,
    _registered_sktime_forecasters,
    _resolve_sktime_forecaster,
    _sktime_forecaster_index_path,
    _store_sktime_forecaster_index,
)
from mtdata.forecast.use_cases.tune import (
    _TUNING_METRICS,
    run_forecast_optimize_hints,
    run_forecast_tune_genetic,
    run_forecast_tune_optuna,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_forecast_backtest",
    "run_forecast_barrier_optimize",
    "run_forecast_barrier_prob",
    "run_forecast_conformal_intervals",
    "run_forecast_generate",
    "run_forecast_optimize_hints",
    "run_forecast_tune_genetic",
    "run_forecast_tune_optuna",
    "run_forecast_volatility_estimate",
    "run_strategy_backtest",
    "_annotate_barrier_prob_context",
    "_apply_barrier_prob_detail",
    "_apply_conformal_intervals_detail",
    "_apply_forecast_generate_detail",
    "_BACKTEST_METRICS_REASON_NOTES",
    "_compact_backtest_result",
    "_DEFAULT_VOLATILITY_PROXY",
    "_discover_sktime_forecasters",
    "_FORECAST_DIRECTION_MIN_THRESHOLD_PCT",
    "_forecast_anchor_freshness",
    "_forecast_generate_volatility_rows",
    "_load_sktime_forecaster_index",
    "_MIN_CONFORMAL_CALIBRATION_POINTS",
    "_normalize_forecast_time_fields",
    "_normalize_forecaster_name",
    "_PRETRAINED_FORECAST_METHODS",
    "_registered_sktime_forecasters",
    "_resolve_sktime_forecaster",
    "_resolve_stored_model_execution_alias",
    "_round_barrier_prob_payload",
    "_SKTIME_INDEX_SCHEMA_VERSION",
    "_sktime_forecaster_index_path",
    "_store_sktime_forecaster_index",
    "_symbol_price_currency",
    "_TUNING_METRICS",
    "_VOLATILITY_PROXY_METHODS",
]
