from __future__ import annotations

import difflib
import importlib
import inspect
import json
import logging
import math
import os
import pkgutil
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.error_envelope import build_error_payload
from ..core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from ..core.output_contract import attach_collection_contract
from ..shared.validators import unknown_mapping_keys_error
from ..utils.coercion import coerce_finite_float as _finite_float
from ..utils.coercion import is_explicit_false as _is_explicit_false
from ..utils.coercion import round_finite
from ..utils.freshness import format_age_seconds as _format_age_seconds
from ..utils.freshness import format_freshness_label
from .backtest import execute_forecast_backtest as _forecast_backtest_impl
from .barriers_shared import (
    BARRIER_EDGE_DEFINITION,
    barrier_method_error,
    normalize_barrier_method,
)
from .capabilities import resolve_capability_request
from .exceptions import ForecastError, raise_if_error_result
from .forecast import execute_forecast as _forecast_impl
from .forecast_methods import (
    get_forecast_method_names,
    get_forecast_methods_snapshot,
    get_method_param_names,
)
from .forecast_registry import ForecastRegistry
from .forecast_validation import format_invalid_method_error
from .requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
    StrategyBacktestRequest,
)
from .tuning_contract import (
    ANNUALIZED_TUNING_METRICS,
    MIN_ANNUALIZED_TUNING_TRADES,
    TUNING_METRIC_DIRECTIONS,
    resolve_tuning_mode,
)

logger = logging.getLogger(__name__)

_BACKTEST_METRICS_REASON_NOTES = {
    "no_non_flat_trades": (
        "No active long/short trades; win_rate and drawdown need at least one trade."
    ),
}
_TUNING_METRICS = frozenset(TUNING_METRIC_DIRECTIONS)
_VOLATILITY_PROXY_METHODS = {"arima", "sarima", "ets", "theta"}
_PRETRAINED_FORECAST_METHODS = ("chronos2", "chronos_bolt", "timesfm")
_DEFAULT_VOLATILITY_PROXY = "squared_return"
_FORECAST_DIRECTION_MIN_THRESHOLD_PCT = 0.05
_SKTIME_INDEX_SCHEMA_VERSION = 1


def _format_forecast_time_utc(value: Any) -> Any:
    if value in (None, ""):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except Exception:
            return value
    text = str(value).strip()
    if not text:
        return value
    if "T" not in text and " " not in text:
        return value
    parse_text = text.replace("Z", "+00:00")
    if "T" not in parse_text and " " in parse_text:
        parse_text = parse_text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(parse_text)
    except Exception:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    parsed = parsed.replace(microsecond=0)
    if parsed.second == 0:
        return parsed.strftime("%Y-%m-%dT%H:%MZ")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_forecast_time_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize every serialized forecast datetime to one UTC representation."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            normalized = {key: normalize(item) for key, item in value.items()}
            if "timezone" in normalized:
                normalized["timezone"] = "UTC"
            return normalized
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalize(item) for item in value)
        if isinstance(value, str):
            return _format_forecast_time_utc(value)
        return value

    out = normalize(payload)
    if any(key in out for key in ("last_observation_time", "forecast_time")):
        out["timezone"] = "UTC"
    return out


def _normalize_trader_detail(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized in {"summary"}:
        return "compact"
    if normalized == "full":
        return "full"
    if normalized == "standard":
        return "standard"
    return "compact"


def _requested_detail_label(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized in {"compact", "standard", "summary", "full"}:
        return normalized
    return str(default)


def _symbol_price_currency(symbol: Any) -> Optional[str]:
    from ..utils.mt5 import symbol_price_currency_for

    return symbol_price_currency_for(symbol)


def _annotate_price_currency(payload: Dict[str, Any], symbol: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error") or payload.get("price_currency"):
        return payload
    currency = _symbol_price_currency(symbol)
    if not currency:
        return payload
    out = dict(payload)
    out["price_currency"] = currency
    return out


def _forecast_interval_summary(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lower_key = next(
        (
            key
            for key in ("lower_price", "lower_return", "lower")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if lower_key is None:
        return None
    upper_key = lower_key.replace("lower", "upper", 1)
    lower_vals = payload.get(lower_key)
    upper_vals = payload.get(upper_key)
    if not isinstance(lower_vals, list) or not isinstance(upper_vals, list) or not lower_vals or not upper_vals:
        return None
    try:
        widths = [
            float(upper) - float(lower)
            for lower, upper in zip(lower_vals, upper_vals, strict=False)
        ]
        if not widths:
            return None
        widths_sorted = sorted(widths)
        return {
            "first_low": float(lower_vals[0]),
            "first_high": float(upper_vals[0]),
            "last_low": float(lower_vals[-1]),
            "last_high": float(upper_vals[-1]),
            "median_width": float(widths_sorted[len(widths_sorted) // 2]),
        }
    except Exception:
        return None


def _forecast_compact_ci(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ci_status = str(payload.get("ci_status") or "").strip().lower()
    if ci_status == "not_requested":
        return {
            "status": "not_requested",
            "mode": "point_only",
            "reason": "ci_alpha was not requested; direction is based on the point estimate only.",
            "recommended_tool": "forecast_conformal_intervals",
        }
    if ci_status == "unavailable":
        out: Dict[str, Any] = {
            "status": "unavailable",
            "mode": "point_only",
            "recommended_tool": "forecast_conformal_intervals",
        }
        if payload.get("ci_alpha") is not None:
            out["requested_alpha"] = payload.get("ci_alpha")
        return out

    lower_key = next(
        (
            key
            for key in ("lower_price", "lower_return", "lower")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if lower_key is None:
        if ci_status:
            return {"status": ci_status}
        return None

    upper_key = lower_key.replace("lower", "upper", 1)
    lower_vals = payload.get(lower_key)
    upper_vals = payload.get(upper_key)
    if not isinstance(lower_vals, list) or not isinstance(upper_vals, list):
        return None

    forecast_key = (
        "forecast_price"
        if lower_key.endswith("_price")
        else "forecast_return"
        if lower_key.endswith("_return")
        else "forecast"
    )
    forecasts = payload.get(forecast_key)
    times = payload.get("forecast_time")
    bar_states = payload.get("forecast_bar_states")
    count = min(len(lower_vals), len(upper_vals))
    if isinstance(forecasts, list):
        count = min(count, len(forecasts))
    intervals: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {}
        if isinstance(times, list) and idx < len(times):
            row["time"] = times[idx]
        if isinstance(bar_states, list) and idx < len(bar_states):
            row["bar_state"] = bar_states[idx]
        if isinstance(forecasts, list):
            row["forecast"] = forecasts[idx]
        row["low"] = lower_vals[idx]
        row["high"] = upper_vals[idx]
        intervals.append(row)

    out = {"status": ci_status or "available", "mode": "interval"}
    if payload.get("ci_alpha") is not None:
        out["alpha"] = payload.get("ci_alpha")
    if intervals:
        out["intervals"] = intervals
    summary = _forecast_interval_summary(payload)
    if summary:
        out["summary"] = summary
    return out


def _forecast_price_digits(payload: Dict[str, Any]) -> Optional[int]:
    for key in ("digits", "price_precision"):
        value = payload.get(key)
        try:
            digits = int(value)
        except Exception:
            continue
        return max(0, digits)
    return None


def _round_forecast_number(value: Any, *, digits: int) -> Any:
    rounded = round_finite(value, digits, on_invalid="passthrough")
    return float(rounded) if isinstance(rounded, (int, float)) and not isinstance(rounded, bool) else rounded


def _round_forecast_list(values: Any, *, digits: int) -> Any:
    if not isinstance(values, list):
        return values
    return [_round_forecast_number(value, digits=digits) for value in values]


def _round_forecast_generate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    digits = _forecast_price_digits(payload)
    if digits is None:
        return payload
    out = dict(payload)
    for key in (
        "forecast_price",
        "lower_price",
        "upper_price",
        "lower",
        "upper",
    ):
        if key in out:
            out[key] = _round_forecast_list(out.get(key), digits=digits)
    for key in ("last_price", "last_price_close"):
        if key in out:
            out[key] = _round_forecast_number(out.get(key), digits=digits)
    return out


def _round_forecast_volatility_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    digits_by_key = {
        "volatility_per_bar": 6,
        "volatility_annualized": 6,
        "volatility_horizon": 6,
        "volatility_horizon_annualized": 6,
        "volatility_per_bar_pct": 4,
        "volatility_annualized_pct": 4,
        "volatility_horizon_pct": 4,
        "volatility_horizon_annualized_pct": 4,
    }
    for key, digits in digits_by_key.items():
        if key in out:
            out[key] = _round_forecast_number(out.get(key), digits=digits)
    return out


def _round_barrier_value(value: Any, *, digits: int) -> Any:
    numeric = _finite_float(value)
    if numeric is None:
        return value
    precision = max(0, int(digits))
    return float(f"{numeric:.{precision}f}")


def _round_barrier_ci(value: Any, *, digits: int) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: _round_barrier_value(item, digits=digits)
        for key, item in value.items()
    }


def _round_barrier_prob_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    price_digits = _forecast_price_digits(payload) or 8
    out = dict(payload)
    for key in ("last_price", "last_price_close", "reference_price", "tp_price", "sl_price", "barrier"):
        if key in out:
            out[key] = _round_barrier_value(out.get(key), digits=price_digits)
    for key in (
        "prob_hit",
        "prob_tp_first",
        "prob_sl_first",
        "prob_tp_strict_first",
        "prob_sl_strict_first",
        "prob_same_bar",
        "prob_no_hit",
        "prob_resolve",
        "prob_unresolved",
        "probability_edge",
        "prob_tp_first_se",
        "prob_sl_first_se",
        "prob_same_bar_se",
        "prob_no_hit_se",
    ):
        if key in out:
            out[key] = _round_barrier_value(out.get(key), digits=6)
    for key in (
        "prob_tp_first_ci95",
        "prob_sl_first_ci95",
        "prob_same_bar_ci95",
        "prob_no_hit_ci95",
    ):
        if key in out:
            out[key] = _round_barrier_ci(out.get(key), digits=6)
    return out


_BARRIER_OPTIMIZE_PRICE_KEYS = {
    "last_price",
    "last_price_close",
    "reference_price",
    "tp_price",
    "sl_price",
    "barrier",
    "entry_price",
}
_BARRIER_OPTIMIZE_METRIC_DIGITS = {
    "tp": 6,
    "sl": 6,
    "rr": 4,
    "prob_win": 6,
    "prob_loss": 6,
    "prob_tp_first": 6,
    "prob_sl_first": 6,
    "prob_no_hit": 6,
    "prob_same_bar": 6,
    "prob_tp_strict_first": 6,
    "prob_sl_strict_first": 6,
    "prob_unresolved": 6,
    "prob_resolve": 6,
    "ev": 6,
    "ev_gross": 6,
    "ev_net": 6,
    "ev_unresolved": 6,
    "ev_cond": 6,
    "edge": 6,
    "edge_vs_breakeven": 6,
    "breakeven_win_rate": 6,
    "profit_factor": 6,
    "kelly": 6,
    "kelly_cond": 6,
    "ev_per_bar": 6,
    "utility": 6,
}


def _round_barrier_optimize_value(value: Any, *, key: str, price_digits: int) -> Any:
    if key in _BARRIER_OPTIMIZE_PRICE_KEYS:
        return _round_barrier_value(value, digits=price_digits)
    digits = _BARRIER_OPTIMIZE_METRIC_DIGITS.get(key)
    if digits is not None:
        return _round_barrier_value(value, digits=digits)
    return value


def _round_barrier_optimize_payload_value(value: Any, *, key: str, price_digits: int) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _round_barrier_optimize_payload_value(
                item_value,
                key=str(item_key),
                price_digits=price_digits,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _round_barrier_optimize_payload_value(item, key=key, price_digits=price_digits)
            for item in value
        ]
    return _round_barrier_optimize_value(value, key=key, price_digits=price_digits)


def _round_barrier_optimize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    price_digits = _forecast_price_digits(payload) or 6
    return {
        key: _round_barrier_optimize_payload_value(
            value,
            key=str(key),
            price_digits=price_digits,
        )
        for key, value in payload.items()
    }


def _with_reference_price_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    reference_price = out.get("reference_price", out.get("last_price"))
    if reference_price not in (None, "", [], {}):
        out.setdefault("reference_price", reference_price)
    reference_source = out.get("reference_price_source", out.get("last_price_source"))
    if reference_source not in (None, "", [], {}):
        out.setdefault("reference_price_source", reference_source)
    return out


_BARRIER_OPTIMIZE_COMPACT_OMIT_KEYS = frozenset(
    {
        "actionability",
        "actionability_flags",
        "actionability_reason",
        "concise",
        "no_action",
        "no_action_reason",
        "no_candidates",
        "output_mode",
        "trade_gate_passed",
        "tradable",
        "viable",
        "viable_only",
        "warning",
    }
)


def _compact_barrier_optimize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        key: value
        for key, value in payload.items()
        if key not in _BARRIER_OPTIMIZE_COMPACT_OMIT_KEYS
    }
    reason = (
        payload.get("status_reason")
        or payload.get("actionability_reason")
        or payload.get("no_action_reason")
        or payload.get("warning")
    )
    if reason not in (None, "", [], {}):
        out["status_reason"] = reason
    trade_gate = payload.get("trade_gate_passed", payload.get("tradable"))
    if trade_gate not in (None, "", [], {}):
        out["tradable"] = bool(trade_gate)
    return out


def _gate_barrier_optimize_live_readiness(payload: Dict[str, Any]) -> None:
    """Require both live inputs and a viable optimizer result for live readiness."""
    if "usable_for_live_trading" not in payload:
        return
    has_best = isinstance(payload.get("best"), dict)
    mathematically_viable = bool(
        has_best
        and payload.get(
            "mathematically_viable",
            payload.get("viable"),
        )
        is True
    )
    viable_result = bool(payload.get("tradable") is True and mathematically_viable)
    payload["usable_for_live_trading"] = bool(
        payload.get("usable_for_live_trading") is True and viable_result
    )
    payload["usable_for_live_trading_basis"] = (
        "model_viability_and_reference_quote"
    )
    if viable_result:
        return
    blockers = list(payload.get("execution_blockers") or [])
    blocker = (
        "risk_actionability_gate_failed"
        if mathematically_viable
        else "optimizer_non_viable"
    )
    if blocker not in blockers:
        blockers.append(blocker)
    if mathematically_viable:
        for flag in payload.get("actionability_flags") or []:
            normalized = str(flag).strip()
            if normalized and normalized not in blockers:
                blockers.append(normalized)
    payload["execution_blockers"] = blockers


def _forecast_vs_last_price(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last_price = _finite_float(payload.get("last_price"))
    prices = payload.get("forecast_price")
    if last_price is None or not isinstance(prices, list) or not prices:
        return None
    first_forecast = _finite_float(prices[0])
    horizon_forecast = _finite_float(prices[-1])
    if first_forecast is None or horizon_forecast is None:
        return None
    first_delta = first_forecast - last_price
    horizon_delta = horizon_forecast - last_price
    digits = _forecast_price_digits(payload)
    delta_digits = digits if digits is not None else 6
    first_delta_pct = None
    horizon_delta_pct = None
    if last_price:
        first_delta_pct = first_delta / last_price * 100.0
        horizon_delta_pct = horizon_delta / last_price * 100.0
    threshold_pct = _finite_float(payload.get("direction_threshold_pct"))
    if threshold_pct is None or threshold_pct < _FORECAST_DIRECTION_MIN_THRESHOLD_PCT:
        threshold_pct = _FORECAST_DIRECTION_MIN_THRESHOLD_PCT
    if horizon_delta_pct is not None and abs(horizon_delta_pct) <= threshold_pct:
        direction = "neutral"
    elif horizon_delta > 0:
        direction = "bullish"
    elif horizon_delta < 0:
        direction = "bearish"
    else:
        direction = "neutral"
    out: Dict[str, Any] = {
        "direction": direction,
        "direction_basis": "horizon_end",
        "direction_threshold_pct": float(round(threshold_pct, 6)),
        "direction_threshold_basis": payload.get("direction_threshold_basis")
        or "minimum_effect_size_0.05_pct",
        "first_step_delta": float(round(first_delta, delta_digits)),
        "horizon_delta": float(round(horizon_delta, delta_digits)),
    }
    if first_delta_pct is not None and horizon_delta_pct is not None:
        out["first_step_delta_pct"] = float(round(first_delta_pct, 4))
        out["horizon_delta_pct"] = float(round(horizon_delta_pct, 4))
    return out


def _gate_forecast_direction(
    payload: Dict[str, Any],
    price_context: Dict[str, Any],
) -> None:
    direction = str(price_context.get("direction") or "").strip().lower()
    if direction not in {"bullish", "bearish"}:
        price_context["direction_status"] = "neutral"
        price_context["direction_actionable"] = False
        return

    interval_excludes_anchor = price_context.get(
        "direction_interval_excludes_last_price"
    )
    if interval_excludes_anchor is True:
        price_context["direction_status"] = "interval_confirmed"
        price_context["direction_actionable"] = True
        return

    price_context["point_estimate_direction"] = direction
    price_context.pop("direction", None)
    price_context["direction_status"] = "unconfirmed"
    price_context["direction_actionable"] = False
    interval_basis = str(
        price_context.get("direction_interval_basis") or ""
    ).strip()
    if interval_basis == "not_available":
        reason = "forecast_uncertainty_not_available"
    elif interval_basis == "not_comparable":
        reason = "interval_not_comparable_to_price_anchor"
    else:
        reason = "horizon_interval_contains_last_price"
    price_context.setdefault("direction_suppressed_reason", reason)
    payload["signal_status"] = "not_actionable"


def _annotate_forecast_direction_interval(
    payload: Dict[str, Any],
    price_context: Dict[str, Any],
) -> None:
    ci_status = str(payload.get("ci_status") or "").strip().lower()
    lower_prices = payload.get("lower_price")
    upper_prices = payload.get("upper_price")
    has_price_interval = (
        ci_status == "available"
        and isinstance(lower_prices, list)
        and bool(lower_prices)
        and isinstance(upper_prices, list)
        and bool(upper_prices)
    )
    if not has_price_interval:
        price_context["direction_interval_excludes_last_price"] = None
        price_context["direction_interval_basis"] = "not_available"
        price_context["direction_interpretation"] = (
            "interval_unavailable"
            if ci_status == "unavailable"
            else "point_estimate_only"
        )
        _gate_forecast_direction(payload, price_context)
        return

    last_price = _finite_float(payload.get("last_price"))
    horizon_low = _finite_float(lower_prices[-1])
    horizon_high = _finite_float(upper_prices[-1])
    if last_price is None or horizon_low is None or horizon_high is None:
        price_context["direction_interval_excludes_last_price"] = None
        price_context["direction_interval_basis"] = "not_comparable"
        price_context["direction_interpretation"] = (
            "interval_not_comparable_to_price_anchor"
        )
        _gate_forecast_direction(payload, price_context)
        return

    direction = str(price_context.get("direction") or "").strip().lower()
    excludes_last_price = (
        horizon_low > last_price
        if direction == "bullish"
        else horizon_high < last_price
        if direction == "bearish"
        else False
    )
    price_context["direction_interval_excludes_last_price"] = excludes_last_price
    price_context["direction_interval_basis"] = (
        "horizon_interval_vs_last_price"
    )
    price_context["direction_interpretation"] = (
        "interval_excludes_last_price"
        if excludes_last_price
        else "interval_contains_last_price_or_direction_is_neutral"
    )
    _gate_forecast_direction(payload, price_context)


def _forecast_path_flatness(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prices = payload.get("forecast_price")
    if not isinstance(prices, list) or len(prices) < 2:
        return None
    finite_prices = [_finite_float(value) for value in prices]
    if any(value is None for value in finite_prices):
        return None
    price_values = [float(value) for value in finite_prices if value is not None]
    path_range = max(price_values) - min(price_values)
    digits = _forecast_price_digits(payload)
    threshold = 0.0 if digits is None else 10.0 ** (-max(0, digits))
    tolerance = max(threshold * 1e-9, 1e-12)
    if path_range > threshold + tolerance:
        return None
    range_digits = digits if digits is not None else 6
    return {
        "path_flat": True,
        "path_range": float(round(path_range, range_digits)),
    }


def _forecast_point_mode(payload: Dict[str, Any]) -> Optional[str]:
    return "flat_model_path" if _forecast_path_flatness(payload) else None


_FORECAST_FLAT_PATH_WARNING = (
    "Forecast path is near-flat at displayed price precision; compare "
    "another method or run forecast_conformal_intervals."
)


def _append_forecast_warning(payload: Dict[str, Any], warning: str) -> None:
    warnings_out = payload.get("warnings")
    if not isinstance(warnings_out, list):
        warnings_out = []
    if warning not in warnings_out:
        warnings_out.append(warning)
    payload["warnings"] = warnings_out


def _annotate_forecast_generate_quality(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    ci_status = str(out.get("ci_status") or "").strip().lower()
    if not ci_status:
        out["ci_status"] = "not_requested"
        out.setdefault(
            "uncertainty",
            {
                "status": "not_requested",
                "mode": "point_only",
                "reason": "ci_alpha was not requested; direction is based on the point estimate only.",
                "recommended_tool": "forecast_conformal_intervals",
            },
        )
    if str(out.get("ci_status") or "").strip().lower() in {
        "not_requested",
        "unavailable",
    }:
        out.setdefault("signal_status", "not_actionable")
    path_flatness = _forecast_path_flatness(out)
    price_context = _forecast_vs_last_price(out)
    if price_context:
        if path_flatness:
            price_context["direction"] = "neutral"
            price_context["direction_basis"] = "flat_path"
            price_context["direction_suppressed_reason"] = "flat_path"
        _annotate_forecast_direction_interval(out, price_context)
        out["forecast_vs_last_price"] = price_context
        out.pop("direction_threshold_pct", None)
        out.pop("direction_threshold_basis", None)
        units = dict(out.get("units") or {})
        units.setdefault(
            "forecast_vs_last_price.*_delta_pct",
            "percent (1.0 = 1%)",
        )
        units.setdefault(
            "forecast_vs_last_price.direction_threshold_pct",
            "percent (1.0 = 1%)",
        )
        out["units"] = units
    if path_flatness:
        out.update(path_flatness)
        out.setdefault("point_forecast_mode", "flat_model_path")
        out["forecast_status"] = "non_informative"
        out["signal_status"] = "not_actionable"
        _append_forecast_warning(out, _FORECAST_FLAT_PATH_WARNING)
    out.setdefault("forecast_reliability_basis", "history_sample_size")
    trust_blockers: List[str] = []
    if str(out.get("forecast_reliability") or "").strip().lower() == "low":
        trust_blockers.append("insufficient_history_sample")
    if out.get("history_policy_ok") is False:
        trust_blockers.append("history_freshness_policy_not_met")
    ci_status = str(out.get("ci_status") or "").strip().lower()
    if ci_status == "unavailable":
        trust_blockers.append("forecast_uncertainty_not_available")
    if path_flatness:
        trust_blockers.append("non_informative_forecast_path")
    out["trust_level"] = (
        "low"
        if any(
            blocker in trust_blockers
            for blocker in (
                "insufficient_history_sample",
                "non_informative_forecast_path",
            )
        )
        else "degraded"
        if trust_blockers
        else "adequate"
    )
    out["trust_level_basis"] = [
        "history_sample_size",
        "history_freshness_policy",
        "forecast_uncertainty",
    ]
    if trust_blockers:
        out["trust_blockers"] = trust_blockers
    return out


def _attach_invalid_method_guidance(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    error = str(payload.get("error") or "").strip()
    if not error.lower().startswith("invalid method:"):
        return payload
    methods = get_forecast_methods_snapshot().get("methods", [])
    available = sorted(
        {
            str(row.get("method"))
            for row in methods
            if isinstance(row, dict)
            and row.get("method")
            and row.get("available") is not False
        }
    )
    out = dict(payload)
    display_limit = 20
    out["valid_values"] = {"method": available[:display_limit]}
    if len(available) > display_limit:
        out["valid_values_truncated"] = len(available) - display_limit
    out["related_tools"] = ["forecast_list_methods"]
    return out


def _forecast_anchor_freshness(payload: Dict[str, Any]) -> Optional[str]:
    policy_relaxed = payload.get("freshness_policy_relaxed") is not False
    label = format_freshness_label(
        data_stale=payload.get("last_price_stale"),
        market_status=payload.get("market_status") if policy_relaxed else None,
        market_status_reason=(
            payload.get("market_status_reason") if policy_relaxed else None
        ),
        age_seconds=payload.get("last_price_age_seconds"),
        age_text=payload.get("last_price_age"),
        item="anchor",
    )
    if not label:
        return None
    policy = _format_age_seconds(payload.get("stale_after_seconds"))
    if policy and label.startswith("stale"):
        return f"{label} (policy: {policy})"
    return label


def _forecast_generate_data_window(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last_observation = payload.get("last_observation_time")
    if last_observation in (None, "", [], {}):
        return None
    last_bar_complete = bool(payload.get("last_bar_complete", True))
    out: Dict[str, Any] = {
        "last_observation": last_observation,
        "last_bar_complete": last_bar_complete,
        "input_bar_policy": (
            "closed_bars_only"
            if last_bar_complete
            else "includes_forming_bar"
        ),
    }
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        for source_key, target_key in (
            ("history_start_time", "history_start"),
            ("history_end_time", "history_end"),
            ("history_bars_used", "history_bars_used"),
        ):
            value = diagnostics.get(source_key)
            if value not in (None, "", [], {}):
                out[target_key] = value
    for source_key, target_key in (
        ("forecast_start_time", "forecast_start"),
        ("forecast_start_gap_bars", "forecast_start_gap_bars"),
        ("forecast_time_semantics", "forecast_time_semantics"),
        ("forecast_value_semantics", "forecast_value_semantics"),
    ):
        value = payload.get(source_key)
        if value not in (None, "", [], {}):
            out[target_key] = value
    bar_states = payload.get("forecast_bar_states")
    if isinstance(bar_states, list) and bar_states:
        out["first_forecast_bar_state"] = bar_states[0]
        out["horizon_includes_forming_bar"] = "forming" in bar_states
    age_seconds = payload.get("last_price_age_seconds")
    if age_seconds not in (None, "", [], {}):
        out["last_observation_age_seconds"] = age_seconds
    age_metric = payload.get("freshness_age_metric")
    if age_metric not in (None, "", [], {}):
        out["last_observation_age_metric"] = age_metric
    stale = payload.get("last_price_stale")
    if isinstance(stale, bool):
        out["last_observation_stale"] = stale
    return out


def _forecast_generate_compact_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = payload.get("forecast_time")
    if not isinstance(times, list):
        return []

    forecast_values = None
    forecast_key = ""
    quantity = str(payload.get("quantity") or "").strip().lower()
    candidate_keys = (
        ("forecast_return", "forecast_price", "forecast")
        if quantity == "return"
        else ("forecast_price", "forecast_return", "forecast")
    )
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            forecast_values = value
            forecast_key = key
            break
    if not isinstance(forecast_values, list):
        return []

    lower_key = "lower_price" if isinstance(payload.get("lower_price"), list) else "lower_return"
    upper_key = "upper_price" if lower_key == "lower_price" else "upper_return"
    lower_values = payload.get(lower_key)
    upper_values = payload.get(upper_key)
    if not isinstance(lower_values, list) or not isinstance(upper_values, list):
        lower_values = payload.get("lower")
        upper_values = payload.get("upper")
    if quantity == "return" and forecast_key == "forecast_return":
        lower_field = "lower_return"
        upper_field = "upper_return"
    elif quantity == "price" and forecast_key == "forecast_price":
        lower_field = "lower_price"
        upper_field = "upper_price"
    else:
        lower_field = "lower"
        upper_field = "upper"
    market_status = payload.get("forecast_market_status")
    bar_states = payload.get("forecast_bar_states")

    count = min(len(times), len(forecast_values))
    price_values = payload.get("forecast_price")
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {"time": _format_forecast_time_utc(times[idx])}
        if isinstance(bar_states, list) and idx < len(bar_states):
            row["bar_state"] = bar_states[idx]
        if quantity == "return" and forecast_key == "forecast_return":
            row["return"] = forecast_values[idx]
            if isinstance(price_values, list) and idx < len(price_values):
                row["price"] = price_values[idx]
        else:
            row["value"] = forecast_values[idx]
        if isinstance(market_status, list) and idx < len(market_status):
            row["market_status"] = market_status[idx]
        if isinstance(lower_values, list) and isinstance(upper_values, list):
            if idx < len(lower_values) and idx < len(upper_values):
                row[lower_field] = lower_values[idx]
                row[upper_field] = upper_values[idx]
        rows.append(row)
    return rows


def _forecast_generate_volatility_rows(
    payload: Dict[str, Any],
    *,
    horizon: Any,
) -> List[Dict[str, Any]]:
    volatility = _finite_float(payload.get("volatility_per_bar"))
    volatility_pct = _finite_float(payload.get("volatility_per_bar_pct"))
    volatility_annualized = _finite_float(payload.get("volatility_annualized"))
    volatility_annualized_pct = _finite_float(payload.get("volatility_annualized_pct"))
    horizon_volatility = _finite_float(payload.get("volatility_horizon"))
    horizon_volatility_pct = _finite_float(payload.get("volatility_horizon_pct"))
    horizon_volatility_annualized = _finite_float(payload.get("volatility_horizon_annualized"))
    horizon_volatility_annualized_pct = _finite_float(payload.get("volatility_horizon_annualized_pct"))
    if all(
        value is None
        for value in (
            volatility,
            volatility_pct,
            volatility_annualized,
            volatility_annualized_pct,
            horizon_volatility,
            horizon_volatility_pct,
            horizon_volatility_annualized,
            horizon_volatility_annualized_pct,
        )
    ):
        return []
    try:
        count = max(1, int(horizon or payload.get("horizon") or 1))
    except Exception:
        count = 1
    times = payload.get("forecast_time")
    if not isinstance(times, list):
        times = payload.get("times") if isinstance(payload.get("times"), list) else []
    row: Dict[str, Any] = {"horizon_steps": count}
    if times:
        row["start_time"] = times[0]
        row["end_time"] = times[min(count - 1, len(times) - 1)]
    if volatility is not None:
        row["volatility_per_bar"] = float(round(volatility, 6))
    if volatility_pct is not None:
        row["volatility_per_bar_pct"] = float(round(volatility_pct, 4))
    if volatility_annualized is not None:
        row["volatility_annualized"] = float(round(volatility_annualized, 6))
    if volatility_annualized_pct is not None:
        row["volatility_annualized_pct"] = float(round(volatility_annualized_pct, 4))
    if horizon_volatility is not None:
        row["volatility_horizon"] = float(round(horizon_volatility, 6))
    if horizon_volatility_pct is not None:
        row["volatility_horizon_pct"] = float(round(horizon_volatility_pct, 4))
    if horizon_volatility_annualized is not None:
        row["volatility_horizon_annualized"] = float(round(horizon_volatility_annualized, 6))
    if horizon_volatility_annualized_pct is not None:
        row["volatility_horizon_annualized_pct"] = float(round(horizon_volatility_annualized_pct, 4))
    return [row]


_ANALOG_COMPACT_COMPONENT_KEYS = (
    "timeframe",
    "role",
    "status",
    "n_paths",
    "component_weight",
    "reason",
)
_ANALOG_COMPACT_METRIC_KEYS = (
    "n_paths",
    "effective_paths",
    "spread",
    "weighted",
)
_ANALOG_VERBOSE_METADATA_KEYS = frozenset(
    {
        "analogs",
        "component_status",
        "ensemble_metrics",
        "timeframe_diagnostics",
    }
)


def _compact_analog_metadata(metadata: Any) -> Dict[str, Any]:
    """Keep the decision-facing analog diagnostics without repeated detail blobs."""
    if not isinstance(metadata, dict):
        return {}

    compact: Dict[str, Any] = {}
    statuses = metadata.get("component_status")
    if isinstance(statuses, list):
        compact_statuses: List[Dict[str, Any]] = []
        for status in statuses:
            if not isinstance(status, dict):
                continue
            row = {
                key: status[key]
                for key in _ANALOG_COMPACT_COMPONENT_KEYS
                if status.get(key) not in (None, "", [], {})
            }
            if row:
                compact_statuses.append(row)
        if compact_statuses:
            compact["component_status"] = compact_statuses

    metrics = metadata.get("ensemble_metrics")
    if isinstance(metrics, dict):
        compact_metrics = {
            key: metrics[key]
            for key in _ANALOG_COMPACT_METRIC_KEYS
            if metrics.get(key) not in (None, "", [], {})
        }
        score_summary = metrics.get("score_summary")
        if isinstance(score_summary, dict):
            compact_scores = {
                key: score_summary[key]
                for key in ("best", "median")
                if score_summary.get(key) is not None
            }
            if compact_scores:
                compact_metrics["score_summary"] = compact_scores
        quality_gate = metrics.get("quality_gate")
        if isinstance(quality_gate, dict):
            compact_quality_gate = {
                key: quality_gate[key]
                for key in ("status", "failed_check")
                if quality_gate.get(key) not in (None, "", [], {})
            }
            if compact_quality_gate:
                compact_metrics["quality_gate"] = compact_quality_gate
        if compact_metrics:
            compact["ensemble_metrics"] = compact_metrics
    return compact


def _compact_ensemble_metadata(metadata: Any) -> Dict[str, Any]:
    """Project nested ensemble metadata while applying analog's compact contract."""
    if not isinstance(metadata, dict):
        return {}
    compact = {
        key: value
        for key, value in metadata.items()
        if key not in _ANALOG_VERBOSE_METADATA_KEYS
    }
    compact.update(_compact_analog_metadata(metadata))
    return compact


def _apply_forecast_generate_detail(  # noqa: C901
    payload: Dict[str, Any],
    request: ForecastGenerateRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    payload = dict(payload)
    payload.setdefault("quantity", request.quantity)
    payload = _round_forecast_generate_payload(payload)
    payload = _normalize_forecast_time_fields(payload)
    if str(payload.get("quantity") or request.quantity or "").strip().lower() == "volatility":
        payload = _round_forecast_volatility_payload(payload)
    payload = _annotate_forecast_generate_quality(payload)
    training_period = _forecast_training_period(payload)
    volatility_rows = _forecast_generate_volatility_rows(
        payload,
        horizon=getattr(request, "horizon", None),
    )
    volatility_summary_mode = bool(
        volatility_rows and str(payload.get("quantity") or request.quantity or "").strip().lower() == "volatility"
    )

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value in {"standard", "full"}:
        out = dict(payload)
        out.pop("ci_available", None)
        out.setdefault("symbol", request.symbol)
        out.setdefault("timeframe", request.timeframe)
        if training_period:
            out.setdefault("training_period", training_period)
        forecast_rows = _forecast_generate_compact_rows(out)
        row_series = forecast_rows or volatility_rows
        if row_series:
            out.setdefault("forecast", row_series)
        if volatility_summary_mode and not forecast_rows:
            out.setdefault("forecast_summary_mode", "scalar_volatility_estimate")
            out.setdefault(
                "quantity_note",
                "forecast contains a single volatility summary row; horizon_steps records the requested horizon "
                "because no distinct per-step volatility path is modeled.",
            )
        out["detail"] = detail_value
        if detail_value == "full":
            out.setdefault("interpretation", _forecast_generate_interpretation(out))
        return attach_collection_contract(
            out,
            collection_kind="time_series",
            series=_forecast_generate_series_rows(out) or row_series,
            include_contract_meta=detail_value == "full",
        )

    compact: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "method": payload.get("method"),
        "horizon": payload.get("horizon"),
        "quantity": payload.get("quantity"),
    }
    is_non_informative = payload.get("path_flat") is True
    if is_non_informative:
        compact["forecast_status"] = "non_informative"
        compact["signal_status"] = "not_actionable"
        compact["suggested_methods"] = ["drift", "analog", "fourier_ols"]
        compact["suggested_uncertainty_tool"] = "forecast_conformal_intervals"
    ci_unavailable = str(payload.get("ci_status") or "").strip().lower() == "unavailable"
    ci_compact = _forecast_compact_ci(payload)
    if ci_compact:
        compact["uncertainty"] = ci_compact
    if ci_unavailable:
        compact["ci_status"] = "unavailable"
        compact["forecast_mode"] = "point_only"
    ci_warning_dedup = ci_unavailable
    for key in (
        "last_observation_time",
        "timezone",
        "forecast_time",
        "forecast_price",
        "forecast_return",
        "last_price",
        "last_price_source",
        "price_basis",
        "last_price_stale",
        "warnings",
    ):
        value = payload.get(key)
        if key == "warnings":
            value = _compact_forecast_warnings(
                value,
                ci_unavailable=ci_warning_dedup,
            )
        if value not in (None, "", [], {}):
            compact[key] = value
    freshness = _forecast_anchor_freshness(payload)
    if freshness:
        compact["freshness"] = freshness
    data_window = _forecast_generate_data_window(payload)
    stale_nested = False
    if data_window:
        compact["data_window"] = data_window
        if "last_observation_stale" in data_window:
            stale_nested = True
            compact.pop("last_price_stale", None)
    if str(compact.get("quantity") or "").strip().lower() == "return":
        compact["return_unit"] = "return_fraction"
        if isinstance(payload.get("forecast_price"), list):
            compact["quantity_note"] = (
                "forecast rows show return; price is the reconstructed price path."
            )
    path_flatness = (
        {
            "path_flat": payload.get("path_flat"),
            "path_range": payload.get("path_range"),
        }
        if payload.get("path_flat") is True
        else None
    )
    price_context = payload.get("forecast_vs_last_price")
    if price_context:
        if path_flatness:
            price_context["direction"] = "neutral"
            price_context["direction_basis"] = "flat_path"
            price_context["direction_suppressed_reason"] = "flat_path"
        compact["forecast_vs_last_price"] = price_context
    if path_flatness:
        compact.update(path_flatness)
        compact.setdefault("point_forecast_mode", "flat_model_path")
    if str(compact.get("quantity") or "").strip().lower() == "volatility":
        for key in (
            "volatility_per_bar",
            "volatility_annualized",
            "volatility_horizon",
            "volatility_horizon_annualized",
            "volatility_unit",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value
    forecast_rows = _forecast_generate_compact_rows(payload)
    ci_has_intervals = isinstance(ci_compact, dict) and bool(ci_compact.get("intervals"))
    if forecast_rows:
        compact["forecast"] = forecast_rows
    elif volatility_rows:
        compact["forecast"] = volatility_rows
        compact["forecast_summary_mode"] = "scalar_volatility_estimate"
        compact["quantity_note"] = (
            "forecast summarizes a single volatility estimate; horizon_steps records the requested "
            "horizon and no distinct per-step path is implied."
        )
        compact.pop("forecast_time", None)
        compact.pop("forecast_price", None)
        compact.pop("forecast_return", None)
    if forecast_rows or ci_has_intervals:
        compact.pop("forecast_time", None)
        compact.pop("forecast_price", None)
        compact.pop("forecast_return", None)
    if path_flatness:
        warnings_out = compact.get("warnings")
        if not isinstance(warnings_out, list):
            warnings_out = []
        if _FORECAST_FLAT_PATH_WARNING not in warnings_out:
            warnings_out.append(_FORECAST_FLAT_PATH_WARNING)
        compact["warnings"] = warnings_out
    for key, value in payload.items():
        if key in compact:
            continue
        if key in {
            "base_col",
            "last_observation_epoch",
            "forecast_start_epoch",
            "forecast_from",
            "forecast_start_time",
            "forecast_start_gap_bars",
            "forecast_start_gap_note",
            "forecast_time",
            "forecast_bar_states",
            "forecast_time_semantics",
            "forecast_value_semantics",
            "forecast_price",
            "forecast_return",
            "forecast_anchor",
            "forecast_step_seconds",
            "forecast_epoch",
            "last_price_close",
            "last_price_source",
            "last_price_age_seconds",
            "last_price_age",
            "freshness_basis",
            "freshness_age_metric",
            "last_observation_close_epoch",
            "stale_after_seconds",
            "stale_warning",
            "lower_price",
            "upper_price",
            "lower_return",
            "upper_return",
            "lower",
            "upper",
            "ci",
            "uncertainty",
            "ci_status",
            "ci_alpha",
            "ci_available",
            "diagnostics",
            "params_used",
            "analogs",
            "component_status",
            "ensemble_metrics",
            "timeframe_diagnostics",
            "ensemble",
            "detail",
        }:
            continue
        if ci_unavailable and str(key).startswith("ci_"):
            continue
        if key == "last_price_stale" and stale_nested:
            continue
        if key == "denoise_applied" and value is False:
            continue
        compact[key] = value
    compact.update(_compact_analog_metadata(payload))
    ensemble = _compact_ensemble_metadata(payload.get("ensemble"))
    if ensemble:
        compact["ensemble"] = ensemble
    return compact


def _forecast_generate_interpretation(payload: Dict[str, Any]) -> Dict[str, str]:
    interpretation: Dict[str, str] = {}
    if payload.get("forecast") not in (None, "", [], {}):
        if payload.get("forecast_summary_mode") == "scalar_volatility_estimate":
            interpretation["forecast"] = (
                "Single summary row for scalar volatility output; horizon_steps records the requested "
                "horizon and no distinct per-step volatility path is implied."
            )
        else:
            interpretation["forecast"] = (
                "Per-step forecast rows for the requested horizon."
            )
    if payload.get("forecast_price") not in (None, "", [], {}):
        interpretation["forecast_price"] = (
            "Predicted price path in instrument price units."
        )
    if payload.get("forecast_return") not in (None, "", [], {}):
        interpretation["forecast_return"] = (
            "Predicted return path as decimal fractions; 0.01 means 1%."
        )
    if payload.get("last_price") not in (None, "", [], {}):
        interpretation["last_price"] = (
            "Reference market price used to anchor forecast comparisons."
        )
    if payload.get("forecast_vs_last_price") not in (None, "", [], {}):
        interpretation["forecast_vs_last_price"] = (
            "Horizon-end forecast versus last_price; first_step_delta shows "
            "only the first bar."
        )
    if (
        payload.get("lower_price") not in (None, "", [], {})
        or payload.get("upper_price") not in (None, "", [], {})
        or payload.get("ci") not in (None, "", [], {})
    ):
        interpretation["confidence_intervals"] = (
            "Forecast uncertainty bands when the selected method supports them."
        )
    return interpretation


def _forecast_training_period(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    out: Dict[str, Any] = {}
    for source_key, target_key in (
        ("history_start_time", "start"),
        ("history_end_time", "end"),
        ("history_bars_used", "history_bars_used"),
        ("target_points_used", "target_points_used"),
        ("lookback_bars_requested", "lookback_bars_requested"),
        ("minimum_history_bars_requested", "minimum_history_bars_requested"),
        ("history_bars_received", "history_bars_received"),
    ):
        value = diagnostics.get(source_key)
        if value not in (None, "", [], {}):
            out[target_key] = value
    if out:
        out.setdefault(
            "note",
            "Forecast was fit on the historical window summarized here.",
        )
    return out or None


def _forecast_generate_series_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = payload.get("forecast_time")
    prices = payload.get("forecast_price")
    if not isinstance(times, list) or not isinstance(prices, list):
        return []

    optional_series = {
        "forecast_return": payload.get("forecast_return"),
        "lower_price": payload.get("lower_price"),
        "upper_price": payload.get("upper_price"),
    }
    rows: List[Dict[str, Any]] = []
    for idx, time_value in enumerate(times):
        row: Dict[str, Any] = {
            "time": time_value,
            "forecast_price": prices[idx] if idx < len(prices) else None,
        }
        for key, values in optional_series.items():
            if isinstance(values, list) and idx < len(values):
                row[key] = values[idx]
        rows.append(row)
    return rows


def _conformal_summary(conformal: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(conformal, dict):
        return None
    out = {
        key: conformal.get(key)
        for key in (
            "interval_method",
            "ci_alpha",
            "calibration_steps",
            "calibration_spacing",
            "empirical_coverage",
            "coverage_target",
            "coverage_evaluation",
            "coverage_note",
            "min_calibration_points",
            "required_calibration_points",
            "calibration_sufficient",
            "interval_usage",
        )
        if conformal.get(key) not in (None, "", [], {})
    }
    return out or None


def _conformal_alpha_warning(ci_alpha: Any) -> Optional[str]:
    alpha = _finite_float(ci_alpha)
    if alpha is None:
        return None
    confidence = 1.0 - float(alpha)
    if alpha < 0.05:
        return (
            f"ci_alpha={alpha:g} gives a {confidence:.0%} interval, which is "
            "unusually wide for trading decisions; typical values are 0.05 or 0.10."
        )
    if alpha > 0.20:
        return (
            f"ci_alpha={alpha:g} gives a {confidence:.0%} interval, which is "
            "unusually narrow for risk management; typical values are 0.05 or 0.10."
        )
    return None


def _apply_conformal_intervals_detail(
    payload: Dict[str, Any],
    request: ForecastConformalIntervalsRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    payload = _round_forecast_generate_payload(payload)
    payload = _normalize_forecast_time_fields(payload)
    payload.setdefault("symbol", request.symbol)
    payload.setdefault("timeframe", request.timeframe)
    forecast_rows = _forecast_generate_compact_rows(payload)
    point_mode = _forecast_point_mode(payload)
    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value == "full":
        out = dict(payload)
        if forecast_rows:
            out.setdefault("forecast", forecast_rows)
        if point_mode:
            out.setdefault("point_forecast_mode", point_mode)
        out["detail"] = "full"
        return out

    out: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "method": payload.get("method", request.method),
        "horizon": request.horizon,
        "detail": detail_value,
    }
    for key in (
        "last_observation_time",
        "timezone",
        "forecast_time",
        "forecast_price",
        "lower_price",
        "upper_price",
        "lower_return",
        "upper_return",
        "interval_method",
        "ci_alpha",
        "nominal_confidence_level",
        "empirical_coverage",
        "coverage_status",
        "coverage_gap",
        "ci_status",
        "ci_available",
        "ci_warning",
        "required_calibration_points",
        "calibration_sufficient",
        "interval_usage",
        "calibration_remediation",
        "last_price",
        "last_price_source",
        "digits",
        "price_precision",
        "last_price_age_seconds",
        "last_price_stale",
        "history_policy_ok",
        "freshness_basis",
        "freshness_age_metric",
        "last_observation_close_epoch",
        "stale_after_seconds",
        "market_status",
        "market_status_reason",
        "retrieved_at",
        "retrieval_time",
        "forecast_vs_last_price",
        "signal_status",
        "units",
        "warnings",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    if "last_price_age_seconds" in out:
        out["data_age_seconds"] = out["last_price_age_seconds"]
    if "last_price_stale" in out:
        out["data_stale"] = out["last_price_stale"]
    conformal = _conformal_summary(payload.get("conformal"))
    if conformal:
        out["conformal"] = conformal
    if point_mode:
        out["point_forecast_mode"] = point_mode
    if forecast_rows:
        out["forecast"] = forecast_rows
        for key in (
            "forecast_time",
            "forecast_price",
            "lower_price",
            "upper_price",
            "lower_return",
            "upper_return",
        ):
            out.pop(key, None)
    return out


def _specific_forecast_method_name(
    *,
    requested_method: str,
    resolved_method: str,
    resolved_library: str,
    params: Dict[str, Any],
) -> str:
    requested = str(requested_method or "").strip()
    if ":" in requested:
        requested = requested.split(":", 1)[1].strip()
    if requested and requested.lower() != str(resolved_method or "").strip().lower():
        return requested

    selector_key_by_library = {
        "statsforecast": "model_name",
        "sktime": "estimator",
        "mlforecast": "model",
    }
    selector_key = selector_key_by_library.get(resolved_library)
    if selector_key:
        selector_value = params.get(selector_key)
        if selector_value not in (None, "", [], {}):
            return str(selector_value)
    return str(resolved_method or requested or "").strip()


def _library_method_error(
    *,
    library: str,
    method: str,
    valid_methods: Iterable[str],
) -> str:
    valid = ", ".join(str(item) for item in valid_methods)
    return f"method '{method}' is not available in library '{library}'. Valid methods: {valid}."


def _annotate_forecast_generate_method(
    payload: Dict[str, Any],
    *,
    requested_method: str,
    resolved_method: str,
    resolved_library: str,
    params: Dict[str, Any],
) -> None:
    if not isinstance(payload, dict) or payload.get("error"):
        return
    library_name = str(resolved_library or "native").strip().lower() or "native"
    if library_name in {"", "native"}:
        return

    payload["library"] = library_name
    adapter_method = str(resolved_method or "").strip().lower()
    output_method = str(payload.get("method") or "").strip().lower()
    if output_method in {"", adapter_method}:
        payload["method"] = _specific_forecast_method_name(
            requested_method=requested_method,
            resolved_method=resolved_method,
            resolved_library=library_name,
            params=params,
        )


def _apply_barrier_prob_detail(
    payload: Dict[str, Any],
    request: ForecastBarrierProbRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    payload = _round_barrier_prob_payload(payload)
    payload = _with_reference_price_context(_annotate_barrier_prob_context(payload, request))

    def _set_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            target[key] = value

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value == "full":
        out = dict(payload)
        out["detail"] = "full"
        out.setdefault("interpretation", _barrier_prob_interpretation(out))
        return out

    if "prob_hit" in payload:
        closed_form: Dict[str, Any] = {
            "success": bool(payload.get("success", True)),
            "detail": detail_value,
        }
        for key in (
            "symbol",
            "timeframe",
            "direction",
            "horizon",
            "barrier",
            "reference_price",
            "reference_price_source",
            "last_price_close",
            "analysis_mode",
            "conditioning_note",
            "prob_hit",
            "mu_annual",
            "log_drift_annual",
            "sigma_annual",
            "bars_per_year",
            "annualization_basis",
            "override_units",
            "already_hit",
            "warnings",
            "denoise_applied",
            "denoise_status",
            "denoise_error",
            "usable_for_live_trading",
            "usable_for_live_trading_basis",
            "execution_blockers",
            "remediation",
            "data_as_of",
            "data_stale",
            "freshness",
        ):
            _set_if_present(closed_form, key, payload.get(key))
        if detail_value == "standard":
            for key in ("already_hit", "mu_annual", "log_drift_annual", "sigma_annual"):
                value = payload.get(key)
                if value not in (None, "", [], {}):
                    closed_form[key] = value
        if set(closed_form) == {"success", "detail"}:
            return dict(payload)
        return closed_form

    if detail_value == "standard":
        out = dict(payload)
        out.pop("last_price", None)
        out.pop("last_price_close", None)
        out.pop("last_price_source", None)
        out.pop("tp_hit_prob_by_t", None)
        out.pop("sl_hit_prob_by_t", None)
        out.pop("sim_meta", None)
        out.pop("model_summary", None)
        out["detail"] = "standard"
        return out

    compact: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "detail": "compact",
    }
    for key in (
        "symbol",
        "timeframe",
        "method",
        "kind",
        "direction",
        "horizon",
        "reference_price",
        "reference_price_source",
        "tp_price",
        "sl_price",
        "prob_tp_first",
        "prob_sl_first",
        "prob_no_hit",
        "prob_same_bar",
        "prob_unresolved",
        "prob_resolve",
        "probability_edge",
        "probability_unit",
        "probability_edge_definition",
        "intra_bar_hit_detection",
        "bridge_correction",
        "bridge_dual_barrier_model",
        "bridge_joint_first_passage",
        "same_bar_policy",
        "n_sims",
        "seed",
        "seed_source",
        "as_of",
        "data_as_of",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "execution_blockers",
        "remediation",
        "verdict",
        "status",
        "status_reason",
        "barrier_unit",
        "tp_pct",
        "sl_pct",
        "tp_ticks",
        "sl_ticks",
    ):
        _set_if_present(compact, key, payload.get(key))
    if payload.get("warnings") not in (None, "", [], {}):
        compact["warnings"] = payload.get("warnings")
    if set(compact) == {"success", "detail"}:
        return dict(payload)
    return compact


def _annotate_barrier_prob_context(
    payload: Dict[str, Any],
    request: ForecastBarrierProbRequest,
) -> Dict[str, Any]:
    out = dict(payload)
    out.setdefault("symbol", request.symbol)
    out.setdefault("timeframe", request.timeframe)
    out.setdefault("horizon", request.horizon)
    out.setdefault("direction", request.direction)
    if request.tp_pct is not None:
        out.setdefault("tp_pct", request.tp_pct)
    if request.sl_pct is not None:
        out.setdefault("sl_pct", request.sl_pct)
    if request.tp_abs is not None:
        out.setdefault("tp_abs", request.tp_abs)
    if request.sl_abs is not None:
        out.setdefault("sl_abs", request.sl_abs)
    if request.tp_ticks is not None:
        out.setdefault("tp_ticks", request.tp_ticks)
    if request.sl_ticks is not None:
        out.setdefault("sl_ticks", request.sl_ticks)

    if out.get("tp_pct") is not None or out.get("sl_pct") is not None:
        out.setdefault("barrier_unit", "percent")
        out.setdefault("barrier_mode", "pct")
    elif out.get("tp_ticks") is not None or out.get("sl_ticks") is not None:
        out.setdefault("barrier_unit", "ticks")
        out.setdefault("barrier_mode", "ticks")
    elif out.get("tp_abs") is not None or out.get("sl_abs") is not None or out.get("barrier") is not None:
        out.setdefault("barrier_unit", "price")
        out.setdefault("barrier_mode", "price")
    out.setdefault("probability_unit", "fraction")
    if out.get("probability_edge") is None:
        tp_prob = _finite_float(out.get("prob_tp_first"))
        sl_prob = _finite_float(out.get("prob_sl_first"))
        if tp_prob is not None and sl_prob is not None:
            out["probability_edge"] = round(tp_prob - sl_prob, 6)
    out.setdefault(
        "probability_edge_definition",
        "prob_tp_first - prob_sl_first",
    )
    units = _barrier_prob_units(out)
    if units:
        out.setdefault("units", units)
    verdict = _barrier_prob_verdict(out)
    if verdict:
        if out.get("usable_for_live_trading") is False:
            out.setdefault("verdict", f"Research only — {verdict}")
        else:
            out.setdefault("verdict", verdict)
    if out.get("usable_for_live_trading") is False:
        out.setdefault("signal_status", "not_actionable")
    return out


def _barrier_prob_units(payload: Dict[str, Any]) -> Dict[str, str]:
    units: Dict[str, str] = {}
    for key in ("horizon", "time_to_tp_bars", "time_to_sl_bars"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "bars"
    price_keys = (
        "reference_price",
        "tp_price",
        "sl_price",
        "tp_abs",
        "sl_abs",
        "barrier",
    )
    for key in price_keys:
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "price"
    for key in ("tp_pct", "sl_pct"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "percent"
    for key in ("tp_ticks", "sl_ticks"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "ticks"
    for key in ("prob_tp_first", "prob_sl_first", "prob_no_hit", "prob_hit"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "probability_fraction"
    if payload.get("probability_edge") not in (None, "", [], {}):
        units["probability_edge"] = "probability_difference"
    return units


def _barrier_prob_verdict(payload: Dict[str, Any]) -> Optional[str]:
    edge_value = _finite_float(payload.get("probability_edge"))
    if edge_value is None:
        tp_prob = _finite_float(payload.get("prob_tp_first"))
        sl_prob = _finite_float(payload.get("prob_sl_first"))
        if tp_prob is not None and sl_prob is not None:
            edge_value = tp_prob - sl_prob
    if edge_value is not None:
        if edge_value > 0:
            return "TP-first probability bias"
        if edge_value < 0:
            return "SL-first probability bias"
        return "Neutral first-hit probabilities"
    if payload.get("prob_hit") not in (None, "", [], {}):
        return "Barrier-hit probability estimated"
    return None


def _barrier_prob_interpretation(payload: Dict[str, Any]) -> Dict[str, str]:
    interpretation: Dict[str, str] = {}
    if payload.get("prob_tp_first") not in (None, "", [], {}):
        interpretation["prob_tp_first"] = (
            "Probability the take-profit barrier is reached before stop-loss."
        )
    if payload.get("prob_sl_first") not in (None, "", [], {}):
        interpretation["prob_sl_first"] = (
            "Probability the stop-loss barrier is reached before take-profit."
        )
    if payload.get("prob_no_hit") not in (None, "", [], {}):
        interpretation["prob_no_hit"] = (
            "Probability neither barrier is reached before the forecast horizon."
        )
    if payload.get("probability_edge") not in (None, "", [], {}):
        interpretation["probability_edge"] = (
            "Take-profit-first probability minus stop-loss-first probability; "
            "this is not expected value."
        )
    if payload.get("prob_hit") not in (None, "", [], {}):
        interpretation["prob_hit"] = (
            "Closed-form probability the requested barrier is touched by horizon."
        )
    if any(str(key).endswith("_ci95") for key in payload):
        interpretation["ci95"] = (
            "Approximate 95% confidence intervals for Monte Carlo probabilities."
        )
    return interpretation


def _barrier_optimize_unit_context(payload: Dict[str, Any]) -> Tuple[str, str]:
    mode = str(
        payload.get("distance_unit")
        or payload.get("mode")
        or payload.get("barrier_mode")
        or ""
    ).strip().lower()
    if mode in {"ticks", "tick"}:
        return "ticks", "ticks"
    if mode in {"pct", "percent", "percentage", "percentage_points"}:
        return "percent", "pct"
    if mode in {"price", "abs", "absolute"}:
        return "price", "price"
    return "percent", "pct"


def _request_has_barrier_inputs(request: ForecastBarrierProbRequest) -> bool:
    return any(
        getattr(request, field_name, None) is not None
        for field_name in (
            "tp_abs",
            "sl_abs",
            "tp_pct",
            "sl_pct",
            "tp_ticks",
            "sl_ticks",
        )
    )


def _closed_form_barrier_input_error(request: ForecastBarrierProbRequest) -> Optional[str]:
    supplied_tp_sl_fields = [
        field_name
        for field_name in (
            "tp_abs",
            "sl_abs",
            "tp_pct",
            "sl_pct",
            "tp_ticks",
            "sl_ticks",
        )
        if getattr(request, field_name, None) is not None
    ]
    try:
        barrier_value = float(request.barrier_level)
    except (TypeError, ValueError):
        barrier_value = 0.0
    if barrier_value > 0.0:
        if supplied_tp_sl_fields:
            return (
                "The closed_form method uses the absolute barrier parameter only "
                "and does not consume TP/SL inputs. Remove "
                f"{', '.join(supplied_tp_sl_fields)} or use a Monte Carlo method "
                "such as mc_gbm for TP/SL barrier inputs."
            )
        return None
    if supplied_tp_sl_fields:
        return (
            "The closed_form method uses the absolute barrier parameter and "
            "does not consume TP/SL inputs such as tp_pct/sl_pct, tp_abs/sl_abs, "
            "or tick-based barriers. Provide barrier as a positive price, or use "
            "a Monte Carlo method such as mc_gbm for TP/SL barrier inputs."
        )
    return None


def _is_interval_unavailable_warning(value: Any) -> bool:
    text = str(value)
    return (
        "forecast_conformal_intervals" in text
        or "confidence intervals are unavailable" in text
    )


def _compact_forecast_warnings(
    warnings: Any,
    *,
    ci_unavailable: bool,
) -> Any:
    if not ci_unavailable:
        return warnings
    if isinstance(warnings, list):
        filtered = [
            warning
            for warning in warnings
            if not _is_interval_unavailable_warning(warning)
        ]
        return filtered
    if warnings not in (None, "", [], {}) and not _is_interval_unavailable_warning(warnings):
        return warnings
    return None


def _compact_backtest_units(
    raw_units: Any,
    method_summaries: list[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(raw_units, dict):
        return {}
    visible_unit_keys = {"forecast_error"}
    for row in method_summaries:
        visible_unit_keys.update(row.keys())
    return {
        key: value
        for key, value in raw_units.items()
        if key in visible_unit_keys
    }


def _compact_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    raw_results = result.get("results")
    if not isinstance(raw_results, dict):
        return result

    metric_digits = {
        "avg_rmse": 6,
        "avg_mae": 6,
        "avg_directional_accuracy": 4,
        "win_rate": 4,
        "win_rate_pct": 4,
        "max_drawdown": 4,
        "max_drawdown_pct": 4,
        "cumulative_return": 6,
        "cumulative_return_pct": 4,
        "avg_return": 6,
        "avg_return_pct": 4,
        "avg_return_per_trade": 6,
        "avg_return_per_trade_pct": 4,
        "avg_win_return": 6,
        "avg_win_return_pct": 4,
        "avg_loss_return": 6,
        "avg_loss_return_pct": 4,
        "avg_loss_magnitude": 6,
        "avg_loss_magnitude_pct": 4,
        "avg_win_loss_ratio": 4,
        "kelly_fraction": 4,
        "half_kelly_fraction": 4,
        "annual_return_pct": 4,
    }

    def _compact_metric(key: str, value: Any) -> Any:
        if isinstance(value, bool):
            return value
        numeric = _finite_float(value)
        if numeric is None:
            return value
        return float(round(numeric, metric_digits.get(key, 6)))

    def _sort_metric(value: Any) -> Optional[float]:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return value_f if math.isfinite(value_f) else None

    method_summaries: list[Dict[str, Any]] = []
    methods_total = 0
    methods_failed: list[str] = []
    for method_name, method_payload in raw_results.items():
        methods_total += 1
        if not isinstance(method_payload, dict):
            method_summaries.append({"method": method_name, "result": method_payload})
            methods_failed.append(str(method_name))
            continue
        if method_payload.get("success") is False:
            methods_failed.append(str(method_name))
        details = method_payload.get("details")
        metrics = (
            method_payload.get("metrics")
            if isinstance(method_payload.get("metrics"), dict)
            else {}
        )
        method_out: Dict[str, Any] = {"method": method_name}
        for key in (
            "success",
            "avg_rmse",
            "avg_mae",
            "avg_directional_accuracy",
            "successful_tests",
            "num_tests",
            "trade_status",
            "directional_accuracy_status",
            "metrics_available",
            "metrics_reason",
        ):
            if key in method_payload:
                method_out[key] = _compact_metric(key, method_payload[key])
        failure_error = method_payload.get("error")
        failure_code = method_payload.get("error_code")
        if not failure_error and isinstance(details, list):
            for detail_row in details:
                if isinstance(detail_row, dict) and detail_row.get("error"):
                    failure_error = detail_row.get("error")
                    failure_code = failure_code or detail_row.get("error_code")
                    break
        if failure_error:
            method_out["error"] = str(failure_error)
            if failure_code:
                method_out["error_code"] = str(failure_code)
        metrics_reason = str(method_out.get("metrics_reason") or "").strip()
        metrics_unavailable = _is_explicit_false(method_out.get("metrics_available"))
        if metrics_unavailable and metrics_reason:
            metrics_note = _BACKTEST_METRICS_REASON_NOTES.get(metrics_reason)
            if metrics_note:
                method_out["metrics_note"] = metrics_note
        if not metrics_unavailable:
            sample_notice = metrics.get("sample_notice")
            low_sample_metrics = (
                isinstance(sample_notice, dict)
                and sample_notice.get("code") == "annualization_suppressed_low_sample"
            )
            metric_keys = (
                (
                    "trades_observed",
                    "metrics_reliability",
                    "metrics_reliability_reason",
                )
                if low_sample_metrics
                else (
                    "win_rate",
                    "win_rate_pct",
                    "cumulative_return",
                    "cumulative_return_pct",
                    "max_drawdown",
                    "max_drawdown_pct",
                    "avg_return",
                    "avg_return_pct",
                    "avg_return_per_trade",
                    "avg_return_per_trade_pct",
                    "avg_win_return",
                    "avg_win_return_pct",
                    "avg_loss_return",
                    "avg_loss_return_pct",
                    "avg_loss_magnitude",
                    "avg_loss_magnitude_pct",
                    "avg_win_loss_ratio",
                    "kelly_fraction",
                    "half_kelly_fraction",
                    "annual_return_pct",
                    "trades_observed",
                    "metrics_reliability",
                    "metrics_reliability_reason",
                )
            )
            if low_sample_metrics:
                method_out.setdefault("metrics_reliability", "low")
                method_out.setdefault("metrics_reliability_reason", "low_sample")
            for key in metric_keys:
                if key in metrics:
                    method_out[key] = _compact_metric(key, metrics[key])
            if isinstance(sample_notice, dict) and sample_notice:
                method_out["sample_notice"] = sample_notice
        if isinstance(details, list) and not metrics_unavailable:
            method_out["details_count"] = len(details)
        ranked_row = dict(method_out)
        ranked_row["_sort_metric"] = _sort_metric(method_payload.get("avg_rmse"))
        method_summaries.append(ranked_row)

    compact_out = dict(result)
    compact_out.pop("request", None)
    compact_out.pop("resolved_request", None)
    compact_out["detail"] = "compact"
    compact_units = _compact_backtest_units(
        compact_out.get("units"),
        method_summaries,
    )
    if compact_units:
        compact_out["units"] = compact_units
    else:
        compact_out.pop("units", None)
    slippage_bps = float(compact_out.get("slippage_bps") or 0.0)
    compact_out["slippage_bps"] = slippage_bps
    compact_out["cost_assumptions"] = {
        "score_basis": (
            "net_of_configured_slippage"
            if slippage_bps > 0.0
            else "gross_before_execution_costs"
        ),
        "slippage_bps_per_side": slippage_bps,
        "spread_and_commission": "not_modeled",
    }
    if compact_out.get("trade_threshold") in (0, 0.0, None):
        compact_out.pop("trade_threshold", None)
    compact_out["methods_total"] = methods_total
    compact_out["methods_succeeded"] = methods_total - len(methods_failed)
    compact_out["methods_failed"] = len(methods_failed)
    if methods_failed:
        compact_out["failed_methods"] = methods_failed
    method_summaries.sort(
        key=lambda row: (
            row.get("_sort_metric") is None,
            row.get("_sort_metric") if row.get("_sort_metric") is not None else 0.0,
            str(row.get("method") or ""),
        )
    )
    ranked_methods: list[Dict[str, Any]] = []
    rank = 0
    for row in method_summaries:
        method = str(row.get("method") or "")
        score = row.get("_sort_metric")
        eligible = score is not None and row.get("success") is not False
        ranked_row: Dict[str, Any] = {
            "method": method,
            "ranking_status": "ranked" if eligible else "unranked",
        }
        if eligible:
            rank += 1
            ranked_row.update(
                {
                    "rank": rank,
                    "avg_rmse": _compact_metric("avg_rmse", score),
                    "trading_metrics_available": not _is_explicit_false(
                        row.get("metrics_available")
                    ),
                }
            )
            if _is_explicit_false(row.get("metrics_available")):
                ranked_row["selection_warning"] = (
                    "ranking_uses_forecast_error_only; trading metrics are unavailable"
                )
                if row.get("metrics_reason"):
                    ranked_row["trading_metrics_reason"] = row["metrics_reason"]
        else:
            ranked_row["unranked_reason"] = (
                row.get("error_code")
                or ("method_failed" if row.get("success") is False else "avg_rmse_unavailable")
            )
            if row.get("error"):
                ranked_row["error"] = row["error"]
        ranked_methods.append(ranked_row)
    compact_out["ranking"] = {
        "metric": "avg_rmse",
        "direction": "ascending",
        "scope": "non_failed_methods_with_finite_avg_rmse",
        "note": "Trading metrics do not affect rank; inspect results for method details.",
    }
    compact_out["ranked_methods"] = ranked_methods
    compact_out["results"] = {
        str(row.get("method")): {
            key: value
            for key, value in row.items()
            if key not in {"method", "_sort_metric"} and value is not None
        }
        for row in method_summaries
    }
    return compact_out


def _attach_backtest_collection_contract(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep method summaries and counters stable across every detail level."""
    compact = _compact_backtest_result(result)
    out = dict(result)
    for key in (
        "ranked_methods",
        "ranking",
        "methods_total",
        "methods_succeeded",
        "methods_failed",
        "failed_methods",
        "cost_assumptions",
    ):
        if key in compact:
            out[key] = compact[key]
    out["slippage_bps"] = float(result.get("slippage_bps") or 0.0)
    return out


def _sktime_forecaster_index_path() -> Optional[Path]:
    """Return the version-scoped persistent class index path."""
    try:
        sktime_version = metadata.version("sktime")
    except metadata.PackageNotFoundError:
        return None
    safe_version = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in str(sktime_version)
    )
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return (
        base
        / "mtdata"
        / "forecast-indices-v1"
        / f"sktime-{safe_version}-py{sys.version_info.major}{sys.version_info.minor}.json"
    )


def _valid_sktime_forecaster_mapping(value: Any) -> Dict[str, Tuple[str, str]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        class_name, dotted_path = item
        if (
            isinstance(class_name, str)
            and class_name
            and isinstance(dotted_path, str)
            and dotted_path.startswith("sktime.forecasting.")
        ):
            out[key.lower()] = (class_name, dotted_path)
    return out


def _load_sktime_forecaster_index() -> Dict[str, Tuple[str, str]]:
    path = _sktime_forecaster_index_path()
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SKTIME_INDEX_SCHEMA_VERSION
    ):
        return {}
    return _valid_sktime_forecaster_mapping(payload.get("forecasters"))


def _store_sktime_forecaster_index(mapping: Dict[str, Tuple[str, str]]) -> None:
    path = _sktime_forecaster_index_path()
    if path is None or not mapping:
        return
    temporary_path: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            json.dump(
                {
                    "schema_version": _SKTIME_INDEX_SCHEMA_VERSION,
                    "forecasters": mapping,
                },
                stream,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _registered_sktime_forecasters() -> Dict[str, Tuple[str, str]]:
    """Build cheap exact-name routes from registered sktime aliases."""
    mapping: Dict[str, Tuple[str, str]] = {}
    for method_name in ForecastRegistry.get_all_method_names():
        try:
            method_class = ForecastRegistry.get_class(method_name)
        except ValueError:
            continue
        dotted_path = str(
            getattr(method_class, "CAPABILITY_SELECTOR_VALUE", "") or ""
        )
        if not dotted_path.startswith("sktime.forecasting."):
            continue
        class_name = dotted_path.rsplit(".", 1)[-1]
        value = (class_name, dotted_path)
        for alias in (
            class_name,
            method_name,
            *tuple(getattr(method_class, "CAPABILITY_ALIASES", ()) or ()),
        ):
            alias_text = str(alias or "").strip()
            if alias_text:
                mapping.setdefault(alias_text.lower(), value)
    return mapping


@lru_cache(maxsize=1)
def _discover_sktime_forecasters() -> Dict[str, Tuple[str, str]]:
    """Return mapping of forecaster class name (lower) -> (class_name, dotted path)."""
    try:
        # sktime 1.0+ forecasting package eagerly imports torch-backed aliases.
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings(
                "ignore",
                message=r".*swigvarlink.*",
                category=DeprecationWarning,
            )
            import sktime.forecasting as _sf  # type: ignore
            from sktime.forecasting.base import BaseForecaster  # type: ignore
    except Exception:
        return {}

    mapping: Dict[str, Tuple[str, str]] = {}

    def _skip_module(mod_name: str) -> bool:
        parts = mod_name.split(".")
        if "tests" in parts:
            return True
        if any(part.startswith("test") for part in parts):
            return True
        return False

    for mod in pkgutil.walk_packages(getattr(_sf, "__path__", []), _sf.__name__ + "."):
        mod_name = getattr(mod, "name", None)
        if not isinstance(mod_name, str) or _skip_module(mod_name):
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings(
                    "ignore",
                    message=r".*swigvarlink.*",
                    category=DeprecationWarning,
                )
                module = importlib.import_module(mod_name)
        except Exception:
            continue
        for _, obj in vars(module).items():
            if not isinstance(obj, type):
                continue
            if obj is BaseForecaster:
                continue
            name = getattr(obj, "__name__", None)
            if not isinstance(name, str) or not name or name.startswith("_"):
                continue
            try:
                if not issubclass(obj, BaseForecaster):
                    continue
            except Exception:
                continue
            if inspect.isabstract(obj):
                continue
            try:
                constructor = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            required_constructor_params = [
                parameter
                for parameter in constructor.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            ]
            if required_constructor_params:
                continue
            key = name.lower()
            if key not in mapping:
                mapping[key] = (name, f"{obj.__module__}.{name}")
    _store_sktime_forecaster_index(mapping)
    return mapping


def _normalize_forecaster_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_MIN_CONFORMAL_CALIBRATION_POINTS = 30


def _finite_sample_conformal_quantile(values: List[float], alpha: float) -> float:
    if not values:
        return float("nan")

    import numpy as _np

    arr = _np.asarray(values, dtype=float)
    if _np.isnan(arr).any():
        return float("nan")

    n = int(arr.size)
    rank = max(1, min(n, math.ceil((n + 1) * (1.0 - float(alpha)))))
    return float(_np.partition(arr, rank - 1)[rank - 1])


def _leave_one_out_conformal_coverage(
    values: List[float],
    alpha: float,
) -> Optional[float]:
    if len(values) < 2:
        return None
    covered = 0
    evaluated = 0
    for index, value in enumerate(values):
        calibration = values[:index] + values[index + 1 :]
        quantile = _finite_sample_conformal_quantile(calibration, alpha)
        if not math.isfinite(quantile):
            continue
        evaluated += 1
        covered += int(float(value) <= quantile)
    return float(covered / evaluated) if evaluated else None


def _resolve_sktime_forecaster(method: str) -> Optional[Tuple[str, str]]:
    """Resolve a user-provided method name to (class_name, dotted_path)."""
    method_s = str(method or "").strip()
    if not method_s:
        return None

    module_name, separator, class_name = method_s.rpartition(".")
    if (
        separator
        and module_name.startswith("sktime.forecasting")
        and class_name
    ):
        return class_name, method_s

    method_key = method_s.lower()
    exact_mapping = _registered_sktime_forecasters()
    exact = exact_mapping.get(method_key)
    if exact:
        return exact

    persistent_mapping = _load_sktime_forecaster_index()
    exact = persistent_mapping.get(method_key)
    if exact:
        return exact

    mapping = _discover_sktime_forecasters()
    if not mapping:
        return None

    exact = mapping.get(method_s.lower())
    if exact:
        return exact

    norm_map: Dict[str, Tuple[str, str]] = {}
    for _, (cls_name, dotted) in mapping.items():
        norm_map.setdefault(_normalize_forecaster_name(cls_name), (cls_name, dotted))

    query_norm = _normalize_forecaster_name(method_s)
    if query_norm in norm_map:
        return norm_map[query_norm]

    starts = [value for key, value in norm_map.items() if key.startswith(query_norm)]
    if starts:
        return sorted(starts, key=lambda item: len(item[0]))[0]

    contains = [value for key, value in norm_map.items() if query_norm and query_norm in key]
    if contains:
        return sorted(contains, key=lambda item: len(item[0]))[0]

    candidates = difflib.get_close_matches(query_norm, list(norm_map), n=1, cutoff=0.6)
    if candidates:
        return norm_map[candidates[0]]
    return None


def _resolve_stored_model_execution_alias(
    *,
    library: str,
    requested_method: str,
    resolved_method: str,
    params: Dict[str, Any],
    original_params: Dict[str, Any],
    model_id: Any,
) -> tuple[str, Dict[str, Any]]:
    """Execute a compatible stored wrapper under its canonical model ID."""
    parts = str(model_id or "").split("/")
    if len(parts) != 3:
        return resolved_method, params
    stored_method = parts[0]
    if stored_method == resolved_method:
        return resolved_method, params
    try:
        stored_class = ForecastRegistry.get_class(stored_method)
    except ValueError:
        return resolved_method, params
    selector_key = str(
        getattr(stored_class, "CAPABILITY_SELECTOR_KEY", "") or ""
    )
    selector_value = str(
        getattr(stored_class, "CAPABILITY_SELECTOR_VALUE", "") or ""
    )
    supplied_selector = str(original_params.get(selector_key) or "")
    if (
        selector_key
        and selector_value
        and selector_key in original_params
        and supplied_selector.lower() != selector_value.lower()
    ):
        raise ForecastError(
            f"model_id '{model_id}' identifies method '{stored_method}' with "
            f"{selector_key}={selector_value!r}, but the request supplied "
            f"{selector_key}={supplied_selector!r}. Remove the conflicting selector."
        )
    requested_selector = str(params.get(selector_key) or "")
    method_matches = requested_method.strip().lower() == stored_method.lower()
    selector_matches = bool(
        selector_key
        and selector_value
        and requested_selector.lower() == selector_value.lower()
    )
    execution_library = str(
        getattr(stored_class, "CAPABILITY_EXECUTION_LIBRARY", "") or ""
    ).lower()
    library_matches = not execution_library or execution_library == library
    if not library_matches or not (method_matches or selector_matches):
        return resolved_method, params
    alias_params = dict(params)
    if selector_key and selector_key not in original_params:
        alias_params.pop(selector_key, None)
    return stored_method, alias_params


def run_forecast_generate(  # noqa: C901
    request: ForecastGenerateRequest,
    *,
    forecast_impl: Any = _forecast_impl,
    resolve_sktime_forecaster: Any = _resolve_sktime_forecaster,
    log_events: bool = True,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    lib = str(request.library or "native").strip().lower()
    method = str(request.method or "").strip()
    params = dict(request.params or {})
    if log_events:
        log_operation_start(
            logger,
            operation="forecast_generate",
            symbol=request.symbol,
            timeframe=request.timeframe,
            library=lib or "native",
            method=method or None,
        )

    def _finish(result: Dict[str, Any], *, resolved_method: Optional[str] = None) -> Dict[str, Any]:
        if log_events:
            log_operation_finish(
                logger,
                operation="forecast_generate",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                library=lib or "native",
                method=method or None,
                resolved_method=resolved_method,
            )
        return result

    try:
        capability_requested = ":" in method
        requested_method = method
        original_resolution = (lib, method, dict(params))
        lib, method, params = resolve_capability_request(
            library=lib,
            method=method,
            params=params,
            discover_sktime_forecasters=_discover_sktime_forecasters,
        )
        capability_requested = capability_requested or (lib, method, params) != original_resolution
        if capability_requested:
            if lib in ("", "native"):
                resolved_method = method or "theta"
            elif lib == "statsforecast":
                resolved_method = "statsforecast"
            elif lib == "sktime":
                resolved_method = "sktime"
            elif lib == "pretrained":
                resolved_method = method or "chronos2"
            elif lib == "mlforecast":
                resolved_method = "mlforecast"
            else:
                raise ForecastError(f"Unsupported library: {lib}")
        elif lib in ("", "native"):
            resolved_method = method or "theta"
        elif lib == "statsforecast":
            if not method:
                raise ForecastError("method is required for library=statsforecast")
            resolved_method = "statsforecast"
            params.setdefault("model_name", method)
        elif lib == "sktime":
            query = method.strip() if method else "ThetaForecaster"
            if "." in query:
                resolved_method = "sktime"
                params.setdefault("estimator", query)
            else:
                found = resolve_sktime_forecaster(query)
                if not found:
                    raise ForecastError(f"Unknown sktime forecaster '{query}'")
                _, dotted = found
                resolved_method = "sktime"
                params.setdefault("estimator", dotted)
        elif lib == "pretrained":
            if method and method.strip().lower() not in _PRETRAINED_FORECAST_METHODS:
                raise ForecastError(
                    _library_method_error(
                        library="pretrained",
                        method=method,
                        valid_methods=_PRETRAINED_FORECAST_METHODS,
                    )
                )
            resolved_method = method or "chronos2"
        elif lib == "mlforecast":
            if not method:
                raise ForecastError("method is required for library=mlforecast")
            method_key = method.strip().lower()
            if (
                "." not in method
                and method_key not in {"mlforecast", "mlf_rf", "mlf_lightgbm"}
            ):
                raise ForecastError(
                    _library_method_error(
                        library="mlforecast",
                        method=method,
                        valid_methods=(
                            "mlf_lightgbm",
                            "mlf_rf",
                            "mlforecast with params.model=<approved dotted class>",
                        ),
                    )
                )
            if method_key in {"mlf_rf", "mlf_lightgbm"}:
                resolved_method = method_key
            else:
                resolved_method = "mlforecast"
                params.setdefault("model", method)
        else:
            raise ForecastError(f"Unsupported library: {request.library}")

        resolved_method, params = _resolve_stored_model_execution_alias(
            library=lib,
            requested_method=requested_method,
            resolved_method=resolved_method,
            params=params,
            original_params=original_resolution[2],
            model_id=getattr(request, "model_id", None),
        )

        proxy_value = request.proxy
        proxy_defaulted = False
        if str(request.quantity).strip().lower() == "volatility":
            if proxy_value is None and isinstance(params, dict):
                proxy_candidate = params.get("proxy")
                if proxy_candidate not in (None, ""):
                    proxy_value = str(proxy_candidate).strip().lower()
                    params.pop("proxy", None)
            if (
                proxy_value is None
                and str(resolved_method).strip().lower() in _VOLATILITY_PROXY_METHODS
            ):
                proxy_value = _DEFAULT_VOLATILITY_PROXY
                proxy_defaulted = True

        parameter_error = unknown_mapping_keys_error(
            params,
            get_method_param_names(str(resolved_method)),
            subject=f"forecast params for method '{resolved_method}'",
        )
        if parameter_error is not None:
            parameter_error["operation"] = "forecast_generate"
            parameter_error["details"] = {
                "library": lib or "native",
                "method": str(resolved_method),
            }
            return _finish(parameter_error, resolved_method=str(resolved_method))

        out = forecast_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(resolved_method),
            horizon=request.horizon,
            lookback=request.lookback,
            as_of=request.as_of,
            start=request.start,
            end=request.end,
            params=params,
            ci_alpha=request.effective_ci_alpha,
            quantity=request.quantity,
            proxy=proxy_value,
            denoise=request.denoise,
            features=request.features or {},
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            target_spec=request.target_spec,
            async_mode=getattr(request, 'async_mode', False),
            model_id=getattr(request, 'model_id', None),
            model_cache=getattr(request, 'model_cache', 'reuse'),
        )
        if isinstance(out, dict):
            out = _attach_invalid_method_guidance(out)
        if isinstance(out, dict) and "success" not in out and infer_result_success(out):
            out["success"] = True
        if proxy_defaulted and isinstance(out, dict) and not out.get("error"):
            warnings_out = out.get("warnings")
            if not isinstance(warnings_out, list):
                warnings_out = []
            default_warning = (
                "quantity=volatility defaulted proxy=squared_return; set proxy "
                "explicitly to use abs_return or log_r2."
            )
            if default_warning not in warnings_out:
                warnings_out.append(default_warning)
            out["warnings"] = warnings_out

        if (
            isinstance(out, dict)
            and lib in ("", "native")
            and str(resolved_method).strip().lower() == "theta"
        ):
            detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
            warning = (
                "Using native theta. StatsForecast theta is available via the "
                "statsforecast library (--library statsforecast --method Theta) "
                "and may produce different forecasts or interval behavior."
            )
            if detail_value != "compact":
                warning = (
                    warning
                    + " Example: "
                    + f"mtdata-cli forecast_generate {request.symbol} --timeframe {request.timeframe} "
                    + f"--library statsforecast --method Theta --horizon {request.horizon}"
                )
            warnings_out = out.get("warnings")
            if not isinstance(warnings_out, list):
                warnings_out = []
            has_interval_warning = any(
                _is_interval_unavailable_warning(item) for item in warnings_out
            )
            if warning not in warnings_out and not has_interval_warning:
                warnings_out.append(warning)
            out["warnings"] = warnings_out
        if isinstance(out, dict):
            out = _annotate_price_currency(out, request.symbol)
            _annotate_forecast_generate_method(
                out,
                requested_method=requested_method,
                resolved_method=str(resolved_method),
                resolved_library=lib,
                params=params,
            )
        out = _apply_forecast_generate_detail(out, request)
        return _finish(out, resolved_method=str(resolved_method))
    except Exception as exc:
        if log_events:
            log_operation_exception(
                logger,
                operation="forecast_generate",
                started_at=started_at,
                exc=exc,
                symbol=request.symbol,
                timeframe=request.timeframe,
                library=lib or "native",
                method=method or None,
            )
        raise


def run_forecast_backtest(
    request: ForecastBacktestRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        methods=len(request.methods or []),
    )
    try:
        result = backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            steps=request.steps,
            spacing=request.spacing,
            lookback=request.lookback,
            start=request.start,
            end=request.end,
            methods=request.methods,
            params_per_method=request.params_per_method,
            quantity=request.quantity,
            denoise=request.denoise,
            params=request.params,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            slippage_bps=request.slippage_bps,
            trade_threshold=request.trade_threshold,
            detail=request.detail,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_backtest",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_backtest",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        methods=len(request.methods or []),
    )
    if isinstance(result, dict):
        backtest_plan = result.get("backtest_plan")
        if isinstance(backtest_plan, dict):
            backtest_plan["actual_runtime_seconds"] = round(
                time.perf_counter() - started_at,
                6,
            )
        result = _attach_analysis_time_window(result, request)
    requested_detail = _requested_detail_label(request.detail)
    if str(request.detail or "compact").strip().lower() == "compact":
        return _compact_backtest_result(result)
    if isinstance(result, dict):
        result = _attach_backtest_collection_contract(result)
        result["detail"] = requested_detail
    return result


def run_strategy_backtest(
    request: StrategyBacktestRequest,
    *,
    strategy_backtest_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="strategy_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        lookback=request.lookback,
    )
    try:
        result = strategy_backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            lookback=request.lookback,
            start=request.start,
            end=request.end,
            detail=request.detail,
            position_mode=request.position_mode,
            fast_period=request.fast_period,
            slow_period=request.slow_period,
            rsi_length=request.rsi_length,
            oversold=request.oversold,
            overbought=request.overbought,
            max_hold_bars=request.max_hold_bars,
            cost_model=request.cost_model,
            spread_bps=request.spread_bps,
            slippage_bps=request.slippage_bps,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="strategy_backtest",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
        )
        raise
    log_operation_finish(
        logger,
        operation="strategy_backtest",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        lookback=request.lookback,
    )
    return result


def _analysis_time_kwargs(request: Any) -> Dict[str, Any]:
    return {
        key: value
        for key, value in {
            "as_of": getattr(request, "as_of", None),
            "start": getattr(request, "start", None),
            "end": getattr(request, "end", None),
        }.items()
        if value not in (None, "")
    }


def _attach_analysis_time_window(
    result: Dict[str, Any],
    request: Any,
) -> Dict[str, Any]:
    """Disclose the historical cutoff/range used by replayable analytics."""
    values = {
        "as_of": getattr(request, "as_of", None),
        "start": getattr(request, "start", None),
        "end": getattr(request, "end", None),
    }
    existing_window = result.get("analysis_time_window")
    if not any(value not in (None, "") for value in values.values()) and not isinstance(
        existing_window,
        dict,
    ):
        return result
    out = dict(result)
    out["analysis_time_window"] = (
        dict(existing_window) if isinstance(existing_window, dict) else {}
    )
    window = out["analysis_time_window"]
    window.update(
        {key: value for key, value in values.items() if value not in (None, "")}
    )
    data_window = out.get("data_window")
    if not isinstance(data_window, dict):
        data_window = out.get("history_window")
    if isinstance(data_window, dict):
        if data_window.get("start") is not None:
            window["effective_start"] = data_window.get("start")
        if data_window.get("end") is not None:
            window["effective_end"] = data_window.get("end")
    elif out.get("data_as_of") is not None:
        window["effective_end"] = out.get("data_as_of")
    window["timezone"] = "UTC"
    window["input_bar_policy"] = "closed_bars_only"
    window["reference_policy"] = "historical_candle_close"
    return out


def run_forecast_conformal_intervals(
    request: ForecastConformalIntervalsRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
    forecast_impl: Any = _forecast_impl,
) -> Dict[str, Any]:
    """Build residual-quantile forecast bands from rolling backtest residuals.

    Not true split-conformal prediction: residuals come from rolling-origin
    backtest fits (different models per anchor), bands are symmetric absolute-
    residual quantiles, and reported coverage is empirical leave-one-out on
    those residuals—not a guaranteed finite-sample coverage bound.
    """
    requested_method = str(request.method or "").strip()
    valid_methods = list(get_forecast_method_names())
    if requested_method.lower() not in {
        str(method).lower() for method in valid_methods
    }:
        return build_error_payload(
            format_invalid_method_error(requested_method, valid_methods),
            code="invalid_method",
            operation="forecast_conformal_intervals",
            details={"method": requested_method},
            related_tools=["forecast_list_methods"],
        )
    started_at = time.perf_counter()
    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    log_operation_start(
        logger,
        operation="forecast_conformal_intervals",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    try:
        # 1) Rolling backtest to collect residuals.
        bt = raise_if_error_result(backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            **(
                {"lookback": int(request.lookback)}
                if request.lookback is not None
                else {}
            ),
            **_analysis_time_kwargs(request),
            methods=[str(request.method)],
            denoise=request.denoise,
            params_per_method={str(request.method): dict(request.params or {})},
            detail="full",
        ))
        res = bt.get("results", {}).get(str(request.method))
        if not res or not res.get("details"):
            raise ForecastError(
                "Residual-quantile interval calibration failed: no backtest details"
            )

        # Build per-step residuals |y_hat_i - y_i|.
        fh = int(request.horizon)
        errs: List[List[float]] = [[] for _ in range(fh)]
        for detail in res["details"]:
            fc = detail.get("forecast")
            act = detail.get("actual")
            if not fc or not act:
                continue
            width = min(len(fc), len(act), fh)
            for i in range(width):
                try:
                    errs[i].append(abs(float(fc[i]) - float(act[i])))
                except Exception:
                    continue

        import numpy as _np

        qerrs = [
            _finite_sample_conformal_quantile(err, float(request.ci_alpha))
            for err in errs
        ]
        calibration_points = [len(err) for err in errs]
        coverage_per_step = [
            _leave_one_out_conformal_coverage(err, float(request.ci_alpha))
            for err in errs
        ]
        finite_coverage = [value for value in coverage_per_step if value is not None]
        empirical_coverage = (
            float(sum(finite_coverage) / len(finite_coverage))
            if finite_coverage
            else None
        )
        min_calibration_points = min(calibration_points) if calibration_points else 0

        # 2) Forecast now (latest).
        out = raise_if_error_result(forecast_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=int(request.horizon),
            **(
                {"lookback": int(request.lookback)}
                if request.lookback is not None
                else {}
            ),
            params=request.params,
            denoise=request.denoise,
            **_analysis_time_kwargs(request),
        ))
        yhat = out.get("forecast_price") or []
        if not yhat:
            raise ForecastError("Empty point forecast for residual-quantile intervals")
        yhat_arr = _np.array(yhat, dtype=float)
        fh_eff = min(fh, yhat_arr.size)
        lo = _np.empty(fh_eff, dtype=float)
        hi = _np.empty(fh_eff, dtype=float)
        for i in range(fh_eff):
            err = qerrs[i] if i < len(qerrs) and _np.isfinite(qerrs[i]) else 0.0
            lo[i] = yhat_arr[i] - err
            hi[i] = yhat_arr[i] + err

        result = dict(out)
        result["detail"] = detail_value
        result["interval_method"] = "rolling_residual_quantiles"
        result["conformal"] = {
            "interval_method": "rolling_residual_quantiles",
            "ci_alpha": float(request.ci_alpha),
            "calibration_steps": int(request.steps),
            "calibration_spacing": int(request.spacing),
            "per_step_q": [float(v) for v in qerrs],
            "calibration_points_per_step": calibration_points,
            "min_calibration_points": int(min_calibration_points),
            "required_calibration_points": _MIN_CONFORMAL_CALIBRATION_POINTS,
            "calibration_sufficient": (
                min_calibration_points >= _MIN_CONFORMAL_CALIBRATION_POINTS
            ),
            "empirical_coverage_per_step": coverage_per_step,
            "empirical_coverage": empirical_coverage,
            "coverage_target": round(1.0 - float(request.ci_alpha), 6),
            "coverage_evaluation": "leave_one_out_calibration_residuals",
            "coverage_note": (
                "Empirical residual quantiles from rolling backtest; not a "
                "finite-sample conformal coverage guarantee."
            ),
        }
        result["lower_price"] = [float(v) for v in lo.tolist()]
        result["upper_price"] = [float(v) for v in hi.tolist()]
        result["ci_alpha"] = float(request.ci_alpha)
        nominal_confidence = round(1.0 - float(request.ci_alpha), 6)
        result["nominal_confidence_level"] = nominal_confidence
        result["empirical_coverage"] = empirical_coverage
        if empirical_coverage is None:
            result["coverage_status"] = "not_evaluated"
        elif empirical_coverage + 1e-12 < nominal_confidence:
            result["coverage_status"] = "below_nominal_target"
            result["coverage_gap"] = round(
                float(empirical_coverage) - nominal_confidence,
                6,
            )
        else:
            result["coverage_status"] = "at_or_above_nominal_target"
        calibration_sufficient = (
            min_calibration_points >= _MIN_CONFORMAL_CALIBRATION_POINTS
            and empirical_coverage is not None
        )
        result["required_calibration_points"] = _MIN_CONFORMAL_CALIBRATION_POINTS
        result["calibration_sufficient"] = calibration_sufficient
        if calibration_sufficient:
            result["ci_status"] = "available"
            result["ci_available"] = True
            result["interval_usage"] = "calibrated"
        else:
            result["ci_status"] = "insufficient_calibration"
            result["ci_available"] = False
            result["interval_usage"] = "diagnostic_only"
            result["calibration_remediation"] = (
                "Increase --steps until every forecast horizon has at least "
                f"{_MIN_CONFORMAL_CALIBRATION_POINTS} calibration residuals."
            )
        result["conformal"]["interval_usage"] = result["interval_usage"]
        result = _attach_analysis_time_window(result, request)
        alpha_warning = _conformal_alpha_warning(request.ci_alpha)
        warnings_out = result.get("warnings")
        if isinstance(warnings_out, list):
            filtered_warnings = [
                item for item in warnings_out if not _is_interval_unavailable_warning(item)
            ]
            if filtered_warnings:
                result["warnings"] = filtered_warnings
            else:
                result.pop("warnings", None)
        if alpha_warning:
            result["ci_warning"] = alpha_warning
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if alpha_warning not in warnings_list:
                warnings_list.append(alpha_warning)
            result["warnings"] = warnings_list
        if min_calibration_points < _MIN_CONFORMAL_CALIBRATION_POINTS:
            sample_warning = (
                "Residual-quantile calibration has as few as "
                f"{min_calibration_points} residual(s) per forecast step; "
                f"at least {_MIN_CONFORMAL_CALIBRATION_POINTS} are required before "
                "intervals are available for decision use. Returned bounds are "
                "diagnostic only and are not true conformal prediction intervals."
            )
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if sample_warning not in warnings_list:
                warnings_list.append(sample_warning)
            result["warnings"] = warnings_list
        if result.get("coverage_status") == "below_nominal_target":
            coverage_warning = (
                f"Empirical coverage {float(empirical_coverage):.3f} is below the "
                f"nominal target {nominal_confidence:.3f}; use empirical_coverage "
                "when assessing historical calibration quality."
            )
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if coverage_warning not in warnings_list:
                warnings_list.append(coverage_warning)
            result["warnings"] = warnings_list
        result = _annotate_forecast_generate_quality(result)
        result = _apply_conformal_intervals_detail(result, request)
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_conformal_intervals",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_conformal_intervals",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    return result


def _resolve_tuning_search_space(
    request: ForecastTuneGeneticRequest | ForecastTuneOptunaRequest,
) -> tuple[Optional[str], Dict[str, Any]]:
    method_for_search: Optional[str] = request.method
    from ..forecast.tune import default_search_space as _default_search_space

    search_space = dict(request.search_space or {})
    if not search_space:
        if isinstance(request.methods, (list, tuple)) and len(request.methods) > 0:
            return None, _default_search_space(method=None, methods=request.methods)
        return method_for_search, _default_search_space(method=method_for_search, methods=None)
    if isinstance(request.methods, (list, tuple)) and len(request.methods) > 0:
        method_for_search = None
    return method_for_search, search_space


def _validate_tuning_methods(
    request: (
        ForecastTuneGeneticRequest
        | ForecastTuneOptunaRequest
        | ForecastOptimizeHintsRequest
    ),
) -> Optional[Dict[str, Any]]:
    request_methods = getattr(request, "methods", None)
    if isinstance(request_methods, (list, tuple)) and request_methods:
        requested = list(request_methods)
    else:
        default_method = getattr(request, "method", None)
        if default_method in (None, ""):
            return None
        requested = [default_method]
    methods = [str(method or "").strip() for method in requested if str(method or "").strip()]
    valid_methods = list(get_forecast_method_names())
    valid_lookup = {str(method).lower(): str(method) for method in valid_methods}
    for method in methods:
        if method.lower() in valid_lookup:
            continue
        return {
            "success": False,
            "error": format_invalid_method_error(method, valid_methods),
            "error_code": "unsupported_method",
            "method": method,
            "valid_methods_tool": "forecast_list_methods",
        }
    return None


def _validate_tuning_metric(metric: Any) -> Optional[Dict[str, Any]]:
    metric_value = str(metric or "").strip()
    metric_key = metric_value.lower()
    if metric_key in _TUNING_METRICS:
        return None
    suggestions = difflib.get_close_matches(metric_key, sorted(_TUNING_METRICS), n=3, cutoff=0.45)
    message = (
        f"Unsupported tuning metric: {metric_value or '<empty>'}. "
        f"Supported metrics: {', '.join(sorted(_TUNING_METRICS))}."
    )
    if suggestions:
        message += f" Did you mean: {', '.join(suggestions)}?"
    return {
        "success": False,
        "error": message,
        "error_code": "unsupported_metric",
        "metric": metric_value,
        "supported_metrics": sorted(_TUNING_METRICS),
    }


def _validate_tuning_sample(metric: Any, steps: int) -> Optional[Dict[str, Any]]:
    metric_key = str(metric or "").strip().lower()
    if (
        metric_key not in ANNUALIZED_TUNING_METRICS
        or int(steps) >= MIN_ANNUALIZED_TUNING_TRADES
    ):
        return None
    return {
        "success": False,
        "error": (
            f"Metric '{metric_key}' requires at least "
            f"{MIN_ANNUALIZED_TUNING_TRADES} successful trades, but steps={int(steps)} "
            "cannot produce that sample. Increase --steps before starting the search."
        ),
        "error_code": "insufficient_tuning_sample",
        "metric": metric_key,
        "steps": int(steps),
        "minimum_steps": MIN_ANNUALIZED_TUNING_TRADES,
        "remediation": f"Retry with --steps {MIN_ANNUALIZED_TUNING_TRADES} or greater.",
    }


def _attach_tuning_assumptions(
    result: Dict[str, Any],
    *,
    slippage_bps: float,
    trade_threshold: float,
) -> Dict[str, Any]:
    out = dict(result)
    out["cost_assumptions"] = {
        "score_basis": (
            "net_of_configured_slippage"
            if float(slippage_bps) > 0.0
            else "gross_before_execution_costs"
        ),
        "slippage_bps_per_side": float(slippage_bps),
        "trade_threshold": float(trade_threshold),
        "spread_and_commission": "not_modeled",
    }
    return out


def _attach_tuning_context(
    result: Dict[str, Any],
    request: ForecastTuneGeneticRequest | ForecastTuneOptunaRequest,
) -> Dict[str, Any]:
    """Echo the immutable evaluation identity on every tuning result."""
    out = dict(result)
    context: Dict[str, Any] = {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "quantity": request.quantity,
        "horizon": int(request.horizon),
        "steps": int(request.steps),
        "spacing": int(request.spacing),
        "methods": list(request.methods),
        "metric": str(request.metric),
        "seed": int(request.seed),
    }
    for key in ("as_of", "start", "end"):
        value = getattr(request, key, None)
        if value is not None:
            context[key] = value
    for key in ("symbol", "timeframe", "quantity", "horizon", "steps", "spacing"):
        out[key] = context[key]
    out["methods"] = list(context["methods"])
    out["tuning_context"] = context
    summary = out.get("best_result_summary")
    if isinstance(summary, dict):
        summary = dict(summary)
        summary["horizon"] = int(request.horizon)
        out["best_result_summary"] = summary
    return out


def _validate_tuning_param_spec(path: str, spec: Any) -> Optional[str]:
    if not isinstance(spec, dict):
        return f"{path} must be an object with type/min/max or choices."
    spec_type = str(spec.get("type", "float")).strip().lower()
    if spec_type not in {"int", "float", "categorical"}:
        return f"{path}.type must be int, float, or categorical."
    if spec_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, (list, tuple)) or len(choices) == 0:
            return f"{path}.choices must be a non-empty list."
        return None
    if "min" not in spec or "max" not in spec:
        return f"{path} must include min and max."
    try:
        lower = float(spec.get("min"))
        upper = float(spec.get("max"))
    except Exception:
        return f"{path}.min and {path}.max must be numeric."
    if upper < lower:
        return f"{path}.max must be >= min."
    if bool(spec.get("log", False)) and (lower <= 0.0 or upper <= 0.0):
        return f"{path}.log=true requires positive min and max."
    return None


def _validate_tuning_search_space(search_space: Any) -> Optional[Dict[str, Any]]:
    if search_space in (None, {}):
        return None
    if not isinstance(search_space, dict):
        return {
            "success": False,
            "error": "search_space must be an object mapping parameter names to specs.",
            "error_code": "invalid_search_space",
        }
    flat = any(
        isinstance(value, dict)
        and any(key in value for key in ("type", "min", "max", "choices"))
        for key, value in search_space.items()
        if key != "_method_spaces"
    )
    errors: List[str] = []
    if flat:
        for name, spec in search_space.items():
            if name == "_method_spaces":
                continue
            error = _validate_tuning_param_spec(str(name), spec)
            if error:
                errors.append(error)
    else:
        for method_name, method_space in search_space.items():
            if method_name == "_method_spaces":
                continue
            if not isinstance(method_space, dict):
                errors.append(f"{method_name} must map to a parameter-spec object.")
                continue
            for param_name, spec in method_space.items():
                error = _validate_tuning_param_spec(f"{method_name}.{param_name}", spec)
                if error:
                    errors.append(error)
    method_spaces = search_space.get("_method_spaces")
    if method_spaces is not None and not isinstance(method_spaces, dict):
        errors.append("_method_spaces must be an object.")
    elif isinstance(method_spaces, dict):
        for method_name, method_space in method_spaces.items():
            if not isinstance(method_space, dict):
                errors.append(f"_method_spaces.{method_name} must be an object.")
                continue
            for param_name, spec in method_space.items():
                error = _validate_tuning_param_spec(
                    f"_method_spaces.{method_name}.{param_name}",
                    spec,
                )
                if error:
                    errors.append(error)
    if not errors:
        return None
    return {
        "success": False,
        "error": "Invalid search_space: " + "; ".join(errors[:5]),
        "error_code": "invalid_search_space",
        "errors": errors[:10],
    }


def _validate_tuning_parameter_names(
    search_space: Dict[str, Any],
    methods: Iterable[str],
) -> Optional[Dict[str, Any]]:
    """Reject native tuner genes that the selected method cannot consume."""
    method_names = [str(method).strip() for method in methods if str(method).strip()]
    allowed_by_method: Dict[str, set[str]] = {}
    for method in method_names:
        # Library adapters validate their model-specific parameters at runtime.
        if method.startswith(("sf_", "skt_", "mlf_")) or method in {
            "statsforecast",
            "sktime",
            "mlforecast",
        }:
            continue
        try:
            params = getattr(ForecastRegistry.get_class(method), "PARAMS", ())
        except Exception:
            continue
        allowed_by_method[method] = {
            str(spec.get("name"))
            for spec in params
            if isinstance(spec, dict) and spec.get("name")
        }

    if not allowed_by_method:
        return None

    flat = any(
        isinstance(value, dict)
        and any(key in value for key in ("type", "min", "max", "choices"))
        for key, value in search_space.items()
        if key not in {"_method_spaces", "_shared"}
    )
    invalid: List[str] = []
    if flat:
        allowed_sets = list(allowed_by_method.values())
        allowed = set.intersection(*allowed_sets) if allowed_sets else set()
        invalid.extend(
            str(name)
            for name in search_space
            if name not in {"method", "_method_spaces", "_shared"}
            and name not in allowed
        )
    else:
        sections = dict(search_space)
        method_spaces = sections.pop("_method_spaces", None)
        if isinstance(method_spaces, dict):
            sections.update(method_spaces)
        shared = sections.pop("_shared", {})
        if isinstance(shared, dict):
            allowed_sets = list(allowed_by_method.values())
            shared_allowed = (
                set.intersection(*allowed_sets) if allowed_sets else set()
            )
            invalid.extend(
                f"_shared.{name}"
                for name in shared
                if name != "method" and name not in shared_allowed
            )
        for method, space in sections.items():
            allowed = allowed_by_method.get(str(method))
            if allowed is None or not isinstance(space, dict):
                continue
            invalid.extend(
                f"{method}.{name}"
                for name in space
                if name != "method" and name not in allowed
            )
    if not invalid:
        return None
    return {
        "success": False,
        "error": (
            "Invalid search_space parameter name(s): "
            + ", ".join(sorted(invalid))
            + ". Use forecast_list_methods to inspect the selected method's canonical parameters."
        ),
        "error_code": "invalid_search_space",
        "invalid_parameters": sorted(invalid),
    }


def _apply_tuning_detail(result: Dict[str, Any], detail: str) -> Dict[str, Any]:
    detail_value = _requested_detail_label(detail)
    out = dict(result)
    out["detail"] = detail_value
    if detail_value == "full":
        return out
    if "history_tail" in out:
        out["history_tail_count"] = len(out.get("history_tail") or [])
        out.pop("history_tail", None)
    if "best_result_summary" in out:
        summary = out.get("best_result_summary")
        if isinstance(summary, dict) and summary.get("horizon") is not None:
            out.setdefault("best_horizon", summary.get("horizon"))
        out["best_result_summary_omitted"] = "Use detail=full for nested backtest result details."
        out.pop("best_result_summary", None)
    return out


def run_forecast_tune_genetic(
    request: ForecastTuneGeneticRequest,
    *,
    genetic_search_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_tune_genetic",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    invalid_method = _validate_tuning_methods(request)
    if invalid_method is not None:
        result = _apply_tuning_detail(invalid_method, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_metric = _validate_tuning_metric(request.metric)
    if invalid_metric is not None:
        result = _apply_tuning_detail(invalid_metric, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_sample = _validate_tuning_sample(request.metric, request.steps)
    if invalid_sample is not None:
        result = _apply_tuning_detail(invalid_sample, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_search_space = _validate_tuning_search_space(request.search_space)
    if invalid_search_space is not None:
        result = _apply_tuning_detail(invalid_search_space, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    method_for_search, search_space = _resolve_tuning_search_space(request)
    invalid_parameter_names = (
        _validate_tuning_parameter_names(search_space, request.methods)
        if request.search_space
        else None
    )
    if invalid_parameter_names is not None:
        result = _attach_tuning_context(invalid_parameter_names, request)
        result = _attach_analysis_time_window(result, request)
        result = _apply_tuning_detail(result, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    try:
        result = genetic_search_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(method_for_search) if method_for_search is not None else None,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            quantity=str(request.quantity),
            **_analysis_time_kwargs(request),
            search_space=search_space,
            metric=str(request.metric),
            mode=resolve_tuning_mode(str(request.metric), str(request.mode)),
            population=int(request.population),
            generations=int(request.generations),
            crossover_rate=float(request.crossover_rate),
            mutation_rate=float(request.mutation_rate),
            seed=int(request.seed),
            max_search_time_seconds=(
                float(request.max_search_time_seconds)
                if request.max_search_time_seconds is not None
                else None
            ),
            slippage_bps=float(request.slippage_bps),
            trade_threshold=float(request.trade_threshold),
            denoise=request.denoise,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
        )
        raise
    result = _attach_tuning_assumptions(
        result,
        slippage_bps=request.slippage_bps,
        trade_threshold=request.trade_threshold,
    )
    result = _attach_tuning_context(result, request)
    result = _attach_analysis_time_window(result, request)
    result = _apply_tuning_detail(result, request.detail)
    log_operation_finish(
        logger,
        operation="forecast_tune_genetic",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    return result


def run_forecast_tune_optuna(
    request: ForecastTuneOptunaRequest,
    *,
    optuna_search_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_tune_optuna",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    invalid_method = _validate_tuning_methods(request)
    if invalid_method is not None:
        result = _apply_tuning_detail(invalid_method, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_metric = _validate_tuning_metric(request.metric)
    if invalid_metric is not None:
        result = _apply_tuning_detail(invalid_metric, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_sample = _validate_tuning_sample(request.metric, request.steps)
    if invalid_sample is not None:
        result = _apply_tuning_detail(invalid_sample, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_search_space = _validate_tuning_search_space(request.search_space)
    if invalid_search_space is not None:
        result = _apply_tuning_detail(invalid_search_space, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    method_for_search, search_space = _resolve_tuning_search_space(request)
    invalid_parameter_names = (
        _validate_tuning_parameter_names(search_space, request.methods)
        if request.search_space
        else None
    )
    if invalid_parameter_names is not None:
        result = _attach_tuning_context(invalid_parameter_names, request)
        result = _attach_analysis_time_window(result, request)
        result = _apply_tuning_detail(result, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    try:
        result = optuna_search_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(method_for_search) if method_for_search is not None else None,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            quantity=str(request.quantity),
            **_analysis_time_kwargs(request),
            search_space=search_space,
            metric=str(request.metric),
            mode=resolve_tuning_mode(str(request.metric), str(request.mode)),
            n_trials=int(request.n_trials),
            timeout=float(request.timeout) if request.timeout is not None else None,
            n_jobs=int(request.n_jobs),
            sampler=str(request.sampler),
            study_name=str(request.study_name) if request.study_name is not None else None,
            storage=str(request.storage) if request.storage is not None else None,
            seed=int(request.seed),
            slippage_bps=float(request.slippage_bps),
            trade_threshold=float(request.trade_threshold),
            denoise=request.denoise,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
        )
        raise
    result = _attach_tuning_assumptions(
        result,
        slippage_bps=request.slippage_bps,
        trade_threshold=request.trade_threshold,
    )
    result = _attach_tuning_context(result, request)
    result = _attach_analysis_time_window(result, request)
    result = _apply_tuning_detail(result, request.detail)
    log_operation_finish(
        logger,
        operation="forecast_tune_optuna",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    return result


def run_forecast_barrier_prob(
    request: ForecastBarrierProbRequest,
    *,
    build_barrier_kwargs: Any,
    normalize_trade_direction: Any,
    barrier_hit_probabilities_impl: Any,
    barrier_closed_form_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    method_val = normalize_barrier_method(
        request.method or "hmm_mc",
        allow_closed_form=True,
    )
    if method_val is None:
        method_val = str(request.method or "hmm_mc").lower().strip()
    mc_methods = {
        "auto",
        "bootstrap",
        "garch",
        "heston",
        "hmm_mc",
        "jump_diffusion",
        "mc_gbm",
        "mc_gbm_bb",
    }
    log_operation_start(
        logger,
        operation="forecast_barrier_prob",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )

    direction, direction_error = normalize_trade_direction(request.direction)
    if direction_error:
        result = {"error": direction_error}
        log_operation_finish(
            logger,
            operation="forecast_barrier_prob",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        return result

    try:
        if method_val in mc_methods:
            barrier_kwargs = build_barrier_kwargs(request.barrier_kwargs())
            has_resolved_barriers = any(
                barrier_kwargs.get(field_name) is not None
                for field_name in (
                    "tp_abs",
                    "sl_abs",
                    "tp_pct",
                    "sl_pct",
                    "tp_ticks",
                    "sl_ticks",
                )
            )
            if not has_resolved_barriers:
                result = build_error_payload(
                    (
                        "Barrier probabilities require an explicit take-profit and "
                        "stop-loss pair."
                    ),
                    code="barrier_parameters_missing",
                    operation="forecast_barrier_prob",
                    remediation=(
                        "Provide tp_pct/sl_pct, tp_abs/sl_abs, or tp_ticks/sl_ticks "
                        "scaled to the symbol and forecast horizon. Use "
                        "forecast_barrier_optimize for data-driven candidates."
                    ),
                    related_tools=[
                        "forecast_barrier_optimize",
                        "labels_triple_barrier",
                    ],
                )
                log_operation_finish(
                    logger,
                    operation="forecast_barrier_prob",
                    started_at=started_at,
                    success=False,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    method=method_val,
                    direction=request.direction,
                )
                return result
            result = barrier_hit_probabilities_impl(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                method=method_val,
                direction=direction,
                same_bar_policy=request.same_bar_policy,
                **barrier_kwargs,
                params=request.params,
                denoise=request.denoise,
                **_analysis_time_kwargs(request),
            )
            if isinstance(result, dict):
                result = _annotate_price_currency(result, request.symbol)
            result = _attach_analysis_time_window(result, request)
            result = _apply_barrier_prob_detail(result, request)
            log_operation_finish(
                logger,
                operation="forecast_barrier_prob",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                method=method_val,
                direction=direction,
            )
            return result

        if method_val == "closed_form":
            input_error = _closed_form_barrier_input_error(request)
            if input_error is not None:
                result = {"error": input_error, "error_code": "invalid_input"}
                log_operation_finish(
                    logger,
                    operation="forecast_barrier_prob",
                    started_at=started_at,
                    success=False,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    method=method_val,
                    direction=direction,
                )
                return result
            result = barrier_closed_form_impl(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                direction=direction,
                barrier=request.barrier_level,
                mu=request.mu,
                sigma=request.sigma,
                denoise=request.denoise,
                **_analysis_time_kwargs(request),
            )
            if isinstance(result, dict):
                result = _annotate_price_currency(result, request.symbol)
            result = _attach_analysis_time_window(result, request)
            result = _apply_barrier_prob_detail(result, request)
            log_operation_finish(
                logger,
                operation="forecast_barrier_prob",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                method=method_val,
                direction=direction,
            )
            return result
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_barrier_prob",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=direction,
        )
        raise

    result = {
        "error": barrier_method_error(request.method, allow_closed_form=True),
        "error_code": "unsupported_method",
    }
    log_operation_finish(
        logger,
        operation="forecast_barrier_prob",
        started_at=started_at,
        success=False,
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=direction,
    )
    return result


def run_forecast_barrier_optimize(
    request: ForecastBarrierOptimizeRequest,
    *,
    parse_kv_or_json: Any,
    barrier_optimize_impl: Any,
    cpu_count: Any = os.cpu_count,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    method_val = normalize_barrier_method(request.method or "auto", allow_ensemble=True)
    method_supported = method_val is not None
    if method_val is None:
        method_val = str(request.method or "auto").lower().strip()
    log_operation_start(
        logger,
        operation="forecast_barrier_optimize",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )
    if not method_supported:
        result = {
            "error": barrier_method_error(request.method, allow_ensemble=True),
            "error_code": "unsupported_method",
        }
        log_operation_finish(
            logger,
            operation="forecast_barrier_optimize",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        return result
    params_norm = parse_kv_or_json(request.params)
    if not isinstance(params_norm, dict):
        params_norm = {}
    params_norm["same_bar_policy"] = request.same_bar_policy
    for threshold_key in ("min_ev", "min_edge", "min_kelly"):
        threshold_value = getattr(request, threshold_key, None)
        if threshold_value is not None:
            params_norm[threshold_key] = threshold_value
    if bool(getattr(request, "tradable_only", False)):
        params_norm["tradable_only"] = True
    if str(params_norm.get("optimizer", "")).strip().lower() == "optuna":
        optuna_defaults = {
            "sampler": "tpe",
            "pruner": "median",
            "n_jobs": int((cpu_count() or 1)),
        }
        for key, value in optuna_defaults.items():
            if key not in params_norm:
                params_norm[key] = value

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value == "full":
        format_value = "full"
        concise_value = False
        return_grid_value = True
    elif detail_value == "standard":
        format_value = "summary"
        concise_value = False
        return_grid_value = True
    else:
        format_value = "summary"
        concise_value = True
        return_grid_value = False

    try:
        result = barrier_optimize_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=method_val,
            direction=request.direction,
            mode=request.mode,
            tp_min=0.25,
            tp_max=1.5,
            tp_steps=None,
            sl_min=0.25,
            sl_max=2.5,
            sl_steps=None,
            params=params_norm,
            denoise=request.denoise,
            **_analysis_time_kwargs(request),
            objective=request.objective,
            return_grid=return_grid_value,
            top_k=request.top_k,
            output_mode=format_value,
            viable_only=request.viable_only,
            concise=concise_value,
            grid_style=request.grid_style,
            preset=request.preset,
            vol_window=250,
            vol_min_mult=0.5,
            vol_max_mult=4.0,
            vol_steps=None,
            vol_sl_multiplier=1.8,
            vol_floor_pct=0.15,
            vol_floor_ticks=8.0,
            ratio_min=0.5,
            ratio_max=4.0,
            ratio_steps=None,
            refine=None,
            refine_radius=0.3,
            refine_steps=5,
            min_prob_win=None,
            max_prob_no_hit=None,
            max_median_time=None,
            fast_defaults=False,
            search_profile=request.search_profile,
            statistical_robustness=False,
            target_ci_width=0.05,
            n_seeds_stability=3,
            enable_bootstrap=False,
            n_bootstrap=200,
            enable_convergence_check=True,
            convergence_window=100,
            convergence_threshold=0.01,
            enable_power_analysis=False,
            power_effect_size=0.05,
            enable_sensitivity_analysis=False,
            sensitivity_params=None,
        )
        if isinstance(result, dict) and not result.get("error"):
            result = _with_reference_price_context(
                _round_barrier_optimize_payload(dict(result))
            )
            result["detail"] = detail_value
            result = _attach_analysis_time_window(result, request)
            if detail_value != "full":
                result.pop("last_price", None)
                result.pop("last_price_close", None)
                result.pop("last_price_source", None)
            _gate_barrier_optimize_live_readiness(result)
            if detail_value == "compact":
                result = _compact_barrier_optimize_payload(result)
            barrier_unit, barrier_mode = _barrier_optimize_unit_context(result)
            result.setdefault("barrier_unit", barrier_unit)
            result.setdefault("barrier_mode", barrier_mode)
            result.setdefault("probability_unit", "fraction")
            result.setdefault(
                "edge_definition",
                BARRIER_EDGE_DEFINITION,
            )
            result.setdefault(
                "ev_definition",
                "Expected value uses the optimizer objective and candidate barrier returns; "
                "probabilities are decimal fractions.",
            )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_barrier_optimize",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_barrier_optimize",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )
    return result


def run_forecast_volatility_estimate(
    request: ForecastVolatilityEstimateRequest,
    *,
    forecast_volatility_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_volatility_estimate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    try:
        result = forecast_volatility_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=request.method,
            proxy=request.proxy,
            params=request.params,
            as_of=request.as_of,
            start=request.start,
            end=request.end,
            denoise=request.denoise,
            detail=request.detail,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_volatility_estimate",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_volatility_estimate",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    return result


def run_forecast_optimize_hints(
    request: ForecastOptimizeHintsRequest,
    *,
    optimize_hints_impl: Any,
) -> Dict[str, Any]:
    """Run genetic search for optimal forecast settings across multiple dimensions.

    Searches across timeframes, methods, parameters, and optionally feature indicators
    to find top-N configurations ranked by composite fitness score.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_optimize_hints",
        symbol=request.symbol,
        timeframe=request.timeframe,
        methods=len(request.methods or []),
    )

    # Resolve timeframes to search
    timeframes_to_search = request.timeframes
    if not timeframes_to_search and request.timeframe:
        timeframes_to_search = [request.timeframe]
    if not timeframes_to_search:
        timeframes_to_search = ['H1', 'H4', 'D1', 'W1']

    invalid_method = _validate_tuning_methods(request)
    if invalid_method is not None:
        return _apply_tuning_detail(invalid_method, request.detail)

    invalid_sample = _validate_tuning_sample(request.fitness_metric, request.steps)
    if invalid_sample is not None:
        return _apply_tuning_detail(invalid_sample, request.detail)

    try:
        result = optimize_hints_impl(
            symbol=request.symbol,
            timeframes=timeframes_to_search,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            **_analysis_time_kwargs(request),
            fitness_metric=str(request.fitness_metric or 'composite'),
            fitness_weights=request.fitness_weights,
            population=int(request.population),
            generations=int(request.generations),
            crossover_rate=float(request.crossover_rate),
            mutation_rate=float(request.mutation_rate),
            seed=int(request.seed),
            max_search_time_seconds=float(request.max_search_time_seconds)
            if request.max_search_time_seconds is not None
            else None,
            slippage_bps=float(request.slippage_bps),
            trade_threshold=float(request.trade_threshold),
            denoise=request.denoise,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            top_n=int(request.top_n),
            include_feature_genes=bool(request.include_feature_genes),
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_optimize_hints",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
        )
        raise
    result = _attach_tuning_assumptions(
        result,
        slippage_bps=request.slippage_bps,
        trade_threshold=request.trade_threshold,
    )
    result = _attach_analysis_time_window(result, request)
    result = _apply_tuning_detail(result, request.detail)
    log_operation_finish(
        logger,
        operation="forecast_optimize_hints",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        methods=len(request.methods or []),
    )
    return result
