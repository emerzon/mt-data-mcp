from __future__ import annotations

import difflib
import importlib
import logging
import math
import os
import pkgutil
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from ..core.output_contract import attach_collection_contract
from ..utils.coercion import coerce_finite_float as _finite_float
from ..utils.coercion import is_explicit_false as _is_explicit_false
from ..utils.freshness import format_age_seconds as _format_age_seconds
from ..utils.freshness import format_freshness_label
from .backtest import execute_forecast_backtest as _forecast_backtest_impl
from .barriers_shared import barrier_method_error, normalize_barrier_method
from .capabilities import resolve_capability_request
from .exceptions import ForecastError, raise_if_error_result
from .forecast import execute_forecast as _forecast_impl
from .forecast_methods import get_forecast_method_names
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

logger = logging.getLogger(__name__)

_BACKTEST_METRICS_REASON_NOTES = {
    "no_non_flat_trades": (
        "No active long/short trades; win_rate and drawdown need at least one trade."
    ),
}
_TUNING_METRICS = frozenset(
    {
        "avg_rmse",
        "avg_mae",
        "avg_directional_accuracy",
        "win_rate",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "annual_return",
        "avg_return_per_trade",
        "avg_win_loss_ratio",
        "kelly_fraction",
        "half_kelly_fraction",
    }
)
_VOLATILITY_PROXY_METHODS = {"arima", "sarima", "ets", "theta"}
_PRETRAINED_FORECAST_METHODS = ("chronos2", "chronos_bolt", "timesfm")
_DEFAULT_VOLATILITY_PROXY = "squared_return"
_FORECAST_DIRECTION_NEUTRAL_THRESHOLD_PCT = 0.01
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
    out = dict(payload)
    for key in (
        "last_observation_time",
        "forecast_from",
        "forecast_start_time",
    ):
        if key in out:
            out[key] = _format_forecast_time_utc(out.get(key))
    value = out.get("forecast_time")
    if isinstance(value, list):
        out["forecast_time"] = [_format_forecast_time_utc(item) for item in value]
    elif value not in (None, ""):
        out["forecast_time"] = _format_forecast_time_utc(value)
    if any(key in out for key in ("last_observation_time", "forecast_time")):
        out.setdefault("timezone", "UTC")
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
    symbol_text = str(symbol or "").strip()
    if not symbol_text:
        return None
    try:
        from ..utils.mt5 import get_symbol_info_cached

        info = get_symbol_info_cached(symbol_text)
    except Exception:
        return None
    for attr in ("currency_profit", "currency_margin"):
        try:
            value = getattr(info, attr, None)
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
    count = min(len(lower_vals), len(upper_vals))
    if isinstance(forecasts, list):
        count = min(count, len(forecasts))
    intervals: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {}
        if isinstance(times, list) and idx < len(times):
            row["time"] = times[idx]
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
    numeric = _finite_float(value)
    if numeric is None:
        return value
    return float(round(numeric, max(0, int(digits))))


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
        "mathematically_viable",
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
    if horizon_delta_pct is not None and abs(horizon_delta_pct) <= _FORECAST_DIRECTION_NEUTRAL_THRESHOLD_PCT:
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
        "direction_threshold_pct": _FORECAST_DIRECTION_NEUTRAL_THRESHOLD_PCT,
        "first_step_delta": float(round(first_delta, delta_digits)),
        "horizon_delta": float(round(horizon_delta, delta_digits)),
    }
    if first_delta_pct is not None and horizon_delta_pct is not None:
        out["first_step_delta_pct"] = float(round(first_delta_pct, 4))
        out["horizon_delta_pct"] = float(round(horizon_delta_pct, 4))
    return out


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
    path_flatness = _forecast_path_flatness(out)
    price_context = _forecast_vs_last_price(out)
    if price_context:
        if path_flatness:
            price_context["direction"] = "neutral"
            price_context["direction_basis"] = "flat_path"
            price_context["direction_suppressed_reason"] = "flat_path"
        out.setdefault("forecast_vs_last_price", price_context)
    if path_flatness:
        out.update(path_flatness)
        out.setdefault("point_forecast_mode", "flat_model_path")
        out["forecast_status"] = "non_informative"
        out["signal_status"] = "not_actionable"
        _append_forecast_warning(out, _FORECAST_FLAT_PATH_WARNING)
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
    out: Dict[str, Any] = {
        "last_observation": last_observation,
        "last_bar_complete": True,
        "input_bar_policy": "closed_bars_only",
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
    ):
        value = payload.get(source_key)
        if value not in (None, "", [], {}):
            out[target_key] = value
    age_seconds = payload.get("last_price_age_seconds")
    if age_seconds not in (None, "", [], {}):
        out["last_observation_age_seconds"] = age_seconds
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
    market_status = payload.get("forecast_market_status")

    count = min(len(times), len(forecast_values))
    price_values = payload.get("forecast_price")
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {"time": _format_forecast_time_utc(times[idx])}
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
                row["lower"] = lower_values[idx]
                row["upper"] = upper_values[idx]
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


def _apply_forecast_generate_detail(
    payload: Dict[str, Any],
    request: ForecastGenerateRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
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
    if forecast_rows and not ci_has_intervals:
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
        ("lookback_bars_fetched", "lookback_bars_fetched"),
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
        "confidence_level",
        "ci_status",
        "ci_available",
        "ci_warning",
        "last_price",
        "last_price_source",
        "warnings",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
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
            "prob_hit",
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
        "direction",
        "horizon",
        "reference_price",
        "reference_price_source",
        "tp_price",
        "sl_price",
        "prob_tp_first",
        "prob_sl_first",
        "prob_no_hit",
        "probability_edge",
    ):
        _set_if_present(compact, key, payload.get(key))
    confidence: Dict[str, Any] = {}
    for key in (
        "prob_tp_first_ci95",
        "prob_sl_first_ci95",
        "prob_no_hit_ci95",
        "prob_tp_first_se",
        "prob_sl_first_se",
        "prob_no_hit_se",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            confidence[key] = value
    if confidence:
        compact["confidence"] = confidence
    timing: Dict[str, Any] = {}
    for source_key, target_key in (
        ("time_to_tp_bars", "tp"),
        ("time_to_sl_bars", "sl"),
    ):
        value = payload.get(source_key)
        if isinstance(value, dict) and any(val not in (None, "") for val in value.values()):
            timing[target_key] = {
                key: value.get(key)
                for key in ("mean", "median")
                if value.get(key) not in (None, "")
            }
    if timing:
        compact["timing_bars"] = timing
    if payload.get("warnings") not in (None, "", [], {}):
        compact["warnings"] = payload.get("warnings")
    for key, value in payload.items():
        if key in compact:
            continue
        if key in {
            "prob_tp_first_ci95",
            "prob_tp_first_se",
            "prob_sl_first_ci95",
            "prob_sl_first_se",
            "prob_no_hit_ci95",
            "prob_same_bar",
            "prob_same_bar_se",
            "prob_no_hit_se",
            "prob_same_bar_ci95",
            "last_price",
            "last_price_close",
            "last_price_source",
            "tp_hit_prob_by_t",
            "sl_hit_prob_by_t",
            "time_to_tp_bars",
            "time_to_sl_bars",
            "sim_meta",
            "model_summary",
        }:
            continue
        compact[key] = value
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
        out.setdefault("verdict", verdict)
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
            units[key] = "percentage_points"
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
        barrier_value = float(request.barrier)
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


def _compact_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:
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

    ranked_methods: list[Dict[str, Any]] = []
    methods_total = 0
    methods_failed: list[str] = []
    for method_name, method_payload in raw_results.items():
        methods_total += 1
        if not isinstance(method_payload, dict):
            ranked_methods.append({"method": method_name, "result": method_payload})
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
        ranked_row["_sort_metric"] = _sort_metric(
            method_payload.get("avg_rmse", method_payload.get("avg_mae"))
        )
        ranked_methods.append(ranked_row)

    compact_out = dict(result)
    compact_out.pop("results", None)
    compact_out.pop("request", None)
    compact_out.pop("resolved_request", None)
    compact_out.pop("detail", None)
    if isinstance(compact_out.pop("units", None), dict):
        compact_out["units_profile"] = "forecast_backtest_v1"
    if compact_out.get("slippage_bps") in (0, 0.0, None):
        compact_out.pop("slippage_bps", None)
    if compact_out.get("trade_threshold") in (0, 0.0, None):
        compact_out.pop("trade_threshold", None)
    compact_out["methods_total"] = methods_total
    compact_out["methods_succeeded"] = methods_total - len(methods_failed)
    compact_out["methods_failed"] = len(methods_failed)
    if methods_failed:
        compact_out["failed_methods"] = methods_failed
    ranked_methods.sort(
        key=lambda row: (
            row.get("_sort_metric") is None,
            row.get("_sort_metric") if row.get("_sort_metric") is not None else 0.0,
            str(row.get("method") or ""),
        )
    )
    compact_out["ranked_methods"] = [
        {key: value for key, value in row.items() if key != "_sort_metric" and value is not None}
        for row in ranked_methods
    ]
    return compact_out


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
            key = name.lower()
            if key not in mapping:
                mapping[key] = (name, f"{obj.__module__}.{name}")
    return mapping


def _normalize_forecaster_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


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


def run_forecast_generate(
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
            ci_alpha=request.ci_alpha,
            quantity=request.quantity,
            proxy=proxy_value,
            denoise=request.denoise,
            features=request.features or {},
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            target_spec=request.target_spec,
            async_mode=getattr(request, 'async_mode', False),
            model_id=getattr(request, 'model_id', None),
        )
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
                "Using native theta. StatsForecast theta is available via "
                "and may produce different forecasts/interval behavior."
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
    requested_detail = _requested_detail_label(request.detail)
    if str(request.detail or "compact").strip().lower() == "compact":
        return _compact_backtest_result(result)
    if isinstance(result, dict) and not result.get("error"):
        result = dict(result)
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
            params=request.params,
            denoise=request.denoise,
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
        result["confidence_level"] = round(1.0 - float(request.ci_alpha), 6)
        result["ci_status"] = "available"
        result["ci_available"] = True
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
        if min_calibration_points < 30:
            sample_warning = (
                "Residual-quantile calibration has as few as "
                f"{min_calibration_points} residual(s) per forecast step; "
                "tail quantiles and empirical coverage may be unstable below 30. "
                "These bands are not true conformal prediction intervals."
            )
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if sample_warning not in warnings_list:
                warnings_list.append(sample_warning)
            result["warnings"] = warnings_list
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
    request: ForecastTuneGeneticRequest | ForecastTuneOptunaRequest,
) -> Optional[Dict[str, Any]]:
    requested = (
        list(request.methods)
        if isinstance(request.methods, (list, tuple)) and len(request.methods) > 0
        else [request.method]
    )
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
    try:
        result = genetic_search_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(method_for_search) if method_for_search is not None else None,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            search_space=search_space,
            metric=str(request.metric),
            mode=str(request.mode),
            population=int(request.population),
            generations=int(request.generations),
            crossover_rate=float(request.crossover_rate),
            mutation_rate=float(request.mutation_rate),
            seed=int(request.seed),
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
    try:
        result = optuna_search_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(method_for_search) if method_for_search is not None else None,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            search_space=search_space,
            metric=str(request.metric),
            mode=str(request.mode),
            n_trials=int(request.n_trials),
            timeout=float(request.timeout) if request.timeout is not None else None,
            n_jobs=int(request.n_jobs),
            sampler=str(request.sampler),
            pruner=str(request.pruner),
            study_name=str(request.study_name) if request.study_name is not None else None,
            storage=str(request.storage) if request.storage is not None else None,
            seed=int(request.seed),
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
            barrier_kwargs = build_barrier_kwargs(request.model_dump())
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
                result = {
                    "success": False,
                    "error": (
                        "Barrier probabilities require an explicit take-profit and "
                        "stop-loss pair."
                    ),
                    "error_code": "barrier_parameters_missing",
                    "operation": "forecast_barrier_prob",
                    "remediation": (
                        "Provide tp_pct/sl_pct, tp_abs/sl_abs, or tp_ticks/sl_ticks "
                        "scaled to the symbol and forecast horizon. Use "
                        "forecast_barrier_optimize for data-driven candidates."
                    ),
                    "related_tools": ["forecast_barrier_optimize", "labels_triple_barrier"],
                }
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
            )
            if isinstance(result, dict):
                result = _annotate_price_currency(result, request.symbol)
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
                barrier=request.barrier,
                mu=request.mu,
                sigma=request.sigma,
                denoise=request.denoise,
            )
            if isinstance(result, dict):
                result = _annotate_price_currency(result, request.symbol)
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
            if detail_value != "full":
                result.pop("last_price", None)
                result.pop("last_price_close", None)
                result.pop("last_price_source", None)
            if detail_value == "compact":
                result = _compact_barrier_optimize_payload(result)
            barrier_unit, barrier_mode = _barrier_optimize_unit_context(result)
            result.setdefault("barrier_unit", barrier_unit)
            result.setdefault("barrier_mode", barrier_mode)
            result.setdefault("probability_unit", "fraction")
            result.setdefault(
                "edge_definition",
                "Expected reward/risk edge for the candidate TP/SL barrier pair.",
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
        timeframes_to_search = ['H1', 'H4', 'D1']

    try:
        result = optimize_hints_impl(
            symbol=request.symbol,
            timeframes=timeframes_to_search,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
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
