from __future__ import annotations

import difflib
import importlib
import logging
import math
import os
import pkgutil
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from ..core.output_contract import attach_collection_contract
from .backtest import execute_forecast_backtest as _forecast_backtest_impl
from .barriers_shared import barrier_method_error, normalize_barrier_method
from .capabilities import resolve_capability_request
from .exceptions import ForecastError, raise_if_error_result
from .forecast import execute_forecast as _forecast_impl
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
_VOLATILITY_PROXY_METHODS = {"arima", "sarima", "ets", "theta"}
_DEFAULT_VOLATILITY_PROXY = "squared_return"


def _normalize_trader_detail(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized in {"summary", "summary_only"}:
        return "compact"
    if normalized == "full":
        return "full"
    if normalized == "standard":
        return "standard"
    return "compact"


def _requested_detail_label(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized == "summary_only":
        return "summary"
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
            for lower, upper in zip(lower_vals, upper_vals)
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
        return None

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

    out = {"status": ci_status or "available"}
    if payload.get("ci_alpha") is not None:
        out["alpha"] = payload.get("ci_alpha")
    if intervals:
        out["intervals"] = intervals
    summary = _forecast_interval_summary(payload)
    if summary:
        out["summary"] = summary
    return out


def _strip_volatility_impl_aliases(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    for key in (
        "sigma_bar_return",
        "sigma_annual_return",
        "horizon_sigma_return",
        "horizon_sigma_annual",
    ):
        out.pop(key, None)
    return out


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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
        "prob_tie",
        "prob_no_hit",
        "edge",
        "prob_tp_first_se",
        "prob_sl_first_se",
        "prob_tie_se",
        "prob_no_hit_se",
    ):
        if key in out:
            out[key] = _round_barrier_value(out.get(key), digits=6)
    for key in (
        "prob_tp_first_ci95",
        "prob_sl_first_ci95",
        "prob_tie_ci95",
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
    "prob_tie": 6,
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


def _forecast_vs_last_price(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    last_price = _finite_float(payload.get("last_price"))
    prices = payload.get("forecast_price")
    if last_price is None or not isinstance(prices, list) or not prices:
        return None
    first_forecast = _finite_float(prices[0])
    if first_forecast is None:
        return None
    delta = first_forecast - last_price
    digits = _forecast_price_digits(payload)
    delta_digits = digits if digits is not None else 6
    out: Dict[str, float] = {
        "first_forecast_delta": float(round(delta, delta_digits)),
    }
    if last_price:
        out["first_forecast_delta_pct"] = float(round(delta / last_price * 100.0, 4))
    last_forecast = _finite_float(prices[-1])
    if last_forecast is not None and last_forecast != first_forecast:
        out["last_forecast_delta"] = float(round(last_forecast - last_price, delta_digits))
    return out


def _forecast_generate_compact_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = payload.get("forecast_time")
    if not isinstance(times, list):
        return []

    forecast_values = None
    for key in ("forecast_price", "forecast_return", "forecast"):
        value = payload.get(key)
        if isinstance(value, list):
            forecast_values = value
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
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {
            "time": times[idx],
            "value": forecast_values[idx],
        }
        if isinstance(market_status, list) and idx < len(market_status):
            row["market_status"] = market_status[idx]
        if isinstance(lower_values, list) and isinstance(upper_values, list):
            if idx < len(lower_values) and idx < len(upper_values):
                row["lower"] = lower_values[idx]
                row["upper"] = upper_values[idx]
        rows.append(row)
    return rows


def _theta_flatness_context(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if str(payload.get("method") or "").strip().lower() != "theta":
        return None
    params_used = payload.get("params_used")
    if not isinstance(params_used, dict):
        return None
    trend_slope = _finite_float(params_used.get("trend_slope"))
    if trend_slope is None or trend_slope == 0.0:
        return None
    prices = payload.get("forecast_price")
    if not isinstance(prices, list) or len(prices) < 2:
        return None
    finite_prices = [_finite_float(value) for value in prices]
    if any(value is None for value in finite_prices):
        return None
    price_values = [float(value) for value in finite_prices if value is not None]
    drift_per_step = 0.5 * trend_slope
    digits = _forecast_price_digits(payload)
    drift_digits = max(8, (digits if digits is not None else 6) + 3)
    return {
        "target_drift_per_step": float(round(drift_per_step, drift_digits)),
        "appears_flat_at_price_precision": len(set(price_values)) == 1,
    }


def _apply_forecast_generate_detail(
    payload: Dict[str, Any],
    request: ForecastGenerateRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    payload = _round_forecast_generate_payload(payload)
    training_period = _forecast_training_period(payload)

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value in {"standard", "full"}:
        out = dict(payload)
        out.pop("ci_available", None)
        out.setdefault("symbol", request.symbol)
        out.setdefault("timeframe", request.timeframe)
        if training_period:
            out.setdefault("training_period", training_period)
        out["detail"] = detail_value
        return attach_collection_contract(
            out,
            collection_kind="time_series",
            series=_forecast_generate_series_rows(out),
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
        compact["ci"] = ci_compact
    ci_warning_dedup = (
        isinstance(ci_compact, dict)
        and str(ci_compact.get("status") or "").strip().lower() == "unavailable"
    )
    for key in (
        "last_observation_time",
        "timezone",
        "forecast_time",
        "forecast_price",
        "forecast_return",
        "last_price",
        "last_price_source",
        "last_price_age_seconds",
        "last_price_age",
        "last_price_stale",
        "freshness_basis",
        "stale_after_seconds",
        "stale_warning",
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
    price_context = _forecast_vs_last_price(payload)
    if price_context:
        compact["forecast_vs_last_price"] = price_context
    forecast_rows = _forecast_generate_compact_rows(payload)
    if forecast_rows:
        compact["forecast"] = forecast_rows
    theta_context = _theta_flatness_context(payload)
    if theta_context and theta_context.get("appears_flat_at_price_precision"):
        warning = "Native theta forecast is near-flat at displayed price precision."
        warnings_out = compact.get("warnings")
        if not isinstance(warnings_out, list):
            warnings_out = []
        if warning not in warnings_out:
            warnings_out.append(warning)
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
            "forecast_anchor",
            "forecast_step_seconds",
            "forecast_epoch",
            "last_price_close",
            "lower_price",
            "upper_price",
            "lower_return",
            "upper_return",
            "lower",
            "upper",
            "ci",
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
        compact[key] = value
    return compact


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
        "edge",
    ):
        _set_if_present(compact, key, payload.get(key))
    confidence: Dict[str, Any] = {}
    for key in ("prob_tp_first_ci95", "prob_sl_first_ci95", "prob_no_hit_ci95"):
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
            "prob_tie",
            "prob_tie_se",
            "prob_no_hit_se",
            "prob_tie_ci95",
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
    elif out.get("tp_ticks") is not None or out.get("sl_ticks") is not None:
        out.setdefault("barrier_unit", "ticks")
    elif out.get("tp_abs") is not None or out.get("sl_abs") is not None or out.get("barrier") is not None:
        out.setdefault("barrier_unit", "price")
    out.setdefault("probability_unit", "fraction")
    out.setdefault("edge_definition", "prob_tp_first - prob_sl_first")
    return out


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


def _append_default_barrier_warning(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    warning = (
        "Default 1% symmetrical barriers applied; pass tp_pct/sl_pct, "
        "tp_abs/sl_abs, or tp_ticks/sl_ticks to customize."
    )
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        if warning not in warnings:
            warnings.append(warning)
    elif warnings in (None, "", [], {}):
        payload["warnings"] = [warning]
    else:
        payload["warnings"] = [warnings, warning]
    return payload


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
        "max_drawdown": 4,
        "avg_return": 6,
        "avg_return_per_trade": 6,
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

    def _is_explicit_false(value: Any) -> bool:
        if value is None:
            return False
        try:
            return bool(value) is False
        except Exception:
            return False

    ranked_methods: list[Dict[str, Any]] = []
    for method_name, method_payload in raw_results.items():
        if not isinstance(method_payload, dict):
            ranked_methods.append({"method": method_name, "result": method_payload})
            continue
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
            for key in (
                "win_rate",
                "max_drawdown",
                "avg_return",
                "avg_return_per_trade",
                "trades_observed",
            ):
                if key in metrics:
                    method_out[key] = _compact_metric(key, metrics[key])
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
    if compact_out.get("slippage_bps") in (0, 0.0, None):
        compact_out.pop("slippage_bps", None)
    if compact_out.get("trade_threshold") in (0, 0.0, None):
        compact_out.pop("trade_threshold", None)
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
            resolved_method = method or "chronos2"
        elif lib == "mlforecast":
            if not method:
                raise ForecastError("method is required for library=mlforecast")
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
            warning = (
                "Using native theta. StatsForecast theta is available via "
                f"`mtdata-cli forecast_generate {request.symbol} --timeframe {request.timeframe} "
                f"--library statsforecast --method Theta --horizon {request.horizon}` "
                "and may produce different forecasts/interval behavior."
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
            raise ForecastError("Conformal calibration failed: no backtest details")

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
            raise ForecastError("Empty point forecast for conformal intervals")
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
        result["conformal"] = {
            "ci_alpha": float(request.ci_alpha),
            "calibration_steps": int(request.steps),
            "calibration_spacing": int(request.spacing),
            "per_step_q": [float(v) for v in qerrs],
        }
        result["lower_price"] = [float(v) for v in lo.tolist()]
        result["upper_price"] = [float(v) for v in hi.tolist()]
        result["ci_alpha"] = float(request.ci_alpha)
        result["confidence_level"] = round(1.0 - float(request.ci_alpha), 6)
        result["ci_status"] = "available"
        result["ci_available"] = True
        warnings_out = result.get("warnings")
        if isinstance(warnings_out, list):
            filtered_warnings = [
                item for item in warnings_out if not _is_interval_unavailable_warning(item)
            ]
            if filtered_warnings:
                result["warnings"] = filtered_warnings
            else:
                result.pop("warnings", None)
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


def _apply_tuning_detail(result: Dict[str, Any], detail: str) -> Dict[str, Any]:
    detail_value = _requested_detail_label(detail)
    out = dict(result)
    out["detail"] = detail_value
    if detail_value == "full":
        return out
    if "history_tail" in out:
        out["history_tail_count"] = len(out.get("history_tail") or [])
        out.pop("history_tail", None)
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
            default_barriers_applied = False
            if not _request_has_barrier_inputs(request):
                barrier_kwargs = {
                    **barrier_kwargs,
                    "tp_pct": 1.0,
                    "sl_pct": 1.0,
                }
                default_barriers_applied = True
            result = barrier_hit_probabilities_impl(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                method=method_val,
                direction=direction,
                **barrier_kwargs,
                params=request.params,
                denoise=request.denoise,
            )
            if default_barriers_applied:
                result = _append_default_barrier_warning(result)
                if isinstance(result, dict) and not result.get("error"):
                    result.setdefault("tp_pct", 1.0)
                    result.setdefault("sl_pct", 1.0)
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
            result.setdefault("barrier_unit", "percent")
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
    return _strip_volatility_impl_aliases(result)


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
