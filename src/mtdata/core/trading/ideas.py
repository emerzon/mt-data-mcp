"""Preview-only trade-idea composer.

Composes session context, forecast, volatility, barriers, optional confluence,
sizing, and a forced dry-run ``trade_place`` into one ``TradeIdea`` artifact.
This module never sends a live order.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ...forecast.requests import MAX_FORECAST_HORIZON
from ...shared.schema import normalize_required_symbol
from ...utils.barriers import barrier_prices_are_valid
from ...utils.coercion import coerce_finite_float
from ...utils.time import format_datetime_utc
from .._mcp_instance import mcp
from ..error_envelope import build_error_payload
from ..execution_logging import run_logged_operation
from ..output_contract import attach_success_guidance
from ..runtime_metadata import attach_mt5_source
from ..tool_calling import call_tool_sync_structured
from .ideas_requests import (
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    TradeIdeaComposeRequest,
)

logger = logging.getLogger(__name__)

SectionCaller = Callable[[str, Dict[str, Any]], Any]

_QUICK_SECTIONS = (
    "session",
    "forecast",
    "volatility",
    "barriers",
    "sizing",
    "preview",
)
_STANDARD_SECTIONS = (
    "session",
    "confluence",
    "forecast",
    "volatility",
    "barriers",
    "sizing",
    "preview",
)
_HISTORICAL_SKIP = frozenset({"session", "sizing", "preview"})
_SNAP_DISTANCE_FRACTION = 0.25
_COMPACT_KEYS = (
    "success",
    "symbol",
    "timeframe",
    "horizon",
    "template",
    "as_of",
    "assembled_at",
    "timezone",
    "direction",
    "direction_basis",
    "suggested_direction",
    "actionability",
    "narrative",
    "quote",
    "structure",
    "forecast",
    "volatility",
    "barriers",
    "geometry",
    "sizing",
    "gates",
    "preview",
    "partial_failure",
    "failed_sections",
    "section_errors",
    "warnings",
    "error",
    "error_code",
    "remediation",
    "related_tools",
    "source",
)


def _as_float(value: Any) -> Optional[float]:
    return coerce_finite_float(value)


def _section_failed(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return True
    if payload.get("error"):
        return True
    return payload.get("success") is False


def _section_error_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "section failed"
    error = payload.get("error")
    if error not in (None, ""):
        return str(error)
    message = payload.get("message") or payload.get("details")
    return str(message) if message not in (None, "") else "section failed"


def _forecast_values(payload: Any) -> List[float]:
    if not isinstance(payload, dict):
        return []
    for key in ("forecast_price", "forecast", "forecast_series", "values", "predictions"):
        raw = payload.get(key)
        if not isinstance(raw, list):
            continue
        values: List[float] = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("value") or item.get("forecast_price")
            number = _as_float(item)
            if number is not None:
                values.append(number)
        if values:
            return values
    return []


def _forecast_trend(values: List[float]) -> Optional[str]:
    if len(values) < 2:
        return None
    first = values[0]
    last = values[-1]
    if last > first:
        return "up"
    if last < first:
        return "down"
    return "flat"


def _forecast_direction(payload: Any) -> tuple[Optional[str], str]:
    if not isinstance(payload, dict):
        return None, "forecast direction metadata is unavailable"
    context = payload.get("forecast_vs_last_price")
    if not isinstance(context, dict):
        return None, "forecast direction metadata is unavailable"
    if context.get("direction_actionable") is not True:
        reason = str(
            context.get("direction_suppressed_reason")
            or context.get("direction_status")
            or "forecast direction is neutral or unconfirmed"
        ).replace("_", " ")
        return None, reason
    direction = str(context.get("direction") or "").strip().lower()
    if direction == "bullish":
        return "long", ""
    if direction == "bearish":
        return "short", ""
    return None, "forecast direction is neutral or unconfirmed"


def _gate(status: str, reason: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": status}
    if reason:
        payload["reason"] = reason
    return payload


def _looks_like_invalid_symbol(message: str, symbol: str) -> bool:
    text = str(message or "").lower()
    if "symbol" not in text:
        return False
    symbol_text = str(symbol or "").strip().lower()
    if symbol_text and symbol_text not in text:
        return False
    return any(
        phrase in text
        for phrase in ("not found", "not available", "unavailable", "unknown symbol")
    )


def _extract_quote(session: Any) -> Dict[str, Any]:
    if not isinstance(session, dict):
        return {}
    quote = session.get("quote")
    if not isinstance(quote, dict):
        quote = session if any(key in session for key in ("bid", "ask", "mid")) else {}
    keys = (
        "symbol",
        "bid",
        "ask",
        "mid",
        "last",
        "spread",
        "spread_pips",
        "spread_quality",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "data_stale",
        "data_age_seconds",
        "freshness_state",
        "execution_blockers",
        "quote_not_live_ready",
        "time",
        "time_epoch",
    )
    compact = {key: quote[key] for key in keys if key in quote}
    if session.get("is_tradable") is not None:
        compact["is_tradable"] = session.get("is_tradable")
    if session.get("can_open_new_positions") is not None:
        compact["can_open_new_positions"] = session.get("can_open_new_positions")
    if session.get("market_status") not in (None, ""):
        compact["market_status"] = session.get("market_status")
    if session.get("market_status_reason") not in (None, ""):
        compact["market_status_reason"] = session.get("market_status_reason")
    blockers = compact.get("execution_blockers")
    not_live = (
        compact.get("usable_for_live_trading") is False
        or compact.get("data_stale") is True
        or compact.get("quote_not_live_ready") is True
        or (isinstance(blockers, list) and bool(blockers))
    )
    compact["quote_not_live_ready"] = bool(not_live)
    return compact


def _reference_price(quote: Dict[str, Any], direction: Optional[str]) -> Optional[float]:
    if direction == "long":
        return _as_float(quote.get("ask")) or _as_float(quote.get("mid"))
    if direction == "short":
        return _as_float(quote.get("bid")) or _as_float(quote.get("mid"))
    return (
        _as_float(quote.get("mid"))
        or _as_float(quote.get("last"))
        or _as_float(quote.get("ask"))
        or _as_float(quote.get("bid"))
    )


def _session_tradable(session: Any, quote: Dict[str, Any]) -> bool:
    if quote.get("can_open_new_positions") is False:
        return False
    if quote.get("is_tradable") is False:
        return False
    if isinstance(session, dict):
        if session.get("can_open_new_positions") is False:
            return False
        if session.get("is_tradable") is False:
            return False
        trade_ready = session.get("trade_ready")
        if isinstance(trade_ready, dict) and trade_ready.get("can_open_new_positions") is False:
            return False
    return True


def _compact_structure(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("levels")
    if not isinstance(rows, list):
        return []
    compact: List[Dict[str, Any]] = []
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        price = _as_float(row.get("price") if "price" in row else row.get("value"))
        if price is None:
            continue
        item: Dict[str, Any] = {"price": price}
        for key in ("type", "score", "source_families"):
            if row.get(key) not in (None, "", []):
                item[key] = row[key]
        range_payload = row.get("range")
        if isinstance(range_payload, dict):
            compact_range = {
                name: _as_float(range_payload.get(name))
                for name in ("low", "high", "width")
                if _as_float(range_payload.get(name)) is not None
            }
            if compact_range:
                item["range"] = compact_range
        compact.append(item)
    return compact


def _prices_from_percent(
    *,
    entry: float,
    direction: str,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[Optional[float], Optional[float]]:
    tp_frac = abs(float(take_profit_pct)) / 100.0
    sl_frac = abs(float(stop_loss_pct)) / 100.0
    if direction == "long":
        return entry * (1.0 + tp_frac), entry * (1.0 - sl_frac)
    if direction == "short":
        return entry * (1.0 - tp_frac), entry * (1.0 + sl_frac)
    return None, None


def _barrier_prices(payload: Any, *, entry: Optional[float], direction: str) -> tuple[Optional[float], Optional[float]]:
    if not isinstance(payload, dict):
        return None, None
    tp = _as_float(payload.get("tp_price") or payload.get("tp_abs"))
    sl = _as_float(payload.get("sl_price") or payload.get("sl_abs"))
    if tp is not None and sl is not None:
        return tp, sl
    if entry is None:
        return tp, sl
    tp_pct = _as_float(payload.get("tp_pct"))
    sl_pct = _as_float(payload.get("sl_pct"))
    if tp_pct is None or sl_pct is None:
        return tp, sl
    computed_tp, computed_sl = _prices_from_percent(
        entry=entry,
        direction=direction,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
    )
    return tp if tp is not None else computed_tp, sl if sl is not None else computed_sl


def _snap_exit(
    *,
    entry: float,
    level: float,
    direction: str,
    kind: str,
    structure: List[Dict[str, Any]],
) -> tuple[float, Optional[Dict[str, Any]]]:
    want_type = "resistance" if (kind == "tp") == (direction == "long") else "support"
    max_distance = abs(level - entry) * _SNAP_DISTANCE_FRACTION
    best: Optional[Dict[str, Any]] = None
    best_distance: Optional[float] = None
    for row in structure:
        price = _as_float(row.get("price"))
        if price is None:
            continue
        row_type = str(row.get("type") or "").strip().lower()
        if row_type and row_type != want_type:
            continue
        if kind == "tp":
            if direction == "long" and not (entry < price):
                continue
            if direction == "short" and not (price < entry):
                continue
        else:
            if direction == "long" and not (price < entry):
                continue
            if direction == "short" and not (entry < price):
                continue
        distance = abs(price - level)
        if distance > max_distance:
            continue
        if best_distance is None or distance < best_distance:
            best = row
            best_distance = distance
    if best is None:
        return level, None
    snapped = float(best["price"])
    return snapped, {
        "from": level,
        "to": snapped,
        "source": "confluence",
        "type": best.get("type"),
        "score": best.get("score"),
    }


def _compact_forecast(payload: Any, values: List[float], trend: Optional[str]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in ("method", "library", "quantity", "horizon", "interval_status"):
            if payload.get(key) not in (None, ""):
                compact[key] = payload[key]
    if values:
        compact["first"] = values[0]
        compact["last"] = values[-1]
        compact["bars"] = len(values)
    if trend:
        compact["trend"] = trend
    if isinstance(payload, dict) and isinstance(payload.get("forecast_vs_last_price"), dict):
        context = payload["forecast_vs_last_price"]
        direction_context = {
            key: context[key]
            for key in (
                "direction",
                "direction_basis",
                "direction_actionable",
                "direction_status",
                "direction_suppressed_reason",
                "point_estimate_direction",
                "horizon_delta",
                "horizon_delta_pct",
            )
            if context.get(key) not in (None, "")
        }
        if direction_context:
            compact["forecast_vs_last_price"] = direction_context
    return compact


def _compact_volatility(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key in (
        "method",
        "horizon",
        "volatility_per_bar",
        "volatility_horizon",
        "volatility_annualized",
        "volatility_unit",
    ):
        if payload.get(key) not in (None, ""):
            compact[key] = payload[key]
    return compact


def _compact_barriers(payload: Any, *, take_profit: Optional[float], stop_loss: Optional[float]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in (
            "method",
            "direction",
            "horizon",
            "prob_tp_first",
            "prob_sl_first",
            "prob_no_hit",
            "probability_edge",
            "expected_value",
            "tp_pct",
            "sl_pct",
            "reference_price",
        ):
            if payload.get(key) not in (None, ""):
                compact[key] = payload[key]
    if take_profit is not None:
        compact["take_profit"] = take_profit
    if stop_loss is not None:
        compact["stop_loss"] = stop_loss
    return compact


def _compact_sizing(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("position_sizing")
    source = nested if isinstance(nested, dict) else payload
    compact: Dict[str, Any] = {}
    for key in (
        "suggested_volume",
        "status",
        "candidate_valid",
        "requested_risk_pct",
        "risk_pct",
        "risk_currency",
        "entry",
        "sl",
        "tp",
        "rr_ratio",
        "message",
    ):
        if source.get(key) not in (None, ""):
            compact[key] = source[key]
    if payload.get("candidate_valid") is not None and "candidate_valid" not in compact:
        compact["candidate_valid"] = payload.get("candidate_valid")
    if payload.get("error_code") not in (None, ""):
        compact["error_code"] = payload.get("error_code")
    return compact


def _compact_preview(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key in (
        "dry_run",
        "preview_ok",
        "actionability",
        "blockers",
        "would_send_order",
        "no_action",
        "guardrails_preview",
    ):
        if payload.get(key) not in (None, ""):
            compact[key] = payload[key]
    validation = payload.get("validation")
    if isinstance(validation, dict) and validation.get("live_submission_eligible") is not None:
        compact["live_submission_eligible"] = validation.get("live_submission_eligible")
    elif payload.get("live_submission_eligible") is not None:
        compact["live_submission_eligible"] = payload.get("live_submission_eligible")
    return compact


def _build_narrative(
    *,
    symbol: str,
    direction: str,
    trend: Optional[str],
    barriers: Dict[str, Any],
    stand_down_reasons: List[str],
) -> str:
    parts = [f"{symbol} research idea."]
    if direction == "stand_down":
        reason = "; ".join(stand_down_reasons) if stand_down_reasons else "gates did not clear"
        parts.append(f"Stand down: {reason}.")
    else:
        parts.append(f"Suggested {direction} geometry for study only.")
    if trend and direction != "stand_down":
        parts.append(f"Forecast path is {trend}.")
    tp_prob = barriers.get("prob_tp_first")
    sl_prob = barriers.get("prob_sl_first")
    no_hit = barriers.get("prob_no_hit")
    if tp_prob is not None and sl_prob is not None:
        parts.append(
            f"Barrier sketch: TP-first {tp_prob}, SL-first {sl_prob}"
            + (f", no-hit {no_hit}." if no_hit is not None else ".")
        )
    parts.append("This is not an order or financial advice.")
    return " ".join(parts)


def _default_call_section(name: str, kwargs: Dict[str, Any]) -> Any:
    if name == "session":
        from .context import trade_session_context
        from .requests import TradeSessionContextRequest

        return call_tool_sync_structured(
            trade_session_context,
            request=TradeSessionContextRequest(
                symbol=kwargs["symbol"],
                detail=kwargs.get("detail", "compact"),
            ),
        )
    if name == "confluence":
        from ..pivot import confluence_levels

        return call_tool_sync_structured(
            confluence_levels,
            symbol=kwargs["symbol"],
            pivot_timeframe="D1",
            sr_timeframe="auto",
            detail="compact",
            **({"end": kwargs["as_of"]} if kwargs.get("as_of") else {}),
        )
    if name == "forecast":
        from ..forecast import forecast_generate

        payload = {
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "horizon": kwargs["horizon"],
            "library": "native",
            "method": "theta",
            "quantity": "price",
            "detail": "compact",
        }
        if kwargs.get("as_of"):
            payload["as_of"] = kwargs["as_of"]
        return call_tool_sync_structured(forecast_generate, **payload)
    if name == "volatility":
        from ..forecast import forecast_volatility_estimate

        payload = {
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "horizon": kwargs["horizon"],
            "method": "ewma",
            "detail": "compact",
        }
        if kwargs.get("as_of"):
            payload["as_of"] = kwargs["as_of"]
        return call_tool_sync_structured(forecast_volatility_estimate, **payload)
    if name == "barriers":
        from ..forecast import forecast_barrier_prob

        payload = {
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "horizon": kwargs["horizon"],
            "direction": kwargs["direction"],
            "method": "mc_gbm_bb",
            "detail": "compact",
            "barrier": {
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": DEFAULT_TAKE_PROFIT_PCT,
                "stop_loss": DEFAULT_STOP_LOSS_PCT,
            },
            "params": {"n_sims": 500},
        }
        if kwargs.get("as_of"):
            payload["as_of"] = kwargs["as_of"]
        return call_tool_sync_structured(forecast_barrier_prob, **payload)
    if name == "sizing":
        from .requests import FixedFractionSizing, TradeRiskAnalyzeRequest
        from .risk import trade_risk_analyze

        return call_tool_sync_structured(
            trade_risk_analyze,
            request=TradeRiskAnalyzeRequest(
                symbol=kwargs["symbol"],
                direction=kwargs["direction"],
                entry=kwargs.get("entry"),
                stop_loss=kwargs.get("stop_loss"),
                take_profit=kwargs.get("take_profit"),
                sizing=FixedFractionSizing(risk_pct=float(kwargs["risk_pct"])),
                detail="compact",
            ),
        )
    if name == "preview":
        from . import trade_place
        from .requests import TradePlaceRequest

        return call_tool_sync_structured(
            trade_place,
            request=TradePlaceRequest(
                symbol=kwargs["symbol"],
                volume=float(kwargs["volume"]),
                order_type=kwargs["order_type"],
                stop_loss=kwargs.get("stop_loss"),
                take_profit=kwargs.get("take_profit"),
                dry_run=True,
                require_sl_tp=True,
                detail="compact",
            ),
        )
    raise ValueError(f"Unsupported trade-idea section {name!r}.")


def run_trade_idea_compose(  # noqa: C901
    request: TradeIdeaComposeRequest,
    *,
    call_section: Optional[SectionCaller] = None,
) -> Dict[str, Any]:
    """Assemble a preview-only TradeIdea from existing research tools."""
    caller = call_section or _default_call_section
    try:
        symbol = normalize_required_symbol(request.symbol)
    except ValueError as exc:
        return build_error_payload(
            str(exc),
            code="invalid_symbol",
            operation="trade_idea_compose",
        )
    if not 1 <= int(request.horizon) <= MAX_FORECAST_HORIZON:
        return build_error_payload(
            f"horizon must be between 1 and {MAX_FORECAST_HORIZON}.",
            code="trade_idea_invalid_horizon",
            operation="trade_idea_compose",
            details={"horizon": request.horizon},
        )

    historical = bool(str(request.as_of or "").strip())
    planned = list(_STANDARD_SECTIONS if request.template == "standard" else _QUICK_SECTIONS)
    if historical:
        planned = [name for name in planned if name not in _HISTORICAL_SKIP]

    common = {
        "symbol": symbol,
        "timeframe": request.timeframe,
        "horizon": int(request.horizon),
        "as_of": request.as_of,
        "detail": "compact",
    }
    sections: Dict[str, Any] = {}
    failed: List[str] = []
    section_errors: Dict[str, Dict[str, Any]] = {}
    source_calls: List[Dict[str, Any]] = []

    def _run_section(name: str, kwargs: Dict[str, Any]) -> Any:
        try:
            payload = caller(name, kwargs)
        except Exception as exc:
            payload = {
                "success": False,
                "error": str(exc),
                "error_code": "trade_idea_section_error",
            }
        sections[name] = payload
        failed_now = _section_failed(payload)
        record: Dict[str, Any] = {
            "name": name,
            "status": "failed" if failed_now else "ok",
        }
        if failed_now:
            failed.append(name)
            text = _section_error_text(payload)
            record["error"] = text
            summary: Dict[str, Any] = {"reason": text}
            if isinstance(payload, dict):
                if payload.get("error_code") not in (None, ""):
                    summary["error_code"] = payload["error_code"]
                    record["error_code"] = payload["error_code"]
                if payload.get("remediation") not in (None, ""):
                    summary["remediation"] = payload["remediation"]
            section_errors[name] = summary
        source_calls.append(record)
        return payload

    early = [name for name in planned if name in {"session", "confluence", "forecast", "volatility"}]
    for name in early:
        payload = _run_section(name, dict(common))
        if name == "session" and isinstance(payload, dict):
            if payload.get("error_code") == "symbol_not_found" or _looks_like_invalid_symbol(
                _section_error_text(payload),
                symbol,
            ):
                return {
                    **build_error_payload(
                        payload.get("error") or f"Symbol '{symbol}' was not found.",
                        code="symbol_not_found",
                        operation="trade_idea_compose",
                        details={"symbol": symbol},
                    ),
                    "symbol": symbol,
                    "timeframe": request.timeframe,
                }

    session = sections.get("session")
    quote = _extract_quote(session) if not _section_failed(session) else {}
    structure = (
        _compact_structure(sections.get("confluence"))
        if request.template == "standard" and not _section_failed(sections.get("confluence"))
        else []
    )
    forecast_payload = sections.get("forecast")
    forecast_values = _forecast_values(forecast_payload)
    trend = _forecast_trend(forecast_values)
    if trend is None and isinstance(forecast_payload, dict):
        raw_trend = str(forecast_payload.get("trend") or "").strip().lower()
        if raw_trend in {"up", "down", "flat"}:
            trend = raw_trend

    requested_direction = request.direction
    suggested_direction, forecast_direction_reason = _forecast_direction(
        forecast_payload
    )

    stand_down_reasons: List[str] = []
    gates: Dict[str, Dict[str, Any]] = {
        "quote_fresh": _gate("skip", "historical research cutoff") if historical else _gate("pass"),
        "session": _gate("skip", "historical research cutoff") if historical else _gate("pass"),
        "structure": (
            _gate("pass")
            if structure
            else _gate("skip", "quick template omits confluence")
            if request.template == "quick"
            else _gate("fail", "confluence was unavailable")
        ),
        "forecast": _gate("pass") if forecast_values else _gate("fail", "forecast values missing"),
        "barriers": _gate("skip", "direction not resolved yet"),
        "sl_tp": _gate("skip", "exits not resolved yet"),
        "sizing": _gate("skip"),
        "preview": _gate("skip"),
        "alignment": _gate("skip"),
    }

    if not historical:
        if not quote:
            gates["quote_fresh"] = _gate("fail", "session quote unavailable")
            stand_down_reasons.append("no live quote")
        elif quote.get("quote_not_live_ready"):
            gates["quote_fresh"] = _gate("fail", "quote is not live-ready")
            stand_down_reasons.append("quote is not live-ready")
        if session is None or _section_failed(session):
            gates["session"] = _gate("fail", "session context unavailable")
            stand_down_reasons.append("session context unavailable")
        elif not _session_tradable(session, quote):
            gates["session"] = _gate("fail", "market is not accepting new positions")
            stand_down_reasons.append("market is not accepting new positions")

    direction = "stand_down"
    direction_basis = "forecast_vs_last_price"
    if requested_direction in {"long", "short"}:
        direction = requested_direction
        direction_basis = "requested"
        if suggested_direction and suggested_direction != requested_direction:
            gates["alignment"] = _gate(
                "fail",
                f"forecast direction disagrees with requested {requested_direction}",
            )
        elif suggested_direction:
            gates["alignment"] = _gate("pass")
        else:
            gates["alignment"] = _gate("fail", forecast_direction_reason)
    elif suggested_direction:
        direction = suggested_direction
        gates["alignment"] = _gate("pass")
    else:
        stand_down_reasons.append(forecast_direction_reason)
        gates["alignment"] = _gate("fail", forecast_direction_reason)

    barriers_payload: Any = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    snaps: List[Dict[str, Any]] = []
    if direction in {"long", "short"} and "barriers" in planned:
        barriers_payload = _run_section(
            "barriers",
            {**common, "direction": direction},
        )
        if _section_failed(barriers_payload):
            gates["barriers"] = _gate("fail", "barrier probabilities unavailable")
        else:
            entry_for_barriers = _as_float(
                barriers_payload.get("reference_price") if isinstance(barriers_payload, dict) else None
            ) or _reference_price(quote, direction)
            take_profit, stop_loss = _barrier_prices(
                barriers_payload,
                entry=entry_for_barriers,
                direction=direction,
            )
            if take_profit is None or stop_loss is None:
                take_profit, stop_loss = _prices_from_percent(
                    entry=entry_for_barriers or 0.0,
                    direction=direction,
                    take_profit_pct=DEFAULT_TAKE_PROFIT_PCT,
                    stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
                ) if entry_for_barriers else (None, None)
            if entry_for_barriers is not None and structure:
                if take_profit is not None:
                    take_profit, snap = _snap_exit(
                        entry=entry_for_barriers,
                        level=take_profit,
                        direction=direction,
                        kind="tp",
                        structure=structure,
                    )
                    if snap:
                        snaps.append({"kind": "take_profit", **snap})
                if stop_loss is not None:
                    stop_loss, snap = _snap_exit(
                        entry=entry_for_barriers,
                        level=stop_loss,
                        direction=direction,
                        kind="sl",
                        structure=structure,
                    )
                    if snap:
                        snaps.append({"kind": "stop_loss", **snap})
            tp_prob = _as_float(barriers_payload.get("prob_tp_first")) if isinstance(barriers_payload, dict) else None
            sl_prob = _as_float(barriers_payload.get("prob_sl_first")) if isinstance(barriers_payload, dict) else None
            if tp_prob is not None and sl_prob is not None and sl_prob > tp_prob:
                gates["barriers"] = _gate("fail", "stop is more likely to hit first")
                if requested_direction == "auto":
                    direction = "stand_down"
                    stand_down_reasons.append("barriers disagree with the forecast path")
                    gates["alignment"] = _gate("fail", "forecast and barriers disagree")
            else:
                gates["barriers"] = _gate("pass")

    entry = None
    if isinstance(barriers_payload, dict):
        entry = _as_float(barriers_payload.get("reference_price"))
    if entry is None:
        entry = _reference_price(quote, direction if direction in {"long", "short"} else None)

    if direction in {"long", "short"} and take_profit is not None and stop_loss is not None and entry is not None:
        if barrier_prices_are_valid(
            price=entry,
            direction=direction,  # type: ignore[arg-type]
            tp_price=take_profit,
            sl_price=stop_loss,
        ):
            gates["sl_tp"] = _gate("pass")
        else:
            gates["sl_tp"] = _gate("fail", "TP/SL are not on the correct side of entry")
            take_profit = None
            stop_loss = None
    elif direction == "stand_down":
        gates["sl_tp"] = _gate("skip", "stand down")
    else:
        gates["sl_tp"] = _gate("fail", "missing take-profit or stop-loss")
        stand_down_reasons.append("missing take-profit or stop-loss")
        direction = "stand_down"

    safety_blocked = any(
        gates[name]["status"] == "fail"
        for name in ("quote_fresh", "session", "sl_tp")
    )
    if safety_blocked and direction != "stand_down":
        direction = "stand_down"
        if gates["quote_fresh"]["status"] == "fail":
            stand_down_reasons.append("quote is not live-ready")
        if gates["session"]["status"] == "fail":
            stand_down_reasons.append("session is not tradable")

    suggested_volume = 0.0
    sizing_payload: Any = None
    preview_payload: Any = None
    can_size = (
        direction in {"long", "short"}
        and not historical
        and take_profit is not None
        and stop_loss is not None
        and entry is not None
        and gates["quote_fresh"]["status"] != "fail"
        and gates["session"]["status"] != "fail"
    )
    if can_size and "sizing" in planned:
        sizing_payload = _run_section(
            "sizing",
            {
                **common,
                "direction": direction,
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_pct": float(request.risk_pct),
            },
        )
        sizing_compact = _compact_sizing(sizing_payload)
        volume = _as_float(sizing_compact.get("suggested_volume"))
        candidate_valid = sizing_compact.get("candidate_valid")
        if volume is not None and volume > 0.0 and candidate_valid is not False:
            suggested_volume = float(volume)
            gates["sizing"] = _gate("pass")
        else:
            gates["sizing"] = _gate("fail", "no valid suggested volume")
            suggested_volume = 0.0
    elif historical:
        gates["sizing"] = _gate("skip", "historical ideas do not size against the live account")
        gates["preview"] = _gate("skip", "historical ideas are research-only")
    elif direction == "stand_down":
        gates["sizing"] = _gate("skip", "stand down")
        gates["preview"] = _gate("skip", "stand down")
    else:
        gates["sizing"] = _gate("skip", "sizing not attempted")
        gates["preview"] = _gate("skip", "preview not attempted")

    if can_size and suggested_volume > 0.0 and "preview" in planned:
        preview_payload = _run_section(
            "preview",
            {
                **common,
                "volume": suggested_volume,
                "order_type": "BUY" if direction == "long" else "SELL",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
        )
        preview_compact = _compact_preview(preview_payload)
        if preview_compact.get("dry_run") is False:
            gates["preview"] = _gate("fail", "composer rejected a non-dry-run preview")
            preview_payload = {
                "success": False,
                "error": "trade_idea_compose cannot send live orders",
                "error_code": "trade_idea_live_send_forbidden",
                "preview_ok": False,
                "dry_run": True,
            }
            suggested_volume = 0.0
            direction = "stand_down"
            stand_down_reasons.append("live send is forbidden")
        elif preview_compact.get("preview_ok") is True:
            gates["preview"] = _gate("pass")
        else:
            gates["preview"] = _gate("fail", "dry-run preview is not eligible")
            blockers = preview_compact.get("blockers")
            if isinstance(blockers, list) and blockers:
                stand_down_reasons.append("preview blockers: " + ", ".join(str(item) for item in blockers))

    assembled_at = format_datetime_utc(datetime.now(timezone.utc))
    barriers_compact = _compact_barriers(
        barriers_payload,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )
    if snaps:
        barriers_compact["snapped_to_structure"] = snaps
    actionability = "research" if historical or direction == "stand_down" else "preview_only"
    unique_reasons: List[str] = []
    for reason in stand_down_reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    idea: Dict[str, Any] = {
        "success": True,
        "symbol": symbol,
        "timeframe": request.timeframe,
        "horizon": int(request.horizon),
        "template": request.template,
        "as_of": assembled_at if not historical else request.as_of,
        "assembled_at": assembled_at,
        "timezone": "UTC",
        "direction": direction,
        "direction_basis": direction_basis,
        "actionability": actionability,
        "narrative": _build_narrative(
            symbol=symbol,
            direction=direction,
            trend=trend,
            barriers=barriers_compact,
            stand_down_reasons=unique_reasons,
        ),
        "gates": gates,
    }
    if suggested_direction:
        idea["suggested_direction"] = suggested_direction
    if quote:
        idea["quote"] = quote
    if structure:
        idea["structure"] = {"levels": structure}
    forecast_compact = _compact_forecast(forecast_payload, forecast_values, trend)
    if forecast_compact:
        idea["forecast"] = forecast_compact
    vol_compact = _compact_volatility(sections.get("volatility"))
    if vol_compact:
        idea["volatility"] = vol_compact
    if barriers_compact:
        idea["barriers"] = barriers_compact
    if entry is not None and (take_profit is not None or stop_loss is not None):
        geometry: Dict[str, Any] = {"entry": entry}
        if take_profit is not None:
            geometry["take_profit"] = take_profit
        if stop_loss is not None:
            geometry["stop_loss"] = stop_loss
        if direction in {"long", "short"}:
            geometry["direction"] = direction
        idea["geometry"] = geometry
    sizing_compact = _compact_sizing(sizing_payload) if sizing_payload is not None else {}
    if direction == "stand_down":
        sizing_compact["suggested_volume"] = 0.0
    if sizing_compact:
        idea["sizing"] = sizing_compact
    elif direction == "stand_down":
        idea["sizing"] = {"suggested_volume": 0.0}
    preview_compact = _compact_preview(preview_payload) if preview_payload is not None else {}
    if preview_compact:
        preview_compact.setdefault("dry_run", True)
        preview_compact.setdefault("would_send_order", False)
        idea["preview"] = preview_compact
    elif actionability == "preview_only":
        idea["preview"] = {
            "dry_run": True,
            "preview_ok": False,
            "would_send_order": False,
        }
    else:
        idea["preview"] = {
            "dry_run": True,
            "preview_ok": False,
            "would_send_order": False,
            "skipped": True,
        }

    if failed:
        idea["failed_sections"] = list(failed)
        idea["section_errors"] = section_errors
        if len(failed) == len(source_calls):
            idea["success"] = False
            idea["partial_failure"] = False
            idea["error"] = "All requested trade-idea sections failed."
            idea["error_code"] = "trade_idea_all_sections_failed"
        else:
            idea["partial_failure"] = True
    if request.detail == "full":
        idea["source_tool_calls"] = source_calls
    if historical:
        idea.setdefault("warnings", []).append(
            "Historical as_of ideas are research-only and never request a live preview."
        )

    if isinstance(session, dict) and isinstance(session.get("source"), dict):
        idea["source"] = dict(session["source"])
    idea = attach_mt5_source(idea)
    idea = attach_success_guidance(idea, tool_name="trade_idea_compose")
    if request.detail != "full":
        idea = {key: idea[key] for key in _COMPACT_KEYS if key in idea}
    return idea


@mcp.tool()
def trade_idea_compose(request: TradeIdeaComposeRequest) -> Dict[str, Any]:
    """Compose a preview-only trade idea from existing research tools.

    Combines session context, a Theta price forecast, EWMA volatility, one
    take-profit/stop-loss barrier pair (0.40%/0.60% by default), optional
    confluence, fixed-fraction sizing, and a forced dry-run order preview.
    The composer never sends a live order. Use template=standard to add
    confluence and snap exits toward nearby structure. Historical as_of
    ideas stay research-only.

    This is a research artifact, not a trade instruction.
    """

    def _run() -> Dict[str, Any]:
        return run_trade_idea_compose(request)

    return run_logged_operation(
        logger,
        operation="trade_idea_compose",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        template=request.template,
        func=_run,
    )
