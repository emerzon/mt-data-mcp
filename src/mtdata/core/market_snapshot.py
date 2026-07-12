"""Unified market snapshot tool."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..shared.schema import CompactStandardFullDetailLiteral, TimeframeLiteral
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .tool_calling import call_tool_sync_structured

logger = logging.getLogger(__name__)

_DEFAULT_SECTIONS = ("quote", "levels", "patterns")
_SNAPSHOT_PATTERN_LAST_N_BARS = 3
_VALID_SECTIONS = frozenset(
    {
        "quote",
        "levels",
        "patterns",
        "regime",
        "forecast",
    }
)


def _parse_snapshot_sections(value: Optional[str]) -> tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return _DEFAULT_SECTIONS
    raw_parts = str(value).replace(";", ",").split(",")
    sections = []
    for part in raw_parts:
        item = part.strip().lower()
        if not item:
            continue
        if item == "all":
            return tuple(sorted(_VALID_SECTIONS))
        if item not in _VALID_SECTIONS:
            raise ValueError(
                "sections must contain only: "
                + ", ".join(sorted(_VALID_SECTIONS))
            )
        if item not in sections:
            sections.append(item)
    return tuple(sections or _DEFAULT_SECTIONS)


def _section_error(exc: Exception) -> Dict[str, Any]:
    return {"error": str(exc)}


def _compact_quote(quote: Any) -> Any:
    if not isinstance(quote, dict) or quote.get("error"):
        return quote
    normalized_quote = dict(quote)
    raw_time = normalized_quote.get("time")
    display_time = normalized_quote.pop("time_display", None)
    if isinstance(raw_time, (int, float)):
        normalized_quote["time_epoch"] = raw_time
        if display_time not in (None, ""):
            normalized_quote["time"] = display_time
        else:
            normalized_quote["time"] = (
                datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
    elif display_time not in (None, "") and raw_time in (None, ""):
        normalized_quote["time"] = display_time
    keys = (
        "symbol",
        "bid",
        "ask",
        "mid",
        "last",
        "spread",
        "spread_points",
        "spread_pips",
        "spread_pct",
        "freshness",
        "time",
        "time_epoch",
        "data_stale",
    )
    return {key: normalized_quote[key] for key in keys if key in normalized_quote}


def _utc_iso_text(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _snapshot_quote_as_of(sections: Dict[str, Any]) -> Optional[str]:
    quote = sections.get("quote")
    if not isinstance(quote, dict):
        return None
    raw_epoch = quote.get("time_epoch")
    if isinstance(raw_epoch, (int, float)):
        return _utc_iso_text(float(raw_epoch))
    raw_time = quote.get("time")
    if isinstance(raw_time, str):
        text = raw_time.strip()
        if text.endswith("Z"):
            return text
    return None


def _latest_direction(forecast: Any) -> Optional[str]:
    if not isinstance(forecast, dict):
        return None
    values = forecast.get("forecast") or forecast.get("values") or forecast.get("predictions")
    if not isinstance(values, list) or len(values) < 2:
        return None
    try:
        first = float(values[0])
        last = float(values[-1])
    except Exception:
        return None
    if last > first:
        return "up"
    if last < first:
        return "down"
    return "flat"


def _section_failed(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("error"):
        return True
    return payload.get("success") is False


def _section_error_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if error not in (None, ""):
        return str(error)
    if payload.get("success") is False:
        message = payload.get("message") or payload.get("details")
        return str(message) if message not in (None, "") else "section failed"
    return ""


def _looks_like_invalid_symbol_error(message: str, symbol: str) -> bool:
    text = str(message or "").lower()
    if "symbol" not in text:
        return False
    symbol_text = str(symbol or "").strip().lower()
    if symbol_text and symbol_text not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "not found",
            "not available",
            "unavailable",
            "unknown symbol",
        )
    )


def _snapshot_health(
    symbol: str,
    selected: tuple[str, ...],
    sections: Dict[str, Any],
) -> Dict[str, Any]:
    failed = [name for name in selected if _section_failed(sections.get(name))]
    if not failed:
        return {"success": True}

    errors = {
        name: text
        for name in failed
        if (text := _section_error_text(sections.get(name)))
    }
    invalid_symbol = any(
        _looks_like_invalid_symbol_error(message, symbol)
        for message in errors.values()
    )

    health: Dict[str, Any] = {"failed_sections": failed}
    if invalid_symbol:
        health.update(
            {
                "success": False,
                "failure_reason": "invalid_symbol",
                "error": (
                    f"Symbol {symbol!r} was not found or is not available."
                ),
            }
        )
    elif len(failed) == len(selected):
        health.update(
            {
                "success": False,
                "failure_reason": "all_sections_failed",
                "error": "All requested snapshot sections failed.",
            }
        )
    else:
        health.update({"success": True, "partial_failure": True})
    return health


def _snapshot_summary(
    symbol: str,
    sections: Dict[str, Any],
    failed_sections: Optional[list[str]] = None,
) -> str:
    parts = [f"{symbol} snapshot"]
    quote = sections.get("quote")
    if isinstance(quote, dict):
        mid = quote.get("mid")
        if mid is not None:
            parts.append(f"mid={mid}")
        spread_pips = quote.get("spread_pips")
        if spread_pips is not None:
            parts.append(f"spread_pips={spread_pips}")
    forecast = sections.get("forecast")
    direction = _latest_direction(forecast)
    if direction:
        parts.append(f"forecast={direction}")
    if failed_sections:
        parts.append("failed=" + ",".join(failed_sections))
    return "; ".join(parts) + "."


def _coerce_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _first_level_value(levels: Any) -> Any:
    if not isinstance(levels, list):
        return None
    for level in levels:
        if not isinstance(level, dict):
            continue
        value = level.get("value")
        if value is not None:
            return value
    return None


def _nearest_level_from_side(
    levels: Any,
    side: str,
    reference_price: Optional[float],
) -> Any:
    if not isinstance(levels, list):
        return None
    if reference_price is None:
        return _first_level_value(levels)

    candidates: list[tuple[float, Any]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        value = level.get("value")
        numeric_value = _coerce_float(value)
        if numeric_value is None:
            continue
        if side == "support" and numeric_value > reference_price:
            continue
        if side == "resistance" and numeric_value < reference_price:
            continue
        candidates.append((abs(numeric_value - reference_price), value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _nearest_level_value(
    payload: Any,
    side: str,
    *,
    reference_price: Optional[float] = None,
) -> Any:
    if not isinstance(payload, dict):
        return None

    direct_key = f"nearest_{side}"
    direct = payload.get(direct_key)
    if isinstance(direct, dict):
        value = direct.get("value")
        numeric_value = _coerce_float(value)
        if (
            numeric_value is not None
            and reference_price is not None
            and (
                (side == "support" and numeric_value > reference_price)
                or (side == "resistance" and numeric_value < reference_price)
            )
        ):
            value = None
        if value is not None:
            return value
    elif direct is not None:
        numeric_value = _coerce_float(direct)
        if not (
            numeric_value is not None
            and reference_price is not None
            and (
                (side == "support" and numeric_value > reference_price)
                or (side == "resistance" and numeric_value < reference_price)
            )
        ):
            return direct

    nearest = payload.get("nearest")
    if isinstance(nearest, dict):
        nested = nearest.get(side)
        if isinstance(nested, dict):
            value = nested.get("value")
            numeric_value = _coerce_float(value)
            if (
                numeric_value is not None
                and reference_price is not None
                and (
                    (side == "support" and numeric_value > reference_price)
                    or (side == "resistance" and numeric_value < reference_price)
                )
            ):
                value = None
            if value is not None:
                return value
        elif nested is not None:
            numeric_value = _coerce_float(nested)
            if not (
                numeric_value is not None
                and reference_price is not None
                and (
                    (side == "support" and numeric_value > reference_price)
                    or (side == "resistance" and numeric_value < reference_price)
                )
            ):
                return nested

    return _nearest_level_from_side(payload.get(f"{side}s"), side, reference_price)


def _bias_from_signal_bias(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in ("net_bias", "bias", "direction"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip().lower()
    return None


def _pattern_row_bias(row: Any) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for key in ("pattern_bias", "bias", "direction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    details = row.get("details")
    if isinstance(details, dict):
        return _pattern_row_bias(details)
    return None


def _pattern_bias(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    for key in ("pattern_bias", "bias", "direction"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    bias = _bias_from_signal_bias(payload.get("signal_bias"))
    if bias:
        return bias

    summary = payload.get("summary")
    if isinstance(summary, dict):
        bias = _bias_from_signal_bias(summary.get("signal_bias"))
        if bias:
            return bias
        for key in ("pattern_bias", "bias", "direction"):
            value = summary.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for rows_key in ("highlights", "patterns", "data"):
        rows = payload.get(rows_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            row_bias = _pattern_row_bias(row)
            if row_bias in counts:
                counts[row_bias] += 1

    if counts["bullish"] > counts["bearish"]:
        return "bullish"
    if counts["bearish"] > counts["bullish"]:
        return "bearish"
    if counts["bullish"] or counts["bearish"]:
        return "mixed"
    if counts["neutral"]:
        return "neutral"
    if payload.get("n_patterns") == 0:
        return "none"
    return None


def _snapshot_summary_payload(sections: Dict[str, Any]) -> Dict[str, Any]:
    quote = sections.get("quote")
    levels = sections.get("levels")
    patterns = sections.get("patterns")
    regime = sections.get("regime")
    forecast = sections.get("forecast")

    out: Dict[str, Any] = {}
    if isinstance(quote, dict):
        for key in (
            "bid",
            "ask",
            "mid",
            "spread",
            "spread_points",
            "spread_pips",
            "spread_pct",
            "freshness",
            "time",
        ):
            value = quote.get(key)
            if value is not None:
                out[key] = value

    reference_price = _coerce_float(out.get("mid"))
    if reference_price is None:
        bid = _coerce_float(out.get("bid"))
        ask = _coerce_float(out.get("ask"))
        if bid is not None and ask is not None:
            reference_price = (bid + ask) / 2.0

    support = _nearest_level_value(levels, "support", reference_price=reference_price)
    if support is not None:
        out["nearest_support"] = support
    resistance = _nearest_level_value(levels, "resistance", reference_price=reference_price)
    if resistance is not None:
        out["nearest_resistance"] = resistance

    pattern_bias = _pattern_bias(patterns)
    if pattern_bias:
        out["pattern_bias"] = pattern_bias
        out["pattern_is_signal"] = False
        out["pattern_usage"] = "information_only"
        out["pattern_window_bars"] = _SNAPSHOT_PATTERN_LAST_N_BARS
    if isinstance(patterns, dict):
        for source_key, output_key in (
            ("pattern_status", "pattern_status"),
            ("pattern_confidence", "pattern_confidence"),
            ("conflict", "pattern_conflict"),
            ("n_patterns", "pattern_count"),
        ):
            value = patterns.get(source_key)
            if value is not None:
                out[output_key] = value
        if "is_signal" in patterns:
            out["pattern_is_signal"] = bool(patterns["is_signal"])
        usage = patterns.get("usage")
        if usage not in (None, ""):
            out["pattern_usage"] = usage
        applied_window = patterns.get("applied_last_n_bars")
        if applied_window is not None:
            out["pattern_window_bars"] = applied_window
    if isinstance(regime, dict):
        compact_regime = {
            key: regime[key]
            for key in ("current_regime", "regime", "label", "probabilities", "confidence")
            if key in regime
        }
        if compact_regime:
            out["regime"] = compact_regime
    if isinstance(forecast, dict):
        compact_forecast = {
            key: forecast[key]
            for key in ("method", "forecast", "values", "predictions", "horizon", "quantity")
            if key in forecast
        }
        if compact_forecast:
            out["forecast"] = compact_forecast
    return out


def _embedded_section_payload(name: str, payload: Any) -> Any:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    out = dict(payload)
    for key in ("symbol", "timeframe", "detail"):
        out.pop(key, None)
    if name == "levels":
        out.pop("mode", None)
    elif name == "patterns":
        for key in ("mode", "calibration"):
            out.pop(key, None)
    return out


def _call_section(name: str, symbol: str, timeframe: str, horizon: int, detail: str) -> Any:
    try:
        if name == "quote":
            from .market_depth import market_ticker

            return _compact_quote(
                call_tool_sync_structured(
                    market_ticker,
                    symbol=symbol,
                    detail=detail,
                    raw_tool_output=True,
                )
            )
        if name == "levels":
            from .pivot import support_resistance_levels

            return call_tool_sync_structured(
                support_resistance_levels,
                symbol=symbol,
                timeframe=timeframe,
                detail="compact",
                lookback=200,
                max_levels=4,
                raw_tool_output=True,
            )
        if name == "patterns":
            from .patterns import patterns_detect

            return call_tool_sync_structured(
                patterns_detect,
                symbol=symbol,
                timeframe=timeframe,
                mode="candlestick",
                detail="summary",
                limit=150,
                top_k=3,
                last_n_bars=_SNAPSHOT_PATTERN_LAST_N_BARS,
                raw_tool_output=True,
            )
        if name == "regime":
            from .regime import regime_detect

            return call_tool_sync_structured(
                regime_detect,
                symbol=symbol,
                timeframe=timeframe,
                method="hmm",
                detail="summary",
                raw_tool_output=True,
            )
        if name == "forecast":
            from .forecast import forecast_generate

            return call_tool_sync_structured(
                forecast_generate,
                symbol=symbol,
                timeframe=timeframe,
                method="theta",
                horizon=horizon,
                detail="compact",
                raw_tool_output=True,
            )
    except Exception as exc:
        return _section_error(exc)
    return {"error": f"Unsupported snapshot section {name!r}."}


@mcp.tool()
def market_snapshot(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    sections: Optional[str] = None,
    horizon: int = 8,
    detail: CompactStandardFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Return a unified pre-trade market snapshot with selectable analysis sections.

    Default sections are quote,levels,patterns; pass sections=quote for quote-only
    or sections=all for quote,levels,patterns,regime,forecast. `sections`
    selects analysis modules; `detail` controls field verbosity inside the
    selected modules and does not add heavier sections by itself. The optional
    regime section uses HMM and the optional forecast section uses Theta.
    `horizon` applies only to that built-in forecast section. Call the dedicated
    regime or forecast tool for custom methods and parameters.

    Timestamp semantics: `as_of` tracks the latest quote time when available,
    `quote_as_of` duplicates that normalized quote timestamp explicitly, and
    `assembled_at` records when this snapshot payload was built.
    """

    def _run() -> Dict[str, Any]:
        selected = _parse_snapshot_sections(sections)
        detail_mode = str(detail or "compact").strip().lower()
        section_payloads = {
            name: _call_section(name, symbol, str(timeframe), int(horizon), detail_mode)
            for name in selected
        }
        health = _snapshot_health(symbol, selected, section_payloads)
        assembled_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        quote_as_of = _snapshot_quote_as_of(section_payloads)
        payload: Dict[str, Any] = {
            "success": bool(health.get("success")),
            "symbol": symbol,
            "timeframe": timeframe,
            "as_of": quote_as_of or assembled_at,
            "assembled_at": assembled_at,
            "sections": list(selected),
            **{key: value for key, value in health.items() if key != "success"},
        }
        if quote_as_of is not None:
            payload["quote_as_of"] = quote_as_of
        if detail_mode in {"summary", "summary_only", "compact"}:
            summary_payload = _snapshot_summary_payload(section_payloads)
            if summary_payload:
                payload["snapshot"] = summary_payload
            payload["summary"] = _snapshot_summary(
                symbol,
                section_payloads,
                health.get("failed_sections"),
            )
        else:
            payload.update(
                {
                    name: _embedded_section_payload(name, section_payload)
                    for name, section_payload in section_payloads.items()
                }
            )
        if detail_mode == "full":
            payload["section_notes"] = {
                "default": "quote,levels,patterns",
                "heavy_opt_in": "Add regime or forecast to sections when needed.",
            }
        return payload

    return run_logged_operation(
        logger,
        operation="market_snapshot",
        symbol=symbol,
        timeframe=timeframe,
        sections=sections,
        horizon=horizon,
        detail=detail,
        func=_run,
    )
