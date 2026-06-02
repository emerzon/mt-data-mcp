"""Unified market snapshot tool."""

from __future__ import annotations

import logging
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
        "data_stale",
    )
    return {key: quote[key] for key in keys if key in quote}


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
    """Return a unified pre-trade market snapshot with selectable analysis sections."""

    def _run() -> Dict[str, Any]:
        selected = _parse_snapshot_sections(sections)
        detail_mode = str(detail or "compact").strip().lower()
        section_payloads = {
            name: _call_section(name, symbol, str(timeframe), int(horizon), detail_mode)
            for name in selected
        }
        health = _snapshot_health(symbol, selected, section_payloads)
        payload: Dict[str, Any] = {
            "success": bool(health.get("success")),
            "symbol": symbol,
            "timeframe": timeframe,
            "as_of": (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "sections": list(selected),
            **{key: value for key, value in health.items() if key != "success"},
            **section_payloads,
        }
        payload["summary"] = _snapshot_summary(
            symbol,
            section_payloads,
            health.get("failed_sections"),
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
