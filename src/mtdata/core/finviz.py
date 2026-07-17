"""
Finviz MCP tools for stock screening, fundamentals, news, and market data.

Exposes finvizfinance library functionality as MCP tools.
Note: Data is delayed 15-20 minutes; US stocks only.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from datetime import time as datetime_time
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from zoneinfo import ZoneInfo

from ..services.finviz import (
    get_crypto_performance,
    get_dividends_calendar_api,
    get_earnings_calendar,
    get_earnings_calendar_api,
    get_economic_calendar,
    get_forex_performance,
    get_futures_performance,
    get_general_news,
    get_insider_activity,
    get_stock_description,
    get_stock_fundamentals,
    get_stock_insider_trades,
    get_stock_news,
    get_stock_peers,
    get_stock_ratings,
    screen_stocks,
)
from ..services.news_text import normalize_news_text
from ..services.finviz.symbols import looks_like_non_equity_symbol
from ..shared.schema import DetailLiteral
from ..shared.symbols import finviz_forex_symbol_to_mt5
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import (
    build_pagination_meta,
    normalize_output_detail,
    normalize_output_extras,
    normalize_output_verbosity_detail,
)

logger = logging.getLogger(__name__)

_FINVIZ_EQUITY_BROKER_SUFFIXES = {
    "AMEX",
    "ARCA",
    "BATS",
    "L",
    "NAS",
    "NASDAQ",
    "NQ",
    "NY",
    "NYSE",
    "O",
    "OTC",
    "TQ",
    "US",
}
_FINVIZ_SCREEN_FILTERS_EXAMPLE = '{"Exchange":"NASDAQ","Sector":"Technology"}'
_FINVIZ_FUNDAMENTAL_CATEGORIES: Dict[str, tuple[str, ...]] = {
    "summary": (
        "Company",
        "Sector",
        "Industry",
        "Market Cap",
        "Price",
        "Change",
        "P/E",
        "Forward P/E",
        "EPS (ttm)",
        "52W High",
        "52W Low",
        "RSI (14)",
    ),
    "valuation": (
        "Market Cap",
        "P/E",
        "Forward P/E",
        "PEG",
        "P/S",
        "P/B",
        "P/C",
        "P/FCF",
        "EPS (ttm)",
        "EPS next Y",
        "EPS next Q",
    ),
    "performance": (
        "Perf Week",
        "Perf Month",
        "Perf Quarter",
        "Perf Half Y",
        "Perf Year",
        "Perf YTD",
        "Perf 3Y",
        "Perf 5Y",
        "Perf 10Y",
        "52W High",
        "52W Low",
    ),
    "technical": (
        "RSI (14)",
        "SMA20",
        "SMA50",
        "SMA200",
        "ATR (14)",
        "Beta",
        "Volatility W",
        "Volatility M",
        "Price",
        "Change",
        "Volume",
        "Avg Volume",
        "Rel Volume",
    ),
    "dividends": (
        "Dividend Est.",
        "Dividend TTM",
        "Dividend Ex-Date",
        "Dividend Gr. 3Y",
        "Dividend Gr. 5Y",
        "Payout",
    ),
    "ownership": (
        "Insider Own",
        "Insider Trans",
        "Inst Own",
        "Inst Trans",
        "Short Float",
        "Short Ratio",
    ),
    "profile": (
        "Company",
        "Sector",
        "Industry",
        "Country",
        "Exchange",
        "Index",
        "Employees",
        "IPO",
    ),
}
_FINVIZ_FUNDAMENTAL_CATEGORY_ALIASES = {
    "overview": "summary",
    "tech": "technical",
    "valuation_metrics": "valuation",
}


def _finviz_error_payload(
    message: Any,
    *,
    code: str,
    operation: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_error_payload(
        message,
        code=code,
        operation=operation,
        details=details,
    )


def _validate_positive_finviz_limit(
    limit: Any,
    *,
    operation: str,
) -> Optional[Dict[str, Any]]:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 0
    if value >= 1:
        return None
    return _finviz_error_payload(
        "limit must be greater than or equal to 1.",
        code=f"{operation}_invalid_limit",
        operation=operation,
        details={"limit": limit, "minimum": 1},
    )


def _normalize_finviz_equity_symbol_text(symbol: str) -> str:
    symbol_norm = str(symbol or "").strip().upper()
    if "." not in symbol_norm:
        return symbol_norm
    base, suffix = symbol_norm.rsplit(".", 1)
    if (
        suffix in _FINVIZ_EQUITY_BROKER_SUFFIXES
        and re.fullmatch(r"[A-Z]{1,6}", base or "") is not None
    ):
        return base
    return symbol_norm


def _normalize_equity_symbol(symbol: str, *, tool_name: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    symbol_norm = _normalize_finviz_equity_symbol_text(symbol)
    if not symbol_norm:
        return None, _finviz_error_payload(
            f"{tool_name} requires a symbol.",
            code="finviz_symbol_required",
            operation=tool_name,
            details={"tool": tool_name},
        )
    if looks_like_non_equity_symbol(symbol_norm):
        return None, _finviz_error_payload(
            (
                f"{symbol_norm} is not a Finviz-supported equity ticker. "
                f"{tool_name} only supports US equities."
            ),
            code="finviz_unsupported_symbol",
            operation=tool_name,
            details={"symbol": symbol_norm, "tool": tool_name},
        )
    return symbol_norm, None


def _require_equity_symbol(
    symbol: str,
    *,
    tool_name: str,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    symbol_norm, error = _normalize_equity_symbol(symbol, tool_name=tool_name)
    if error is not None:
        return None, error
    if symbol_norm is None:
        return None, _finviz_error_payload(
            f"{tool_name} could not normalize symbol.",
            code="finviz_symbol_invalid",
            operation=tool_name,
            details={"tool": tool_name},
        )
    return symbol_norm, None


def _finviz_data_fetched_at() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _attach_finviz_fetch_timestamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in payload or payload.get("success") is False:
        return payload
    out = dict(payload)
    out.setdefault("data_fetched_at", _finviz_data_fetched_at())
    return out


def _run_logged_tool(
    operation: str,
    fields: Dict[str, Any],
    fn: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    return run_logged_operation(
        logger,
        operation=operation,
        func=lambda: _attach_finviz_fetch_timestamp(fn()),
        **fields,
    )


def _snake_finviz_market_key(value: Any) -> str:
    key = str(value).strip().lower()
    for old, new in (("%", "pct"), ("/", "_"), ("&", "and"), ("-", "_")):
        key = key.replace(old, new)
    return "_".join(part for part in key.replace(".", "").split() if part)


_FOREX_CURRENCY_NAMES = {
    "AUD": "Australian Dollar",
    "CAD": "Canadian Dollar",
    "CHF": "Swiss Franc",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "NZD": "New Zealand Dollar",
    "USD": "US Dollar",
}

_FINVIZ_MARKET_COMPACT_FIELDS = (
    "symbol",
    "display_symbol",
    "name",
    "price",
    "price_currency",
    "price_source",
    "data_delayed",
    "delay_minutes_min",
    "delay_minutes_max",
    "group",
    "perf_day",
    "perf_week",
    "perf_month",
    "perf_quart",
    "perf_year",
)
_FINVIZ_MARKET_PERFORMANCE_PERIOD_FIELDS = (
    ("day", "perf_day_pct"),
    ("week", "perf_week_pct"),
    ("month", "perf_month_pct"),
    ("quarter", "perf_quart_pct"),
    ("year", "perf_year_pct"),
)
_FINVIZ_SCREEN_COMPACT_FIELDS_BY_VIEW = {
    "overview": (
        "symbol",
        "price",
        "change_pct",
        "volume",
        "pe_ratio",
    ),
    "valuation": (
        "symbol",
        "price",
        "market_cap",
        "pe_ratio",
        "forward_pe",
        "peg",
        "price_to_sales",
        "price_to_book",
    ),
    "financial": (
        "symbol",
        "profit_margin",
        "operating_margin",
        "gross_margin",
        "return_on_assets",
        "return_on_equity",
        "current_ratio",
        "debt_to_equity",
    ),
    "ownership": (
        "symbol",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "short_float",
        "short_ratio",
    ),
    "performance": (
        "symbol",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
    ),
    "technical": (
        "symbol",
        "price",
        "change_pct",
        "volume",
        "rsi_14",
        "sma20_distance_pct",
        "sma50_distance_pct",
        "atr_14",
        "beta",
    ),
}
_FINVIZ_SCREEN_FRACTION_PERCENT_FIELDS = frozenset(
    {
        "profit_margin",
        "operating_margin",
        "gross_margin",
        "return_on_assets",
        "return_on_equity",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
    }
)
_FINVIZ_SCREEN_PERCENT_FIELDS = _FINVIZ_SCREEN_FRACTION_PERCENT_FIELDS | frozenset(
    {
        "change_pct",
        "short_float",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
        "rsi_14",
        "sma20_distance_pct",
        "sma50_distance_pct",
    }
)
_FINVIZ_DETAIL_ERROR = (
    "detail must be one of: compact, standard, summary, full. "
    "Finviz standard/summary output uses the compact shape."
)
_FINVIZ_DELAYED_FRESHNESS = "finviz_delayed"
_FINVIZ_DELAY_MINUTES_MIN = 15
_FINVIZ_DELAY_MINUTES_MAX = 20
_FINVIZ_DELAYED_DATA_QUALITY = "delayed_15_to_20_min"
_FINVIZ_FOREX_DELAYED_PRICE_WARNING = (
    "Finviz forex prices are delayed web quotes, not executable MT5 bid/ask; "
    "use market_ticker before order placement."
)
_FINVIZ_FUTURES_DELAYED_WARNING = (
    "Finviz futures performance rows are delayed web data, not executable MT5 "
    "quotes; use market_ticker before order placement."
)
_FINVIZ_USD_PRICE_CURRENCY = "USD"
_FINVIZ_CALENDAR_LOCAL_TIMEZONE = "America/New_York"
_FINVIZ_CALENDAR_LOCAL_TZ = ZoneInfo(_FINVIZ_CALENDAR_LOCAL_TIMEZONE)


def _derive_forex_pair_name(symbol: Any) -> Optional[str]:
    text = str(symbol or "").strip().upper()
    if "/" in text:
        left, right = text.split("/", 1)
    elif len(text) == 6:
        left, right = text[:3], text[3:]
    else:
        return None
    left_name = _FOREX_CURRENCY_NAMES.get(left)
    right_name = _FOREX_CURRENCY_NAMES.get(right)
    if left_name and right_name:
        return f"{left_name} / {right_name}"
    return None


def _forex_pair_currencies(symbol: Any) -> Optional[tuple[str, str]]:
    text = str(symbol or "").strip().upper()
    if "/" in text:
        left, right = text.split("/", 1)
    elif len(text) == 6:
        left, right = text[:3], text[3:]
    else:
        return None
    if left in _FOREX_CURRENCY_NAMES and right in _FOREX_CURRENCY_NAMES:
        return left, right
    return None


def _normalize_finviz_forex_symbol(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    source_symbol = str(row.get("symbol") or "").strip().upper()
    mt5_symbol = finviz_forex_symbol_to_mt5(source_symbol)
    if mt5_symbol is not None:
        out["symbol"] = mt5_symbol
        if source_symbol and source_symbol != mt5_symbol:
            out["display_symbol"] = source_symbol
    currencies = (
        _forex_pair_currencies(source_symbol or out.get("symbol"))
        if out.get("price") not in (None, "")
        else None
    )
    if currencies is not None:
        base_currency, quote_currency = currencies
        out["base_currency"] = base_currency
        out["price_currency"] = quote_currency
    return out


def _append_finviz_warning(payload: Dict[str, Any], warning: str) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    payload["warnings"] = warnings


def _finviz_percent_value(
    value: Any,
    *,
    fraction_input: bool = True,
) -> Optional[float]:
    if value is None or value == "":
        return None
    parsed = _parse_finviz_numeric_value(value)
    if parsed is None:
        return None
    if not fraction_input:
        return round(float(parsed), 6)
    if isinstance(value, (int, float)) and abs(float(parsed)) <= 1.0:
        parsed = float(parsed) * 100.0
        return round(float(parsed), 6)
    text = str(value).strip()
    if text and not text.endswith("%") and abs(float(parsed)) <= 1.0:
        parsed = float(parsed) * 100.0
    return round(float(parsed), 6)


def _compact_finviz_market_row(row: Dict[str, Any], *, rows_key: str) -> Dict[str, Any]:
    compact = dict(row)
    if "perf_day" not in compact and "perf_pct" in compact:
        compact["perf_day"] = compact["perf_pct"]
    if rows_key == "pairs" and not compact.get("name"):
        derived_name = _derive_forex_pair_name(compact.get("symbol"))
        if derived_name is not None:
            compact["name"] = derived_name
    if "perf_week" not in compact and compact.get("perf_wtd") not in (None, ""):
        compact["perf_week"] = compact["perf_wtd"]
        compact["_perf_week_basis"] = "week_to_date"
    fields = _FINVIZ_MARKET_COMPACT_FIELDS
    if rows_key == "pairs" and compact.get("price") not in (None, ""):
        compact["delayed_price"] = compact.pop("price")
        fields = tuple(
            "delayed_price" if field == "price" else field
            for field in fields
        )
    out = {
        field: compact[field]
        for field in fields
        if field in compact and compact[field] not in (None, "")
    }
    for field in tuple(out):
        if field.startswith("perf_"):
            pct_value = _finviz_percent_value(
                out.get(field),
                fraction_input=rows_key != "futures",
            )
            if pct_value is not None:
                out[f"{field}_pct"] = pct_value
            out.pop(field, None)
    if compact.get("_perf_week_basis") and "perf_week_pct" in out:
        out["perf_week_basis"] = compact["_perf_week_basis"]
    return out


def _is_known_forex_pair_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return _derive_forex_pair_name(row.get("symbol")) is not None


def _finviz_market_performance_periods(rows: Any) -> List[str]:
    if not isinstance(rows, list):
        return []
    periods: List[str] = []
    for period, field in _FINVIZ_MARKET_PERFORMANCE_PERIOD_FIELDS:
        if any(
            isinstance(row, dict) and row.get(field) not in (None, "")
            for row in rows
        ):
            periods.append(period)
    return periods


def _finviz_screen_compact_fields(view: Any) -> tuple[str, ...]:
    view_key = str(view or "overview").strip().lower()
    return _FINVIZ_SCREEN_COMPACT_FIELDS_BY_VIEW.get(
        view_key,
        _FINVIZ_SCREEN_COMPACT_FIELDS_BY_VIEW["overview"],
    )


def _compact_finviz_screen_row(
    row: Dict[str, Any],
    *,
    view: str = "overview",
) -> Dict[str, Any]:
    out = {
        field: row[field]
        for field in _finviz_screen_compact_fields(view)
        if field in row and row[field] not in (None, "")
    }
    for field in (
        "price_source",
        "data_delayed",
        "delay_minutes_min",
        "delay_minutes_max",
    ):
        if row.get(field) not in (None, ""):
            out[field] = row[field]
    return out


def _mark_finviz_delayed_price(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    price = _parse_finviz_numeric_value(out.get("price"))
    if price is None:
        return out
    out["price"] = price
    out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
    out["data_delayed"] = True
    out["delay_minutes_min"] = _FINVIZ_DELAY_MINUTES_MIN
    out["delay_minutes_max"] = _FINVIZ_DELAY_MINUTES_MAX
    return out


def _attach_finviz_delayed_root_metadata(out: Dict[str, Any]) -> None:
    out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
    out["freshness"] = _FINVIZ_DELAYED_FRESHNESS
    out["data_quality"] = _FINVIZ_DELAYED_DATA_QUALITY
    out["data_delayed"] = True
    out["delay_minutes_min"] = _FINVIZ_DELAY_MINUTES_MIN
    out["delay_minutes_max"] = _FINVIZ_DELAY_MINUTES_MAX


def _finviz_screen_units_for_rows(rows: Any) -> Dict[str, str]:
    if not isinstance(rows, list):
        return {}
    seen_fields = {
        key
        for row in rows
        if isinstance(row, dict)
        for key, value in row.items()
        if value not in (None, "")
    }
    units = {
        key: "percentage_points (1.0 = 1%)"
        for key in seen_fields
        if key in _FINVIZ_SCREEN_PERCENT_FIELDS or str(key).endswith("_pct")
    }
    if "short_ratio" in seen_fields:
        units["short_ratio"] = "days_to_cover"
    return units


def _normalize_finviz_market_payload(
    result: Dict[str, Any],
    *,
    rows_key: str,
    limit: Optional[int] = None,
    detail: str = "compact",
    tool: str,
    request: Dict[str, Any],
    symbol_filter: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict) or "error" in result:
        return result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    rows = result.get(rows_key, [])
    key_normalizer = (
        _normalize_finviz_output_key
        if rows_key == "stocks"
        else _snake_finviz_market_key
    )
    normalized_rows = [
        _canonicalize_finviz_market_row(
            {key_normalizer(key): value for key, value in row.items()}
        )
        if isinstance(row, dict)
        else row
        for row in (rows if isinstance(rows, list) else [])
    ]
    upstream_count = len(normalized_rows)
    symbol_filter_norm: Optional[str] = None
    if rows_key == "pairs":
        normalized_rows = [
            row for row in normalized_rows if _is_known_forex_pair_row(row)
        ]
        if symbol_filter not in (None, ""):
            symbol_filter_norm = finviz_forex_symbol_to_mt5(symbol_filter)
            if symbol_filter_norm is not None:
                normalized_rows = [
                    row
                    for row in normalized_rows
                    if str(row.get("symbol") or "").upper() == symbol_filter_norm
                ]
    limit_value = _coerce_finviz_limit(limit, default=len(normalized_rows))
    limited_rows = normalized_rows[:limit_value]
    if detail_mode != "full" and rows_key in {"pairs", "coins", "futures"}:
        output_rows = [
            _compact_finviz_market_row(row, rows_key=rows_key)
            if isinstance(row, dict)
            else row
            for row in limited_rows
        ]
    elif detail_mode != "full" and rows_key == "stocks":
        view = str(request.get("view") or result.get("view") or "overview")
        output_rows = [
            _compact_finviz_screen_row(row, view=view)
            if isinstance(row, dict)
            else row
            for row in limited_rows
        ]
    else:
        output_rows = limited_rows
    out = {key: value for key, value in result.items() if key != rows_key}
    out["items"] = output_rows
    out["count"] = len(output_rows)
    if symbol_filter_norm is not None:
        out["symbol"] = symbol_filter_norm
    available = len(normalized_rows)
    if rows_key != "stocks":
        out["available_count"] = available
    out["pagination"] = build_pagination_meta(
        total=(int(out.get("total") or 0) if rows_key == "stocks" else available),
        returned=len(output_rows),
        offset=0,
        limit=limit_value,
    )
    if rows_key == "stocks" and out.get("total") not in (None, ""):
        omitted = max(0, int(out.get("total") or 0) - int(out["count"]))
    else:
        omitted = max(0, available - len(limited_rows))
    if omitted:
        out["omitted_item_count"] = omitted
    out["detail"] = detail_mode
    has_price = any(
        isinstance(row, dict) and row.get("price") not in (None, "")
        for row in normalized_rows
    )
    if has_price and rows_key in {"stocks", "coins"}:
        out["price_currency"] = _FINVIZ_USD_PRICE_CURRENCY
    if has_price and rows_key == "pairs":
        out["price_currency_basis"] = "quote_currency"
    if has_price and rows_key in {"stocks", "pairs", "coins"}:
        out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
        out["freshness"] = _FINVIZ_DELAYED_FRESHNESS
    if has_price and rows_key in {"pairs", "coins"}:
        _attach_finviz_delayed_root_metadata(out)
    if has_price and rows_key == "pairs":
        _append_finviz_warning(out, _FINVIZ_FOREX_DELAYED_PRICE_WARNING)
    if rows_key == "pairs" and symbol_filter_norm is not None and not output_rows:
        _append_finviz_warning(
            out,
            f"No Finviz forex row matched symbol {symbol_filter_norm}.",
        )
    if rows_key == "futures":
        _attach_finviz_delayed_root_metadata(out)
        _append_finviz_warning(out, _FINVIZ_FUTURES_DELAYED_WARNING)
    if detail_mode != "full" and rows_key in {"pairs", "coins", "futures"}:
        out["performance_format"] = "percentage_points"
    units = _finviz_screen_units_for_rows(output_rows)
    if units:
        out["units"] = units
    if rows_key in {"pairs", "coins", "futures"}:
        limitations: Dict[str, Any] = {}
        periods = _finviz_market_performance_periods(output_rows)
        if periods:
            limitations["performance_periods"] = periods
        if rows_key == "pairs" and has_price:
            limitations["price"] = "delayed_web_quote_not_executable"
        if rows_key == "futures":
            if not has_price:
                limitations["price"] = "not_available_from_source"
        if limitations:
            out["data_limitations"] = limitations
    if detail_mode == "full":
        out["meta"] = _build_tool_contract_meta(
            tool=tool,
            request=request,
            stats={
                "available": available,
                "returned": len(limited_rows),
                "filtered_non_forex": max(0, upstream_count - available),
            },
        )
    return out


def _canonicalize_finviz_symbol_key(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if "symbol" not in out:
        for source_key in ("ticker", "pair"):
            if source_key in out:
                out["symbol"] = out.pop(source_key)
                break
    return out


def _canonicalize_finviz_market_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = _canonicalize_finviz_symbol_key(row)
    if "name" not in out and "label" in out:
        out["name"] = out.pop("label")
    if "p_e" in out and "pe_ratio" not in out:
        out["pe_ratio"] = out.pop("p_e")
    if "perf" in out and not any(key.startswith("perf_") for key in out):
        out["perf_pct"] = out.pop("perf")
    if "change" in out and "change_pct" not in out:
        out["change_pct"] = out.pop("change")
    change_pct = _finviz_percent_value(out.get("change_pct"))
    if change_pct is not None:
        out["change_pct"] = change_pct
    for field in _FINVIZ_SCREEN_PERCENT_FIELDS:
        if field not in out:
            continue
        pct_value = _finviz_percent_value(
            out.get(field),
            fraction_input=field in _FINVIZ_SCREEN_FRACTION_PERCENT_FIELDS,
        )
        if pct_value is not None:
            out[field] = pct_value
    if _is_known_forex_pair_row(out):
        out = _normalize_finviz_forex_symbol(out)
    out = _mark_finviz_delayed_price(out)
    return out


def _coerce_finviz_limit(limit: Optional[int], *, default: int) -> int:
    if limit is None:
        return max(0, int(default))
    return max(0, int(limit))


def _coerce_finviz_offset(offset: Optional[int]) -> int:
    try:
        return max(0, int(offset or 0))
    except Exception:
        return 0


def _build_tool_contract_meta(
    *,
    tool: str,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, Any]] = None,
    legends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tool": tool}
    if stats:
        out["stats"] = {
            key: value for key, value in stats.items() if value is not None
        }
    if pagination:
        out["pagination"] = {
            key: value for key, value in pagination.items() if value is not None
        }
    if legends:
        out["legends"] = legends
    return out


def _finviz_earnings_error_code(message: str) -> str:
    text = str(message or "")
    if "Invalid period" in text:
        return "finviz_earnings_invalid_period"
    if "No earnings calendar data available" in text:
        return "finviz_earnings_no_data"
    return "finviz_earnings_failed"


_FINVIZ_EARNINGS_PERIODS = {
    "this-week": "This Week",
    "next-week": "Next Week",
    "previous-week": "Previous Week",
    "this-month": "This Month",
}
_FINVIZ_EARNINGS_PERIOD_ALIASES = {
    value.lower(): key for key, value in _FINVIZ_EARNINGS_PERIODS.items()
}


def _normalize_finviz_earnings_period(value: Any) -> Optional[tuple[str, str]]:
    text = str(value or "").strip()
    key = text.lower()
    key = key.replace("_", "-")
    if key in _FINVIZ_EARNINGS_PERIODS:
        return key, _FINVIZ_EARNINGS_PERIODS[key]
    label_key = text.lower()
    if label_key in _FINVIZ_EARNINGS_PERIOD_ALIASES:
        canonical = _FINVIZ_EARNINGS_PERIOD_ALIASES[label_key]
        return canonical, _FINVIZ_EARNINGS_PERIODS[canonical]
    return None


def _invalid_finviz_screen_filters_error(
    filters: Any,
    *,
    reason: Optional[str] = None,
    invalid_tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if isinstance(filters, str):
        raw = filters.strip()
        if raw and not raw.startswith("{"):
            message = (
                "Invalid filters format. Received a string value, but finviz_screen "
                "expects a JSON object (dict) with filter names as keys, "
                "key=value or key:value pairs, or Finviz screener shorthand tokens "
                "like "
                "'cap_largeover,exch_nyse'. Examples: "
                "'country=USA,marketcap=mega', "
                "'country:USA,marketcap:mega', "
                "{'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
                "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
                f"Got: {filters!r}"
            )
        else:
            message = (
                "Invalid filters format. Provide filters as key=value or key:value "
                "pairs, a JSON object (dict), or JSON string with filter names as keys "
                "and filter values as values. Example: 'country=USA,marketcap=mega', "
                "{'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
                "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
                f"Got: {filters}"
            )
    else:
        message = (
            "Invalid filters format. Provide filters as key=value or key:value pairs, "
            "a JSON object (dict), or JSON string with filter names as keys and filter "
            "values as values. Example: 'country=USA,marketcap=mega', "
            "{'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
            "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
            f"Got: {filters}"
        )
    if reason:
        message = f"{message} {reason}"
    details: Dict[str, Any] = {"received_type": type(filters).__name__}
    if invalid_tokens:
        details["invalid_tokens"] = list(invalid_tokens)
    examples = _finviz_screen_filter_name_examples()
    if examples:
        details["valid_filter_examples"] = examples
    payload = _finviz_error_payload(
        message,
        code="finviz_screen_filters_invalid",
        operation="finviz_screen",
        details=details,
    )
    payload["related_tools"] = ["finviz_filters_list"]
    payload["remediation"] = (
        "Run finviz_filters_list(filter_name='<filter>') to inspect accepted "
        "values, or use shorthand tokens such as fa_pe_under_20."
    )
    return payload


def _finviz_screen_shorthand_token_map() -> Optional[Dict[str, tuple[str, str]]]:
    try:
        from finvizfinance.screener.base import filter_dict
    except ImportError:
        return None

    reverse_filters: Dict[str, tuple[str, str]] = {}
    for filter_name, spec in filter_dict.items():
        prefix = str(spec.get("prefix") or "").strip()
        for option_name, option_code in (spec.get("option") or {}).items():
            code = str(option_code or "").strip()
            if prefix and code:
                reverse_filters[f"{prefix}_{code}"] = (str(filter_name), str(option_name))
    return reverse_filters


def _finviz_screen_filter_name_examples(limit: int = 12) -> List[str]:
    try:
        from finvizfinance.screener.base import filter_dict
    except ImportError:
        return []
    return [str(name) for name in list(filter_dict.keys())[: max(1, int(limit))]]


def _parse_finviz_screen_shorthand(raw: str) -> Optional[Dict[str, Any]]:
    reverse_filters = _finviz_screen_shorthand_token_map()
    if reverse_filters is None:
        return None

    filters: Dict[str, Any] = {}
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        match = reverse_filters.get(token)
        if match is None:
            return None
        filters[match[0]] = match[1]
    return filters or None


def _unknown_finviz_screen_shorthand_tokens(raw: str) -> List[str]:
    reverse_filters = _finviz_screen_shorthand_token_map()
    if reverse_filters is None:
        return []
    tokens = [part.strip() for part in raw.split(",") if part.strip()]
    return [token for token in tokens if token not in reverse_filters]


def _compact_finviz_filter_token(value: Any, *, keep_sign: bool = False) -> str:
    text = str(value or "").strip().lower()
    return "".join(
        ch for ch in text if ch.isalnum() or (keep_sign and ch in {"+", "-"})
    )


def _resolve_finviz_filter_option(spec: Dict[str, Any], raw_value: str) -> Optional[str]:
    aliases: Dict[str, str] = {}
    for option_name, option_code in (spec.get("option") or {}).items():
        option_name_text = str(option_name)
        option_code_text = str(option_code)
        aliases[_compact_finviz_filter_token(option_name_text, keep_sign=True)] = option_name_text
        aliases[_compact_finviz_filter_token(option_code_text, keep_sign=True)] = option_name_text
        first_word = option_name_text.split(maxsplit=1)[0]
        if first_word.startswith(("+", "-")):
            aliases[_compact_finviz_filter_token(first_word, keep_sign=True)] = option_name_text

    value_key = _compact_finviz_filter_token(raw_value, keep_sign=True)
    if value_key in aliases:
        return aliases[value_key]
    if value_key.startswith(("+", "-")):
        return aliases.get(value_key[1:])
    return None


def _split_finviz_filter_operator_key(raw_key: str) -> tuple[str, Optional[str]]:
    key = str(raw_key or "").strip()
    compact_key = _compact_finviz_filter_token(key)
    for suffix, option_prefix in (
        ("under", "Under"),
        ("below", "Under"),
        ("over", "Over"),
        ("above", "Over"),
    ):
        marker = f"_{suffix}"
        if key.lower().endswith(marker):
            return key[: -len(marker)], option_prefix
        compact_marker = suffix
        if compact_key.endswith(compact_marker) and len(compact_key) > len(compact_marker):
            return compact_key[: -len(compact_marker)], option_prefix
    return key, None


def _parse_finviz_screen_key_value_filters(raw: str) -> Optional[Dict[str, Any]]:
    if "=" not in raw and ":" not in raw:
        return None
    try:
        from finvizfinance.screener.base import filter_dict
    except ImportError:
        return None

    filter_names = {
        _compact_finviz_filter_token(name): str(name)
        for name in filter_dict
    }
    parsed: Dict[str, Any] = {}
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if "=" in token:
            key_raw, value_raw = token.split("=", 1)
        elif ":" in token:
            key_raw, value_raw = token.split(":", 1)
        else:
            return None
        filter_name = filter_names.get(_compact_finviz_filter_token(key_raw))
        option_prefix = None
        if filter_name is None:
            base_key, option_prefix = _split_finviz_filter_operator_key(key_raw)
            filter_name = filter_names.get(_compact_finviz_filter_token(base_key))
        if filter_name is None:
            return None
        option_value = (
            f"{option_prefix} {value_raw}".strip()
            if option_prefix
            else value_raw
        )
        option_name = _resolve_finviz_filter_option(
            filter_dict[filter_name],
            option_value,
        )
        if option_name is None:
            return None
        parsed[filter_name] = option_name
    return parsed or None


def _normalize_finviz_screen_filter_dict(
    filters: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        from finvizfinance.screener.base import filter_dict
    except ImportError:
        return filters, None

    filter_names = {
        _compact_finviz_filter_token(name): str(name)
        for name in filter_dict
    }
    normalized: Dict[str, Any] = {}
    invalid_tokens: List[str] = []
    for key, value in filters.items():
        filter_name = filter_names.get(_compact_finviz_filter_token(key))
        if filter_name is None:
            invalid_tokens.append(str(key))
            continue
        option_name = _resolve_finviz_filter_option(filter_dict[filter_name], value)
        if option_name is None:
            invalid_tokens.append(f"{filter_name}={value}")
            continue
        normalized[filter_name] = option_name
    if invalid_tokens:
        return None, _invalid_finviz_screen_filters_error(
            filters,
            reason=(
                "Unknown filter key or value. Use exact Finviz filter names, "
                "supported shorthand, or key=value aliases such as pe_under=15."
            ),
            invalid_tokens=invalid_tokens,
        )
    return normalized, None


def _resolve_finviz_screen_filters(filters: Any) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if filters is None:
        return None, None
    if isinstance(filters, dict):
        return _normalize_finviz_screen_filter_dict(filters)
    if not isinstance(filters, str):
        return None, _invalid_finviz_screen_filters_error(filters)

    raw = filters.strip()
    if not raw:
        return None, None
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None, _invalid_finviz_screen_filters_error(filters)
        if not isinstance(parsed, dict):
            return None, _invalid_finviz_screen_filters_error(filters)
        return parsed, None
    if "=" in raw or ":" in raw:
        parsed = _parse_finviz_screen_key_value_filters(raw)
        if parsed is not None:
            return parsed, None
        invalid_tokens = [part.strip() for part in raw.split(",") if part.strip()]
        return None, _invalid_finviz_screen_filters_error(
            filters,
            reason=(
                "Unsupported Finviz key=value filter or option. Use Finviz "
                "discrete filters such as beta_under=1, pe_under=15, "
                "country=USA, or native shorthand like cap_largeover."
            ),
            invalid_tokens=invalid_tokens,
        )
    if "_" in raw:
        parsed = _parse_finviz_screen_shorthand(raw)
        if parsed is not None:
            return parsed, None
        invalid_tokens = _unknown_finviz_screen_shorthand_tokens(raw)
        if invalid_tokens:
            return None, _invalid_finviz_screen_filters_error(
                filters,
                reason=(
                    "Unrecognized Finviz shorthand token(s): "
                    f"{', '.join(invalid_tokens)}."
                ),
                invalid_tokens=invalid_tokens,
            )
    return None, _invalid_finviz_screen_filters_error(filters)


@mcp.tool()
def finviz_filters_list(
    search: Optional[str] = None,
    filter_name: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """List valid Finviz screener filters and accepted values."""
    try:
        from finvizfinance.screener.base import filter_dict
    except ImportError as exc:
        return {"error": f"Unable to load Finviz filter metadata: {exc}"}

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    query = str(search or "").strip().lower()
    filter_query = str(filter_name or "").strip().lower()
    rows: List[Dict[str, Any]] = []
    for display_name, spec in filter_dict.items():
        prefix = str(spec.get("prefix") or "").strip()
        options = [
            {"value": str(option_name), "token": f"{prefix}_{option_code}"}
            for option_name, option_code in (spec.get("option") or {}).items()
            if str(option_name).strip()
        ]
        haystack = " ".join(
            [str(display_name), prefix]
            + [str(option.get("value") or "") for option in options]
            + [str(option.get("token") or "") for option in options]
        ).lower()
        if query and query not in haystack:
            continue
        if filter_query and filter_query not in {
            str(display_name).strip().lower(),
            prefix.lower(),
        }:
            continue
        row: Dict[str, Any] = {
            "filter": str(display_name),
            "prefix": prefix,
            "value_count": len(options),
        }
        if detail_mode == "full" or filter_query or query:
            row["values"] = options
        rows.append(row)

    try:
        limit_value = max(1, int(limit or 20))
    except Exception:
        return {"error": "limit must be a positive integer."}
    offset_value = _coerce_finviz_offset(offset)
    limited_rows = rows[offset_value: offset_value + limit_value]
    out: Dict[str, Any] = {
        "success": True,
        "items": limited_rows,
        "count": len(limited_rows),
        "total": len(rows),
        "limit": limit_value,
        "offset": offset_value,
        "has_more": offset_value + len(limited_rows) < len(rows),
        "detail": detail_mode,
        "hint": (
            "Use finviz_screen filters as Filter=Value pairs or shorthand "
            "tokens such as cap_largeover; pass filter_name or detail=full "
            "for accepted values."
        ),
    }
    if search not in (None, ""):
        out["search"] = search
    if filter_name not in (None, ""):
        out["filter_name"] = filter_name
    if len(rows) > len(limited_rows):
        out["omitted_item_count"] = max(0, len(rows) - offset_value - len(limited_rows))
    return out


def _clean_finviz_text_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_news_text(value)
    return value


def _normalize_finviz_published_at(value: Any, *, now: Optional[datetime] = None) -> Any:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text

    iso_text = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue

    for fmt in ("%I:%M%p", "%I:%M %p"):
        try:
            parsed_time = datetime.strptime(text.upper(), fmt).time()
        except ValueError:
            continue
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        reference = reference.astimezone(timezone.utc)
        dt = datetime.combine(
            reference.date(),
            datetime_time(
                parsed_time.hour,
                parsed_time.minute,
                parsed_time.second,
                tzinfo=timezone.utc,
            ),
        )
        if dt > reference + timedelta(hours=1):
            dt -= timedelta(days=1)
        return dt.isoformat()

    return text


def _finviz_relative_time_from_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if " ago" in text.lower():
        return text
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 90:
        return "just now"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes} minutes ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hours ago"
    days = hours // 24
    if days < 14:
        return f"{days} days ago"
    weeks = days // 7
    return f"{weeks} weeks ago"


def _normalize_finviz_news_item(item: Any, *, kind: str = "headline") -> Any:
    if not isinstance(item, dict):
        return item

    out: Dict[str, Any] = {}
    raw_published_at: Any = None
    for source_key, target_key in (
        ("Title", "title"),
        ("Source", "source"),
        ("Date", "published_at"),
        ("Link", "url"),
        ("title", "title"),
        ("source", "source"),
        ("published_at", "published_at"),
        ("relative_time", "relative_time"),
        ("kind", "kind"),
        ("url", "url"),
    ):
        if source_key not in item:
            continue
        value = _clean_finviz_text_value(item.get(source_key))
        if value in (None, ""):
            continue
        if target_key == "published_at":
            raw_published_at = value
            value = _normalize_finviz_published_at(value)
        out[target_key] = value
    if "relative_time" not in out:
        relative_time = (
            _finviz_relative_time_from_text(raw_published_at)
            or _finviz_relative_time_from_text(out.get("published_at"))
        )
        if relative_time:
            out["relative_time"] = relative_time
    out.setdefault("kind", kind)
    out["content_type"] = "blog" if str(kind).lower() == "blog" else "news"
    return out


def _normalize_finviz_news_payload(
    result: Dict[str, Any],
    *,
    detail: DetailLiteral = "compact",  # type: ignore
    kind: str = "headline",
) -> Dict[str, Any]:
    out = dict(result)
    out.pop("tool_scope", None)
    out.pop("preferred_tool", None)
    out.pop("output_shape", None)
    out.pop("timezone", None)
    detail_mode = normalize_output_detail(detail, default="compact")
    out["detail"] = detail_mode
    out["provider"] = "finviz"
    out["delivery"] = "aggregated_web_feed"
    out["is_realtime"] = False
    out["freshness_note"] = (
        "Finviz aggregates third-party headlines and does not guarantee real-time delivery."
    )

    news_rows = result.get("news")
    items_rows = result.get("items")
    if not isinstance(news_rows, list) and not isinstance(items_rows, list):
        return out

    source_rows = news_rows if isinstance(news_rows, list) else items_rows
    normalized_items = [
        _normalize_finviz_news_item(item, kind=kind)
        for item in source_rows
    ]
    out.pop("news", None)
    if detail_mode == "summary":
        out.pop("items", None)
        out["count"] = int(out.get("count") or len(normalized_items))
        return out
    if detail_mode == "compact":
        compact_fields = {
            "title",
            "source",
            "published_at",
            "relative_time",
            "url",
            "kind",
            "content_type",
        }
        out["items"] = [
            {
                key: value
                for key, value in item.items()
                if key in compact_fields and value not in (None, "")
            }
            if isinstance(item, dict)
            else item
            for item in normalized_items
        ]
    else:
        out["items"] = normalized_items
    out["row_key"] = "items"
    return out


_FINVIZ_OUTPUT_KEY_MAP = {
    "#Shares": "shares",
    "#Shares Total": "shares_total",
    "Datetime": "datetime",
    "For": "for_currency",
    "Market Cap": "market_cap",
    "Market Cap.": "market_cap",
    "ReferenceDate": "reference_date",
    "SEC Form 4": "sec_form_4",
    "SEC Form 4 Link": "sec_form_4_link",
    "Insider Trading": "owner",
    "Insider_id": "insider_id",
    "Ticker": "symbol",
    "ticker": "symbol",
    "Value ($)": "value_usd",
    "dateFrom": "date_from",
    "dateTo": "date_to",
    "earningsdate": "earnings_date",
    "EarningsDate": "earnings_date",
    "isearningdateestimate": "is_earning_date_estimate",
    "IsEarningDateEstimate": "is_earning_date_estimate",
    "epsestimate": "eps_estimate",
    "EPSEstimate": "eps_estimate",
    "epsactual": "eps_actual",
    "EPSActual": "eps_actual",
    "epssurprise": "eps_surprise",
    "EPSSurprise": "eps_surprise",
    "epsreportedsurprise": "eps_reported_surprise",
    "EPSReportedSurprise": "eps_reported_surprise",
    "salesestimate": "sales_estimate",
    "SalesEstimate": "sales_estimate",
    "salesactual": "sales_actual",
    "SalesActual": "sales_actual",
    "salessurprise": "sales_surprise",
    "SalesSurprise": "sales_surprise",
    "marketcap": "market_cap",
    "MarketCap": "market_cap",
    "P/E": "pe_ratio",
    "Forward P/E": "forward_pe",
    "P/S": "price_to_sales",
    "P/B": "price_to_book",
    "P/C": "price_to_cash",
    "P/FCF": "price_to_free_cash_flow",
    "Price/Cash": "price_to_cash",
    "Price/Free Cash Flow": "price_to_free_cash_flow",
    "EPS past 3/5Y": "eps_past_3_5_y",
    "Sales past 3/5Y": "sales_past_3_5_y",
    "EPS (ttm)": "eps_ttm",
    "EPS next Y": "eps_next_y",
    "EPS next Q": "eps_next_q",
    "52W High": "high_52w",
    "52W Low": "low_52w",
    "RSI (14)": "rsi_14",
    "SMA20": "sma20_distance_pct",
    "SMA50": "sma50_distance_pct",
    "SMA200": "sma200_distance_pct",
    "ATR (14)": "atr_14",
    "ROA": "return_on_assets",
    "ROE": "return_on_equity",
    "ROI": "return_on_investment",
    "ROIC": "return_on_invested_capital",
    "Curr R": "current_ratio",
    "Quick R": "quick_ratio",
    "LT Debt/Eq": "long_term_debt_to_equity",
    "LTDebt/Eq": "long_term_debt_to_equity",
    "Debt/Eq": "debt_to_equity",
    "Outer": "firm",
    "outer": "firm",
    "Gross M": "gross_margin",
    "Oper M": "operating_margin",
    "Profit M": "profit_margin",
    "Book/sh": "book_value_per_share",
    "Shs Outstand": "shares_outstanding",
    "Shs Float": "shares_float",
    "Perf Week": "performance_week",
    "Perf Month": "performance_month",
    "Perf Quarter": "performance_quarter",
    "Perf Half Y": "performance_half_year",
    "Perf Year": "performance_year",
    "Perf YTD": "performance_ytd",
    "Perf 3Y": "performance_3y",
    "Perf 5Y": "performance_5y",
    "Perf 10Y": "performance_10y",
    "Volatility W": "volatility_w_pct",
    "Volatility M": "volatility_m_pct",
    "Dividend %": "dividend_yield",
    "Dividend Est.": "dividend_est",
    "Dividend TTM": "dividend_ttm",
    "Dividend Ex-Date": "dividend_ex_date",
    "Dividend Gr. 3Y": "dividend_growth_3y",
    "Dividend Gr. 5Y": "dividend_growth_5y",
    "Dividend Gr. 3/5Y": "dividend_growth_3_5_y",
}
_FINVIZ_MARKET_CAP_BUCKETS = {
    "nano",
    "micro",
    "small",
    "mid",
    "large",
    "mega",
}

_FINVIZ_52W_COMPOUND_FIELDS = {
    "high_52w": ("high_52w_price", "high_52w_distance_pct"),
    "low_52w": ("low_52w_price", "low_52w_distance_pct"),
}
_FINVIZ_DUAL_PERIOD_FIELDS = {
    "eps_past_3_5_y": ("eps_past_3y_cagr_pct", "eps_past_5y_cagr_pct"),
    "sales_past_3_5_y": ("sales_past_3y_cagr_pct", "sales_past_5y_cagr_pct"),
    "dividend_gr_3_5_y": ("dividend_growth_3y_cagr_pct", "dividend_growth_5y_cagr_pct"),
    "dividend_growth_3_5_y": ("dividend_growth_3y_cagr_pct", "dividend_growth_5y_cagr_pct"),
}

_FINVIZ_FUNDAMENTAL_NUMERIC_KEYS = frozenset(
    {
        "market_cap",
        "price",
        "change_pct",
        "change_price",
        "enterprise_value",
        "income",
        "sales",
        "pe_ratio",
        "forward_pe",
        "peg",
        "price_to_sales",
        "price_to_book",
        "price_to_cash",
        "price_to_free_cash_flow",
        "eps_ttm",
        "eps_this_y",
        "eps_next_y",
        "eps_next_q",
        "rsi_14",
        "sma20_distance_pct",
        "sma50_distance_pct",
        "sma200_distance_pct",
        "atr_14",
        "beta",
        "volatility_w_pct",
        "volatility_m_pct",
        "volume",
        "avg_volume",
        "rel_volume",
        "return_on_assets",
        "return_on_equity",
        "return_on_investment",
        "return_on_invested_capital",
        "current_ratio",
        "quick_ratio",
        "long_term_debt_to_equity",
        "debt_to_equity",
        "gross_margin",
        "operating_margin",
        "profit_margin",
        "book_value_per_share",
        "shares_outstanding",
        "shares_float",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
        "performance_3y",
        "performance_5y",
        "performance_10y",
        "dividend_yield",
        "dividend_est",
        "dividend_ttm",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "eps_past_3y_cagr_pct",
        "eps_past_5y_cagr_pct",
        "sales_past_3y_cagr_pct",
        "sales_past_5y_cagr_pct",
        "dividend_growth_3y_cagr_pct",
        "dividend_growth_5y_cagr_pct",
        "payout",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "short_float",
        "short_ratio",
    }
)
_FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS = {
    "K": 1_000.0,
    "M": 1_000_000.0,
    "B": 1_000_000_000.0,
    "T": 1_000_000_000_000.0,
}
_FINVIZ_INTEGER_NUMERIC_KEYS = frozenset(
    {
        "market_cap",
        "enterprise_value",
        "income",
        "sales",
        "volume",
        "avg_volume",
        "shares_outstanding",
        "shares_float",
        "employees",
    }
)
_FINVIZ_PERCENT_FUNDAMENTAL_KEYS = frozenset(
    {
        "change_pct",
        "return_on_assets",
        "return_on_equity",
        "return_on_investment",
        "return_on_invested_capital",
        "gross_margin",
        "operating_margin",
        "profit_margin",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
        "performance_3y",
        "performance_5y",
        "performance_10y",
        "dividend_yield",
        "payout",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "short_float",
    }
)
_FINVIZ_LARGE_NUMBER_FORMAT_KEYS = frozenset(
    {
        "market_cap",
        "enterprise_value",
        "income",
        "sales",
        "volume",
        "avg_volume",
        "shares_outstanding",
        "shares_float",
        "employees",
    }
)

_FINVIZ_EARNINGS_COMPACT_FIELDS = (
    "symbol",
    "company",
    "earnings_date",
    "earnings",
    "earnings_timing",
    "eps_estimate",
    "market_cap",
    "price",
    "change",
    "volume",
)
_FINVIZ_EARNINGS_TIMING_SUFFIXES = {
    "/b": "before_market",
    "/a": "after_market",
}
_FINVIZ_CALENDAR_COMPACT_FIELDS = (
    "symbol",
    "country",
    "country_code",
    "event",
    "category",
    "date",
    "local_time",
    "local_timezone",
    "earnings_date",
    "ex_dividend_date",
    "exdate",
    "ex_date",
    "pay_date",
    "record_date",
    "reference",
    "reference_date",
    "actual",
    "previous",
    "forecast",
    "eps_estimate",
    "eps_actual",
    "eps_surprise",
    "sales_estimate",
    "sales_actual",
    "dividend",
    "amount",
    "dividend_amount",
    "ordinary_amount",
    "special_amount",
    "yield_pct",
    "impact",
)

_FINVIZ_CALENDAR_IMPORTANCE_LABELS = {
    1: "low",
    2: "medium",
    3: "high",
}


def _normalize_finviz_output_key(key: Any) -> str:
    text = str(key).strip()
    mapped = _FINVIZ_OUTPUT_KEY_MAP.get(text)
    if mapped:
        return mapped
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = text.replace("%", " pct ").replace("&", " and ").replace("/", " ")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or str(key)


def _parse_finviz_numeric_value(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "n/a", "None", "none", "null"}:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    multiplier = 1.0
    if text and text[-1].upper() in _FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS:
        multiplier = _FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS[text[-1].upper()]
        text = text[:-1].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    return float(text) * multiplier


def _parse_finviz_numeric_tokens(value: Any) -> list[float]:
    tokens = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?", str(value or ""))
    parsed: list[float] = []
    for token in tokens:
        number = _parse_finviz_numeric_value(token)
        if number is not None:
            parsed.append(number)
    return parsed


def _expand_finviz_compound_fundamental(
    key: str,
    value: Any,
) -> Optional[Dict[str, Any]]:
    values = _parse_finviz_numeric_tokens(value)
    if key in _FINVIZ_52W_COMPOUND_FIELDS and len(values) >= 2:
        price_key, distance_key = _FINVIZ_52W_COMPOUND_FIELDS[key]
        return {
            price_key: values[0],
            distance_key: values[1],
        }
    if key in _FINVIZ_DUAL_PERIOD_FIELDS and len(values) >= 2:
        first_key, second_key = _FINVIZ_DUAL_PERIOD_FIELDS[key]
        return {
            first_key: values[0],
            second_key: values[1],
        }
    return None


def _finviz_compound_output_keys(key: str) -> tuple[str, ...]:
    if key in _FINVIZ_52W_COMPOUND_FIELDS:
        return _FINVIZ_52W_COMPOUND_FIELDS[key]
    if key in _FINVIZ_DUAL_PERIOD_FIELDS:
        return _FINVIZ_DUAL_PERIOD_FIELDS[key]
    return ()


def _normalize_finviz_fundamental_value(key: str, value: Any) -> Any:
    if key not in _FINVIZ_FUNDAMENTAL_NUMERIC_KEYS:
        return value
    parsed = _parse_finviz_numeric_value(value)
    if parsed is None:
        return None
    if key in _FINVIZ_INTEGER_NUMERIC_KEYS:
        rounded = round(float(parsed))
        if abs(float(parsed) - float(rounded)) <= 1e-6 * max(1.0, abs(float(parsed))):
            return int(rounded)
    return parsed


def _format_finviz_large_number(value: Any) -> Optional[str]:
    number = _parse_finviz_numeric_value(value)
    if number is None:
        return None
    abs_number = abs(float(number))
    for threshold, suffix in (
        (1_000_000_000_000.0, "T"),
        (1_000_000_000.0, "B"),
        (1_000_000.0, "M"),
        (1_000.0, "K"),
    ):
        if abs_number >= threshold:
            text = f"{float(number) / threshold:.2f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{float(number):.0f}"


def _add_finviz_large_number_formats(fundamentals: Dict[str, Any]) -> None:
    for key in sorted(_FINVIZ_LARGE_NUMBER_FORMAT_KEYS):
        if key not in fundamentals:
            continue
        formatted_key = f"{key}_formatted"
        if formatted_key in fundamentals:
            continue
        formatted = _format_finviz_large_number(fundamentals.get(key))
        if formatted:
            fundamentals[formatted_key] = formatted


def _finviz_fundamental_units(fundamentals: Dict[str, Any]) -> Dict[str, str]:
    units: Dict[str, str] = {}
    for key in fundamentals:
        if key.endswith("_pct") or key in _FINVIZ_PERCENT_FUNDAMENTAL_KEYS:
            units[key] = "percentage_points (1.0 = 1%)"
    return units


def _compact_finviz_fundamentals(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in fundamentals.items()
        if not str(key).endswith("_recomputed")
    }


def _finite_finviz_float(value: Any) -> Optional[float]:
    parsed = _parse_finviz_numeric_value(value)
    if parsed is None:
        return None
    try:
        numeric = float(parsed)
    except Exception:
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _add_finviz_52w_quality_flags(
    fundamentals: Dict[str, Any],
    *,
    include_diagnostics: bool = False,
) -> None:
    price = _finite_finviz_float(fundamentals.get("price"))
    if price is None or price <= 0:
        return
    warnings_out: List[str] = []
    high = _finite_finviz_float(fundamentals.get("high_52w_price"))
    if high is not None and high > 0 and price > high:
        fundamentals["new_52w_high"] = False
        fundamentals["new_52w_high_unconfirmed"] = True
        if include_diagnostics:
            fundamentals["high_52w_distance_pct_recomputed"] = round(
                ((price - high) / high) * 100.0,
                2,
            )
        warnings_out.append(
            "Current price is above the reported 52-week high; upstream 52-week data may be delayed."
        )
    low = _finite_finviz_float(fundamentals.get("low_52w_price"))
    if low is not None and low > 0 and price < low:
        fundamentals["new_52w_low"] = False
        fundamentals["new_52w_low_unconfirmed"] = True
        if include_diagnostics:
            fundamentals["low_52w_distance_pct_recomputed"] = round(
                ((price - low) / low) * 100.0,
                2,
            )
        warnings_out.append(
            "Current price is below the reported 52-week low; upstream 52-week data may be delayed."
        )
    if warnings_out:
        existing = fundamentals.get("data_quality_warnings")
        if not isinstance(existing, list):
            existing = []
        for warning in warnings_out:
            if warning not in existing:
                existing.append(warning)
        fundamentals["data_quality_warnings"] = existing


def _normalize_finviz_output_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return {_normalize_finviz_output_key(key): value for key, value in row.items()}


def _normalize_finviz_output_rows(rows: Any) -> Any:
    if not isinstance(rows, list):
        return rows
    return [_normalize_finviz_output_row(row) for row in rows]


def _normalize_finviz_earnings_rows(rows: Any) -> List[Any]:
    normalized = _normalize_finviz_output_rows(rows)
    if not isinstance(normalized, list):
        return []
    for row in normalized:
        if not isinstance(row, dict):
            continue
        earnings_text = str(row.get("earnings") or "").strip().lower()
        for suffix, timing in _FINVIZ_EARNINGS_TIMING_SUFFIXES.items():
            if earnings_text.endswith(suffix):
                row["earnings_timing"] = timing
                break
        earnings_date = _finviz_earnings_date_from_token(row.get("earnings"))
        if earnings_date:
            row["earnings_date"] = earnings_date
        if "market_cap" not in row:
            continue
        market_cap_source = row.get("market_cap")
        market_cap = _normalize_finviz_fundamental_value(
            "market_cap",
            market_cap_source,
        )
        market_cap_formatted = _format_finviz_large_number(market_cap_source)
        if market_cap is not None:
            row["market_cap"] = market_cap
        if market_cap_formatted:
            row["market_cap_formatted"] = market_cap_formatted
    return normalized


def _finviz_earnings_date_from_token(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    token = str(value).strip()
    if not token:
        return None
    date_part = token.split("/", 1)[0].strip()
    if not date_part:
        return None
    for fmt in ("%Y-%m-%d",):
        try:
            parsed = datetime.strptime(date_part, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()
    reference_year = datetime.now(timezone.utc).year
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            parsed = datetime.strptime(f"{date_part} {reference_year}", fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()
    return None


def _normalize_finviz_date_value(value: Any) -> Any:
    if value in (None, ""):
        return value
    if hasattr(value, "date") and callable(value.date):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    text = str(value).strip()
    if not text:
        return text
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return text


def _parse_finviz_calendar_time(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_FINVIZ_CALENDAR_LOCAL_TZ)
    return parsed


def _normalize_finviz_economic_calendar_time(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    parsed = _parse_finviz_calendar_time(normalized.get("date"))
    if parsed is None:
        return normalized
    local_dt = parsed.astimezone(_FINVIZ_CALENDAR_LOCAL_TZ)
    normalized["local_time"] = local_dt.replace(microsecond=0).isoformat()
    normalized["local_timezone"] = _FINVIZ_CALENDAR_LOCAL_TIMEZONE
    utc_time = parsed.astimezone(timezone.utc)
    normalized["date"] = (
        utc_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return normalized


def _finviz_calendar_importance_label(value: Any) -> Optional[str]:
    try:
        importance = int(value)
    except (TypeError, ValueError):
        return None
    return _FINVIZ_CALENDAR_IMPORTANCE_LABELS.get(importance)


def _add_finviz_calendar_impact_label(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    if normalized.get("impact") in (None, ""):
        impact = _finviz_calendar_importance_label(normalized.get("importance"))
        if impact is not None:
            normalized["impact"] = impact
    return normalized


_FINVIZ_PRICE_TARGET_ARROW_TOKENS = (
    "\u2192",
    "\u00e2\u0086\u0092",
    "\u00e2\u2020\u2019",
    "\u00d4\u00e5\u00c6",
)


def _clean_finviz_price_target_display(value: Any) -> str:
    display = str(value or "").strip()
    for token in _FINVIZ_PRICE_TARGET_ARROW_TOKENS:
        display = display.replace(token, " -> ")
    return re.sub(r"\s+", " ", display).strip()


def _finviz_price_target_fields(value: Any) -> Dict[str, Any]:
    if value in (None, ""):
        return {}
    display = _clean_finviz_price_target_display(value)
    if not display:
        return {}
    prices = [
        float(match.group(1).replace(",", ""))
        for match in re.finditer(
            r"[$]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            display,
        )
    ]
    if not prices:
        return {"price_target_display": display}
    previous = prices[0] if len(prices) > 1 else None
    latest = prices[-1]
    out: Dict[str, Any] = {
        "price_target_display": display,
        "price_target_new": latest,
    }
    if previous is not None:
        out["price_target_previous"] = previous
        if previous > 0:
            out["price_target_change_pct"] = round(
                ((latest - previous) / previous) * 100.0,
                2,
            )
    return out


def _normalize_finviz_rating_rows(rows: Any) -> List[Any]:
    normalized = _normalize_finviz_output_rows(rows)
    if not isinstance(normalized, list):
        return []
    for row in normalized:
        if not isinstance(row, dict):
            continue
        if "date" in row:
            row["date"] = _normalize_finviz_date_value(row.get("date"))
        if row.get("price") not in (None, ""):
            row["price"] = _clean_finviz_price_target_display(row.get("price"))
            row.update(_finviz_price_target_fields(row["price"]))
    return normalized


def _compact_finviz_rating_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    compact = dict(row)
    if compact.get("price_target_new") not in (None, ""):
        compact.pop("price", None)
        compact.pop("price_target_display", None)
    return compact


def _compact_finviz_earnings_items(items: Any) -> List[Any]:
    if not isinstance(items, list):
        return []
    compact_rows: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            compact_rows.append(item)
            continue
        row = {
            field: item[field]
            for field in _FINVIZ_EARNINGS_COMPACT_FIELDS
            if field in item
        }
        if "change" in row:
            row["change_pct"] = row.pop("change")
        change_pct = _finviz_percent_value(row.get("change_pct"))
        if change_pct is not None:
            row["change_pct"] = change_pct
        if "market_cap" in row:
            market_cap_formatted = _format_finviz_large_number(row.get("market_cap"))
            if market_cap_formatted:
                row["market_cap"] = market_cap_formatted
        compact_rows.append(row)
    return compact_rows


_FINVIZ_CALENDAR_COUNTRY_PREFIXES = (
    ("UNITEDSTA", "United States", "US"),
    ("USA", "United States", "US"),
    ("USD", "United States", "US"),
    ("CANADA", "Canada", "CA"),
    ("CAD", "Canada", "CA"),
    ("GERMANY", "Germany", "DE"),
    ("DEU", "Germany", "DE"),
    ("EUROZONE", "Eurozone", "EU"),
    ("EUR", "Eurozone", "EU"),
    ("JAPAN", "Japan", "JP"),
    ("JPY", "Japan", "JP"),
    ("UNITEDKINGDOM", "United Kingdom", "GB"),
    ("UK", "United Kingdom", "GB"),
    ("GBP", "United Kingdom", "GB"),
    ("AUSTRALIA", "Australia", "AU"),
    ("AUD", "Australia", "AU"),
    ("NEWZEALAND", "New Zealand", "NZ"),
    ("NZD", "New Zealand", "NZ"),
    ("SWITZERLAND", "Switzerland", "CH"),
    ("CHF", "Switzerland", "CH"),
    ("CHINA", "China", "CN"),
    ("CNY", "China", "CN"),
)
_FINVIZ_CALENDAR_EVENT_COUNTRY_KEYWORDS = (
    ("FEDERAL RESERVE", "United States", "US"),
    ("FOMC", "United States", "US"),
    ("FED ", "United States", "US"),
)
_FINVIZ_CALENDAR_CURRENCY_TO_COUNTRY_CODE = {
    "USD": "US",
    "EUR": "EU",
    "GBP": "GB",
    "JPY": "JP",
    "CAD": "CA",
    "AUD": "AU",
    "NZD": "NZ",
    "CHF": "CH",
    "CNY": "CN",
}


def _resolve_finviz_calendar_country_filter(
    *,
    country: Optional[str],
    currency: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    country_text = str(country or "").strip()
    currency_text = str(currency or "").strip().upper()
    resolved_code: Optional[str] = None
    if currency_text:
        resolved_code = _FINVIZ_CALENDAR_CURRENCY_TO_COUNTRY_CODE.get(currency_text)
        if resolved_code is None:
            return None, f"Unsupported currency filter '{currency}'."
    if country_text:
        compact_country = re.sub(r"[^A-Za-z]", "", country_text).upper()
        country_code = None
        for prefix, country_name, code in _FINVIZ_CALENDAR_COUNTRY_PREFIXES:
            compact_name = re.sub(r"[^A-Za-z]", "", country_name).upper()
            if compact_country in {prefix, compact_name, code}:
                country_code = code
                break
        if country_code is None:
            return None, f"Unsupported country filter '{country}'."
        if resolved_code is not None and country_code != resolved_code:
            return None, "country and currency filters refer to different regions."
        resolved_code = country_code
    return resolved_code, None


def _infer_finviz_calendar_country(item: Dict[str, Any]) -> tuple[Any, Any]:
    existing_country = item.get("country")
    existing_code = item.get("country_code")
    if existing_country not in (None, "") or existing_code not in (None, ""):
        return existing_country, existing_code

    source_id = str(item.get("symbol") or item.get("source_id") or "").strip()
    compact_source = re.sub(r"[^A-Za-z]", "", source_id).upper()
    for prefix, country, code in _FINVIZ_CALENDAR_COUNTRY_PREFIXES:
        if compact_source.startswith(prefix):
            return country, code
    event_text = " ".join(
        str(item.get(field) or "").upper()
        for field in ("event", "title", "category")
    )
    for keyword, country, code in _FINVIZ_CALENDAR_EVENT_COUNTRY_KEYWORDS:
        if keyword in event_text:
            return country, code
    return None, None


def _compact_finviz_calendar_item(
    item: Any,
    *,
    source_id_only: bool = True,
) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    source_id = normalized.get("source_id") or normalized.get("symbol")
    if source_id_only and source_id not in (None, ""):
        normalized["source_id"] = source_id
        normalized.pop("symbol", None)
    country, country_code = _infer_finviz_calendar_country(normalized)
    if country not in (None, ""):
        normalized["country"] = country
    if country_code not in (None, ""):
        normalized["country_code"] = country_code
    return {
        field: normalized[field]
        for field in _FINVIZ_CALENDAR_COMPACT_FIELDS
        if field in normalized and normalized[field] not in (None, "")
    }


def _normalize_finviz_dividend_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    for source, target in (
        ("amount", "dividend_amount"),
        ("ordinary", "ordinary_amount"),
        ("special", "special_amount"),
        ("yield", "yield_pct"),
    ):
        if source in normalized and target not in normalized:
            normalized[target] = normalized.pop(source)
    return normalized


def _enrich_finviz_calendar_country(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    had_country = normalized.get("country") not in (None, "")
    had_country_code = normalized.get("country_code") not in (None, "")
    country, country_code = _infer_finviz_calendar_country(normalized)
    if country not in (None, ""):
        normalized["country"] = country
    if country_code not in (None, ""):
        normalized["country_code"] = country_code
    if (country not in (None, "") or country_code not in (None, "")) and not (
        had_country or had_country_code
    ):
        normalized["country_inferred"] = True
    return normalized


def _normalize_finviz_calendar_payload(
    result: Dict[str, Any],
    *,
    detail: str = "compact",
    calendar_type: str = "economic",
    country_code_filter: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("error"):
        return result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    out: Dict[str, Any] = {}
    for key, value in result.items():
        normalized_key = _normalize_finviz_output_key(key)
        out[normalized_key] = value
    if isinstance(out.get("items"), list):
        normalized_items = [
            _enrich_finviz_calendar_country(item)
            for item in _normalize_finviz_output_rows(out["items"])
        ]
        calendar_mode = str(calendar_type or "economic").strip().lower()
        if calendar_mode == "dividends":
            normalized_items = [
                _normalize_finviz_dividend_item(item)
                for item in normalized_items
            ]
        if calendar_mode == "economic":
            normalized_items = [
                _normalize_finviz_economic_calendar_time(item)
                if isinstance(item, dict)
                else item
                for item in normalized_items
            ]
            normalized_items = [
                _add_finviz_calendar_impact_label(item)
                if isinstance(item, dict)
                else item
                for item in normalized_items
            ]
        if country_code_filter:
            normalized_items = [
                item
                for item in normalized_items
                if str(item.get("country_code") or "").upper()
                == str(country_code_filter).upper()
            ]
        if detail_mode == "full":
            out["items"] = normalized_items
        else:
            out["items"] = [
                _compact_finviz_calendar_item(
                    item,
                    source_id_only=str(calendar_type or "economic").strip().lower()
                    == "economic",
                )
                for item in normalized_items
            ]
        out["count"] = len(out["items"])
        if out["count"] == 0:
            cal_type = str(calendar_type or "economic").strip().lower()
            if cal_type == "earnings":
                out["message"] = "No detailed earnings calendar rows matched the date range."
                out["hint"] = (
                    "Use finviz_earnings for the period-based earnings view with price/volume context."
                )
            elif cal_type == "dividends":
                out["message"] = "No dividend calendar rows matched the date range."
            else:
                out["message"] = "No economic calendar events matched the filters."
                out["hint"] = "Relax impact, country, currency, start, or end filters."
    if country_code_filter:
        out["country_filter"] = str(country_code_filter).upper()
    if str(calendar_type or "economic").strip().lower() == "economic":
        out["timezone"] = "UTC"
    else:
        out.setdefault("timezone", _FINVIZ_CALENDAR_LOCAL_TIMEZONE)
    if str(calendar_type or "economic").strip().lower() == "dividends":
        out["currency_basis"] = "listing_currency"
        out["units"] = {
            "dividend_amount": "listing_currency_per_share",
            "ordinary_amount": "listing_currency_per_share",
            "special_amount": "listing_currency_per_share",
            "yield_pct": "percentage_points (1.0 = 1%)",
        }
    out["detail"] = detail_mode
    return out


def _validate_finviz_detail(detail: str, *, operation: str) -> Optional[Dict[str, Any]]:
    normalized = str(detail or "compact").strip().lower()
    if normalized in {"compact", "standard", "summary", "full"}:
        return None
    return _finviz_error_payload(
        _FINVIZ_DETAIL_ERROR,
        code=f"{operation}_invalid_detail",
        operation=operation,
        details={"detail": detail},
    )


def _transaction_text(row: Dict[str, Any]) -> str:
    parts = [
        str(value)
        for key, value in row.items()
        if "transaction" in str(key).lower() or "trade" in str(key).lower()
    ]
    return " ".join(parts).lower()


def _coerce_finviz_number(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(str(value).replace("$", "").replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _summarize_insider_activity_tickers(rows: List[Any]) -> List[Dict[str, Any]]:
    by_ticker: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
        if not symbol:
            continue
        item = by_ticker.setdefault(
            symbol,
            {"symbol": symbol, "transactions": 0, "shares": 0.0, "value_usd": 0.0},
        )
        item["transactions"] += 1
        item["shares"] += abs(_coerce_finviz_number(row.get("shares")))
        item["value_usd"] += abs(_coerce_finviz_number(row.get("value_usd")))
    ranked = sorted(
        by_ticker.values(),
        key=lambda item: (item["value_usd"], item["shares"], item["transactions"]),
        reverse=True,
    )
    return [
        {
            "symbol": item["symbol"],
            "transactions": int(item["transactions"]),
            "shares": round(float(item["shares"]), 2),
            "value_usd": round(float(item["value_usd"]), 2),
        }
        for item in ranked[:5]
    ]


def _compact_finviz_insider_row(row: Dict[str, Any], *, include_symbol: bool) -> Dict[str, Any]:
    normalized = dict(row)
    if "price_per_share" not in normalized and normalized.get("cost") not in (None, ""):
        normalized["price_per_share"] = normalized["cost"]
    fields = (
        ("symbol",)
        if include_symbol
        else ()
    ) + (
        "owner",
        "date",
        "transaction",
        "price_per_share",
        "shares",
        "value_usd",
    )
    return {field: normalized[field] for field in fields if field in normalized}


def _normalize_finviz_insider_rows(rows: Any) -> List[Any]:
    normalized = _normalize_finviz_output_rows(rows)
    if not isinstance(normalized, list):
        return []
    for row in normalized:
        if isinstance(row, dict) and row.get("cost") not in (None, ""):
            row.setdefault("price_per_share", row["cost"])
    return normalized


def _compact_finviz_insider_payload(result: Dict[str, Any], *, detail: str) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_insider")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    rows = result.get("insider_trades")
    if not isinstance(rows, list):
        return result
    normalized_rows = _normalize_finviz_insider_rows(rows)
    out = {key: value for key, value in result.items() if key != "insider_trades"}
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["items"] = normalized_rows
        out["count"] = len(normalized_rows)
        return out
    compact_rows = [
        _compact_finviz_insider_row(row, include_symbol=False)
        for row in normalized_rows[:3]
    ]
    transaction_texts = [_transaction_text(row) for row in normalized_rows if isinstance(row, dict)]
    buys = sum(1 for text in transaction_texts if "buy" in text or "purchase" in text)
    sells = sum(1 for text in transaction_texts if "sell" in text or "sale" in text)
    out["items"] = compact_rows
    out["count"] = len(compact_rows)
    out["summary"] = {
        "buy_transactions": buys,
        "sell_transactions": sells,
    }
    out["hint"] = (
        "Single-symbol insider trades; use finviz_insider_activity for "
        "market-wide scans."
    )
    out["omitted_item_count"] = max(0, len(normalized_rows) - len(compact_rows))
    return out


def _compact_finviz_insider_activity_payload(
    result: Dict[str, Any], *, detail: str
) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_insider_activity")
    if error is not None or not result.get("success"):
        return error or result
    rows = result.get("insider_trades")
    if not isinstance(rows, list):
        return result

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    normalized_rows = _normalize_finviz_insider_rows(rows)
    out = {key: value for key, value in result.items() if key != "insider_trades"}
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["items"] = normalized_rows
        out["count"] = len(normalized_rows)
        return out

    compact_rows: List[Any] = []
    for row in normalized_rows[:5]:
        if not isinstance(row, dict):
            compact_rows.append(row)
            continue
        compact_rows.append(_compact_finviz_insider_row(row, include_symbol=True))

    transaction_texts = [
        _transaction_text(row) for row in normalized_rows if isinstance(row, dict)
    ]
    buys = sum(1 for text in transaction_texts if "buy" in text or "purchase" in text)
    sells = sum(1 for text in transaction_texts if "sell" in text or "sale" in text)
    out["items"] = compact_rows
    out["count"] = len(compact_rows)
    out["summary"] = {
        "buy_transactions": buys,
        "sell_transactions": sells,
        "top_symbols": _summarize_insider_activity_tickers(normalized_rows),
    }
    out["hint"] = "Market-wide insider activity; use finviz_insider SYMBOL for one ticker."
    out["omitted_item_count"] = max(0, len(normalized_rows) - len(compact_rows))
    return out


def _compact_finviz_ratings_payload(
    result: Dict[str, Any], *, detail: str, limit: Optional[int]
) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_ratings")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    rows = result.get("ratings")
    if not isinstance(rows, list):
        return result
    out = dict(result)
    normalized_rows = _normalize_finviz_rating_rows(rows)
    limit_value = _coerce_finviz_limit(limit, default=len(normalized_rows))
    limited_rows = normalized_rows[:limit_value]
    omitted = max(0, len(normalized_rows) - len(limited_rows))
    out["ratings"] = limited_rows
    out["count"] = len(limited_rows)
    out["available_count"] = len(normalized_rows)
    out["truncated"] = omitted > 0
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["omitted_item_count"] = omitted
        if omitted:
            out["show_all_hint"] = f"Increase limit to {len(normalized_rows)} to view all ratings."
        return out
    compact_rows = [_compact_finviz_rating_row(row) for row in limited_rows]
    out["ratings"] = compact_rows
    out["summary"] = {
        "latest": compact_rows[0] if compact_rows else None,
    }
    out["omitted_item_count"] = omitted
    if omitted:
        out["show_all_hint"] = f"Set extras='metadata' or limit={len(normalized_rows)} to view all ratings."
    return out


def _compact_finviz_peers_payload(
    result: Dict[str, Any], *, detail: str, limit: Optional[int], offset: Optional[int] = 0
) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_peers")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    peers = result.get("peers")
    if not isinstance(peers, list):
        return result
    out = dict(result)
    limit_value = _coerce_finviz_limit(limit, default=len(peers))
    offset_value = _coerce_finviz_offset(offset)
    limited_peers = peers[offset_value: offset_value + limit_value]
    omitted = max(0, len(peers) - offset_value - len(limited_peers))
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["peers"] = limited_peers
        out["count"] = len(limited_peers)
        out["available_count"] = len(peers)
        out["offset"] = offset_value
        out["has_more"] = offset_value + len(limited_peers) < len(peers)
        out["omitted_item_count"] = omitted
        return out
    compact_peers = limited_peers
    out["peers"] = compact_peers
    out["count"] = len(compact_peers)
    out["available_count"] = len(peers)
    out["offset"] = offset_value
    out["has_more"] = offset_value + len(compact_peers) < len(peers)
    out["omitted_item_count"] = omitted
    if omitted:
        out["show_all_hint"] = (
            f"{omitted} more peers available; pass offset={offset_value + len(compact_peers)}."
        )
    return out


def _parse_finviz_fields(fields: Optional[Union[str, list[str]]]) -> Optional[list[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [field.strip() for field in fields.split(",") if field.strip()]
    return [str(field).strip() for field in fields if str(field).strip()]


def _finviz_public_fundamental_keys(field: str) -> tuple[str, ...]:
    output_key = _normalize_finviz_output_key(field)
    if output_key == "change":
        output_key = "change_pct"
    keys = [output_key]
    keys.extend(_finviz_compound_output_keys(output_key))
    if output_key == "market_cap":
        keys.append("market_cap_formatted")
    if output_key == "exchange":
        keys.append("market_cap_category")
    return tuple(dict.fromkeys(keys))


def _finviz_fundamental_field_returned(
    field: str,
    filtered: Dict[str, Any],
) -> bool:
    return any(key in filtered for key in _finviz_public_fundamental_keys(field))


def _resolve_finviz_fundamental_fields(
    fundamentals: Dict[str, Any],
    requested_fields: list[str],
) -> tuple[list[str], list[str]]:
    lookup: Dict[str, str] = {}
    for field in fundamentals:
        for candidate in (field, *_finviz_public_fundamental_keys(field)):
            text = str(candidate).strip()
            if text:
                lookup.setdefault(text.lower(), field)

    selected: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for field in requested_fields:
        if field in fundamentals:
            resolved = field
        else:
            resolved = lookup.get(str(field).strip().lower())
        if resolved is None:
            missing.append(field)
            continue
        if resolved not in seen:
            selected.append(resolved)
            seen.add(resolved)
    return selected, missing


def _filter_finviz_fundamentals_payload(
    result: Dict[str, Any],
    *,
    detail: str,
    category: str,
    fields: Optional[Union[str, list[str]]],
) -> Dict[str, Any]:
    fundamentals = result.get("fundamentals")
    if not isinstance(fundamentals, dict):
        return result

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    category_input = str(category or "summary").strip().lower()
    category_mode = _FINVIZ_FUNDAMENTAL_CATEGORY_ALIASES.get(
        category_input,
        category_input,
    )
    if str(detail or "compact").strip().lower() not in {"compact", "standard", "summary", "full"}:
        return _finviz_error_payload(
            _FINVIZ_DETAIL_ERROR,
            code="finviz_fundamentals_invalid_detail",
            operation="finviz_fundamentals",
            details={"detail": detail},
        )
    if category_mode != "all" and category_mode not in _FINVIZ_FUNDAMENTAL_CATEGORIES:
        return _finviz_error_payload(
            (
                "category must be one of: all, "
                + ", ".join(sorted(_FINVIZ_FUNDAMENTAL_CATEGORIES))
                + "."
            ),
            code="finviz_fundamentals_invalid_category",
            operation="finviz_fundamentals",
            details={"category": category},
        )

    requested_fields = _parse_finviz_fields(fields)
    missing_fields: list[str] = []
    if requested_fields is not None:
        selected_fields, missing_fields = _resolve_finviz_fundamental_fields(
            fundamentals,
            requested_fields,
        )
        category_out = "custom"
    elif category_mode != "all":
        if detail_mode == "compact":
            selected_fields = list(_FINVIZ_FUNDAMENTAL_CATEGORIES[category_mode])
        else:
            selected_fields = list(fundamentals.keys())
        category_out = category_mode
    elif detail_mode == "compact":
        selected_fields = list(_FINVIZ_FUNDAMENTAL_CATEGORIES["summary"])
        category_out = "summary"
    else:
        selected_fields = list(fundamentals.keys())
        category_out = "all"

    filtered: Dict[str, Any] = {}
    for field in selected_fields:
        if field not in fundamentals:
            continue
        value = fundamentals[field]
        if value in (None, ""):
            continue
        output_key = _normalize_finviz_output_key(field)
        if output_key == "change":
            output_key = "change_pct"
        if output_key == "exchange" and str(value).strip().lower() in _FINVIZ_MARKET_CAP_BUCKETS:
            output_key = "market_cap_category"
        expanded = _expand_finviz_compound_fundamental(output_key, value)
        if expanded is not None:
            filtered.update(
                {
                    expanded_key: expanded_value
                    for expanded_key, expanded_value in expanded.items()
                    if expanded_value not in (None, "")
                }
            )
            continue
        output_value = _normalize_finviz_fundamental_value(output_key, value)
        if output_value in (None, ""):
            continue
        filtered[output_key] = output_value
    _add_finviz_large_number_formats(filtered)
    if detail_mode == "compact":
        filtered = _compact_finviz_fundamentals(filtered)
    out = dict(result)
    out["currency"] = "USD"
    _add_finviz_52w_quality_flags(
        filtered,
        include_diagnostics=detail_mode == "full",
    )
    out["fundamentals"] = filtered
    if filtered.get("data_quality_warnings"):
        out["trust"] = "degraded"
    units = _finviz_fundamental_units(filtered)
    if units:
        out["units"] = units
    out["detail"] = detail_mode
    out["category"] = category_out
    if "price" in filtered:
        out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
        out["price_currency"] = _FINVIZ_USD_PRICE_CURRENCY
        out["freshness"] = _FINVIZ_DELAYED_FRESHNESS
        filtered["price_source"] = _FINVIZ_DELAYED_FRESHNESS
        filtered["data_delayed"] = True
        filtered["delay_minutes_min"] = _FINVIZ_DELAY_MINUTES_MIN
        filtered["delay_minutes_max"] = _FINVIZ_DELAY_MINUTES_MAX
    if category_input != category_mode:
        out["category_requested"] = category_input
    if detail_mode == "full":
        out["available_field_count"] = len(fundamentals)
        out["omitted_field_count"] = sum(
            1
            for field in fundamentals
            if not _finviz_fundamental_field_returned(field, filtered)
        )
    if requested_fields is not None:
        if missing_fields:
            out["missing_fields"] = missing_fields
    return out


@mcp.tool()
def finviz_fundamentals(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
    category: str = "summary",
    fields: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get fundamental data for a US stock symbol.
    
    Returns metrics like P/E, EPS, market cap, sector, industry, dividend yield,
    52-week range, analyst recommendations, and more.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    
    Returns
    -------
    dict
        Fundamental metrics for the stock. By default this returns a compact
        summary; set `detail="full", category="all"` for the full Finviz field set.
    
    Example
    -------
    >>> finviz_fundamentals("AAPL")
    {"success": True, "symbol": "AAPL", "fundamentals": {"pe_ratio": "28.5", ...}}
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_fundamentals",
        )
        if error is not None:
            return error
        result = get_stock_fundamentals(symbol_norm)
        return _filter_finviz_fundamentals_payload(
            result,
            detail=detail,
            category=category,
            fields=fields,
        )

    return _run_logged_tool(
        "finviz_fundamentals",
        {"symbol": symbol, "detail": detail, "category": category, "fields": fields},
        _run,
    )


_FINVIZ_DESCRIPTION_COMPACT_CHARS = 600


def _apply_finviz_description_detail(
    result: Dict[str, Any], *, detail: str
) -> Dict[str, Any]:
    """Truncate a long company description for compact detail."""
    if not isinstance(result, dict) or result.get("error"):
        return result
    if str(detail or "compact").strip().lower() == "full":
        return result
    description = result.get("description")
    if not isinstance(description, str):
        return result
    full_length = len(description)
    if full_length <= _FINVIZ_DESCRIPTION_COMPACT_CHARS:
        return result
    truncated = description[:_FINVIZ_DESCRIPTION_COMPACT_CHARS].rstrip()
    sentence_cut = truncated.rfind(". ")
    if sentence_cut >= int(_FINVIZ_DESCRIPTION_COMPACT_CHARS * 0.5):
        truncated = truncated[: sentence_cut + 1]
    out = dict(result)
    out["description"] = truncated
    out["description_truncated"] = True
    out["description_full_length"] = full_length
    out["detail_hint"] = "Use detail='full' for the complete description."
    return out


@mcp.tool()
def finviz_description(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get company business description for a US stock.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., AAPL, TSLA)
    detail : str
        Output detail: compact (default) truncates a long description for token
        efficiency; full returns the complete text.

    Returns
    -------
    dict
        Company description text
    """
    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_description")
        if detail_error is not None:
            return detail_error
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_description",
        )
        if error is not None:
            return error
        return _apply_finviz_description_detail(
            get_stock_description(symbol_norm), detail=detail
        )

    return _run_logged_tool(
        "finviz_description",
        {"symbol": symbol, "detail": detail},
        _run,
    )


@mcp.tool()
def finviz_news(
    symbol: str,
    limit: int = 20,
    page: int = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Raw Finviz per-ticker news provider endpoint.

    Prefer `news` for trading workflows because it merges Finviz with MT5/CNBC
    sources, ranks relevance, and buckets general, related, impact, and event
    news. Use `finviz_news` when you specifically need Finviz pagination, URLs,
    or the raw flat provider schema for one US equity ticker. Use
    `finviz_market_news` for raw Finviz general market news/blogs.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., NVDA, META).
    limit : int
        Max news items per page (default 20)
    page : int
        Page number for pagination (default 1)
    
    Returns
    -------
    dict
        Stock-specific normalized `items` rows with `title`, `source`,
        `published_at`, and `url` fields.
    """
    fields = {"symbol": symbol, "limit": limit, "page": page, "detail": detail}

    def _run() -> Dict[str, Any]:
        limit_error = _validate_positive_finviz_limit(
            limit,
            operation="finviz_news",
        )
        if limit_error is not None:
            return limit_error
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_news",
        )
        if error is not None:
            return error
        return _normalize_finviz_news_payload(
            get_stock_news(symbol_norm, limit=limit, page=page),
            detail=detail,
            kind="direct_symbol",
        )

    return _run_logged_tool("finviz_news", fields, _run)


@mcp.tool()
def finviz_insider(
    symbol: str,
    limit: int = 20,
    page: int = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get insider trading activity for a US stock.
    
    Returns recent insider buys/sells with owner name, relationship,
    transaction type, shares, value, and date.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    limit : int
        Max trades per page (default 20)
    page : int
        Page number for pagination (default 1)
    detail : {"compact", "full"}
        "compact" returns the first three rows plus aggregate buy/sell counts.
        "full" preserves all returned trades.
    
    Returns
    -------
    dict
        List of insider trades
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_insider",
        )
        if error is not None:
            return error
        return _compact_finviz_insider_payload(
            get_stock_insider_trades(symbol_norm, limit=limit, page=page),
            detail=detail,
        )

    return _run_logged_tool(
        "finviz_insider",
        {"symbol": symbol, "limit": limit, "page": page, "detail": detail},
        _run,
    )


@mcp.tool()
def finviz_ratings(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
    limit: int = 3,
    extras: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get analyst ratings for a US stock.
    
    Returns ratings history with date, analyst firm, rating action,
    rating, and price target.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    detail : {"compact", "full"}
        "full" preserves the requested ratings fields. "compact" returns the
        latest limited rows plus a latest-rating summary.
    limit : int
        Maximum rating rows to return (default 3).
    extras : str, optional
        Set to "metadata" to return the full available rating history.
    
    Returns
    -------
    dict
        List of analyst ratings
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_ratings",
        )
        if error is not None:
            return error
        extras_value = normalize_output_extras(extras)
        detail_value = detail
        limit_value: Optional[int] = int(limit)
        if extras_value:
            detail_value = "full"  # type: ignore[assignment]
            limit_value = None
        return _compact_finviz_ratings_payload(
            get_stock_ratings(symbol_norm),
            detail=detail_value,
            limit=limit_value,
        )

    return _run_logged_tool(
        "finviz_ratings",
        {"symbol": symbol, "detail": detail, "limit": limit, "extras": extras},
        _run,
    )


@mcp.tool()
def finviz_peers(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
    limit: int = 5,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Get peer companies for a US stock.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    detail : {"compact", "full"}
        "full" preserves the complete peer list. "compact" returns up to
        five peers plus peer counts.
    
    Returns
    -------
    dict
        List of peer ticker symbols
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_peers",
        )
        if error is not None:
            return error
        return _compact_finviz_peers_payload(
            get_stock_peers(symbol_norm),
            detail=detail,
            limit=limit,
            offset=offset,
        )

    return _run_logged_tool(
        "finviz_peers",
        {"symbol": symbol, "detail": detail, "limit": limit, "offset": offset},
        _run,
    )


@mcp.tool()
def finviz_screen(
    filters: Optional[Union[str, Dict[str, Any]]] = None,
    order: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
    view: Literal["overview", "valuation", "financial", "ownership", "performance", "technical"] = "overview",
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """
    Screen stocks using Finviz screener with filters.
    
    Parameters
    ----------
    filters : str or dict, optional
        Filter criteria as a JSON string, dict, or Finviz URL shorthand string.
        Dict filter names should be keys with filter values as values. Use the
        exact filter names and values shown on finviz.com screener.
        
        Can be provided as:
        - Finviz shorthand: "cap_largeover,exch_nyse"
        - JSON string: '{"Exchange": "NASDAQ", "Sector": "Technology"}'
        - Dict object: {"Exchange": "NASDAQ", "Sector": "Technology"}
        
        Common filter names: Exchange, Index, Sector, Industry, Country, Market Cap.,
        P/E, Forward P/E, PEG, P/S, P/B, Price/Cash, Price/Free Cash Flow,
        EPS growth this year, EPS growth next year, Sales growth past 5 years,
        EPS growth past 5 years, Dividend Yield, Return on Assets, Return on Equity,
        Return on Investment, Current Ratio, Quick Ratio, LT Debt/Equity, Debt/Equity,
        Gross Margin, Operating Margin, Net Profit Margin, Payout Ratio,
        Insider Ownership, Insider Transactions, Institutional Ownership,
        Institutional Transactions, Float Short, Analyst Recom., Option/Short,
        Earnings Date, Performance, Performance 2, Volatility, RSI (14),
        Gap, 20-Day Simple Moving Average, 50-Day Simple Moving Average,
        200-Day Simple Moving Average, Change, Change from Open, 20-Day High/Low,
        50-Day High/Low, 52-Week High/Low, Pattern, Candlestick, Beta,
        Average True Range, Average Volume, Relative Volume, Current Volume,
        Price, Target Price, IPO Date, Shares Outstanding, Float
        
    order : str, optional
        Sort order, e.g. "-marketcap" (descending), "price" (ascending)
    limit : int
        Max results per page (default 20)
    page : int
        Page number for pagination (default 1)
    view : str
        Data view: overview, valuation, financial, ownership, performance, technical
    detail : str
        Output detail: compact (default) or full. Full includes request/meta.
    
    Returns
    -------
    dict
        Matching stock rows under `items` with compact market-tool metadata.
    
    Examples
    --------
    Screen for tech stocks on NASDAQ (using dict):
    >>> finviz_screen(filters={"Exchange": "NASDAQ", "Sector": "Technology"})
    
    Screen for large NYSE stocks (using Finviz shorthand):
    >>> finviz_screen(filters="cap_largeover,exch_nyse")

    Screen for tech stocks on NASDAQ (using JSON string):
    >>> finviz_screen(filters='{"Exchange": "NASDAQ", "Sector": "Technology"}')
    
    Screen for undervalued large caps:
    >>> finviz_screen(filters={"Market Cap.": "Large ($10bln to $200bln)", "P/E": "Under 15"})
    
    Screen for high dividend stocks with specific view:
    >>> finviz_screen(filters={"Dividend Yield": "Over 5%"}, view="valuation")
    
    Notes
    -----
    - Filter names must exactly match those used on the Finviz screener website
    - Filter values must match the available options for each filter
    - Visit finviz.com/screener.ashx to see available filters and their values
    """
    fields = {"limit": limit, "page": page, "view": view, "order": order, "detail": detail}

    def _run() -> Dict[str, Any]:
        filters_dict, filter_error = _resolve_finviz_screen_filters(filters)
        if filter_error is not None:
            return filter_error

        result = screen_stocks(filters=filters_dict, order=order, limit=limit, page=page, view=view)
        if result.get("success") and isinstance(result.get("stocks"), list):
            return _normalize_finviz_market_payload(
                result,
                rows_key="stocks",
                limit=limit,
                detail=detail,
                tool="finviz_screen",
                request={
                    "filters": filters_dict,
                    "order": order,
                    "limit": limit,
                    "page": page,
                    "view": view,
                },
            )
        return result

    return _run_logged_tool("finviz_screen", fields, _run)


@mcp.tool()
def finviz_market_news(
    news_type: Literal["news", "blogs"] = "news",
    limit: int = 20,
    page: int = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Raw Finviz general market news/blog provider endpoint.

    Prefer `news` for trader-facing market news because it aggregates Finviz
    with other sources and categorizes relevance/impact. Use
    `finviz_market_news` when you specifically need Finviz-only pagination,
    `news_type` (`news` vs `blogs`), URLs, or the raw flat provider schema.
    
    Parameters
    ----------
    news_type : str
        Type: "news" for headlines, "blogs" for blog posts
    limit : int
        Max items per page (default 20)
    page : int
        Page number for pagination (default 1)
    
    Returns
    -------
    dict
        List of news/blog items. Top-level metadata marks this as a raw Finviz
        provider endpoint and points traders to `news` as the preferred unified
        tool.
    """
    return _run_logged_tool(
        "finviz_market_news",
        {"news_type": news_type, "limit": limit, "page": page, "detail": detail},
        lambda: _normalize_finviz_news_payload(
            get_general_news(news_type=news_type, limit=limit, page=page),
            detail=detail,
            kind="blog" if str(news_type).lower().strip() == "blogs" else "headline",
        ),
    )


@mcp.tool()
def finviz_insider_activity(
    option: Literal["latest", "top week", "top owner trade", "insider buy", "insider sale"] = "latest",
    limit: int = 50,
    page: int = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get general insider trading activity across the market.
    
    Parameters
    ----------
    option : str
        Activity type:
        - "latest": Most recent insider trades
        - "top week": Top trades this week
        - "top owner trade": Largest owner trades
        - "insider buy": Recent insider buys
        - "insider sale": Recent insider sales
    limit : int
        Max items per page (default 50)
    page : int
        Page number for pagination (default 1)
    detail : {"compact", "full"}
        Response detail level. Compact returns a short normalized item list and
        summary; full keeps all normalized rows including SEC link fields.
    
    Returns
    -------
    dict
        List of insider trades with ticker, owner, transaction details
    """
    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_insider_activity")
        if detail_error is not None:
            return detail_error
        return _compact_finviz_insider_activity_payload(
            get_insider_activity(option=option, limit=limit, page=page),
            detail=detail,
        )

    return _run_logged_tool(
        "finviz_insider_activity",
        {"option": option, "limit": limit, "page": page, "detail": detail},
        _run,
    )


@mcp.tool()
def finviz_forex(
    symbol: Optional[str] = None,
    limit: int = 20,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get forex currency pairs performance from Finviz.
    
    Returns performance data for major currency pairs including
    daily change, weekly change, and other metrics.
    
    Returns
    -------
    dict
        Forex pairs performance data
    """
    request = {"symbol": symbol, "limit": limit, "detail": detail}

    def _run() -> Dict[str, Any]:
        symbol_norm = None
        if symbol not in (None, ""):
            symbol_norm = finviz_forex_symbol_to_mt5(symbol)
            if symbol_norm is None:
                return _finviz_error_payload(
                    (
                        f"Invalid forex symbol: {symbol}. Use a six-letter fiat "
                        "pair such as EURUSD or a slash pair such as EUR/USD."
                    ),
                    code="finviz_forex_invalid_symbol",
                    operation="finviz_forex",
                    details={"symbol": symbol},
                )
        limit_error = _validate_positive_finviz_limit(
            limit,
            operation="finviz_forex",
        )
        if limit_error is not None:
            return limit_error
        detail_error = _validate_finviz_detail(detail, operation="finviz_forex")
        if detail_error is not None:
            return detail_error
        return _normalize_finviz_market_payload(
            get_forex_performance(),
            rows_key="pairs",
            limit=limit,
            detail=detail,
            tool="finviz_forex",
            request=request,
            symbol_filter=symbol_norm,
        )

    return _run_logged_tool("finviz_forex", request, _run)


@mcp.tool()
def finviz_crypto(
    limit: int = 20,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get cryptocurrency performance from Finviz.
    
    Returns performance data for major cryptocurrencies including
    price, daily change, volume, and market cap.
    
    Returns
    -------
    dict
        Crypto performance data
    """
    request = {"limit": limit, "detail": detail}

    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_crypto")
        if detail_error is not None:
            return detail_error
        return _normalize_finviz_market_payload(
            get_crypto_performance(),
            rows_key="coins",
            limit=limit,
            detail=detail,
            tool="finviz_crypto",
            request=request,
        )

    return _run_logged_tool("finviz_crypto", request, _run)


@mcp.tool()
def finviz_futures(
    limit: int = 20,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get futures market performance from Finviz.
    
    This endpoint is a performance-only Finviz source. It returns daily percent
    moves for major futures contracts across commodities, indices, bonds, and
    currencies, but Finviz does not expose current price or volume in this
    source. The response includes data_limitations.price when price is absent.
    
    Returns
    -------
    dict
        Futures performance data
    """
    request = {"limit": limit, "detail": detail}

    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_futures")
        if detail_error is not None:
            return detail_error
        return _normalize_finviz_market_payload(
            get_futures_performance(),
            rows_key="futures",
            limit=limit,
            detail=detail,
            tool="finviz_futures",
            request=request,
        )

    return _run_logged_tool("finviz_futures", request, _run)


@mcp.tool()
def finviz_calendar(
    calendar: Literal["economic", "earnings", "dividends"] = "economic",  # type: ignore
    impact: Optional[Literal["low", "medium", "high"]] = None,
    country: Optional[str] = None,
    currency: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get detailed Finviz calendar data (economic, earnings, or dividends).

    Use `calendar="earnings"` for date-range EPS/sales estimate, actual, and
    surprise data. Use `finviz_earnings` for the quick period-based
    earnings view with price/volume context.

    Parameters
    ----------
    calendar : str
        Calendar type: "economic", "earnings", or "dividends".
    impact : str, optional
        Economic only: filter by impact level: "low", "medium", or "high".
    country : str, optional
        Economic only: filter by country name or code (for example "US").
    currency : str, optional
        Economic only: filter by affected currency (for example "USD").
    start : str, optional
        Start date in ISO format: YYYY-MM-DD.
    end : str, optional
        End date in ISO format: YYYY-MM-DD.
    limit : int
        Max events per page (default 20)
    page : int
        Page number for pagination (default 1)
    detail : str
        Use "compact" for trader-facing fields or "full" for raw upstream fields.

    Returns
    -------
    dict
        Calendar entries (schema depends on calendar type).
    """
    fields = {
        "calendar": calendar,
        "impact": impact,
        "country": country,
        "currency": currency,
        "start": start,
        "end": end,
        "limit": limit,
        "page": page,
        "detail": detail,
    }

    def _run() -> Dict[str, Any]:
        start_value = str(start or "").strip() or None
        end_value = str(end or "").strip() or None

        cal = (calendar or "economic").strip().lower()

        country_filter, filter_error = _resolve_finviz_calendar_country_filter(
            country=country,
            currency=currency,
        )
        if filter_error:
            return {"error": filter_error}
        if cal != "economic" and country_filter:
            return {
                "error": "country/currency filters are only supported for economic calendar."
            }

        if cal == "economic":
            return _normalize_finviz_calendar_payload(
                get_economic_calendar(
                    impact=impact,
                    limit=limit,
                    page=page,
                    date_from=start_value,
                    date_to=end_value,
                ),
                detail=detail,
                calendar_type=cal,
                country_code_filter=country_filter,
            )
        if cal == "earnings":
            return _normalize_finviz_calendar_payload(
                get_earnings_calendar_api(
                    limit=limit,
                    page=page,
                    date_from=start_value,
                    date_to=end_value,
                ),
                detail=detail,
                calendar_type=cal,
            )
        if cal == "dividends":
            return _normalize_finviz_calendar_payload(
                get_dividends_calendar_api(
                    limit=limit,
                    page=page,
                    date_from=start_value,
                    date_to=end_value,
                ),
                detail=detail,
                calendar_type=cal,
            )
        return {"error": f"Unsupported calendar '{calendar}'. Expected economic, earnings, or dividends."}

    return _run_logged_tool("finviz_calendar", fields, _run)


@mcp.tool()
def finviz_earnings(
    period: Literal["this-week", "next-week", "previous-week", "this-month"] = "this-week",
    limit: int = 10,
    page: int = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get the quick upcoming earnings calendar from Finviz.
    
    This is the period-based price/volume earnings view. Use
    `finviz_calendar(calendar="earnings")` when you need date-range EPS/sales
    estimates, actuals, and surprises from the detailed calendar API.
    
    Parameters
    ----------
    period : str
        Calendar period: this-week, next-week, previous-week, or this-month.
    limit : int
        Max items per page (default 10)
    page : int
        Page number for pagination (default 1)
    detail : {"compact", "full"}
        Response detail level. Compact returns calendar-focused rows; full keeps
        all normalized provider columns and adds the tool metadata block.
    
    Returns
    -------
    dict
        Earnings calendar data
    """
    def _run() -> Dict[str, Any]:
        normalized_period = _normalize_finviz_earnings_period(period)
        if normalized_period is None:
            return {
                "success": False,
                "error": (
                    "Invalid period. Use one of: "
                    + ", ".join(sorted(_FINVIZ_EARNINGS_PERIODS))
                    + "."
                ),
                "error_code": "finviz_earnings_invalid_period",
                "meta": _build_tool_contract_meta(
                    tool="finviz_earnings",
                    request={"period": period, "limit": limit, "page": page, "detail": detail},
                ),
            }
        period_key, period_value = normalized_period
        request = {
            "period": period_key,
            "limit": limit,
            "page": page,
            "detail": detail,
        }
        detail_error = _validate_finviz_detail(detail, operation="finviz_earnings")
        if detail_error is not None:
            return detail_error
        result = get_earnings_calendar(period=period_value, limit=limit, page=page)
        if not isinstance(result, dict):
            return {
                "success": False,
                "error": "Unexpected earnings calendar response.",
                "error_code": "finviz_earnings_failed",
                "meta": _build_tool_contract_meta(
                    tool="finviz_earnings",
                    request=request,
                ),
            }
        if result.get("error"):
            return {
                "success": False,
                "error": str(result.get("error")),
                "error_code": _finviz_earnings_error_code(str(result.get("error"))),
                "meta": _build_tool_contract_meta(
                    tool="finviz_earnings",
                    request=request,
                ),
            }

        items = result.get("earnings")
        if not isinstance(items, list):
            items = []
        normalized_items = _normalize_finviz_earnings_rows(items)
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        output_items = (
            normalized_items
            if detail_mode == "full"
            else _compact_finviz_earnings_items(normalized_items)
        )
        pagination = {
            "page": result.get("page"),
            "total": result.get("total"),
            "pages": result.get("pages"),
            "has_more": bool(result.get("has_more")),
            "total_lower_bound": result.get("total_lower_bound"),
            "truncated": bool(result.get("truncated")),
        }
        stats = {
            "truncated": result.get("truncated"),
        }
        out: Dict[str, Any] = {
            "success": True,
            "period": period_key,
            "detail": detail_mode,
            "items": output_items,
            "row_key": "items",
            "count": int(result.get("count") or len(output_items)),
            "total": result.get("total"),
            "page": result.get("page"),
            "pages": result.get("pages"),
            "has_more": bool(result.get("has_more")),
            "truncated": bool(result.get("truncated")),
        }
        if result.get("total_lower_bound") is not None:
            out["total_lower_bound"] = result.get("total_lower_bound")
        if out["has_more"] and out.get("page") is not None:
            out["next_page"] = int(out["page"]) + 1
        if out["detail"] != "full":
            out["omitted_item_count"] = (
                None
                if out.get("total") is None
                else max(0, int(out["total"]) - int(out["count"]))
            )
            out["hint"] = (
                "Period-based earnings view; use finviz_calendar(calendar='earnings') "
                "for date-range EPS/sales actuals and surprises."
            )
        if out["detail"] == "full":
            out["meta"] = _build_tool_contract_meta(
                tool="finviz_earnings",
                request=request,
                stats=stats,
                pagination=pagination,
            )
        return out

    return _run_logged_tool(
        "finviz_earnings",
        {"period": period, "limit": limit, "page": page, "detail": detail},
        _run,
    )
