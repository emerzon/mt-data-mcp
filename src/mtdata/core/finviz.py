"""
Finviz MCP tools for stock screening, fundamentals, news, and market data.

Exposes finvizfinance library functionality as MCP tools.
Note: Data is delayed 15-20 minutes; US stocks only.
"""

import json
import logging
import re
from datetime import datetime, time as datetime_time, timezone, timedelta
from typing import Any, Callable, Dict, Literal, Optional, Union
from urllib.parse import parse_qs

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
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import normalize_output_verbosity_detail
from .schema import CompactFullDetailLiteral

logger = logging.getLogger(__name__)

_PAIR_SUFFIXES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
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
    "technicals": (
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


def _looks_like_non_equity_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip().upper()
    if not s:
        return False
    if "/" in s or ":" in s:
        return True
    if len(s) == 6 and s[:3].isalpha() and s[3:].isalpha() and s[3:] in _PAIR_SUFFIXES:
        return True
    return False


def _normalize_equity_symbol(symbol: str, *, tool_name: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    symbol_norm = str(symbol or "").strip().upper()
    if not symbol_norm:
        return None, _finviz_error_payload(
            f"{tool_name} requires a symbol.",
            code="finviz_symbol_required",
            operation=tool_name,
            details={"tool": tool_name},
        )
    if _looks_like_non_equity_symbol(symbol_norm):
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


def _run_logged_tool(
    operation: str,
    fields: Dict[str, Any],
    fn: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    return run_logged_operation(
        logger,
        operation=operation,
        func=fn,
        **fields,
    )


def _snake_finviz_market_key(value: Any) -> str:
    key = str(value).strip().lower()
    for old, new in (("%", "pct"), ("/", "_"), ("&", "and"), ("-", "_")):
        key = key.replace(old, new)
    return "_".join(part for part in key.replace(".", "").split() if part)


def _normalize_finviz_market_payload(
    result: Dict[str, Any],
    *,
    rows_key: str,
    limit: Optional[int] = None,
    detail: str = "compact",
    tool: str,
    request: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(result, dict) or "error" in result:
        return result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    rows = result.get(rows_key, [])
    normalized_rows = [
        _canonicalize_finviz_market_row(
            {_snake_finviz_market_key(key): value for key, value in row.items()}
        )
        if isinstance(row, dict)
        else row
        for row in (rows if isinstance(rows, list) else [])
    ]
    limit_value = _coerce_finviz_limit(limit, default=len(normalized_rows))
    limited_rows = normalized_rows[:limit_value]
    out = {key: value for key, value in result.items() if key != rows_key}
    out["items"] = limited_rows
    out["count"] = len(limited_rows)
    available = len(normalized_rows)
    out["available_count"] = available
    omitted = max(0, available - len(limited_rows))
    if omitted:
        out["omitted_item_count"] = omitted
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["meta"] = _build_tool_contract_meta(
            tool=tool,
            request=request,
            stats={"available": available, "returned": len(limited_rows)},
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
    if "perf" in out and not any(key.startswith("perf_") for key in out):
        out["perf_pct"] = out.pop("perf")
    return out


def _coerce_finviz_limit(limit: Optional[int], *, default: int) -> int:
    if limit is None:
        return max(0, int(default))
    return max(0, int(limit))


def _build_tool_contract_meta(
    *,
    tool: str,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, Any]] = None,
    legends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tool": tool,
        "request": {
            key: value for key, value in request.items() if value is not None
        },
        "runtime": {},
    }
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


def _invalid_finviz_screen_filters_error(filters: Any) -> Dict[str, Any]:
    if isinstance(filters, str):
        raw = filters.strip()
        if raw and not raw.startswith("{"):
            message = (
                "Invalid filters format. Received a string value, but finviz_screen expects a JSON object "
                "(dict) mapping filter names as keys to filter values or Finviz screener shorthand tokens like "
                "'cap_largeover,exch_nyse'. Example: "
                "{'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
                "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
                f"Got: {filters!r}"
            )
        else:
            message = (
                "Invalid filters format. Provide filters as a JSON object (dict) or JSON string with filter names as keys "
                "and filter values as values. Example: {'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
                "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
                f"Got: {filters}"
            )
    else:
        message = (
            "Invalid filters format. Provide filters as a JSON object (dict) or JSON string with filter names as keys "
            "and filter values as values. Example: {'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
            "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
            f"Got: {filters}"
        )
    return _finviz_error_payload(
        message,
        code="finviz_screen_filters_invalid",
        operation="finviz_screen",
        details={"received_type": type(filters).__name__},
    )


def _parse_finviz_screen_shorthand(raw: str) -> Optional[Dict[str, Any]]:
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

    filters: Dict[str, Any] = {}
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        match = reverse_filters.get(token)
        if match is None:
            return None
        filters[match[0]] = match[1]
    return filters or None


def _resolve_finviz_screen_filters(filters: Any) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if filters is None:
        return None, None
    if isinstance(filters, dict):
        return filters, None
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
    if "_" in raw:
        parsed = _parse_finviz_screen_shorthand(raw)
        if parsed is not None:
            return parsed, None
    return None, _invalid_finviz_screen_filters_error(filters)


def _resolve_preferred_text_arg(
    *,
    preferred_name: str,
    preferred_value: Optional[str],
    legacy_name: str,
    legacy_value: Optional[str],
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    preferred = str(preferred_value or "").strip() or None
    legacy = str(legacy_value or "").strip() or None
    if preferred and legacy and preferred != legacy:
        return None, _finviz_error_payload(
            (
                f"Provide either {preferred_name} or {legacy_name}, "
                "not both with different values."
            ),
            code="finviz_conflicting_text_args",
            operation="finviz_calendar",
            details={
                "preferred_name": preferred_name,
                "legacy_name": legacy_name,
            },
        )
    return preferred or legacy, None


def _clean_finviz_text_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
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


def _normalize_finviz_news_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item

    out: Dict[str, Any] = {}
    for source_key, target_key in (
        ("Title", "title"),
        ("Source", "source"),
        ("Date", "published_at"),
        ("Link", "url"),
        ("title", "title"),
        ("source", "source"),
        ("published_at", "published_at"),
        ("url", "url"),
    ):
        if source_key not in item:
            continue
        value = _clean_finviz_text_value(item.get(source_key))
        if value in (None, ""):
            continue
        if target_key == "published_at":
            value = _normalize_finviz_published_at(value)
        out[target_key] = value
    return out


def _normalize_finviz_news_payload(
    result: Dict[str, Any],
    *,
    include_preferred_tool: bool = True,
) -> Dict[str, Any]:
    out = dict(result)
    out.setdefault("tool_scope", "raw_finviz_provider")
    if include_preferred_tool:
        out.setdefault("preferred_tool", "news")
    else:
        out.pop("preferred_tool", None)
    out.setdefault("output_shape", "flat_paginated_items")
    out.setdefault("timezone", "UTC")

    news_rows = result.get("news")
    items_rows = result.get("items")
    if not isinstance(news_rows, list) and not isinstance(items_rows, list):
        return out

    source_rows = news_rows if isinstance(news_rows, list) else items_rows
    out["items"] = [_normalize_finviz_news_item(item) for item in source_rows]
    out.pop("news", None)
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
    "P/E": "pe_ratio",
    "Forward P/E": "forward_pe",
    "P/S": "price_to_sales",
    "P/B": "price_to_book",
    "P/C": "price_to_cash",
    "P/FCF": "price_to_free_cash_flow",
    "Price/Cash": "price_to_cash",
    "Price/Free Cash Flow": "price_to_free_cash_flow",
    "EPS (ttm)": "eps_ttm",
    "EPS next Y": "eps_next_y",
    "EPS next Q": "eps_next_q",
    "52W High": "high_52w",
    "52W Low": "low_52w",
    "RSI (14)": "rsi_14",
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
    "Dividend %": "dividend_yield",
    "Dividend Est.": "dividend_est",
    "Dividend TTM": "dividend_ttm",
    "Dividend Ex-Date": "dividend_ex_date",
    "Dividend Gr. 3Y": "dividend_growth_3y",
    "Dividend Gr. 5Y": "dividend_growth_5y",
}

_FINVIZ_EARNINGS_COMPACT_FIELDS = (
    "symbol",
    "company",
    "earnings",
    "eps_estimate",
    "market_cap",
    "price",
    "change",
    "volume",
)


def _normalize_finviz_output_key(key: Any) -> str:
    text = str(key).strip()
    mapped = _FINVIZ_OUTPUT_KEY_MAP.get(text)
    if mapped:
        return mapped
    text = text.replace("%", " pct ").replace("&", " and ").replace("/", " ")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or str(key)


def _normalize_finviz_output_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return {_normalize_finviz_output_key(key): value for key, value in row.items()}


def _normalize_finviz_output_rows(rows: Any) -> Any:
    if not isinstance(rows, list):
        return rows
    return [_normalize_finviz_output_row(row) for row in rows]


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


def _normalize_finviz_rating_rows(rows: Any) -> List[Any]:
    normalized = _normalize_finviz_output_rows(rows)
    if not isinstance(normalized, list):
        return []
    for row in normalized:
        if isinstance(row, dict) and "date" in row:
            row["date"] = _normalize_finviz_date_value(row.get("date"))
    return normalized


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
        compact_rows.append(row)
    return compact_rows


def _normalize_finviz_calendar_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("error"):
        return result
    out: Dict[str, Any] = {}
    for key, value in result.items():
        normalized_key = _normalize_finviz_output_key(key)
        out[normalized_key] = value
    if isinstance(out.get("items"), list):
        out["items"] = _normalize_finviz_output_rows(out["items"])
    return out


def _validate_finviz_detail(detail: str, *, operation: str) -> Optional[Dict[str, Any]]:
    normalized = str(detail or "full").strip().lower()
    if normalized in {"compact", "standard", "summary", "full"}:
        return None
    return _finviz_error_payload(
        "detail must be 'compact' or 'full'.",
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


def _compact_finviz_insider_payload(result: Dict[str, Any], *, detail: str) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_insider")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="full")
    rows = result.get("insider_trades")
    if not isinstance(rows, list):
        return result
    normalized_rows = _normalize_finviz_output_rows(rows)
    out = {key: value for key, value in result.items() if key != "insider_trades"}
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["items"] = normalized_rows
        out["count"] = len(normalized_rows)
        return out
    compact_rows = normalized_rows[:3]
    transaction_texts = [_transaction_text(row) for row in normalized_rows if isinstance(row, dict)]
    buys = sum(1 for text in transaction_texts if "buy" in text or "purchase" in text)
    sells = sum(1 for text in transaction_texts if "sell" in text or "sale" in text)
    out["items"] = compact_rows
    out["count"] = len(compact_rows)
    out["summary"] = {
        "counts": {
            "returned": len(compact_rows),
            "available": len(normalized_rows),
            "total": result.get("total", len(normalized_rows)),
            "buy_transactions": buys,
            "sell_transactions": sells,
        }
    }
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
    normalized_rows = _normalize_finviz_output_rows(rows)
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
        item = dict(row)
        item.pop("sec_form_4_link", None)
        if str(item.get("sec_form_4") or "").startswith(("http://", "https://")):
            item.pop("sec_form_4", None)
        compact_rows.append(item)

    transaction_texts = [
        _transaction_text(row) for row in normalized_rows if isinstance(row, dict)
    ]
    buys = sum(1 for text in transaction_texts if "buy" in text or "purchase" in text)
    sells = sum(1 for text in transaction_texts if "sell" in text or "sale" in text)
    out["items"] = compact_rows
    out["count"] = len(compact_rows)
    out["summary"] = {
        "counts": {
            "returned": len(compact_rows),
            "available": len(normalized_rows),
            "total": result.get("total", len(normalized_rows)),
            "buy_transactions": buys,
            "sell_transactions": sells,
        },
        "top_symbols": _summarize_insider_activity_tickers(normalized_rows),
    }
    out["omitted_item_count"] = max(0, len(normalized_rows) - len(compact_rows))
    return out


def _compact_finviz_ratings_payload(
    result: Dict[str, Any], *, detail: str, limit: Optional[int]
) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_ratings")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="full")
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
    compact_rows = limited_rows
    out["ratings"] = compact_rows
    out["summary"] = {
        "counts": {
            "returned": len(compact_rows),
            "available": len(normalized_rows),
        },
        "latest": compact_rows[0] if compact_rows else None,
    }
    out["omitted_item_count"] = omitted
    if omitted:
        out["show_all_hint"] = f"Set detail='full' or limit={len(normalized_rows)} to view all ratings."
    return out


def _compact_finviz_peers_payload(
    result: Dict[str, Any], *, detail: str, limit: Optional[int]
) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_peers")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="full")
    peers = result.get("peers")
    if not isinstance(peers, list):
        return result
    out = dict(result)
    limit_value = _coerce_finviz_limit(limit, default=len(peers))
    limited_peers = peers[:limit_value]
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["peers"] = limited_peers
        out["available_count"] = len(peers)
        out["omitted_item_count"] = max(0, len(peers) - len(limited_peers))
        return out
    compact_peers = limited_peers
    out["peers"] = compact_peers
    out["summary"] = {
        "counts": {
            "returned": len(compact_peers),
            "available": len(peers),
        }
    }
    out["omitted_item_count"] = max(0, len(peers) - len(compact_peers))
    return out


def _parse_finviz_fields(fields: Optional[Union[str, list[str]]]) -> Optional[list[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [field.strip() for field in fields.split(",") if field.strip()]
    return [str(field).strip() for field in fields if str(field).strip()]


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
    category_mode = str(category or "summary").strip().lower()
    if str(detail or "compact").strip().lower() not in {"compact", "standard", "summary", "full"}:
        return _finviz_error_payload(
            "detail must be 'compact' or 'full'.",
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
    if requested_fields is not None:
        selected_fields = requested_fields
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

    filtered = {
        _normalize_finviz_output_key(field): fundamentals[field]
        for field in selected_fields
        if field in fundamentals and fundamentals[field] not in (None, "")
    }
    out = dict(result)
    out["fundamentals"] = filtered
    out["detail"] = detail_mode
    out["category"] = category_out
    if detail_mode == "full":
        out["available_field_count"] = len(fundamentals)
        omitted_fields = [
            _normalize_finviz_output_key(field)
            for field in fundamentals
            if _normalize_finviz_output_key(field) not in filtered
        ]
        out["omitted_field_count"] = len(omitted_fields)
        if omitted_fields:
            out["omitted_fields"] = omitted_fields
    if requested_fields is not None:
        missing = [field for field in requested_fields if field not in fundamentals]
        if missing:
            out["missing_fields"] = missing
    return out


@mcp.tool()
def finviz_fundamentals(
    symbol: str,
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
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
        symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_fundamentals")
        if error is not None:
            return error
        assert symbol_norm is not None
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


@mcp.tool()
def finviz_description(symbol: str) -> Dict[str, Any]:
    """
    Get company business description for a US stock.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., AAPL, TSLA)
    
    Returns
    -------
    dict
        Company description text
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_description")
        if error is not None:
            return error
        assert symbol_norm is not None
        return get_stock_description(symbol_norm)

    return _run_logged_tool(
        "finviz_description",
        {"symbol": symbol},
        _run,
    )


@mcp.tool()
def finviz_news(symbol: Optional[str] = None, limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Raw Finviz news provider endpoint.

    Prefer `news` for trading workflows because it merges Finviz with MT5/CNBC
    sources, ranks relevance, and buckets general, related, impact, and event
    news. Use `finviz_news` when you specifically need Finviz pagination, URLs,
    or the raw flat provider schema.
    
    Parameters
    ----------
    symbol : str, optional
        Stock ticker symbol (e.g., NVDA, META). If omitted, returns general market news.
    limit : int
        Max news items per page (default 20)
    page : int
        Page number for pagination (default 1)
    
    Returns
    -------
    dict
        Stock-specific calls return normalized `items` rows with `title`,
        `source`, `published_at`, and `url` fields. Top-level metadata marks
        this as a raw Finviz provider endpoint and points traders to `news` as
        the preferred unified tool.
    """
    fields = {"symbol": symbol, "limit": limit, "page": page}

    def _run() -> Dict[str, Any]:
        if symbol:
            symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_news")
            if error is not None:
                return error
            assert symbol_norm is not None
            return _normalize_finviz_news_payload(
                get_stock_news(symbol_norm, limit=limit, page=page)
            )
        return _normalize_finviz_news_payload(
            get_general_news(news_type="news", limit=limit, page=page)
        )

    return _run_logged_tool("finviz_news", fields, _run)


@mcp.tool()
def finviz_insider(
    symbol: str,
    limit: int = 20,
    page: int = 1,
    detail: CompactFullDetailLiteral = "full",  # type: ignore
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
        "full" preserves all returned trades. "compact" returns the first
        three rows plus aggregate buy/sell counts.
    
    Returns
    -------
    dict
        List of insider trades
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_insider")
        if error is not None:
            return error
        assert symbol_norm is not None
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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
    limit: int = 3,
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
    
    Returns
    -------
    dict
        List of analyst ratings
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_ratings")
        if error is not None:
            return error
        assert symbol_norm is not None
        return _compact_finviz_ratings_payload(
            get_stock_ratings(symbol_norm),
            detail=detail,
            limit=limit,
        )

    return _run_logged_tool(
        "finviz_ratings",
        {"symbol": symbol, "detail": detail, "limit": limit},
        _run,
    )


@mcp.tool()
def finviz_peers(
    symbol: str,
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
    limit: int = 5,
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
        symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_peers")
        if error is not None:
            return error
        assert symbol_norm is not None
        return _compact_finviz_peers_payload(
            get_stock_peers(symbol_norm),
            detail=detail,
            limit=limit,
        )

    return _run_logged_tool(
        "finviz_peers",
        {"symbol": symbol, "detail": detail, "limit": limit},
        _run,
    )


@mcp.tool()
def finviz_screen(
    filters: Optional[Union[str, Dict[str, Any]]] = None,
    order: Optional[str] = None,
    limit: int = 50,
    page: int = 1,
    view: Literal["overview", "valuation", "financial", "ownership", "performance", "technical"] = "overview",
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
        Max results per page (default 50)
    page : int
        Page number for pagination (default 1)
    view : str
        Data view: overview, valuation, financial, ownership, performance, technical
    
    Returns
    -------
    dict
        List of stocks matching filters
    
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
    fields = {"limit": limit, "page": page, "view": view, "order": order}

    def _run() -> Dict[str, Any]:
        filters_dict, filter_error = _resolve_finviz_screen_filters(filters)
        if filter_error is not None:
            return filter_error

        result = screen_stocks(filters=filters_dict, order=order, limit=limit, page=page, view=view)
        if result.get("success") and isinstance(result.get("stocks"), list):
            out = dict(result)
            out["stocks"] = _normalize_finviz_output_rows(result["stocks"])
            return out
        return result

    return _run_logged_tool("finviz_screen", fields, _run)


@mcp.tool()
def finviz_market_news(
    news_type: Literal["news", "blogs"] = "news",
    limit: int = 20,
    page: int = 1,
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
        {"news_type": news_type, "limit": limit, "page": page},
        lambda: _normalize_finviz_news_payload(
            get_general_news(news_type=news_type, limit=limit, page=page),
            include_preferred_tool=False,
        ),
    )


@mcp.tool()
def finviz_insider_activity(
    option: Literal["latest", "top week", "top owner trade", "insider buy", "insider sale"] = "latest",
    limit: int = 50,
    page: int = 1,
    detail: str = "compact",
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
    detail : str
        Response detail level. Compact returns a short normalized item list and
        summary; full keeps all normalized rows including SEC link fields.
    
    Returns
    -------
    dict
        List of insider trades with ticker, owner, transaction details
    """
    return _run_logged_tool(
        "finviz_insider_activity",
        {"option": option, "limit": limit, "page": page, "detail": detail},
        lambda: _compact_finviz_insider_activity_payload(
            get_insider_activity(option=option, limit=limit, page=page),
            detail=detail,
        ),
    )


@mcp.tool()
def finviz_forex(
    limit: int = 20,
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
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
    request = {"limit": limit, "detail": detail}

    def _run() -> Dict[str, Any]:
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
        )

    return _run_logged_tool("finviz_forex", request, _run)


@mcp.tool()
def finviz_crypto(
    limit: int = 20,
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get futures market performance from Finviz.
    
    Returns performance data for major futures contracts including
    commodities, indices, bonds, and currencies.
    
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
    calendar: str = "economic",
    impact: Optional[Literal["low", "medium", "high"]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    params: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    """
    Get the Finviz calendar (economic, earnings, or dividends).

    Parameters
    ----------
    calendar : str
        Calendar type: "economic", "earnings", or "dividends" (also accepts paths like "/calendar/economic").
    impact : str, optional
        Economic only: filter by impact level: "low", "medium", or "high".
    start : str, optional
        Preferred start date alias in ISO format: YYYY-MM-DD.
    end : str, optional
        Preferred end date alias in ISO format: YYYY-MM-DD.
    date_from : str, optional
        Legacy alias for `start`. Maps to the Finviz query param `dateFrom`.
    date_to : str, optional
        Legacy alias for `end`. Maps to the Finviz query param `dateTo`.
    params : str, optional
        Optional query-string style parameters like `?dateFrom=2026-01-05&dateTo=2026-01-12`.
    query : str, optional
        Alias for `params`.
    limit : int
        Max events per page (default 20)
    page : int
        Page number for pagination (default 1)

    Returns
    -------
    dict
        Calendar entries (schema depends on calendar type).
    """
    fields = {
        "calendar": calendar,
        "impact": impact,
        "start": start,
        "end": end,
        "limit": limit,
        "page": page,
    }

    def _run() -> Dict[str, Any]:
        nonlocal impact, limit, page

        start_value, start_error = _resolve_preferred_text_arg(
            preferred_name="start",
            preferred_value=start,
            legacy_name="date_from",
            legacy_value=date_from,
        )
        if start_error is not None:
            return start_error
        end_value, end_error = _resolve_preferred_text_arg(
            preferred_name="end",
            preferred_value=end,
            legacy_name="date_to",
            legacy_value=date_to,
        )
        if end_error is not None:
            return end_error

        raw_q = params or query
        if raw_q:
            q = raw_q.strip()
            if q.startswith("?"):
                q = q[1:]
            parsed = {k: (v[-1] if v else None) for k, v in parse_qs(q).items()}
            start_value = start_value or parsed.get("dateFrom")
            end_value = end_value or parsed.get("dateTo")
            impact = impact or parsed.get("impact")  # type: ignore[assignment]
            if parsed.get("page"):
                try:
                    page = int(str(parsed["page"]))
                except ValueError:
                    pass
            if parsed.get("pageSize"):
                try:
                    limit = int(str(parsed["pageSize"]))
                except ValueError:
                    pass

        cal = (calendar or "").strip().lower()
        if "?" in cal:
            cal = cal.split("?", 1)[0]
        if "/" in cal:
            cal = cal.rstrip("/").split("/")[-1]

        if cal == "economic":
            return _normalize_finviz_calendar_payload(
                get_economic_calendar(
                    impact=impact,
                    limit=limit,
                    page=page,
                    date_from=start_value,
                    date_to=end_value,
                )
            )
        if cal == "earnings":
            return _normalize_finviz_calendar_payload(
                get_earnings_calendar_api(
                    limit=limit,
                    page=page,
                    date_from=start_value,
                    date_to=end_value,
                )
            )
        if cal == "dividends":
            return _normalize_finviz_calendar_payload(
                get_dividends_calendar_api(
                    limit=limit,
                    page=page,
                    date_from=start_value,
                    date_to=end_value,
                )
            )
        return {"error": f"Unsupported calendar '{calendar}'. Expected economic, earnings, or dividends."}

    return _run_logged_tool("finviz_calendar", fields, _run)


@mcp.tool()
def finviz_earnings(
    period: Literal["This Week", "Next Week", "Previous Week", "This Month"] = "This Week",
    limit: int = 10,
    page: int = 1,
    detail: str = "compact",
) -> Dict[str, Any]:
    """
    Get upcoming earnings calendar from Finviz.
    
    Returns scheduled earnings announcements with date, time,
    ticker, and expected EPS.
    
    Parameters
    ----------
    period : str
        Calendar period: "This Week", "Next Week", "Previous Week", "This Month"
    limit : int
        Max items per page (default 10)
    page : int
        Page number for pagination (default 1)
    detail : str
        Response detail level. Compact returns calendar-focused rows; full keeps
        all normalized provider columns and adds the tool metadata block.
    
    Returns
    -------
    dict
        Earnings calendar data
    """
    def _run() -> Dict[str, Any]:
        request = {
            "period": period,
            "limit": limit,
            "page": page,
            "detail": detail,
        }
        detail_error = _validate_finviz_detail(detail, operation="finviz_earnings")
        if detail_error is not None:
            return detail_error
        result = get_earnings_calendar(period=period, limit=limit, page=page)
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
        normalized_items = _normalize_finviz_output_rows(items)
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
        }
        stats = {
            "truncated": result.get("truncated"),
        }
        out: Dict[str, Any] = {
            "success": True,
            "period": result.get("period", period),
            "detail": detail_mode,
            "items": output_items,
            "count": int(result.get("count") or len(output_items)),
            "total": result.get("total"),
            "page": result.get("page"),
            "pages": result.get("pages"),
        }
        if out["detail"] != "full":
            out["omitted_item_count"] = max(0, int(out.get("total") or 0) - int(out["count"]))
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
