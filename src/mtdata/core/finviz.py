"""
Finviz MCP tools for stock screening, fundamentals, news, and market data.

Exposes finvizfinance library functionality as MCP tools.
Note: Data is delayed 15-20 minutes; US stocks only.
"""

import json
import logging
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

logger = logging.getLogger(__name__)

_PAIR_SUFFIXES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
_FINVIZ_SCREEN_FILTERS_EXAMPLE = '{"Exchange":"NASDAQ","Sector":"Technology"}'


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
    return _finviz_error_payload(
        (
            "Invalid filters format. Provide filters as a JSON object (dict) or JSON string with filter names as keys "
            "and filter values as values. Example: {'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
            "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
            f"Got: {filters}"
        ),
        code="finviz_screen_filters_invalid",
        operation="finviz_screen",
        details={"received_type": type(filters).__name__},
    )


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


def _normalize_finviz_news_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item

    out: Dict[str, Any] = {}
    for source_key, target_key in (
        ("Title", "title"),
        ("Source", "source"),
        ("Date", "published_at"),
        ("Link", "url"),
    ):
        if source_key not in item:
            continue
        value = _clean_finviz_text_value(item.get(source_key))
        if value in (None, ""):
            continue
        out[target_key] = value
    return out


def _with_finviz_news_items_alias(result: Dict[str, Any]) -> Dict[str, Any]:
    news_rows = result.get("news")
    if not isinstance(news_rows, list) or "items" in result:
        return result

    out = dict(result)
    out["items"] = [_normalize_finviz_news_item(item) for item in news_rows]
    return out


@mcp.tool()
def finviz_fundamentals(symbol: str) -> Dict[str, Any]:
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
        Fundamental metrics for the stock
    
    Example
    -------
    >>> finviz_fundamentals("AAPL")
    {"success": True, "symbol": "AAPL", "fundamentals": {"P/E": "28.5", ...}}
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_fundamentals")
        if error is not None:
            return error
        assert symbol_norm is not None
        return get_stock_fundamentals(symbol_norm)

    return _run_logged_tool(
        "finviz_fundamentals",
        {"symbol": symbol},
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
    Get latest news. If symbol provided, returns stock-specific news.
    If no symbol, returns general market news.
    
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
        Stock-specific calls preserve the legacy `news` rows and also expose a
        normalized `items` alias with `title`, `source`, `published_at`, and
        `url` fields for easier consumption.
    """
    fields = {"symbol": symbol, "limit": limit, "page": page}

    def _run() -> Dict[str, Any]:
        if symbol:
            symbol_norm, error = _normalize_equity_symbol(symbol, tool_name="finviz_news")
            if error is not None:
                return error
            assert symbol_norm is not None
            return _with_finviz_news_items_alias(
                get_stock_news(symbol_norm, limit=limit, page=page)
            )
        return get_general_news(news_type="news", limit=limit, page=page)

    return _run_logged_tool("finviz_news", fields, _run)


@mcp.tool()
def finviz_insider(symbol: str, limit: int = 20, page: int = 1) -> Dict[str, Any]:
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
        return get_stock_insider_trades(symbol_norm, limit=limit, page=page)

    return _run_logged_tool(
        "finviz_insider",
        {"symbol": symbol, "limit": limit, "page": page},
        _run,
    )


@mcp.tool()
def finviz_ratings(symbol: str) -> Dict[str, Any]:
    """
    Get analyst ratings for a US stock.
    
    Returns ratings history with date, analyst firm, rating action,
    rating, and price target.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    
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
        return get_stock_ratings(symbol_norm)

    return _run_logged_tool(
        "finviz_ratings",
        {"symbol": symbol},
        _run,
    )


@mcp.tool()
def finviz_peers(symbol: str) -> Dict[str, Any]:
    """
    Get peer companies for a US stock.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    
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
        return get_stock_peers(symbol_norm)

    return _run_logged_tool(
        "finviz_peers",
        {"symbol": symbol},
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
        Filter criteria as a JSON string or dict. Filter names should be keys 
        with filter values as values. Use the exact filter names and values shown 
        on finviz.com screener.
        
        Can be provided as:
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
        filters_dict = None
        if filters:
            # If filters is already a dict, use it directly
            if isinstance(filters, dict):
                filters_dict = filters
            # If filters is a string, try to parse it as JSON
            elif isinstance(filters, str):
                try:
                    filters_dict = json.loads(filters)
                except (json.JSONDecodeError, TypeError):
                    return _invalid_finviz_screen_filters_error(filters)
                # Verify parsed JSON is a dict
                if not isinstance(filters_dict, dict):
                    return _invalid_finviz_screen_filters_error(filters)
            else:
                return _invalid_finviz_screen_filters_error(filters)

        return screen_stocks(filters=filters_dict, order=order, limit=limit, page=page, view=view)

    return _run_logged_tool("finviz_screen", fields, _run)


@mcp.tool()
def finviz_market_news(
    news_type: Literal["news", "blogs"] = "news",
    limit: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    """
    Get general financial market news from Finviz.
    
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
        List of news/blog items
    """
    return _run_logged_tool(
        "finviz_market_news",
        {"news_type": news_type, "limit": limit, "page": page},
        lambda: get_general_news(news_type=news_type, limit=limit, page=page),
    )


@mcp.tool()
def finviz_insider_activity(
    option: Literal["latest", "top week", "top owner trade", "insider buy", "insider sale"] = "latest",
    limit: int = 50,
    page: int = 1,
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
    
    Returns
    -------
    dict
        List of insider trades with ticker, owner, transaction details
    """
    return _run_logged_tool(
        "finviz_insider_activity",
        {"option": option, "limit": limit, "page": page},
        lambda: get_insider_activity(option=option, limit=limit, page=page),
    )


@mcp.tool()
def finviz_forex() -> Dict[str, Any]:
    """
    Get forex currency pairs performance from Finviz.
    
    Returns performance data for major currency pairs including
    daily change, weekly change, and other metrics.
    
    Returns
    -------
    dict
        Forex pairs performance data
    """
    return _run_logged_tool("finviz_forex", {}, get_forex_performance)


@mcp.tool()
def finviz_crypto() -> Dict[str, Any]:
    """
    Get cryptocurrency performance from Finviz.
    
    Returns performance data for major cryptocurrencies including
    price, daily change, volume, and market cap.
    
    Returns
    -------
    dict
        Crypto performance data
    """
    return _run_logged_tool("finviz_crypto", {}, get_crypto_performance)


@mcp.tool()
def finviz_futures() -> Dict[str, Any]:
    """
    Get futures market performance from Finviz.
    
    Returns performance data for major futures contracts including
    commodities, indices, bonds, and currencies.
    
    Returns
    -------
    dict
        Futures performance data
    """
    return _run_logged_tool("finviz_futures", {}, get_futures_performance)


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
    limit: int = 100,
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
        Max events per page (default 100)
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
            return get_economic_calendar(
                impact=impact,
                limit=limit,
                page=page,
                date_from=start_value,
                date_to=end_value,
            )
        if cal == "earnings":
            return get_earnings_calendar_api(
                limit=limit,
                page=page,
                date_from=start_value,
                date_to=end_value,
            )
        if cal == "dividends":
            return get_dividends_calendar_api(
                limit=limit,
                page=page,
                date_from=start_value,
                date_to=end_value,
            )
        return {"error": f"Unsupported calendar '{calendar}'. Expected economic, earnings, or dividends."}

    return _run_logged_tool("finviz_calendar", fields, _run)


@mcp.tool()
def finviz_earnings(
    period: Literal["This Week", "Next Week", "Previous Week", "This Month"] = "This Week",
    limit: int = 50,
    page: int = 1,
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
        Max items per page (default 50)
    page : int
        Page number for pagination (default 1)
    
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
        }
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
        pagination = {
            "page": result.get("page"),
            "total": result.get("total"),
            "pages": result.get("pages"),
        }
        stats = {
            "truncated": result.get("truncated"),
        }
        return {
            "success": True,
            "data": {
                "items": items,
            },
            "summary": {
                "counts": {
                    "items": int(result.get("count") or len(items)),
                }
            },
            "meta": _build_tool_contract_meta(
                tool="finviz_earnings",
                request=request,
                stats=stats,
                pagination=pagination,
            ),
        }

    return _run_logged_tool(
        "finviz_earnings",
        {"period": period, "limit": limit, "page": page},
        _run,
    )
