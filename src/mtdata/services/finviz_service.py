"""
Finviz data service wrapper.

Provides structured access to finvizfinance library for stock fundamentals,
news, insider trades, ratings, and screening.

Note: Finviz data is delayed 15-20 minutes; not suitable for real-time trading.
"""

import logging
import datetime
from typing import Any, Dict, List, Optional, Literal
import os
import threading
import importlib
import requests

logger = logging.getLogger(__name__)

_FINVIZ_HTTP_TIMEOUT = float(os.getenv("FINVIZ_HTTP_TIMEOUT", "15"))
_FINVIZ_SCREENER_MAX_ROWS = int(os.getenv("FINVIZ_SCREENER_MAX_ROWS", "5000"))
_FINVIZ_PAGE_LIMIT_MAX = int(os.getenv("FINVIZ_PAGE_LIMIT_MAX", "500"))
_FINVIZ_HTTP_SESSION: Optional[requests.Session] = None
_FINVIZ_HTTP_SESSION_LOCK = threading.Lock()


def _sanitize_pagination(limit: int, page: int) -> tuple[int, int]:
    """Clamp pagination inputs to sane bounds."""
    try:
        safe_limit = int(limit)
    except Exception:
        safe_limit = 50
    try:
        safe_page = int(page)
    except Exception:
        safe_page = 1
    safe_limit = max(1, min(_FINVIZ_PAGE_LIMIT_MAX, safe_limit))
    safe_page = max(1, safe_page)
    return safe_limit, safe_page


def _compute_screener_fetch_limit(limit: int, page: int, max_rows: int) -> int:
    """Rows to fetch from finvizfinance screener to satisfy current page safely."""
    safe_limit, safe_page = _sanitize_pagination(limit, page)
    needed = safe_limit * safe_page
    return max(1, min(max_rows, needed))


def _paginate_finviz_records(
    items: Any,
    *,
    limit: int,
    page: int,
) -> tuple[List[Any], int, int, int, int]:
    safe_limit, safe_page = _sanitize_pagination(limit, page)
    total = len(items) if items is not None else 0
    start_idx = (safe_page - 1) * safe_limit
    end_idx = start_idx + safe_limit

    if hasattr(items, "iloc"):
        rows = items.iloc[start_idx:end_idx].to_dict(orient="records")
    elif isinstance(items, list):
        rows = items[start_idx:end_idx]
    else:
        rows = []

    pages = 0 if total <= 0 else (total + safe_limit - 1) // safe_limit
    return rows, total, safe_limit, safe_page, pages


def _normalize_finviz_date_string(value: Any) -> Any:
    """Normalize Finviz short dates like `Nov 07 '25` to ISO 8601."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    text = text.replace("’", "'")
    for fmt in ("%b %d '%y", "%b %d %Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt).date().isoformat()
        except Exception:
            continue
    return value


def _normalize_finviz_dates_in_rows(rows: List[Dict[str, Any]], *keys: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    wanted = set(keys)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = dict(row)
        for key in wanted:
            if key in row_out:
                row_out[key] = _normalize_finviz_date_string(row_out.get(key))
        out.append(row_out)
    return out


def _run_screener_view(
    screener: Any,
    *,
    order: str = "Ticker",
    limit: int = 50,
    page: int = 1,
) -> Any:
    """Run screener_view with bounded rows and no inter-page sleep."""
    fetch_limit = _compute_screener_fetch_limit(limit=limit, page=page, max_rows=_FINVIZ_SCREENER_MAX_ROWS)
    return screener.screener_view(order=order, limit=fetch_limit, verbose=0, sleep_sec=0), fetch_limit


def _finviz_http_get(url: str, *, headers: Dict[str, str], params: Dict[str, Any]) -> Any:
    """HTTP GET helper with centralized timeout and pooled connections."""
    # Testability: when requests.get is monkeypatched, honor that hook.
    if requests.get is not requests.api.get:
        return requests.get(url, headers=headers, params=params, timeout=_FINVIZ_HTTP_TIMEOUT)

    global _FINVIZ_HTTP_SESSION
    if _FINVIZ_HTTP_SESSION is None:
        with _FINVIZ_HTTP_SESSION_LOCK:
            if _FINVIZ_HTTP_SESSION is None:
                session = requests.Session()
                session.headers.update({"User-Agent": "Mozilla/5.0"})
                _FINVIZ_HTTP_SESSION = session
    return _FINVIZ_HTTP_SESSION.get(url, headers=headers, params=params, timeout=_FINVIZ_HTTP_TIMEOUT)


def _apply_finvizfinance_timeout_patch() -> None:
    """Patch finvizfinance's internal bare requests.get call to include timeout."""
    try:
        import finvizfinance.quote as _fv_quote
    except Exception:
        return

    if bool(getattr(_fv_quote, "_mtdata_timeout_patched", False)):
        return

    _orig_get = _fv_quote.requests.get

    def _patched_get(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", _FINVIZ_HTTP_TIMEOUT)
        return _orig_get(*args, **kwargs)

    _fv_quote.requests.get = _patched_get
    _fv_quote._mtdata_timeout_patched = True


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            out = float(value)
            return out if out == out else None
        except Exception:
            return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        out = float(text)
        return out if out == out else None
    except Exception:
        return None


def _values_equivalent(lhs: Any, rhs: Any) -> bool:
    left_num = _to_float_or_none(lhs)
    right_num = _to_float_or_none(rhs)
    if left_num is not None and right_num is not None:
        scale = max(1.0, abs(left_num), abs(right_num))
        return abs(left_num - right_num) <= (1e-9 * scale)
    return lhs == rhs


def _crypto_day_week_identical(rows: List[Dict[str, Any]]) -> bool:
    matched = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "Perf Day" not in row or "Perf Week" not in row:
            continue
        matched += 1
        if not _values_equivalent(row.get("Perf Day"), row.get("Perf Week")):
            return False
    return matched > 0


def _crypto_price_display(value: Any) -> Optional[str]:
    num = _to_float_or_none(value)
    if num is None:
        return None
    abs_num = abs(num)
    if abs_num >= 1.0:
        decimals = 2
    elif abs_num >= 0.01:
        decimals = 4
    elif abs_num >= 0.0001:
        decimals = 6
    else:
        decimals = 8
    return f"{num:.{decimals}f}"


_FINVIZ_SCREENER_VIEWS = {
    "overview": ("finvizfinance.screener.overview", "Overview"),
    "valuation": ("finvizfinance.screener.valuation", "Valuation"),
    "financial": ("finvizfinance.screener.financial", "Financial"),
    "ownership": ("finvizfinance.screener.ownership", "Ownership"),
    "performance": ("finvizfinance.screener.performance", "Performance"),
    "technical": ("finvizfinance.screener.technical", "Technical"),
}


def _load_finviz_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _get_finviz_stock_quote(symbol: str) -> tuple[str, Any]:
    _apply_finvizfinance_timeout_patch()
    finvizfinance = _load_finviz_attr("finvizfinance.quote", "finvizfinance")
    symbol_norm = str(symbol).upper()
    return symbol_norm, finvizfinance(symbol_norm)


def _build_finviz_screener(view: str) -> Any:
    module_name, class_name = _FINVIZ_SCREENER_VIEWS.get(
        view,
        _FINVIZ_SCREENER_VIEWS["overview"],
    )
    screener_cls = _load_finviz_attr(module_name, class_name)
    return screener_cls()


def _fetch_finviz_market_performance_rows(
    *,
    module_name: str,
    class_name: str,
    empty_error: str,
) -> List[Dict[str, Any]]:
    _apply_finvizfinance_timeout_patch()
    market_cls = _load_finviz_attr(module_name, class_name)
    market_client = market_cls()
    df = market_client.performance()
    if df is None or df.empty:
        raise ValueError(empty_error)
    return df.to_dict(orient="records")


def get_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Get fundamental data for a stock symbol.
    
    Returns metrics like P/E, EPS, market cap, sector, industry, etc.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        fundament = stock.ticker_fundament()
        if fundament is None:
            return {"error": f"No fundamental data found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol_norm,
            "fundamentals": fundament,
        }
    except Exception as e:
        logger.exception(f"Error fetching fundamentals for {symbol}")
        return {"error": f"Failed to fetch fundamentals: {str(e)}"}


def get_stock_description(symbol: str) -> Dict[str, Any]:
    """Get company description for a stock symbol."""
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        desc = stock.ticker_description()
        if not desc:
            return {"error": f"No description found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol_norm,
            "description": desc,
        }
    except Exception as e:
        logger.exception(f"Error fetching description for {symbol}")
        return {"error": f"Failed to fetch description: {str(e)}"}


def get_stock_news(symbol: str, limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Get latest news for a stock symbol.
    
    Returns list of news items with title, link, date, source.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        news_df = stock.ticker_news()
        if news_df is None or news_df.empty:
            return {"error": f"No news found for {symbol}"}
        news_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            news_df,
            limit=limit,
            page=page,
        )
        return {
            "success": True,
            "symbol": symbol_norm,
            "count": len(news_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "news": news_list,
        }
    except Exception as e:
        message = str(e)
        if "404" in message or "quote.ashx?t=" in message:
            return {
                "error": (
                    f"{str(symbol).upper()} is not a Finviz-supported symbol. "
                    "finviz_news only covers US equities."
                )
            }
        logger.warning("Error fetching news for %s: %s", symbol, message)
        return {"error": f"Failed to fetch news: {message}"}


def get_stock_insider_trades(symbol: str, limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Get insider trading activity for a stock symbol.
    
    Returns list of insider trades with owner, relationship, date, transaction, cost, shares, value.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        insider_df = stock.ticker_inside_trader()
        if insider_df is None or insider_df.empty:
            return {"error": f"No insider trades found for {symbol}"}
        trades_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            insider_df,
            limit=limit,
            page=page,
        )
        trades_list = _normalize_finviz_dates_in_rows(trades_list, "Date")
        return {
            "success": True,
            "symbol": symbol_norm,
            "count": len(trades_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "insider_trades": trades_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching insider trades for {symbol}")
        return {"error": f"Failed to fetch insider trades: {str(e)}"}


def get_stock_ratings(symbol: str) -> Dict[str, Any]:
    """
    Get analyst ratings for a stock symbol.
    
    Returns list of ratings with date, status, analyst, rating, price target.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        ratings_df = stock.ticker_outer_ratings()
        if ratings_df is None or ratings_df.empty:
            return {"error": f"No ratings found for {symbol}"}
        ratings_list = ratings_df.to_dict(orient="records")
        return {
            "success": True,
            "symbol": symbol_norm,
            "count": len(ratings_list),
            "ratings": ratings_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching ratings for {symbol}")
        return {"error": f"Failed to fetch ratings: {str(e)}"}


def get_stock_peers(symbol: str) -> Dict[str, Any]:
    """Get peer companies for a stock symbol."""
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        peers = stock.ticker_peer()
        if not peers:
            return {"error": f"No peers found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol_norm,
            "peers": peers if isinstance(peers, list) else [peers],
        }
    except Exception as e:
        logger.exception(f"Error fetching peers for {symbol}")
        return {"error": f"Failed to fetch peers: {str(e)}"}


def screen_stocks(
    filters: Optional[Dict[str, str]] = None,
    order: Optional[str] = None,
    limit: int = 50,
    page: int = 1,
    view: str = "overview",
) -> Dict[str, Any]:
    """
    Screen stocks using Finviz screener.
    
    Parameters
    ----------
    filters : dict, optional
        Filter dictionary, e.g. {"Exchange": "NASDAQ", "Sector": "Technology"}
        Available filters: Exchange, Index, Sector, Industry, Country, Market Cap.,
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
        Sort order, e.g. "-marketcap" for descending market cap
    limit : int
        Max results per page (default 50)
    page : int
        Page number (default 1)
    view : str
        Screener view type: "overview", "valuation", "financial", "ownership",
        "performance", "technical"
    
    Returns
    -------
    dict
        Screener results with stock list
    """
    try:
        _apply_finvizfinance_timeout_patch()
        view_lower = view.lower().strip()
        screener = _build_finviz_screener(view_lower)
        
        if filters:
            screener.set_filter(filters_dict=filters)
        order_name = str(order).strip() if isinstance(order, str) and str(order).strip() else "Ticker"

        df, fetch_limit = _run_screener_view(
            screener,
            order=order_name,
            limit=limit,
            page=page,
        )
        if df is None:
            return {"error": "Failed to fetch screener results from Finviz."}

        stocks_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            df,
            limit=limit,
            page=page,
        )
        if df.empty:
            return {
                "success": True,
                "count": 0,
                "total": 0,
                "page": safe_page,
                "pages": 0,
                "stocks": [],
                "message": "No stocks matched the filter criteria",
            }

        truncated = bool(total >= fetch_limit and fetch_limit >= _FINVIZ_SCREENER_MAX_ROWS)
        return {
            "success": True,
            "view": view_lower,
            "filters": filters or {},
            "count": len(stocks_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "truncated": truncated,
            "stocks": stocks_list,
        }
    except Exception as e:
        logger.warning("Error running stock screener: %s", e)
        return {"error": f"Failed to run screener: {str(e)}"}


def get_general_news(news_type: str = "news", limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Get general financial news from Finviz.
    
    Parameters
    ----------
    news_type : str
        Type of news: "news" or "blogs"
    limit : int
        Max items per page
    page : int
        Page number (default 1)
    """
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.news import News

        fnews = News()
        all_news = fnews.get_news()

        if news_type.lower() == "blogs":
            items = all_news.get("blogs", [])
        else:
            items = all_news.get("news", [])

        # Check if items is empty (handle DataFrame or list)
        if hasattr(items, "empty"):
            if items.empty:
                return {"error": f"No {news_type} found"}
            total = len(items)
        elif not items:
            return {"error": f"No {news_type} found"}
        items_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            items,
            limit=limit,
            page=page,
        )

        return {
            "success": True,
            "type": news_type.lower(),
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "items": items_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching general news")
        return {"error": f"Failed to fetch news: {str(e)}"}


def get_insider_activity(option: str = "latest", limit: int = 50, page: int = 1) -> Dict[str, Any]:
    """
    Get general insider trading activity.
    
    Parameters
    ----------
    option : str
        Type: "latest", "top week", "top owner trade", "insider buy", "insider sale"
    limit : int
        Max items per page
    page : int
        Page number (default 1)
    """
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.insider import Insider

        finsider = Insider(option=option)
        df = finsider.get_insider()

        if df is None or df.empty:
            return {"error": f"No insider activity found for option '{option}'"}

        items_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            df,
            limit=limit,
            page=page,
        )
        items_list = _normalize_finviz_dates_in_rows(items_list, "Date")
        return {
            "success": True,
            "option": option,
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "insider_trades": items_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching insider activity")
        return {"error": f"Failed to fetch insider activity: {str(e)}"}


def get_forex_performance() -> Dict[str, Any]:
    """Get forex currency pairs performance data."""
    try:
        items_list = _fetch_finviz_market_performance_rows(
            module_name="finvizfinance.forex",
            class_name="Forex",
            empty_error="No forex performance data available",
        )
        return {
            "success": True,
            "market": "forex",
            "count": len(items_list),
            "pairs": items_list,
        }
    except Exception as e:
        logger.exception("Error fetching forex performance")
        return {"error": f"Failed to fetch forex performance: {str(e)}"}


def get_crypto_performance() -> Dict[str, Any]:
    """Get cryptocurrency performance data."""
    try:
        items_list = _fetch_finviz_market_performance_rows(
            module_name="finvizfinance.crypto",
            class_name="Crypto",
            empty_error="No crypto performance data available",
        )
        warnings_out: List[str] = []
        for row in items_list:
            if not isinstance(row, dict) or "Price" not in row:
                continue
            price_display = _crypto_price_display(row.get("Price"))
            if price_display is not None:
                row["Price_display"] = price_display
        if _crypto_day_week_identical(items_list):
            for row in items_list:
                if isinstance(row, dict) and "Perf Week" in row and "Perf WTD" not in row:
                    row["Perf WTD"] = row.get("Perf Week")
            warnings_out.append(
                "Finviz returned identical 'Perf Day' and 'Perf Week' values across all rows; "
                "added 'Perf WTD' alias to clarify likely week-to-date semantics."
            )

        out = {
            "success": True,
            "market": "crypto",
            "count": len(items_list),
            "coins": items_list,
        }
        if warnings_out:
            out["warnings"] = warnings_out
        return out
    except Exception as e:
        logger.exception("Error fetching crypto performance")
        return {"error": f"Failed to fetch crypto performance: {str(e)}"}


def get_futures_performance() -> Dict[str, Any]:
    """Get futures market performance data."""
    try:
        items_list = _fetch_finviz_market_performance_rows(
            module_name="finvizfinance.future",
            class_name="Future",
            empty_error="No futures performance data available",
        )
        return {
            "success": True,
            "market": "futures",
            "count": len(items_list),
            "futures": items_list,
        }
    except Exception as e:
        logger.exception("Error fetching futures performance")
        return {"error": f"Failed to fetch futures performance: {str(e)}"}


def get_earnings_calendar(
    period: str = "This Week",
    limit: int = 50,
    page: int = 1,
) -> Dict[str, Any]:
    """Get upcoming earnings calendar from Finviz.

    Notes
    -----
    finvizfinance exposes earnings via ``finvizfinance.earnings.Earnings``.
    Supported periods (per library): "This Week", "Next Week", "Previous Week",
    "This Month".
    """
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.earnings import Earnings

        allowed_periods = {"This Week", "Next Week", "Previous Week", "This Month"}
        if period not in allowed_periods:
            raise ValueError(
                "Invalid period '{period}'. Available period: {periods}".format(
                    period=period,
                    periods=sorted(allowed_periods),
                )
            )

        cal = Earnings(period=period)
        df = getattr(cal, "df", None)
        if df is None:
            # Fallback for finvizfinance variants exposing a method instead of .df
            getter = getattr(cal, "earnings", None)
            if callable(getter):
                df = getter()

        if df is None or df.empty:
            return {"error": "No earnings calendar data available"}

        items_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            df,
            limit=limit,
            page=page,
        )
        return {
            "success": True,
            "period": period,
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "truncated": False,
            "earnings": items_list,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching earnings calendar")
        return {"error": f"Failed to fetch earnings calendar: {str(e)}"}


def get_economic_calendar(
    limit: int = 100,
    page: int = 1,
    impact: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Finviz economic calendar (macro releases)."""
    safe_limit, safe_page = _sanitize_pagination(limit, page)

    impact_norm: Optional[Literal["low", "medium", "high"]] = None
    if impact is not None:
        impact_norm = impact.strip().lower()  # type: ignore[assignment]
        allowed = {"low", "medium", "high"}
        if impact_norm not in allowed:
            return {
                "error": "Invalid impact '{impact}'. Expected one of: low, medium, high".format(
                    impact=impact
                )
            }
    try:
        # Finviz migrated the calendar UI to client-side rendering; the legacy
        # finvizfinance HTML table parser often returns no rows. Prefer the JSON API.
        default_days = 7
        date_from, date_to = _resolve_date_range(
            date_from=date_from,
            date_to=date_to,
            default_days=default_days,
        )

        api_date_from = _align_to_next_monday_if_weekend(date_from)
        raw_items = _fetch_finviz_economic_calendar_items(date_from=api_date_from, date_to=date_to)
        events = _normalize_finviz_economic_calendar_items(raw_items)
        events = _filter_calendar_events_by_date(events, date_from=date_from, date_to=date_to)

        if impact_norm is not None:
            events = [e for e in events if str(e.get("Impact", "")).lower() == impact_norm]

        events.sort(key=lambda e: str(e.get("Datetime", "")))

        total = len(events)
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit
        items_list = events[start_idx:end_idx]

        message = None
        if impact_norm and total == 0:
            message = "No economic releases matched impact='{impact}'".format(impact=impact_norm)

        return {
            "success": True,
            "source": "finviz_api",
            "impact": impact_norm,
            "dateFrom": date_from,
            "dateTo": date_to,
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit if total else 0,
            "items": items_list,
            "message": message,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching economic calendar")
        return {"error": f"Failed to fetch economic calendar: {str(e)}"}


def get_earnings_calendar_api(
    limit: int = 50,
    page: int = 1,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Finviz earnings calendar via the Finviz JSON API."""
    try:
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        default_days = 7 if (date_from is not None and date_to is None) else 30
        date_from, date_to = _resolve_date_range(date_from=date_from, date_to=date_to, default_days=default_days)
        payload = _fetch_finviz_calendar_paged(
            kind="earnings",
            date_from=date_from,
            date_to=date_to,
            page=safe_page,
            page_size=safe_limit,
        )
        items = payload.get("items") or []
        total = int(payload.get("totalItemsCount") or len(items))
        pages = int(payload.get("totalPages") or ((total + safe_limit - 1) // safe_limit if total else 0))
        return {
            "success": True,
            "source": "finviz_api",
            "calendar": "earnings",
            "dateFrom": date_from,
            "dateTo": date_to,
            "count": len(items),
            "total": total,
            "page": int(payload.get("page") or safe_page),
            "pages": pages,
            "items": items,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching earnings calendar (API)")
        return {"error": f"Failed to fetch earnings calendar: {str(e)}"}


def get_dividends_calendar_api(
    limit: int = 50,
    page: int = 1,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Finviz dividends calendar via the Finviz JSON API."""
    try:
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        default_days = 7 if (date_from is not None and date_to is None) else 30
        date_from, date_to = _resolve_date_range(date_from=date_from, date_to=date_to, default_days=default_days)
        payload = _fetch_finviz_calendar_paged(
            kind="dividends",
            date_from=date_from,
            date_to=date_to,
            page=safe_page,
            page_size=safe_limit,
        )
        items = payload.get("items") or []
        total = int(payload.get("totalItemsCount") or len(items))
        pages = int(payload.get("totalPages") or ((total + safe_limit - 1) // safe_limit if total else 0))
        return {
            "success": True,
            "source": "finviz_api",
            "calendar": "dividends",
            "dateFrom": date_from,
            "dateTo": date_to,
            "count": len(items),
            "total": total,
            "page": int(payload.get("page") or safe_page),
            "pages": pages,
            "items": items,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching dividends calendar (API)")
        return {"error": f"Failed to fetch dividends calendar: {str(e)}"}


def _resolve_date_range(*, date_from: Optional[str], date_to: Optional[str], default_days: int) -> tuple[str, str]:
    """Resolve an ISO date range for Finviz API calls."""
    if date_to and not date_from:
        raise ValueError("date_from is required when date_to is provided")

    if date_from:
        try:
            df = datetime.date.fromisoformat(date_from)
        except ValueError as e:
            raise ValueError(f"Invalid date_from '{date_from}'. Expected YYYY-MM-DD") from e
    else:
        df = datetime.date.today()
        date_from = df.isoformat()

    if date_to:
        try:
            dt = datetime.date.fromisoformat(date_to)
        except ValueError as e:
            raise ValueError(f"Invalid date_to '{date_to}'. Expected YYYY-MM-DD") from e
    else:
        dt = df + datetime.timedelta(days=int(default_days))
        date_to = dt.isoformat()

    if dt < df:
        raise ValueError("date_to must be >= date_from")

    return date_from, date_to


def _align_to_next_monday_if_weekend(date_from: str) -> str:
    """Finviz economic calendar API appears to anchor by week; weekend anchors often return the prior week."""
    df = datetime.date.fromisoformat(date_from)
    wd = df.weekday()  # Monday=0 ... Sunday=6
    if wd == 5:  # Saturday
        df = df + datetime.timedelta(days=2)
    elif wd == 6:  # Sunday
        df = df + datetime.timedelta(days=1)
    return df.isoformat()


def _filter_calendar_events_by_date(
    events: List[Dict[str, Any]],
    *,
    date_from: str,
    date_to: str,
) -> List[Dict[str, Any]]:
    """Filter events to the inclusive [date_from, date_to] date range."""
    df = datetime.date.fromisoformat(date_from)
    dt = datetime.date.fromisoformat(date_to)

    filtered: List[Dict[str, Any]] = []
    for event in events:
        raw = event.get("Datetime")
        if not raw:
            continue
        try:
            if isinstance(raw, str):
                s = raw.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                if "T" in s:
                    d = datetime.datetime.fromisoformat(s).date()
                else:
                    d = datetime.date.fromisoformat(s)
            elif isinstance(raw, datetime.datetime):
                d = raw.date()
            elif isinstance(raw, datetime.date):
                d = raw
            else:
                continue
        except ValueError:
            continue

        if df <= d <= dt:
            filtered.append(event)

    return filtered


def _fetch_finviz_economic_calendar_items(date_from: str, date_to: str) -> List[Dict[str, Any]]:
    """Fetch raw economic calendar items from Finviz's JSON API."""
    url = "https://finviz.com/api/calendar/economic"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://finviz.com/calendar.ashx",
    }
    params = {"dateFrom": date_from, "dateTo": date_to}

    resp = _finviz_http_get(url, headers=headers, params=params)
    try:
        resp.raise_for_status()
        data = resp.json()
    finally:
        resp.close()
    if not isinstance(data, list):
        raise TypeError("Unexpected response type from Finviz API: {t}".format(t=type(data).__name__))

    items: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            items.append(item)
    return items


def _fetch_finviz_calendar_paged(
    *,
    kind: Literal["earnings", "dividends"],
    date_from: str,
    date_to: str,
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    """Fetch a paged calendar payload from Finviz's JSON API."""

    url = f"https://finviz.com/api/calendar/{kind}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://finviz.com/calendar.ashx",
    }
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "page": max(1, int(page)),
        "pageSize": max(1, int(page_size)),
    }

    resp = _finviz_http_get(url, headers=headers, params=params)
    try:
        resp.raise_for_status()
        data = resp.json()
    finally:
        resp.close()
    if not isinstance(data, dict):
        raise TypeError("Unexpected response type from Finviz API: {t}".format(t=type(data).__name__))
    if "items" not in data or not isinstance(data.get("items"), list):
        raise TypeError("Unexpected payload shape from Finviz API (missing items list)")
    return data


def _normalize_finviz_economic_calendar_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize Finviz API items to the legacy calendar schema."""
    importance_to_impact: Dict[int, Literal["low", "medium", "high"]] = {
        1: "low",
        2: "medium",
        3: "high",
    }

    normalized: List[Dict[str, Any]] = []
    for item in items:
        importance = item.get("importance")
        impact = importance_to_impact.get(importance) if isinstance(importance, int) else None

        normalized.append(
            {
                "Datetime": item.get("date") or "",
                "Release": item.get("event") or "",
                "Impact": impact or "",
                "For": item.get("ticker") or "",
                "Actual": item.get("actual") or "",
                "Expected": item.get("forecast") or item.get("teforecast") or "",
                "Prior": item.get("previous") or "",
                "Category": item.get("category") or "",
                "Reference": item.get("reference") or "",
                "ReferenceDate": item.get("referenceDate") or "",
            }
        )

    return normalized
