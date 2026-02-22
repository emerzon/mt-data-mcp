"""
Finviz data service wrapper.

Provides structured access to finvizfinance library for stock fundamentals,
news, insider trades, ratings, and screening.

Note: Finviz data is delayed 15-20 minutes; not suitable for real-time trading.
"""

import logging
import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from functools import lru_cache
import time
import os

logger = logging.getLogger(__name__)

# Cache TTL in seconds (5 minutes default)
_CACHE_TTL = 300
_cache_timestamps: Dict[str, float] = {}
_FINVIZ_HTTP_TIMEOUT = float(os.getenv("FINVIZ_HTTP_TIMEOUT", "15"))
_FINVIZ_SCREENER_MAX_ROWS = int(os.getenv("FINVIZ_SCREENER_MAX_ROWS", "5000"))
_FINVIZ_PAGE_LIMIT_MAX = int(os.getenv("FINVIZ_PAGE_LIMIT_MAX", "500"))


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
    """HTTP GET helper with centralized timeout."""
    import requests

    return requests.get(url, headers=headers, params=params, timeout=_FINVIZ_HTTP_TIMEOUT)


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


def _is_cache_valid(key: str) -> bool:
    """Check if cached data is still valid."""
    ts = _cache_timestamps.get(key)
    if ts is None:
        return False
    return (time.time() - ts) < _CACHE_TTL


def _update_cache_ts(key: str) -> None:
    """Update cache timestamp."""
    _cache_timestamps[key] = time.time()


def get_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Get fundamental data for a stock symbol.
    
    Returns metrics like P/E, EPS, market cap, sector, industry, etc.
    """
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol.upper())
        fundament = stock.ticker_fundament()
        if fundament is None:
            return {"error": f"No fundamental data found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol.upper(),
            "fundamentals": fundament,
        }
    except Exception as e:
        logger.exception(f"Error fetching fundamentals for {symbol}")
        return {"error": f"Failed to fetch fundamentals: {str(e)}"}


def get_stock_description(symbol: str) -> Dict[str, Any]:
    """Get company description for a stock symbol."""
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol.upper())
        desc = stock.ticker_description()
        if not desc:
            return {"error": f"No description found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol.upper(),
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
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol.upper())
        news_df = stock.ticker_news()
        if news_df is None or news_df.empty:
            return {"error": f"No news found for {symbol}"}
        # Apply pagination
        total = len(news_df)
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit
        news_list = news_df.iloc[start_idx:end_idx].to_dict(orient="records")
        return {
            "success": True,
            "symbol": symbol.upper(),
            "count": len(news_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit,
            "news": news_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching news for {symbol}")
        return {"error": f"Failed to fetch news: {str(e)}"}


def get_stock_insider_trades(symbol: str, limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Get insider trading activity for a stock symbol.
    
    Returns list of insider trades with owner, relationship, date, transaction, cost, shares, value.
    """
    try:
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol.upper())
        insider_df = stock.ticker_inside_trader()
        if insider_df is None or insider_df.empty:
            return {"error": f"No insider trades found for {symbol}"}
        # Apply pagination
        total = len(insider_df)
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit
        trades_list = insider_df.iloc[start_idx:end_idx].to_dict(orient="records")
        return {
            "success": True,
            "symbol": symbol.upper(),
            "count": len(trades_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit,
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
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol.upper())
        ratings_df = stock.ticker_outer_ratings()
        if ratings_df is None or ratings_df.empty:
            return {"error": f"No ratings found for {symbol}"}
        ratings_list = ratings_df.to_dict(orient="records")
        return {
            "success": True,
            "symbol": symbol.upper(),
            "count": len(ratings_list),
            "ratings": ratings_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching ratings for {symbol}")
        return {"error": f"Failed to fetch ratings: {str(e)}"}


def get_stock_peers(symbol: str) -> Dict[str, Any]:
    """Get peer companies for a stock symbol."""
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol.upper())
        peers = stock.ticker_peer()
        if not peers:
            return {"error": f"No peers found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol.upper(),
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
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        _apply_finvizfinance_timeout_patch()
        view_lower = view.lower().strip()
        if view_lower == "overview":
            from finvizfinance.screener.overview import Overview
            screener = Overview()
        elif view_lower == "valuation":
            from finvizfinance.screener.valuation import Valuation
            screener = Valuation()
        elif view_lower == "financial":
            from finvizfinance.screener.financial import Financial
            screener = Financial()
        elif view_lower == "ownership":
            from finvizfinance.screener.ownership import Ownership
            screener = Ownership()
        elif view_lower == "performance":
            from finvizfinance.screener.performance import Performance
            screener = Performance()
        elif view_lower == "technical":
            from finvizfinance.screener.technical import Technical
            screener = Technical()
        else:
            from finvizfinance.screener.overview import Overview
            screener = Overview()
        
        if filters:
            screener.set_filter(filters_dict=filters)
        order_name = str(order).strip() if isinstance(order, str) and str(order).strip() else "Ticker"

        df, fetch_limit = _run_screener_view(
            screener,
            order=order_name,
            limit=safe_limit,
            page=safe_page,
        )
        if df is None or df.empty:
            return {
                "success": True,
                "count": 0,
                "total": 0,
                "page": safe_page,
                "pages": 0,
                "stocks": [],
                "message": "No stocks matched the filter criteria",
            }
        
        # Apply pagination
        total = len(df)
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit
        stocks_list = df.iloc[start_idx:end_idx].to_dict(orient="records")
        truncated = bool(total >= fetch_limit and fetch_limit >= _FINVIZ_SCREENER_MAX_ROWS)
        return {
            "success": True,
            "view": view_lower,
            "filters": filters or {},
            "count": len(stocks_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit,
            "truncated": truncated,
            "stocks": stocks_list,
        }
    except Exception as e:
        logger.exception("Error running stock screener")
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
        safe_limit, safe_page = _sanitize_pagination(limit, page)
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
        else:
            total = len(items)

        # Apply pagination
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit

        # items is typically a DataFrame
        if hasattr(items, "iloc"):
            items_list = items.iloc[start_idx:end_idx].to_dict(orient="records")
        else:
            items_list = items[start_idx:end_idx] if isinstance(items, list) else []

        return {
            "success": True,
            "type": news_type.lower(),
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit,
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
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.insider import Insider

        finsider = Insider(option=option)
        df = finsider.get_insider()

        if df is None or df.empty:
            return {"error": f"No insider activity found for option '{option}'"}

        # Apply pagination
        total = len(df)
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit
        items_list = df.iloc[start_idx:end_idx].to_dict(orient="records")
        return {
            "success": True,
            "option": option,
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit,
            "insider_trades": items_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching insider activity")
        return {"error": f"Failed to fetch insider activity: {str(e)}"}


def get_forex_performance() -> Dict[str, Any]:
    """Get forex currency pairs performance data."""
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.forex import Forex
        forex = Forex()
        df = forex.performance()
        
        if df is None or df.empty:
            return {"error": "No forex performance data available"}
        
        items_list = df.to_dict(orient="records")
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
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.crypto import Crypto
        crypto = Crypto()
        df = crypto.performance()
        
        if df is None or df.empty:
            return {"error": "No crypto performance data available"}
        
        items_list = df.to_dict(orient="records")
        return {
            "success": True,
            "market": "crypto",
            "count": len(items_list),
            "coins": items_list,
        }
    except Exception as e:
        logger.exception("Error fetching crypto performance")
        return {"error": f"Failed to fetch crypto performance: {str(e)}"}


def get_futures_performance() -> Dict[str, Any]:
    """Get futures market performance data."""
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.future import Future
        future = Future()
        df = future.performance()
        
        if df is None or df.empty:
            return {"error": "No futures performance data available"}
        
        items_list = df.to_dict(orient="records")
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
        safe_limit, safe_page = _sanitize_pagination(limit, page)
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

        # Apply pagination
        total = len(df)
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit
        items_list = df.iloc[start_idx:end_idx].to_dict(orient="records")
        return {
            "success": True,
            "period": period,
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit,
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
            "events": items_list,
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
            "earnings": items,
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
            "dividends": items,
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
    resp.raise_for_status()
    data = resp.json()
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
    resp.raise_for_status()
    data = resp.json()
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
                "CalendarId": item.get("calendarId"),
                "AllDay": item.get("allDay"),
                "Alert": item.get("alert"),
                "HasNoDetail": item.get("hasNoDetail"),
            }
        )

    return normalized
