"""
Finviz MCP tools for stock screening, fundamentals, news, and market data.

Exposes finvizfinance library functionality as MCP tools.
Note: Data is delayed 15-20 minutes; US stocks only.
"""

from typing import Any, Dict, Optional, Literal
import json
import logging
from urllib.parse import parse_qs

from .server import mcp
from ..services.finviz_service import (
    get_stock_fundamentals,
    get_stock_description,
    get_stock_news,
    get_stock_insider_trades,
    get_stock_ratings,
    get_stock_peers,
    screen_stocks,
    get_general_news,
    get_insider_activity,
    get_forex_performance,
    get_crypto_performance,
    get_futures_performance,
    get_earnings_calendar,
    get_economic_calendar,
    get_earnings_calendar_api,
    get_dividends_calendar_api,
)

logger = logging.getLogger(__name__)


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
    return get_stock_fundamentals(symbol)


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
    return get_stock_description(symbol)


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
        List of news items with title, link, date, source
    """
    if symbol:
        return get_stock_news(symbol, limit=limit, page=page)
    else:
        return get_general_news(news_type="news", limit=limit, page=page)


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
    return get_stock_insider_trades(symbol, limit=limit, page=page)


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
    return get_stock_ratings(symbol)


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
    return get_stock_peers(symbol)


@mcp.tool()
def finviz_screen(
    filters: Optional[str] = None,
    order: Optional[str] = None,
    limit: int = 50,
    page: int = 1,
    view: Literal["overview", "valuation", "financial", "ownership", "performance", "technical"] = "overview",
) -> Dict[str, Any]:
    """
    Screen stocks using Finviz screener with filters.
    
    Parameters
    ----------
    filters : str, optional
        JSON string of filter dict, e.g. '{"Exchange": "NASDAQ", "Sector": "Technology"}'
        Common filters: Exchange, Index, Sector, Industry, Country, Market Cap.,
        P/E, Forward P/E, PEG, P/S, P/B, Dividend Yield, EPS growth this year,
        Return on Equity, Current Ratio, Analyst Recom., RSI (14), 
        50-Day Simple Moving Average, Average Volume, Price, Beta
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
    Screen for tech stocks on NASDAQ:
    >>> finviz_screen(filters='{"Exchange": "NASDAQ", "Sector": "Technology"}')
    
    Screen for undervalued large caps:
    >>> finviz_screen(filters='{"Market Cap.": "Large ($10bln to $200bln)", "P/E": "Under 15"}')
    
    Screen for high dividend stocks:
    >>> finviz_screen(filters='{"Dividend Yield": "Over 5%"}', view="valuation")
    """
    filters_dict = None
    if filters:
        try:
            filters_dict = json.loads(filters)
        except (json.JSONDecodeError, TypeError):
            return {"error": f"Invalid filters JSON: {filters}"}
    
    return screen_stocks(filters=filters_dict, order=order, limit=limit, page=page, view=view)


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
    return get_general_news(news_type=news_type, limit=limit, page=page)


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
    return get_insider_activity(option=option, limit=limit, page=page)


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
    return get_forex_performance()


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
    return get_crypto_performance()


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
    return get_futures_performance()


@mcp.tool()
def finviz_calendar(
    calendar: str = "economic",
    impact: Optional[Literal["low", "medium", "high"]] = None,
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
    date_from : str, optional
        ISO date "YYYY-MM-DD" (maps to Finviz query param `dateFrom`).
    date_to : str, optional
        ISO date "YYYY-MM-DD" (maps to Finviz query param `dateTo`).
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
    raw_q = params or query
    if raw_q:
        q = raw_q.strip()
        if q.startswith("?"):
            q = q[1:]
        parsed = {k: (v[-1] if v else None) for k, v in parse_qs(q).items()}
        date_from = date_from or parsed.get("dateFrom")
        date_to = date_to or parsed.get("dateTo")
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
        return get_economic_calendar(impact=impact, limit=limit, page=page, date_from=date_from, date_to=date_to)
    if cal == "earnings":
        return get_earnings_calendar_api(limit=limit, page=page, date_from=date_from, date_to=date_to)
    if cal == "dividends":
        return get_dividends_calendar_api(limit=limit, page=page, date_from=date_from, date_to=date_to)
    return {"error": f"Unsupported calendar '{calendar}'. Expected economic, earnings, or dividends."}


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
    return get_earnings_calendar(period=period, limit=limit, page=page)
