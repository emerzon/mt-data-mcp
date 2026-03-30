"""
MT5 Local News MCP Tool

Reads news from MetaTrader 5's internal news database (news.dat).
This provides access to economic calendar events and broker news feeds
stored locally by the MT5 terminal.
"""

import logging
from typing import Any, Dict, Optional

from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from ..services.news_service import get_mt5_news, get_news_categories

logger = logging.getLogger(__name__)


@mcp.tool()
def mt5_news(
    news_db_path: Optional[str] = None,
    limit: int = 50,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    category: Optional[str] = None,
    search_term: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read news from the MT5 terminal's local news database.
    
    Accesses economic calendar events and broker news feeds stored in the 
    MT5 news.dat file. The tool can auto-detect the news database location
    or accept an explicit path.
    
    Parameters
    ----------
    news_db_path : str, optional
        Full path to the news.dat file. If not provided, attempts to auto-detect
        from standard MT5 installation directories (e.g., AppData/Roaming/MetaQuotes/Terminal/)
    limit : int
        Maximum number of news items to return (default 50, max 500)
    from_date : str, optional
        Filter news from this date onwards (ISO format: YYYY-MM-DD)
    to_date : str, optional
        Filter news up to this date (ISO format: YYYY-MM-DD)
    category : str, optional
        Filter by news category/source (e.g., "FXStreet", "DailyFX", "Reuters")
    search_term : str, optional
        Filter news items containing this text in the headline/subject
    
    Returns
    -------
    dict
        Dictionary containing:
        - news: List of news items with relative_time, subject, category, source
        - count: Number of items returned
        - total_records: Total records in database
        - available_categories: List of available category filters
        - available_sources: List of available news sources
        - database_path: Path to the news database file used
        
    Examples
    --------
    # Get latest 50 news items (auto-detect path)
    mt5_news()
    
    # Get news from specific date with category filter
    mt5_news(from_date="2026-03-01", category="FXStreet")
    
    # Search for specific topics
    mt5_news(search_term="Fed", limit=20)
    
    # Use explicit path
    mt5_news(news_db_path="C:/Users/Admin/AppData/Roaming/MetaQuotes/Terminal/.../news/news.dat")
    """
    fields: Dict[str, Any] = {
        "news_db_path": news_db_path,
        "limit": limit,
        "from_date": from_date,
        "to_date": to_date,
        "category": category,
        "search_term": search_term
    }
    
    def _run() -> Dict[str, Any]:
        # Clamp limit to reasonable range
        safe_limit = max(1, min(limit, 500))
        
        return get_mt5_news(
            news_db_path=news_db_path,
            limit=safe_limit,
            from_date=from_date,
            to_date=to_date,
            category=category,
            search_term=search_term
        )
    
    return run_logged_operation(
        logger,
        operation="mt5_news",
        func=_run,
        **fields,
    )


@mcp.tool()
def mt5_news_categories(news_db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get available news categories and sources from the MT5 news database.
    
    Returns a summary of all news categories and their frequency counts,
    helpful for discovering what filters are available before querying news.
    
    Parameters
    ----------
    news_db_path : str, optional
        Full path to the news.dat file. If not provided, attempts to auto-detect
        from standard MT5 installation directories
    
    Returns
    -------
    dict
        Dictionary containing:
        - categories: List of category names with item counts
        - sources: List of news sources with item counts
        - total_records: Total number of news records in database
        - database_path: Path to the news database file
    """
    fields: Dict[str, Any] = {"news_db_path": news_db_path}
    
    def _run() -> Dict[str, Any]:
        return get_news_categories(news_db_path=news_db_path)
    
    return run_logged_operation(
        logger,
        operation="mt5_news_categories",
        func=_run,
        **fields,
    )
