"""Pagination utilities for Finviz service."""
from typing import Any, Dict, List, Tuple

from .client import get_finviz_page_limit_max, get_finviz_screener_max_rows


def sanitize_pagination(limit: int, page: int) -> Tuple[int, int]:
    """Clamp pagination inputs to sane bounds."""
    try:
        safe_limit = int(limit)
    except Exception:
        safe_limit = 50
    try:
        safe_page = int(page)
    except Exception:
        safe_page = 1
    safe_limit = max(1, min(get_finviz_page_limit_max(), safe_limit))
    safe_page = max(1, safe_page)
    return safe_limit, safe_page


def compute_screener_fetch_limit(limit: int, page: int, max_rows: int) -> int:
    """Rows to fetch from finvizfinance screener to satisfy current page safely."""
    safe_limit, safe_page = sanitize_pagination(limit, page)
    needed = safe_limit * safe_page
    return max(1, min(max_rows, needed))


def paginate_finviz_records(
    items: Any,
    *,
    limit: int,
    page: int,
) -> Tuple[List[Any], int, int, int, int]:
    """Paginate a list or DataFrame of Finviz records."""
    safe_limit, safe_page = sanitize_pagination(limit, page)
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


def run_screener_view(
    screener: Any,
    *,
    order: str = "Ticker",
    limit: int = 50,
    page: int = 1,
) -> Tuple[Any, int]:
    """Run screener_view with bounded rows and no inter-page sleep."""
    fetch_limit = compute_screener_fetch_limit(
        limit=limit, page=page, max_rows=get_finviz_screener_max_rows()
    )
    return screener.screener_view(order=order, limit=fetch_limit, verbose=0, sleep_sec=0), fetch_limit


__all__ = [
    "sanitize_pagination",
    "compute_screener_fetch_limit",
    "paginate_finviz_records",
    "run_screener_view",
]
