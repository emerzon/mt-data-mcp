"""Pagination utilities for Finviz service."""
from typing import Any, List, Optional, Tuple

from .client import get_finviz_page_limit_max, get_finviz_screener_max_rows


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        numeric = int(value)
    except Exception:
        return int(default)
    return max(1, int(numeric))


def sanitize_pagination(
    limit: int,
    page: int,
    *,
    page_limit_max: Optional[int] = None,
) -> Tuple[int, int]:
    """Clamp pagination inputs to sane bounds."""
    try:
        safe_limit = int(limit)
    except Exception:
        safe_limit = 50
    try:
        safe_page = int(page)
    except Exception:
        safe_page = 1
    max_page_limit = _coerce_positive_int(
        get_finviz_page_limit_max() if page_limit_max is None else page_limit_max,
        default=get_finviz_page_limit_max(),
    )
    safe_limit = max(1, min(max_page_limit, safe_limit))
    safe_page = max(1, safe_page)
    return safe_limit, safe_page


def compute_screener_fetch_limit(
    limit: int,
    page: int,
    max_rows: int,
    *,
    page_limit_max: Optional[int] = None,
) -> int:
    """Rows to fetch from finvizfinance screener to satisfy current page safely."""
    safe_limit, safe_page = sanitize_pagination(limit, page, page_limit_max=page_limit_max)
    safe_max_rows = _coerce_positive_int(
        get_finviz_screener_max_rows() if max_rows is None else max_rows,
        default=get_finviz_screener_max_rows(),
    )
    needed = safe_limit * safe_page
    return max(1, min(safe_max_rows, needed))


def paginate_finviz_records(
    items: Any,
    *,
    limit: int,
    page: int,
    page_limit_max: Optional[int] = None,
) -> Tuple[List[Any], int, int, int, int]:
    """Paginate a list or DataFrame of Finviz records."""
    safe_limit, safe_page = sanitize_pagination(limit, page, page_limit_max=page_limit_max)
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
    screener_max_rows: Optional[int] = None,
    page_limit_max: Optional[int] = None,
) -> Tuple[Any, int]:
    """Run screener_view with bounded rows and no inter-page sleep."""
    max_rows = _coerce_positive_int(
        get_finviz_screener_max_rows() if screener_max_rows is None else screener_max_rows,
        default=get_finviz_screener_max_rows(),
    )
    fetch_limit = compute_screener_fetch_limit(
        limit=limit,
        page=page,
        max_rows=max_rows,
        page_limit_max=page_limit_max,
    )
    return screener.screener_view(order=order, limit=fetch_limit, verbose=0, sleep_sec=0), fetch_limit


__all__ = [
    "sanitize_pagination",
    "compute_screener_fetch_limit",
    "paginate_finviz_records",
    "run_screener_view",
]
