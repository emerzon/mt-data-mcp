"""Pagination utilities for Finviz service."""
import math
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

from .client import get_finviz_page_limit_max, get_finviz_screener_max_rows


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        numeric = int(value)
    except Exception:
        return int(default)
    return max(1, int(numeric))


def _sanitize_finviz_cell(value: Any) -> Any:
    """Coerce Finviz/pandas missing markers to ``None``.

    pandas ``to_dict`` emits ``float('nan')`` for missing numerics and the
    finvizfinance upstream often yields the literal string ``"nan"`` / ``"-"``
    for fields it could not parse. Normalize all of these to ``None`` so CLI
    and JSON consumers see a proper null instead of a programming artefact.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "nan" or stripped == "-":
            return None
    return value


def _sanitize_row(row: Any) -> Any:
    if isinstance(row, dict):
        return {key: _sanitize_finviz_cell(val) for key, val in row.items()}
    return row


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
    # Fetch one sentinel row beyond the requested page. Without it, a full
    # prefix is indistinguishable from a complete provider result.
    needed = (safe_limit * safe_page) + 1
    return max(1, min(safe_max_rows, needed))


def screener_pagination_metadata(
    *,
    fetched_count: int,
    fetch_limit: int,
    limit: int,
    page: int,
) -> Dict[str, Any]:
    """Describe exact or lower-bound pagination for a bounded prefix fetch."""
    safe_limit, safe_page = sanitize_pagination(limit, page)
    fetched = max(0, int(fetched_count))
    complete = fetched < int(fetch_limit)
    requested_end = safe_limit * safe_page
    has_more = bool(fetched > requested_end or not complete)
    if complete:
        pages = 0 if fetched == 0 else (fetched + safe_limit - 1) // safe_limit
        return {
            "total": fetched,
            "pages": pages,
            "has_more": has_more,
            "truncated": False,
        }
    return {
        "total": None,
        "pages": None,
        "has_more": True,
        "total_lower_bound": fetched,
        "truncated": True,
    }


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

    rows = [_sanitize_row(row) for row in rows]

    pages = 0 if total <= 0 else (total + safe_limit - 1) // safe_limit
    return rows, total, safe_limit, safe_page, pages


def run_screener_view(
    screener: Any,
    *,
    order: str = "Ticker",
    ascend: bool = True,
    limit: int = 20,
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
    original_get_table = getattr(screener, "_get_table", None)

    def _canonical_ticker_get_table(
        instance: Any,
        rows: Any,
        frame: Any,
        num_col_index: Any,
        table_header: Any,
        limit: int = -1,
    ) -> Any:
        selected_rows = list(rows[1:])
        if limit != -1:
            selected_rows = selected_rows[:limit]
        result = original_get_table(
            rows, frame, num_col_index, table_header, limit
        )
        if result is None or "Ticker" not in getattr(result, "columns", []):
            return result
        start_index = max(0, len(result) - len(selected_rows))
        ticker_column = list(result.columns).index("Ticker")
        for offset, row in enumerate(selected_rows):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            ticker_cell = cells[1]
            ticker = ticker_cell.get("data-boxover-ticker")
            if not ticker:
                ticker_link = ticker_cell.find("a", class_="company-ticker")
                ticker = ticker_link.get_text(strip=True) if ticker_link else None
            if ticker:
                result.iloc[start_index + offset, ticker_column] = str(ticker).strip()
        return result

    if callable(original_get_table):
        screener._get_table = MethodType(_canonical_ticker_get_table, screener)
    try:
        result = screener.screener_view(
            order=order,
            limit=fetch_limit,
            verbose=0,
            sleep_sec=0,
            ascend=ascend,
        )
    finally:
        if callable(original_get_table):
            screener._get_table = original_get_table
    return result, fetch_limit


__all__ = [
    "sanitize_pagination",
    "compute_screener_fetch_limit",
    "paginate_finviz_records",
    "screener_pagination_metadata",
    "run_screener_view",
    "_sanitize_finviz_cell",
]
