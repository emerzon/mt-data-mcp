"""HTTP client and session management for Finviz service."""
import math
import os
import threading
from typing import Any, Dict, Optional

import requests

# Configuration constants
_FINVIZ_HTTP_TIMEOUT = float(os.getenv("FINVIZ_HTTP_TIMEOUT", "15"))
_FINVIZ_SCREENER_MAX_ROWS = int(os.getenv("FINVIZ_SCREENER_MAX_ROWS", "5000"))
_FINVIZ_PAGE_LIMIT_MAX = int(os.getenv("FINVIZ_PAGE_LIMIT_MAX", "500"))

# Thread-safe HTTP session
_FINVIZ_HTTP_SESSION: Optional[requests.Session] = None
_FINVIZ_HTTP_SESSION_LOCK = threading.Lock()


def _normalize_timeout_value(timeout: Any) -> float | tuple[float, float]:
    default_timeout = float(_FINVIZ_HTTP_TIMEOUT)
    if timeout is None:
        return default_timeout
    if isinstance(timeout, (list, tuple)):
        if len(timeout) != 2:
            return default_timeout
        try:
            connect_timeout = float(timeout[0])
            read_timeout = float(timeout[1])
        except Exception:
            return default_timeout
        if (
            not math.isfinite(connect_timeout)
            or not math.isfinite(read_timeout)
            or connect_timeout <= 0.0
            or read_timeout <= 0.0
        ):
            return default_timeout
        return (connect_timeout, read_timeout)
    try:
        timeout_value = float(timeout)
    except Exception:
        return default_timeout
    if not math.isfinite(timeout_value) or timeout_value <= 0.0:
        return default_timeout
    return timeout_value


def get_finviz_http_timeout() -> float:
    """Get the configured HTTP timeout for Finviz requests."""
    return _FINVIZ_HTTP_TIMEOUT


def get_finviz_screener_max_rows() -> int:
    """Get the maximum rows allowed for screener queries."""
    return _FINVIZ_SCREENER_MAX_ROWS


def get_finviz_page_limit_max() -> int:
    """Get the maximum page limit for pagination."""
    return _FINVIZ_PAGE_LIMIT_MAX


def _build_finviz_session() -> requests.Session:
    """Create a configured Finviz HTTP session."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def _reset_finviz_session() -> None:
    """Close and discard the shared Finviz session so the next request rebuilds it."""
    global _FINVIZ_HTTP_SESSION
    with _FINVIZ_HTTP_SESSION_LOCK:
        old = _FINVIZ_HTTP_SESSION
        _FINVIZ_HTTP_SESSION = None
    if old is not None:
        try:
            old.close()
        except Exception:
            pass


def finviz_http_get(
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any],
    timeout: Optional[Any] = None,
) -> Any:
    """HTTP GET helper with centralized timeout and pooled connections."""
    timeout_value = _normalize_timeout_value(timeout)
    # Testability: when requests.get is monkeypatched, honor that hook.
    if requests.get is not requests.api.get:
        return requests.get(url, headers=headers, params=params, timeout=timeout_value)

    global _FINVIZ_HTTP_SESSION
    if _FINVIZ_HTTP_SESSION is None:
        with _FINVIZ_HTTP_SESSION_LOCK:
            if _FINVIZ_HTTP_SESSION is None:
                _FINVIZ_HTTP_SESSION = _build_finviz_session()
    return _FINVIZ_HTTP_SESSION.get(url, headers=headers, params=params, timeout=timeout_value)


__all__ = [
    "get_finviz_http_timeout",
    "get_finviz_screener_max_rows",
    "get_finviz_page_limit_max",
    "finviz_http_get",
    "_build_finviz_session",
    "_reset_finviz_session",
]
