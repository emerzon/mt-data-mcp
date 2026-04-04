"""HTTP client and session management for Finviz service."""
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


def get_finviz_http_timeout() -> float:
    """Get the configured HTTP timeout for Finviz requests."""
    return _FINVIZ_HTTP_TIMEOUT


def get_finviz_screener_max_rows() -> int:
    """Get the maximum rows allowed for screener queries."""
    return _FINVIZ_SCREENER_MAX_ROWS


def get_finviz_page_limit_max() -> int:
    """Get the maximum page limit for pagination."""
    return _FINVIZ_PAGE_LIMIT_MAX


def finviz_http_get(url: str, *, headers: Dict[str, str], params: Dict[str, Any]) -> Any:
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


def reset_finviz_http_session() -> None:
    """Reset the HTTP session (useful for testing)."""
    global _FINVIZ_HTTP_SESSION
    with _FINVIZ_HTTP_SESSION_LOCK:
        _FINVIZ_HTTP_SESSION = None


__all__ = [
    "get_finviz_http_timeout",
    "get_finviz_screener_max_rows",
    "get_finviz_page_limit_max",
    "finviz_http_get",
    "reset_finviz_http_session",
]
