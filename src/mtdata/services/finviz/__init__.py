"""Canonical Finviz service package."""

from __future__ import annotations

import inspect
import types
from functools import update_wrapper

from . import api as _api

_REBOUND_CACHE: dict[int, types.FunctionType] = {}


def _rebind_function(func: types.FunctionType) -> types.FunctionType:
    cached = _REBOUND_CACHE.get(id(func))
    if cached is not None:
        return cached

    rebound = types.FunctionType(
        func.__code__,
        globals(),
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )
    _REBOUND_CACHE[id(func)] = rebound
    update_wrapper(rebound, func)
    rebound.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
    rebound.__dict__.update(getattr(func, "__dict__", {}))
    rebound.__module__ = __name__
    signature = getattr(func, "__signature__", None)
    if signature is not None:
        rebound.__signature__ = signature
    wrapped = getattr(func, "__wrapped__", None)
    if inspect.isfunction(wrapped) and wrapped.__module__ == func.__module__:
        rebound.__wrapped__ = _rebind_function(wrapped)
    return rebound


def _copy_namespace(module) -> None:
    for name, value in vars(module).items():
        if name.startswith("__"):
            continue
        globals()[name] = value

    for name, value in list(vars(module).items()):
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            globals()[name] = _rebind_function(value)


_copy_namespace(_api)

__all__ = [
    "get_stock_fundamentals",
    "get_stock_description",
    "get_stock_news",
    "get_stock_insider_trades",
    "get_stock_ratings",
    "get_stock_peers",
    "screen_stocks",
    "get_general_news",
    "get_insider_activity",
    "get_forex_performance",
    "get_crypto_performance",
    "get_futures_performance",
    "get_earnings_calendar",
    "get_economic_calendar",
    "get_earnings_calendar_api",
    "get_dividends_calendar_api",
    "_sanitize_pagination",
    "_compute_screener_fetch_limit",
    "_paginate_finviz_records",
    "_normalize_finviz_date_string",
    "_normalize_finviz_dates_in_rows",
    "_strip_string_fields_in_rows",
    "_run_screener_view",
    "_finviz_http_get",
    "_apply_finvizfinance_timeout_patch",
    "_to_float_or_none",
    "_values_equivalent",
    "_crypto_day_week_identical",
    "_crypto_price_display",
    "_load_finviz_attr",
    "_get_finviz_stock_quote",
    "_build_finviz_screener",
    "_fetch_finviz_market_performance_rows",
    "_resolve_date_range",
    "_align_to_next_monday_if_weekend",
    "_filter_calendar_events_by_date",
    "_fetch_finviz_economic_calendar_items",
    "_fetch_finviz_calendar_paged",
    "_FINVIZ_HTTP_TIMEOUT",
    "_FINVIZ_SCREENER_MAX_ROWS",
    "_FINVIZ_PAGE_LIMIT_MAX",
]

del _copy_namespace
del _api
