from typing import Any, Dict, Optional, Literal, List
import logging
import math
import time

from ..utils.utils import _table_from_rows, _normalize_limit
from ..utils.utils import _format_time_minimal
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from ..utils.mt5_enums import decode_mt5_enum_label, decode_mt5_bitmask_labels
from ..shared.schema import TimeframeLiteral
from ..shared.validators import invalid_timeframe_error
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from .constants import GROUP_SEARCH_THRESHOLD, DEFAULT_ROW_LIMIT, TIMEFRAME_MAP
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from_pos,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    mt5,
)

logger = logging.getLogger(__name__)


@mcp.tool()
def symbols_list(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups."""

    def _run() -> Dict[str, Any]:
        try:
            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            mode = str(list_mode or "symbols").strip().lower()
            if mode not in ("symbols", "groups"):
                return {"error": "list_mode must be 'symbols' or 'groups'."}
            if mode == "groups":
                return _list_symbol_groups(
                    search_term=search_term, limit=limit, mt5_gateway=mt5_gateway
                )

            matched_symbols = []

            if search_term:
                search_upper = search_term.upper()

                all_symbols = mt5_gateway.symbols_get()
                if all_symbols is None:
                    return {
                        "error": f"Failed to get symbols: {mt5_gateway.last_error()}"
                    }

                groups = {}
                for symbol in all_symbols:
                    group_path = _extract_group_path_util(symbol)
                    if group_path not in groups:
                        groups[group_path] = []
                    groups[group_path].append(symbol)

                matching_groups = []
                group_search_threshold = GROUP_SEARCH_THRESHOLD

                for group_name in groups.keys():
                    if search_upper in group_name.upper():
                        matching_groups.append(group_name)

                if matching_groups and len(matching_groups) <= group_search_threshold:
                    for group_name in matching_groups:
                        matched_symbols.extend(groups[group_name])
                else:
                    symbol_name_matches = []
                    for symbol in all_symbols:
                        if search_upper in symbol.name.upper():
                            symbol_name_matches.append(symbol)

                    if symbol_name_matches:
                        matched_symbols = symbol_name_matches
                    elif matching_groups:
                        for group_name in matching_groups:
                            matched_symbols.extend(groups[group_name])
                    else:
                        description_matches = []
                        for symbol in all_symbols:
                            if hasattr(symbol, "description") and symbol.description:
                                if search_upper in symbol.description.upper():
                                    description_matches.append(symbol)
                                    continue

                            group_path = getattr(symbol, "path", "")
                            if search_upper in group_path.upper():
                                description_matches.append(symbol)

                        matched_symbols = description_matches
            else:
                matched_symbols = list(mt5_gateway.symbols_get() or [])

            only_visible = not bool(search_term)
            symbol_list = []
            for symbol in matched_symbols:
                if only_visible and not symbol.visible:
                    continue
                symbol_list.append(
                    {
                        "name": symbol.name,
                        "group": _extract_group_path_util(symbol),
                        "description": symbol.description,
                    }
                )

            limit_value = _normalize_limit(limit)
            if limit_value:
                symbol_list = symbol_list[:limit_value]
            rows = [[s["name"], s["group"], s["description"]] for s in symbol_list]
            return _table_from_rows(["name", "group", "description"], rows)
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting symbols: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_list",
        search_term=search_term,
        limit=limit,
        list_mode=list_mode,
        func=_run,
    )


def _list_symbol_groups(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    mt5_gateway: Any = None,
) -> Dict[str, Any]:
    """List group paths as a tabular result with a single column: group."""
    try:
        gateway = mt5_gateway or get_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
        # Get all symbols first
        all_symbols = gateway.symbols_get()
        if all_symbols is None:
            return {"error": f"Failed to get symbols: {gateway.last_error()}"}

        # Collect unique groups and counts
        groups = {}
        for symbol in all_symbols:
            group_path = _extract_group_path_util(symbol)
            if group_path not in groups:
                groups[group_path] = {"count": 0}
            groups[group_path]["count"] += 1

        # Filter by search term if provided
        filtered_items = list(groups.items())
        if search_term:
            q = search_term.strip().lower()
            filtered_items = [
                (k, v) for (k, v) in filtered_items if q in (k or "").lower()
            ]

        # Sort groups by count (most symbols first)
        filtered_items.sort(key=lambda x: x[1]["count"], reverse=True)

        # Apply limit
        limit_value = _normalize_limit(limit)
        if limit_value:
            filtered_items = filtered_items[:limit_value]

        # Build tabular result with only group names
        group_names = [name for name, _ in filtered_items]
        rows = [[g] for g in group_names]
        return _table_from_rows(["group"], rows)
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}


@mcp.tool()
def symbols_describe(symbol: str) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
    Parameters: symbol
    Includes information such as Symbol Description, Swap Values, Tick Size/Value, etc.
    """

    def _run() -> Dict[str, Any]:
        try:
            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            symbol_info = mt5_gateway.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            enum_specs = {
                "trade_mode": {"prefixes": ("SYMBOL_TRADE_MODE_",), "bitmask": False},
                "trade_exemode": {
                    "prefixes": ("SYMBOL_TRADE_EXECUTION_",),
                    "bitmask": False,
                },
                "trade_calc_mode": {
                    "prefixes": ("SYMBOL_CALC_MODE_",),
                    "bitmask": False,
                },
                "swap_mode": {"prefixes": ("SYMBOL_SWAP_MODE_",), "bitmask": False},
                "filling_mode": {
                    "prefixes": ("ORDER_FILLING_", "SYMBOL_FILLING_"),
                    "bitmask": True,
                },
                "expiration_mode": {
                    "prefixes": ("SYMBOL_EXPIRATION_",),
                    "bitmask": True,
                },
                "order_mode": {"prefixes": ("SYMBOL_ORDER_",), "bitmask": True},
            }

            symbol_data = {}
            excluded = {
                "spread",
                "ask",
                "bid",
                "visible",
                "custom",
                "n_fields",
                "n_sequence_fields",
            }
            for attr in dir(symbol_info):
                if attr.startswith("_"):
                    continue
                if attr in excluded:
                    continue
                try:
                    value = getattr(symbol_info, attr)
                except Exception:
                    continue
                if callable(value):
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and value == "":
                    continue
                if attr == "time":
                    try:
                        epoch = float(value)
                        symbol_data["time_epoch"] = epoch
                        symbol_data["time"] = _format_time_minimal(epoch)
                    except Exception:
                        symbol_data[attr] = value
                else:
                    symbol_data[attr] = value

                spec = enum_specs.get(attr)
                if not spec:
                    continue
                prefixes = spec.get("prefixes", ())
                is_bitmask = bool(spec.get("bitmask"))
                if is_bitmask:
                    labels = []
                    for prefix in prefixes:
                        labels = decode_mt5_bitmask_labels(
                            mt5_gateway, value, prefix=prefix
                        )
                        if labels:
                            break
                    if labels:
                        symbol_data[f"{attr}_labels"] = labels
                        symbol_data[f"{attr}_label"] = ", ".join(labels)
                else:
                    label = None
                    for prefix in prefixes:
                        label = decode_mt5_enum_label(mt5_gateway, value, prefix=prefix)
                        if label:
                            break
                    if label:
                        symbol_data[f"{attr}_label"] = label

            return {
                "success": True,
                "symbol": symbol_data,
            }
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting symbol info: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_describe",
        symbol=symbol,
        func=_run,
    )


def _market_scan_is_tradable(symbol: Any) -> bool:
    disabled_trade_mode = getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", None)
    if disabled_trade_mode is None:
        return True
    return getattr(symbol, "trade_mode", None) != disabled_trade_mode


def _market_scan_base_row(symbol: Any) -> Dict[str, Any]:
    return {
        "symbol": getattr(symbol, "name", None),
        "group": _extract_group_path_util(symbol),
        "description": getattr(symbol, "description", None),
    }


def _market_scan_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _market_scan_bar_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out


def _market_scan_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), max(0, int(digits)))


def _build_market_scan_spread_row(
    symbol: Any,
    mt5_gateway: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    tick = mt5_gateway.symbol_info_tick(symbol.name)
    if tick is None:
        return None, f"Failed to get tick data: {mt5_gateway.last_error()}"

    bid = _market_scan_float(getattr(tick, "bid", None))
    ask = _market_scan_float(getattr(tick, "ask", None))
    if bid is None or ask is None:
        return None, "Bid/ask quote is unavailable."
    if ask < bid:
        return None, "Bid/ask quote is invalid."

    point = _market_scan_float(getattr(symbol, "point", 0.0)) or 0.0
    tick_size = _market_scan_float(getattr(symbol, "trade_tick_size", 0.0)) or 0.0
    tick_value = _market_scan_float(getattr(symbol, "trade_tick_value", 0.0)) or 0.0
    digits = max(0, int(getattr(symbol, "digits", 0) or 0))

    spread_abs = float(ask - bid)
    mid = (ask + bid) / 2.0
    spread_points = (spread_abs / point) if point > 0 else None
    spread_pct = ((spread_abs / mid) * 100.0) if mid > 0 else None
    spread_usd = None
    if tick_size > 0 and tick_value > 0:
        spread_usd = (spread_abs / tick_size) * tick_value

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "bid": _market_scan_round(bid, digits=digits),
            "ask": _market_scan_round(ask, digits=digits),
            "spread": _market_scan_round(spread_abs, digits=digits),
            "spread_points": _market_scan_round(spread_points, digits=4),
            "spread_pct": _market_scan_round(spread_pct, digits=6),
            "spread_usd": _market_scan_round(spread_usd, digits=6),
        }
    )
    return row, None


def _build_market_scan_bar_row(
    symbol: Any,
    timeframe: str,
    mt5_timeframe: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    rates = _mt5_copy_rates_from_pos(symbol.name, mt5_timeframe, 1, 1)
    if rates is None or len(rates) < 1:
        return None, f"No completed {timeframe} bar data returned."

    latest_bar = rates[-1]
    open_price = _market_scan_float(latest_bar["open"])
    close_price = _market_scan_float(latest_bar["close"])
    if open_price is None or close_price is None:
        return None, "Completed bar is missing open/close prices."
    if open_price == 0:
        return None, "Completed bar open price is zero."

    digits = max(0, int(getattr(symbol, "digits", 0) or 0))
    bar_time = _market_scan_float(latest_bar["time"])
    tick_volume = _market_scan_bar_int(latest_bar["tick_volume"])
    real_volume = _market_scan_bar_int(latest_bar["real_volume"])
    row = _market_scan_base_row(symbol)
    row.update(
        {
            "timeframe": timeframe,
            "bar_time": _format_time_minimal(bar_time)
            if bar_time is not None
            else None,
            "open": _market_scan_round(open_price, digits=digits),
            "close": _market_scan_round(close_price, digits=digits),
            "tick_volume": tick_volume,
            "real_volume": real_volume,
            "price_change_pct": _market_scan_round(
                ((close_price - open_price) / open_price) * 100.0,
                digits=6,
            ),
        }
    )
    return row, None


def _market_scan_table(
    headers: List[str], rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    ordered_rows = [[row.get(header) for header in headers] for row in rows]
    return _table_from_rows(headers, ordered_rows)


@mcp.tool()
def symbols_top_markets(
    rank_by: Literal["all", "spread", "volume", "price_change"] = "all",  # type: ignore
    limit: Optional[int] = 10,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
) -> Dict[str, Any]:
    """Scan MT5 symbols and rank the top markets by spread, recent volume, or recent price change.

    Defaults to visible tradable symbols for responsiveness. Set `universe="all"` to
    include hidden tradable symbols too; that mode is slower because MT5 may need to
    activate quotes for instruments that are not already visible. Volume and
    price-change rankings use the most recent completed bar on `timeframe`.
    """

    def _run() -> Dict[str, Any]:
        try:
            rank_by_value = str(rank_by or "all").strip().lower()
            if rank_by_value not in {"all", "spread", "volume", "price_change"}:
                return {
                    "error": "rank_by must be one of: all, spread, volume, price_change."
                }

            universe_value = str(universe or "visible").strip().lower()
            if universe_value not in {"visible", "all"}:
                return {"error": "universe must be 'visible' or 'all'."}

            timeframe_value = str(timeframe or "H1").strip().upper()
            needs_bar_data = rank_by_value in {"all", "volume", "price_change"}
            if needs_bar_data and timeframe_value not in TIMEFRAME_MAP:
                return {
                    "error": invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP)
                }
            mt5_timeframe = TIMEFRAME_MAP.get(timeframe_value)

            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()

            raw_symbols = mt5_gateway.symbols_get()
            if raw_symbols is None:
                return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}
            all_symbols = list(raw_symbols)

            selected_symbols = []
            for symbol in all_symbols:
                if not _market_scan_is_tradable(symbol):
                    continue
                if universe_value == "visible" and not bool(
                    getattr(symbol, "visible", False)
                ):
                    continue
                selected_symbols.append(symbol)

            limit_value = _normalize_limit(limit) or 10
            started_at = time.perf_counter()

            spread_rows: List[Dict[str, Any]] = []
            volume_rows: List[Dict[str, Any]] = []
            price_change_rows: List[Dict[str, Any]] = []
            metric_issues: Dict[str, List[Dict[str, str]]] = {
                "spread": [],
                "volume": [],
                "price_change": [],
            }
            metric_skips: Dict[str, int] = {
                "spread": 0,
                "volume": 0,
                "price_change": 0,
            }

            def _record_issue(metric_name: str, symbol_name: str, reason: str) -> None:
                metric_skips[metric_name] += 1
                if len(metric_issues[metric_name]) < 10:
                    metric_issues[metric_name].append(
                        {"symbol": symbol_name, "reason": reason}
                    )

            def _collect_for_symbol(symbol: Any) -> None:
                symbol_name = str(getattr(symbol, "name", "") or "")

                if rank_by_value in {"all", "spread"}:
                    spread_row, spread_error = _build_market_scan_spread_row(
                        symbol, mt5_gateway
                    )
                    if spread_error:
                        _record_issue("spread", symbol_name, spread_error)
                    elif spread_row is not None:
                        spread_rows.append(spread_row)

                if needs_bar_data and mt5_timeframe is not None:
                    bar_row, bar_error = _build_market_scan_bar_row(
                        symbol,
                        timeframe=timeframe_value,
                        mt5_timeframe=mt5_timeframe,
                    )
                    if bar_error:
                        if rank_by_value in {"all", "volume"}:
                            _record_issue("volume", symbol_name, bar_error)
                        if rank_by_value in {"all", "price_change"}:
                            _record_issue("price_change", symbol_name, bar_error)
                    elif bar_row is not None:
                        if rank_by_value in {"all", "volume"}:
                            volume_rows.append(dict(bar_row))
                        if rank_by_value in {"all", "price_change"}:
                            price_change_rows.append(dict(bar_row))

            for symbol in selected_symbols:
                symbol_name = str(getattr(symbol, "name", "") or "")
                is_hidden = not bool(getattr(symbol, "visible", False))
                if universe_value == "all" and is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol) as (
                        err,
                        _,
                    ):
                        if err:
                            if rank_by_value in {"all", "spread"}:
                                _record_issue("spread", symbol_name, err)
                            if rank_by_value in {"all", "volume"}:
                                _record_issue("volume", symbol_name, err)
                            if rank_by_value in {"all", "price_change"}:
                                _record_issue("price_change", symbol_name, err)
                            continue
                        _collect_for_symbol(symbol)
                    continue
                _collect_for_symbol(symbol)

            spread_rows.sort(
                key=lambda row: (
                    row.get("spread_pct") is None,
                    row.get("spread_pct")
                    if row.get("spread_pct") is not None
                    else float("inf"),
                    row.get("symbol") or "",
                )
            )
            volume_rows.sort(
                key=lambda row: (
                    row.get("tick_volume") is None,
                    -(row.get("tick_volume") or 0),
                    row.get("symbol") or "",
                )
            )
            price_change_rows.sort(
                key=lambda row: (
                    row.get("price_change_pct") is None,
                    -(row.get("price_change_pct") or 0.0),
                    row.get("symbol") or "",
                )
            )

            evaluated_counts = {
                "spread": len(spread_rows),
                "volume": len(volume_rows),
                "price_change": len(price_change_rows),
            }

            spread_rows = spread_rows[:limit_value]
            volume_rows = volume_rows[:limit_value]
            price_change_rows = price_change_rows[:limit_value]

            scan_meta = {
                "success": True,
                "rank_by": rank_by_value,
                "limit": limit_value,
                "universe": universe_value,
                "timeframe": timeframe_value if needs_bar_data else None,
                "scanned_symbols": len(selected_symbols),
                "query_latency_ms": round(
                    (time.perf_counter() - started_at) * 1000.0, 3
                ),
            }

            if rank_by_value == "spread":
                out = _market_scan_table(
                    [
                        "symbol",
                        "group",
                        "description",
                        "bid",
                        "ask",
                        "spread",
                        "spread_points",
                        "spread_pct",
                        "spread_usd",
                    ],
                    spread_rows,
                )
                out.update(scan_meta)
                out["evaluated_symbols"] = evaluated_counts["spread"]
                out["skipped_symbols"] = metric_skips["spread"]
                out["skipped_examples"] = metric_issues["spread"]
                out["ranking"] = "lowest_spread"
                return out

            if rank_by_value == "volume":
                out = _market_scan_table(
                    [
                        "symbol",
                        "group",
                        "description",
                        "timeframe",
                        "bar_time",
                        "tick_volume",
                        "real_volume",
                        "open",
                        "close",
                        "price_change_pct",
                    ],
                    volume_rows,
                )
                out.update(scan_meta)
                out["evaluated_symbols"] = evaluated_counts["volume"]
                out["skipped_symbols"] = metric_skips["volume"]
                out["skipped_examples"] = metric_issues["volume"]
                out["ranking"] = "highest_volume"
                return out

            if rank_by_value == "price_change":
                out = _market_scan_table(
                    [
                        "symbol",
                        "group",
                        "description",
                        "timeframe",
                        "bar_time",
                        "open",
                        "close",
                        "price_change_pct",
                        "tick_volume",
                        "real_volume",
                    ],
                    price_change_rows,
                )
                out.update(scan_meta)
                out["evaluated_symbols"] = evaluated_counts["price_change"]
                out["skipped_symbols"] = metric_skips["price_change"]
                out["skipped_examples"] = metric_issues["price_change"]
                out["ranking"] = "highest_price_change"
                return out

            return {
                **scan_meta,
                "results": {
                    "lowest_spread": _market_scan_table(
                        [
                            "symbol",
                            "group",
                            "description",
                            "bid",
                            "ask",
                            "spread",
                            "spread_points",
                            "spread_pct",
                            "spread_usd",
                        ],
                        spread_rows,
                    ),
                    "highest_volume": _market_scan_table(
                        [
                            "symbol",
                            "group",
                            "description",
                            "timeframe",
                            "bar_time",
                            "tick_volume",
                            "real_volume",
                            "open",
                            "close",
                            "price_change_pct",
                        ],
                        volume_rows,
                    ),
                    "highest_price_change": _market_scan_table(
                        [
                            "symbol",
                            "group",
                            "description",
                            "timeframe",
                            "bar_time",
                            "open",
                            "close",
                            "price_change_pct",
                            "tick_volume",
                            "real_volume",
                        ],
                        price_change_rows,
                    ),
                },
                "scan_stats": {
                    "spread": {
                        "evaluated_symbols": evaluated_counts["spread"],
                        "skipped_symbols": metric_skips["spread"],
                        "skipped_examples": metric_issues["spread"],
                    },
                    "volume": {
                        "evaluated_symbols": evaluated_counts["volume"],
                        "skipped_symbols": metric_skips["volume"],
                        "skipped_examples": metric_issues["volume"],
                    },
                    "price_change": {
                        "evaluated_symbols": evaluated_counts["price_change"],
                        "skipped_symbols": metric_skips["price_change"],
                        "skipped_examples": metric_issues["price_change"],
                    },
                },
            }
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error scanning top markets: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_top_markets",
        rank_by=rank_by,
        limit=limit,
        universe=universe,
        timeframe=timeframe,
        func=_run,
    )
