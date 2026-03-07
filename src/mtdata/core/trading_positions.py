"""Trading position resolution and read-only views."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ._mcp_instance import mcp
from . import trading_comments, trading_validation
from .constants import DEFAULT_ROW_LIMIT
from ..utils.mt5 import _auto_connect_wrapper, _mt5_epoch_to_utc, mt5_adapter
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _normalize_limit,
    _use_client_tz,
)


def _position_sort_key(position: Any) -> float:
    """Prefer the most recently updated position when multiple candidates exist."""
    for field in ("time_update_msc", "time_msc", "time_update", "time"):
        try:
            value = float(getattr(position, field, 0.0) or 0.0)
            if math.isfinite(value):
                return value
        except Exception:
            continue
    return 0.0


def _position_side_matches(position: Any, side: Optional[str], mt5: Any) -> bool:
    if side not in {"BUY", "SELL"}:
        return True
    expected_buy = getattr(mt5, "ORDER_TYPE_BUY", getattr(mt5, "POSITION_TYPE_BUY", None))
    expected_sell = getattr(mt5, "ORDER_TYPE_SELL", getattr(mt5, "POSITION_TYPE_SELL", None))
    expected = expected_buy if side == "BUY" else expected_sell
    if expected is None:
        return True
    try:
        return int(getattr(position, "type", -99999)) == int(expected)
    except Exception:
        return False


def _position_ticket_fields(position: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        ticket = trading_validation._safe_int_ticket(getattr(position, field, None))
        if ticket is not None:
            out[field] = ticket
    return out


def _select_position_candidate(
    rows: List[Any],
    *,
    symbol: Optional[str],
    side: Optional[str],
    volume: Optional[float],
    mt5: Any,
) -> Optional[Any]:
    if not rows:
        return None
    candidates = list(rows)
    if symbol:
        symbol_upper = str(symbol).upper()
        symbol_filtered = [pos for pos in candidates if str(getattr(pos, "symbol", "")).upper() == symbol_upper]
        if symbol_filtered:
            candidates = symbol_filtered
    side_filtered = [pos for pos in candidates if _position_side_matches(pos, side, mt5)]
    if side_filtered:
        candidates = side_filtered
    if volume is not None:
        volume_filtered: List[Any] = []
        for pos in candidates:
            try:
                if abs(float(getattr(pos, "volume", float("nan"))) - float(volume)) <= 1e-9:
                    volume_filtered.append(pos)
            except Exception:
                continue
        if volume_filtered:
            candidates = volume_filtered
    candidates.sort(key=_position_sort_key, reverse=True)
    return candidates[0] if candidates else None


def _resolve_open_position(
    mt5: Any,
    *,
    ticket_candidates: Optional[List[int]] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    volume: Optional[float] = None,
) -> Tuple[Optional[Any], Optional[int], Dict[str, Any]]:
    """Resolve an open position robustly across ticket/identifier mismatches."""
    candidate_ids: List[int] = []
    for raw in list(ticket_candidates or []):
        ticket = trading_validation._safe_int_ticket(raw)
        if ticket is not None and ticket not in candidate_ids:
            candidate_ids.append(ticket)

    for candidate in candidate_ids:
        try:
            rows = mt5.positions_get(ticket=int(candidate))
        except Exception:
            rows = None
        rows_list = list(rows) if rows else []
        picked = _select_position_candidate(
            rows_list,
            symbol=symbol,
            side=side,
            volume=volume,
            mt5=mt5,
        )
        if picked is not None:
            resolved = trading_validation._safe_int_ticket(getattr(picked, "ticket", None)) or candidate
            return picked, resolved, {"method": "positions_get(ticket)", "candidate": candidate}

    try:
        rows_fallback = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    except Exception:
        rows_fallback = None
    rows_list = list(rows_fallback) if rows_fallback else []
    if not rows_list:
        return None, None, {"method": "positions_get", "candidate_ids": candidate_ids, "matched": False}

    exact_matches: List[Tuple[Any, str, int]] = []
    if candidate_ids:
        for pos in rows_list:
            for field, value in _position_ticket_fields(pos).items():
                if value in candidate_ids:
                    exact_matches.append((pos, field, value))
        if exact_matches:
            exact_matches.sort(key=lambda item: _position_sort_key(item[0]), reverse=True)
            pos, field, matched_value = exact_matches[0]
            resolved = trading_validation._safe_int_ticket(getattr(pos, "ticket", None))
            return pos, resolved, {
                "method": "positions_get(fallback_exact)",
                "matched_field": field,
                "matched_value": matched_value,
            }

    picked = _select_position_candidate(
        rows_list,
        symbol=symbol,
        side=side,
        volume=volume,
        mt5=mt5,
    )
    if picked is None:
        return None, None, {"method": "positions_get(fallback_heuristic)", "candidate_ids": candidate_ids, "matched": False}
    resolved = trading_validation._safe_int_ticket(getattr(picked, "ticket", None))
    return picked, resolved, {"method": "positions_get(fallback_heuristic)"}


@mcp.tool()
def trade_get_open(
    symbol: Optional[str] = None,
    ticket: Optional[Union[int, str]] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
) -> List[Dict[str, Any]]:
    """Get open positions."""
    mt5 = mt5_adapter
    import pandas as pd

    @_auto_connect_wrapper
    def _get_open():
        try:
            use_client_tz = _use_client_tz()
            fmt_time = _format_time_minimal_local if use_client_tz else _format_time_minimal

            def _mt5_int_const(name: str, fallback: int) -> int:
                value = getattr(mt5, name, None)
                return value if isinstance(value, int) else fallback

            position_type_text_map = {
                _mt5_int_const("POSITION_TYPE_BUY", 0): "BUY",
                _mt5_int_const("POSITION_TYPE_SELL", 1): "SELL",
            }

            def _pick_series(df: "pd.DataFrame", *names: str) -> "pd.Series":
                out = None
                for name in names:
                    if name in df.columns:
                        out = df[name] if out is None else out.combine_first(df[name])
                if out is None:
                    return pd.Series([None] * len(df))
                return out

            if ticket is not None:
                ticket_int = int(ticket)
                rows = mt5.positions_get(ticket=ticket_int)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No position found with ID {ticket}"}]
            elif symbol is not None:
                rows = mt5.positions_get(symbol=symbol)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No open positions for {symbol}"}]
            else:
                rows = mt5.positions_get()
                if rows is None or len(rows) == 0:
                    return [{"message": "No open positions"}]
            df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())

            time_src = None
            if "time_update" in df.columns:
                time_src = pd.to_numeric(df["time_update"], errors="coerce")
            elif "time" in df.columns:
                time_src = pd.to_numeric(df["time"], errors="coerce")
            if time_src is None:
                time_utc = pd.Series([float("nan")] * len(df))
                time_txt = pd.Series([None] * len(df))
            else:
                time_utc = time_src.apply(lambda x: _mt5_epoch_to_utc(float(x)) if pd.notna(x) else float("nan"))
                time_txt = time_utc.apply(lambda x: fmt_time(float(x)) if pd.notna(x) else None)

            for col in ("time_msc", "time_update", "time_update_msc"):
                if col in df.columns:
                    df = df.drop(columns=[col])

            if "type" in df.columns:
                mapped = df["type"].map(position_type_text_map)
                df["type"] = mapped.fillna(df["type"].astype(str))

            out_df = pd.DataFrame(
                {
                    "Symbol": _pick_series(df, "symbol"),
                    "Ticket": _pick_series(df, "ticket"),
                    "Time": time_txt,
                    "Type": _pick_series(df, "type"),
                    "Volume": _pick_series(df, "volume"),
                    "Open Price": _pick_series(df, "price_open"),
                    "SL": _pick_series(df, "sl"),
                    "TP": _pick_series(df, "tp"),
                    "Current Price": _pick_series(df, "price_current"),
                    "Swap": pd.to_numeric(_pick_series(df, "swap"), errors="coerce").fillna(0.0),
                    "Profit": pd.to_numeric(_pick_series(df, "profit"), errors="coerce").fillna(0.0),
                    "Comments": _pick_series(df, "comment"),
                    "Magic": _pick_series(df, "magic"),
                }
            )
            comment_meta = _pick_series(df, "comment").apply(trading_comments._comment_row_metadata)
            out_df["Comment Length"] = comment_meta.apply(lambda value: value.get("comment_visible_length"))
            out_df["Comment Limit"] = comment_meta.apply(lambda value: value.get("comment_max_length"))
            out_df["Comment May Be Truncated"] = comment_meta.apply(lambda value: value.get("comment_may_be_truncated"))
            out_df["__time_utc"] = time_utc

            limit_value = _normalize_limit(limit)
            if limit_value and len(out_df) > limit_value:
                out_df = out_df.sort_values("__time_utc").tail(limit_value)
            if "__time_utc" in out_df.columns:
                out_df = out_df.drop(columns=["__time_utc"])
            return out_df.to_dict(orient="records")
        except Exception as exc:
            return [{"error": str(exc)}]

    return _get_open()


@mcp.tool()
def trade_get_pending(
    symbol: Optional[str] = None,
    ticket: Optional[Union[int, str]] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
) -> List[Dict[str, Any]]:
    """Get pending orders (open orders)."""
    mt5 = mt5_adapter
    import pandas as pd

    @_auto_connect_wrapper
    def _get_pending():
        try:
            use_client_tz = _use_client_tz()
            fmt_time = _format_time_minimal_local if use_client_tz else _format_time_minimal

            def _mt5_int_const(name: str, fallback: int) -> int:
                value = getattr(mt5, name, None)
                return value if isinstance(value, int) else fallback

            order_type_map = {
                _mt5_int_const("ORDER_TYPE_BUY", 0): "BUY",
                _mt5_int_const("ORDER_TYPE_SELL", 1): "SELL",
                _mt5_int_const("ORDER_TYPE_BUY_LIMIT", 2): "BUY_LIMIT",
                _mt5_int_const("ORDER_TYPE_SELL_LIMIT", 3): "SELL_LIMIT",
                _mt5_int_const("ORDER_TYPE_BUY_STOP", 4): "BUY_STOP",
                _mt5_int_const("ORDER_TYPE_SELL_STOP", 5): "SELL_STOP",
                _mt5_int_const("ORDER_TYPE_BUY_STOP_LIMIT", 6): "BUY_STOP_LIMIT",
                _mt5_int_const("ORDER_TYPE_SELL_STOP_LIMIT", 7): "SELL_STOP_LIMIT",
            }

            def _pick_series(df: "pd.DataFrame", *names: str) -> "pd.Series":
                out = None
                for name in names:
                    if name in df.columns:
                        out = df[name] if out is None else out.combine_first(df[name])
                if out is None:
                    return pd.Series([None] * len(df))
                return out

            if ticket is not None:
                ticket_int = int(ticket)
                rows = mt5.orders_get(ticket=ticket_int)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No pending order found with ID {ticket}"}]
            elif symbol is not None:
                rows = mt5.orders_get(symbol=symbol)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No pending orders for {symbol}"}]
            else:
                rows = mt5.orders_get()
                if rows is None or len(rows) == 0:
                    return [{"message": "No pending orders"}]

            df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())

            time_src = None
            if "time_setup" in df.columns:
                time_src = pd.to_numeric(df["time_setup"], errors="coerce")
            elif "time" in df.columns:
                time_src = pd.to_numeric(df["time"], errors="coerce")
            if time_src is None:
                time_utc = pd.Series([float("nan")] * len(df))
                time_txt = pd.Series([None] * len(df))
            else:
                time_utc = time_src.apply(lambda x: _mt5_epoch_to_utc(float(x)) if pd.notna(x) else float("nan"))
                time_txt = time_utc.apply(lambda x: fmt_time(float(x)) if pd.notna(x) else None)

            if "time_expiration" in df.columns:
                exp_raw = pd.to_numeric(df["time_expiration"], errors="coerce")
                exp_utc = exp_raw.apply(lambda x: _mt5_epoch_to_utc(float(x)) if pd.notna(x) and float(x) > 0 else float("nan"))
                expiration = exp_raw.apply(lambda x: "GTC" if pd.notna(x) and float(x) <= 0 else None)
                expiration = expiration.where(
                    exp_raw.isna() | (exp_raw <= 0),
                    other=exp_utc.apply(lambda x: fmt_time(float(x)) if pd.notna(x) else None),
                )
            else:
                expiration = pd.Series([None] * len(df))

            for col in (
                "time_setup",
                "time_setup_msc",
                "time_done",
                "time_done_msc",
                "time_expiration",
                "time_msc",
            ):
                if col in df.columns:
                    df = df.drop(columns=[col])

            if "type" in df.columns:
                mapped = df["type"].map(order_type_map)
                df["type"] = mapped.fillna(df["type"].astype(str))

            out_df = pd.DataFrame(
                {
                    "Symbol": _pick_series(df, "symbol"),
                    "Ticket": _pick_series(df, "ticket"),
                    "Time": time_txt,
                    "Expiration": expiration,
                    "Type": _pick_series(df, "type"),
                    "Volume": _pick_series(df, "volume", "volume_current", "volume_initial"),
                    "Open Price": _pick_series(df, "price_open"),
                    "SL": _pick_series(df, "sl"),
                    "TP": _pick_series(df, "tp"),
                    "Current Price": _pick_series(df, "price_current"),
                    "Comments": _pick_series(df, "comment"),
                    "Magic": _pick_series(df, "magic"),
                }
            )
            comment_meta = _pick_series(df, "comment").apply(trading_comments._comment_row_metadata)
            out_df["Comment Length"] = comment_meta.apply(lambda value: value.get("comment_visible_length"))
            out_df["Comment Limit"] = comment_meta.apply(lambda value: value.get("comment_max_length"))
            out_df["Comment May Be Truncated"] = comment_meta.apply(lambda value: value.get("comment_may_be_truncated"))
            out_df["__time_utc"] = time_utc

            limit_value = _normalize_limit(limit)
            if limit_value and len(out_df) > limit_value:
                out_df = out_df.sort_values("__time_utc").tail(limit_value) if "__time_utc" in out_df.columns else out_df.head(limit_value)
            if "__time_utc" in out_df.columns:
                out_df = out_df.drop(columns=["__time_utc"])
            return out_df.to_dict(orient="records")
        except Exception as exc:
            return [{"error": str(exc)}]

    return _get_pending()
