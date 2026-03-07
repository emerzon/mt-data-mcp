"""Trading account and history views."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from ._mcp_instance import mcp
from . import trading_comments, trading_validation
from .config import mt5_config
from .constants import DEFAULT_ROW_LIMIT
from .trading_common import _build_trade_preflight
from ..utils.mt5 import _auto_connect_wrapper, _mt5_epoch_to_utc, mt5_adapter
from ..utils.mt5_enums import decode_mt5_enum_label
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _normalize_limit,
    _parse_start_datetime,
    _use_client_tz,
)


@mcp.tool()
def trade_account_info() -> dict:
    """Get account information (balance, equity, profit, margin level, free margin, account type, leverage, currency)."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
    def _get_account_info():
        info = mt5.account_info()
        if info is None:
            return {"error": "Failed to get account info"}
        preflight = _build_trade_preflight(mt5, account_info=info)
        margin_level: Optional[float] = getattr(info, "margin_level", None)
        margin_level_note: Optional[str] = None
        try:
            margin_val = float(getattr(info, "margin", 0.0) or 0.0)
            ml_val = float(getattr(info, "margin_level", 0.0) or 0.0)
            if margin_val <= 0.0 and ml_val <= 0.0:
                margin_level = None
                margin_level_note = "N/A (no open margin/positions)"
            elif not math.isfinite(ml_val):
                margin_level = None
        except Exception:
            pass

        payload = {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": margin_level,
            "currency": info.currency,
            "leverage": info.leverage,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
            "server": preflight.get("server"),
            "company": preflight.get("company"),
            "trade_mode": preflight.get("trade_mode"),
            "terminal_trade_allowed": preflight.get("terminal_trade_allowed"),
            "terminal_tradeapi_disabled": preflight.get("terminal_tradeapi_disabled"),
            "terminal_connected": preflight.get("terminal_connected"),
            "auto_trading_enabled": preflight.get("auto_trading_enabled"),
            "execution_ready": preflight.get("execution_ready"),
            "execution_ready_strict": preflight.get("execution_ready_strict"),
            "execution_hard_blockers": preflight.get("execution_hard_blockers"),
            "execution_soft_blockers": preflight.get("execution_soft_blockers"),
            "execution_blockers": preflight.get("execution_blockers"),
        }
        if margin_level_note:
            payload["margin_level_note"] = margin_level_note
        return payload

    return _get_account_info()


@mcp.tool()
def trade_history(
    history_kind: Literal["deals", "orders"] = "deals",  # type: ignore
    start: Optional[str] = None,
    end: Optional[str] = None,
    symbol: Optional[str] = None,
    position_ticket: Optional[Union[int, str]] = None,
    deal_ticket: Optional[Union[int, str]] = None,
    order_ticket: Optional[Union[int, str]] = None,
    minutes_back: Optional[int] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
) -> List[Dict[str, Any]]:
    """Get deal or order history as tabular data."""
    mt5 = mt5_adapter
    import pandas as pd

    @_auto_connect_wrapper
    def _get_history():
        try:
            use_client_tz = _use_client_tz()
            fmt_time = _format_time_minimal_local if use_client_tz else _format_time_minimal
            trigger_pattern = re.compile(r"\[(sl|tp)\s+([+-]?\d+(?:\.\d+)?)\]", re.IGNORECASE)

            def _normalize_time_col(df: "pd.DataFrame", col: str) -> Optional["pd.Series"]:
                if col not in df.columns:
                    return None
                raw = pd.to_numeric(df[col], errors="coerce")
                utc = raw.apply(lambda x: _mt5_epoch_to_utc(float(x)) if pd.notna(x) else float("nan"))
                df[col] = utc.apply(lambda x: fmt_time(float(x)) if pd.notna(x) else None)
                return utc

            if start and minutes_back not in (None, ""):
                return {"error": "Use either start or minutes_back, not both."}

            position_ticket_value, position_ticket_error = trading_validation._normalize_ticket_filter(position_ticket, name="position_ticket")
            if position_ticket_error:
                return {"error": position_ticket_error}
            deal_ticket_value, deal_ticket_error = trading_validation._normalize_ticket_filter(deal_ticket, name="deal_ticket")
            if deal_ticket_error:
                return {"error": deal_ticket_error}
            order_ticket_value, order_ticket_error = trading_validation._normalize_ticket_filter(order_ticket, name="order_ticket")
            if order_ticket_error:
                return {"error": order_ticket_error}
            minutes_back_value, minutes_back_error = trading_validation._normalize_minutes_back(minutes_back)
            if minutes_back_error:
                return {"error": minutes_back_error}

            if end:
                to_dt = _parse_start_datetime(end)
                if not to_dt:
                    return {"error": "Invalid end time."}
            else:
                to_dt = datetime.now(timezone.utc).replace(tzinfo=None)

            if minutes_back_value is not None:
                from_dt = to_dt - timedelta(minutes=minutes_back_value)
            elif start:
                from_dt = _parse_start_datetime(start)
                if not from_dt:
                    return {"error": "Invalid start time."}
            else:
                from_dt = datetime(2020, 1, 1)

            if from_dt > to_dt:
                return {"error": "start must be before end."}

            kind = str(history_kind or "deals").strip().lower()
            if kind not in ("deals", "orders"):
                return {"error": "history_kind must be 'deals' or 'orders'."}
            if kind == "orders" and deal_ticket_value is not None:
                return {"error": "deal_ticket is only valid when history_kind='deals'."}

            def _decode_enum_column(col: str, prefix: str) -> None:
                if col not in df.columns:
                    return
                raw = df[col]
                numeric = pd.to_numeric(raw, errors="coerce")
                if numeric.notna().any():
                    df[f"{col}_code"] = numeric.astype("Int64")
                df[col] = raw.apply(lambda v: decode_mt5_enum_label(mt5, v, prefix=prefix) or v)

            def _reason_to_exit_trigger(reason: Any) -> Optional[str]:
                txt = str(reason or "").strip().lower()
                if not txt:
                    return None
                if re.search(r"\bsl\b|stop\s*loss", txt):
                    return "SL"
                if re.search(r"\btp\b|take\s*profit", txt):
                    return "TP"
                return None

            def _extract_exit_trigger(comment: Any, reason: Any, entry: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
                entry_txt = str(entry or "").strip().lower()
                if entry_txt and "out" not in entry_txt:
                    return None, None, None

                if isinstance(comment, str) and comment:
                    match = trigger_pattern.search(comment)
                    if match:
                        trigger = str(match.group(1)).upper()
                        try:
                            price = float(match.group(2))
                        except Exception:
                            price = None
                        return trigger, price, "comment"

                reason_trigger = _reason_to_exit_trigger(reason)
                if reason_trigger:
                    return reason_trigger, None, "reason"
                return None, None, None

            def _filter_by_ticket_columns(df_in: "pd.DataFrame", ticket_value: Optional[int], *, columns: Tuple[str, ...]) -> "pd.DataFrame":
                if ticket_value is None:
                    return df_in
                masks: List["pd.Series"] = []
                for col in columns:
                    if col not in df_in.columns:
                        continue
                    masks.append(pd.to_numeric(df_in[col], errors="coerce").eq(ticket_value))
                if not masks:
                    return df_in.iloc[0:0]
                mask = masks[0]
                for extra in masks[1:]:
                    mask = mask | extra
                return df_in.loc[mask]

            def _is_non_informative_series(series: "pd.Series") -> bool:
                vals = pd.Series(series)
                if vals.dropna().empty:
                    return True
                for value in vals:
                    if value is None:
                        continue
                    if isinstance(value, str):
                        if not value.strip():
                            continue
                        return False
                    try:
                        numeric = float(value)
                        if math.isfinite(numeric) and numeric == 0.0:
                            continue
                        return False
                    except Exception:
                        return False
                return True

            if kind == "deals":
                rows = mt5.history_deals_get(from_dt, to_dt, symbol=symbol) if symbol else mt5.history_deals_get(from_dt, to_dt)
                if rows is None or len(rows) == 0:
                    return {"message": "No deals found"}
                df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())
                if symbol and "symbol" in df.columns:
                    df = df.loc[df["symbol"].astype(str).str.upper() == str(symbol).upper()]
                df = _filter_by_ticket_columns(df, deal_ticket_value, columns=("ticket",))
                df = _filter_by_ticket_columns(df, order_ticket_value, columns=("order",))
                df = _filter_by_ticket_columns(df, position_ticket_value, columns=("position_id", "position_by_id"))
                if len(df) == 0:
                    return {"message": f"No deals found for {symbol}" if symbol else "No deals found"}
                sort_src = _normalize_time_col(df, "time")
                _decode_enum_column("type", "DEAL_TYPE_")
                _decode_enum_column("entry", "DEAL_ENTRY_")
                _decode_enum_column("reason", "DEAL_REASON_")
                if len(df) > 0:
                    triggers = df.apply(
                        lambda row: _extract_exit_trigger(row.get("comment"), row.get("reason"), row.get("entry")),
                        axis=1,
                        result_type="expand",
                    )
                    if isinstance(triggers, pd.DataFrame) and triggers.shape[1] == 3:
                        triggers.columns = ["exit_trigger", "exit_trigger_price", "exit_trigger_source"]
                        for col in triggers.columns:
                            df[col] = triggers[col]
                for noise_col in ("time_msc", "external_id", "fee"):
                    if noise_col in df.columns and _is_non_informative_series(df[noise_col]):
                        df = df.drop(columns=[noise_col])
            else:
                rows = mt5.history_orders_get(from_dt, to_dt, symbol=symbol) if symbol else mt5.history_orders_get(from_dt, to_dt)
                if rows is None or len(rows) == 0:
                    return {"message": "No orders found"}
                df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())
                if symbol and "symbol" in df.columns:
                    df = df.loc[df["symbol"].astype(str).str.upper() == str(symbol).upper()]
                df = _filter_by_ticket_columns(df, order_ticket_value, columns=("ticket",))
                df = _filter_by_ticket_columns(df, position_ticket_value, columns=("position_id", "position_by_id"))
                if len(df) == 0:
                    return {"message": f"No orders found for {symbol}" if symbol else "No orders found"}
                sort_src = _normalize_time_col(df, "time_setup")
                if sort_src is None:
                    sort_src = _normalize_time_col(df, "time")
                _normalize_time_col(df, "time_done")
                _decode_enum_column("type", "ORDER_TYPE_")
                _decode_enum_column("state", "ORDER_STATE_")
                _decode_enum_column("type_time", "ORDER_TIME_")
                _decode_enum_column("type_filling", "ORDER_FILLING_")
                _decode_enum_column("reason", "ORDER_REASON_")

            df["__sort_utc"] = sort_src if sort_src is not None else pd.Series([float("nan")] * len(df))

            limit_value = _normalize_limit(limit)
            if limit_value and len(df) > limit_value:
                if "__sort_utc" in df.columns:
                    df = df.sort_values("__sort_utc").tail(limit_value)
                else:
                    df = df.tail(limit_value)
            if "__sort_utc" in df.columns:
                df = df.drop(columns=["__sort_utc"])

            df = df.replace([float("inf"), float("-inf")], pd.NA)
            records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
            timezone_label = "UTC"
            if use_client_tz:
                try:
                    tz_obj = mt5_config.get_client_tz()
                    timezone_label = str(getattr(tz_obj, "zone", None) or tz_obj or "client_local")
                except Exception:
                    timezone_label = "client_local"
            for row in records:
                if isinstance(row, dict):
                    row["timestamp_timezone"] = timezone_label
                    row.update(trading_comments._comment_row_metadata(row.get("comment")))
            return records
        except Exception as exc:
            return {"error": str(exc)}

    return _get_history()
