"""Open-position and pending-order query use cases."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mtdata.core.error_envelope import build_error_payload
from mtdata.core.execution_logging import run_logged_operation
from mtdata.core.trading import validation
from mtdata.core.trading.requests import TradeGetOpenRequest, TradeGetPendingRequest
from mtdata.core.trading.use_cases.common import (
    _epoch_series_to_utc_and_text,
    _trade_rows_to_dataframe,
    logger,
)
from mtdata.utils.mt5 import MT5ConnectionError, _ensure_symbol_ready


def run_trade_get_open(
    request: TradeGetOpenRequest,
    *,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
) -> List[Dict[str, Any]]:
    import pandas as pd

    result = run_logged_operation(
        logger,
        operation="trade_get_open",
        symbol=request.symbol,
        ticket=request.ticket,
        limit=request.limit,
        func=lambda: _run_trade_get_open_impl(
            request=request,
            gateway=gateway,
            use_client_tz=use_client_tz,
            format_time_minimal=format_time_minimal,
            format_time_minimal_local=format_time_minimal_local,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            normalize_limit=normalize_limit,
            comment_row_metadata=comment_row_metadata,
            pd_module=pd,
        ),
    )
    return result


def run_trade_get_pending(
    request: TradeGetPendingRequest,
    *,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
) -> List[Dict[str, Any]]:
    import pandas as pd

    result = run_logged_operation(
        logger,
        operation="trade_get_pending",
        symbol=request.symbol,
        ticket=request.ticket,
        limit=request.limit,
        func=lambda: _run_trade_get_pending_impl(
            request=request,
            gateway=gateway,
            use_client_tz=use_client_tz,
            format_time_minimal=format_time_minimal,
            format_time_minimal_local=format_time_minimal_local,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            normalize_limit=normalize_limit,
            comment_row_metadata=comment_row_metadata,
            pd_module=pd,
        ),
    )
    return result


def _mt5_int_const(gateway: Any, name: str, fallback: int) -> int:
    return validation._safe_int_attr(gateway, name, fallback)


def _pick_trade_series(df: Any, pd_module: Any, *names: str):
    out = None
    for name in names:
        if name in df.columns:
            out = df[name] if out is None else out.where(out.notna(), df[name])
    if out is None:
        return pd_module.Series([None] * len(df), index=df.index)
    return out


def _filter_trade_query_magic(df: Any, request: Any) -> Any:
    magic = getattr(request, "magic", None)
    if magic is None or "magic" not in df.columns:
        return df
    magic_value = validation._safe_int_magic(magic)
    if magic_value is None:
        return df.iloc[0:0].copy()
    mask = df["magic"].map(validation._safe_int_magic) == magic_value
    return df.loc[mask].copy()


def _filter_trade_query_profit(df: Any, request: Any, pd_module: Any) -> Any:
    profit_only = bool(getattr(request, "profit_only", False))
    loss_only = bool(getattr(request, "loss_only", False))
    if not profit_only and not loss_only:
        return df
    profit = pd_module.to_numeric(
        _pick_trade_series(df, pd_module, "profit"),
        errors="coerce",
    ).fillna(0.0)
    if profit_only:
        return df.loc[profit > 0.0].copy()
    return df.loc[profit < 0.0].copy()


def _filter_trade_query_side_and_type(df: Any, request: Any) -> Any:
    type_field = "order_type" if "order_type" in df.columns else "type"
    if type_field not in df.columns and "side" not in df.columns:
        return df
    out = df
    side = str(getattr(request, "side", "") or "").strip().upper()
    order_type = str(getattr(request, "order_type", "") or "").strip().upper()
    if side in {"BUY", "SELL"}:
        if "side" in out.columns:
            side_text = out["side"].astype(str).str.upper()
            out = out.loc[side_text.eq(side)].copy()
        else:
            type_text = out[type_field].astype(str).str.upper()
            out = out.loc[
                type_text.eq(side) | type_text.str.startswith(f"{side}_")
            ].copy()
    if order_type:
        type_text = out[type_field].astype(str).str.upper()
        out = out.loc[type_text.eq(order_type)].copy()
    return out


def _sort_trade_query_close_priority(df: Any, request: Any, pd_module: Any) -> Any:
    priority = str(getattr(request, "close_priority", "") or "").strip().lower()
    if priority not in {"loss_first", "profit_first", "largest_first"}:
        return df
    sort_field = "volume" if priority == "largest_first" else "profit"
    values = pd_module.to_numeric(
        _pick_trade_series(df, pd_module, sort_field),
        errors="coerce",
    ).fillna(0.0)
    out = df.copy()
    out["__trade_query_sort"] = values
    out = out.sort_values(
        "__trade_query_sort",
        ascending=priority == "loss_first",
        kind="stable",
    )
    return out.drop(columns=["__trade_query_sort"])


def _trade_query_empty_filter_message(request: Any) -> Optional[str]:
    if bool(getattr(request, "profit_only", False)):
        return "No rows matched profit_only=true"
    if bool(getattr(request, "loss_only", False)):
        return "No rows matched loss_only=true"
    magic = getattr(request, "magic", None)
    if magic is not None:
        return f"No rows matched magic={magic}"
    side = getattr(request, "side", None)
    if side not in (None, ""):
        return f"No rows matched side={side}"
    order_type = getattr(request, "order_type", None)
    if order_type not in (None, ""):
        return f"No rows matched order_type={order_type}"
    return None


def _fetch_trade_query_rows(
    request: Any,
    *,
    gateway: Any,
    snapshot: str,
    fetch_rows: Any,
    no_ticket_message: Any,
    no_symbol_message: Any,
    no_rows_message: str,
) -> tuple[Optional[Any], Optional[List[Dict[str, Any]]]]:
    if request.ticket is not None:
        ticket_int = int(request.ticket)
        rows = fetch_rows(ticket=ticket_int)
        if rows is None:
            return None, [
                validation.snapshot_unavailable_error(
                    gateway,
                    snapshot=snapshot,
                    context="read trading state",
                )
            ]
        if len(rows) == 0:
            return None, [{"message": no_ticket_message(request.ticket)}]
    elif request.symbol is not None:
        # Validate symbol before querying
        symbol_error = _ensure_symbol_ready(request.symbol)
        if symbol_error:
            return None, [{"error": symbol_error}]
        rows = fetch_rows(symbol=request.symbol)
        if rows is None:
            return None, [
                validation.snapshot_unavailable_error(
                    gateway,
                    snapshot=snapshot,
                    context="read trading state",
                )
            ]
        if len(rows) == 0:
            return None, [{"message": no_symbol_message(request.symbol)}]
    else:
        rows = fetch_rows()
        if rows is None:
            return None, [
                validation.snapshot_unavailable_error(
                    gateway,
                    snapshot=snapshot,
                    context="read trading state",
                )
            ]
        if len(rows) == 0:
            return None, [{"message": no_rows_message}]
    return rows, None


def _build_trade_time_columns(
    df: Any,
    *,
    time_source_fields: tuple[str, ...],
    pd_module: Any,
    mt5_epoch_to_utc: Any,
    fmt_time: Any,
) -> tuple[Any, Any]:
    time_src = None
    for field in time_source_fields:
        if field in df.columns:
            time_src = df[field]
            break
    if time_src is None:
        time_utc = pd_module.Series([float("nan")] * len(df), index=df.index)
        time_txt = pd_module.Series([None] * len(df), index=df.index)
    else:
        time_utc, time_txt = _epoch_series_to_utc_and_text(
            time_src,
            pd_module=pd_module,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            fmt_time=fmt_time,
        )
    return time_utc, time_txt


def _append_trade_comment_metadata(
    out_df: Any,
    *,
    comment_series: Any,
    comment_row_metadata: Any,
) -> None:
    comment_lengths: List[Any] = []
    comment_limits: List[Any] = []
    comment_truncation: List[Any] = []
    for comment_value in comment_series.tolist():
        metadata = comment_row_metadata(comment_value)
        if not isinstance(metadata, dict):
            metadata = {}
        comment_lengths.append(metadata.get("comment_visible_length"))
        comment_limits.append(metadata.get("comment_max_length"))
        comment_truncation.append(metadata.get("comment_may_be_truncated"))
    out_df["comment_visible_length"] = comment_lengths
    out_df["comment_max_length"] = comment_limits
    out_df["comment_may_be_truncated"] = comment_truncation


def _apply_trade_query_limit(
    out_df: Any,
    *,
    time_utc: Any,
    limit: Any,
    normalize_limit: Any,
    preserve_order: bool = False,
) -> Any:
    limit_value = normalize_limit(limit)
    if not limit_value or len(out_df) <= limit_value:
        return out_df
    if preserve_order:
        return out_df.head(limit_value).copy()
    sorted_index = (
        time_utc.reindex(out_df.index).sort_values(
            kind="stable",
            na_position="first",
        )
        .tail(limit_value)
        .index
    )
    return out_df.loc[sorted_index].copy()


def _build_trade_get_open_output(
    *,
    df: Any,
    gateway: Any,
    request: Any,
    time_txt: Any,
    pd_module: Any,
    timezone_label: str = "UTC",
    **_kwargs: Any,
) -> Any:
    open_df = df.drop(
        columns=[
            col
            for col in ("time_msc", "time_update", "time_update_msc")
            if col in df.columns
        ]
    ).copy()
    if "type" in open_df.columns:
        mapped = open_df["type"].map(
            {
                _mt5_int_const(gateway, "POSITION_TYPE_BUY", 0): "BUY",
                _mt5_int_const(gateway, "POSITION_TYPE_SELL", 1): "SELL",
            }
        )
        open_df["side"] = mapped.fillna(open_df["type"].astype(str))
    return pd_module.DataFrame(
        {
            "symbol": _pick_trade_series(open_df, pd_module, "symbol"),
            "ticket": _pick_trade_series(open_df, pd_module, "ticket"),
            "time": time_txt,
            "side": _pick_trade_series(open_df, pd_module, "side"),
            "volume": _pick_trade_series(open_df, pd_module, "volume"),
            "entry_price": _pick_trade_series(open_df, pd_module, "price_open"),
            "sl": _pick_trade_series(open_df, pd_module, "sl"),
            "tp": _pick_trade_series(open_df, pd_module, "tp"),
            "price_current": _pick_trade_series(open_df, pd_module, "price_current"),
            "swap": pd_module.to_numeric(
                _pick_trade_series(open_df, pd_module, "swap"),
                errors="coerce",
            ).fillna(0.0),
            "profit": pd_module.to_numeric(
                _pick_trade_series(open_df, pd_module, "profit"),
                errors="coerce",
            ).fillna(0.0),
            "comment": _pick_trade_series(open_df, pd_module, "comment"),
            "magic": _pick_trade_series(open_df, pd_module, "magic"),
            "timezone": timezone_label,
        }
    )


def _build_trade_get_pending_output(
    *,
    df: Any,
    gateway: Any,
    request: Any,
    time_txt: Any,
    pd_module: Any,
    fmt_time: Any,
    mt5_epoch_to_utc: Any,
    timezone_label: str = "UTC",
    **_kwargs: Any,
) -> Any:
    pending_df = df.copy()
    if "time_expiration" in pending_df.columns:
        exp_raw = pd_module.to_numeric(pending_df["time_expiration"], errors="coerce")
        _, exp_text = _epoch_series_to_utc_and_text(
            exp_raw,
            pd_module=pd_module,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            fmt_time=fmt_time,
            require_positive=True,
        )
        expiration = pd_module.Series(
            [
                None
                if pd_module.isna(raw_value)
                else "GTC"
                if float(raw_value) <= 0.0
                else text_value
                for raw_value, text_value in zip(exp_raw.tolist(), exp_text.tolist())
            ],
            index=exp_raw.index,
        )
    else:
        expiration = pd_module.Series([None] * len(pending_df), index=pending_df.index)

    pending_df = pending_df.drop(
        columns=[
            col
            for col in (
                "time_setup",
                "time_setup_msc",
                "time_done",
                "time_done_msc",
                "time_expiration",
                "time_msc",
            )
            if col in pending_df.columns
        ]
    ).copy()
    if "type" in pending_df.columns:
        mapped = pending_df["type"].map(
            {
                _mt5_int_const(gateway, "ORDER_TYPE_BUY", 0): "BUY",
                _mt5_int_const(gateway, "ORDER_TYPE_SELL", 1): "SELL",
                _mt5_int_const(gateway, "ORDER_TYPE_BUY_LIMIT", 2): "BUY_LIMIT",
                _mt5_int_const(gateway, "ORDER_TYPE_SELL_LIMIT", 3): "SELL_LIMIT",
                _mt5_int_const(gateway, "ORDER_TYPE_BUY_STOP", 4): "BUY_STOP",
                _mt5_int_const(gateway, "ORDER_TYPE_SELL_STOP", 5): "SELL_STOP",
                _mt5_int_const(
                    gateway, "ORDER_TYPE_BUY_STOP_LIMIT", 6
                ): "BUY_STOP_LIMIT",
                _mt5_int_const(
                    gateway, "ORDER_TYPE_SELL_STOP_LIMIT", 7
                ): "SELL_STOP_LIMIT",
            }
        )
        pending_df["order_type"] = mapped.fillna(pending_df["type"].astype(str))
        pending_df["side"] = pending_df["order_type"].astype(str).str.split("_").str[0]
    return pd_module.DataFrame(
        {
            "symbol": _pick_trade_series(pending_df, pd_module, "symbol"),
            "ticket": _pick_trade_series(pending_df, pd_module, "ticket"),
            "time": time_txt,
            "expiration": expiration,
            "side": _pick_trade_series(pending_df, pd_module, "side"),
            "order_type": _pick_trade_series(pending_df, pd_module, "order_type"),
            "volume": _pick_trade_series(
                pending_df,
                pd_module,
                "volume",
                "volume_current",
                "volume_initial",
            ),
            "trigger_price": _pick_trade_series(pending_df, pd_module, "price_open"),
            "stop_limit_price": _pick_trade_series(
                pending_df,
                pd_module,
                "price_stoplimit",
            ),
            "sl": _pick_trade_series(pending_df, pd_module, "sl"),
            "tp": _pick_trade_series(pending_df, pd_module, "tp"),
            "price_current": _pick_trade_series(pending_df, pd_module, "price_current"),
            "comment": _pick_trade_series(pending_df, pd_module, "comment"),
            "magic": _pick_trade_series(pending_df, pd_module, "magic"),
            "timezone": timezone_label,
        }
    )


def _run_trade_query_impl(
    *,
    request: Any,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    pd_module: Any,
    snapshot: str,
    fetch_rows: Any,
    no_ticket_message: Any,
    no_symbol_message: Any,
    no_rows_message: str,
    time_source_fields: tuple[str, ...],
    build_output: Any,
) -> Any:
    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return [
            build_error_payload(
                str(exc),
                code="MT5_CONNECTION",
                operation="trade_query",
            )
        ]

    try:
        use_client_tz_value = bool(use_client_tz())
        fmt_time = format_time_minimal_local if use_client_tz_value else format_time_minimal
        timezone_label = "client_local" if use_client_tz_value else "UTC"
        rows, empty_response = _fetch_trade_query_rows(
            request,
            gateway=gateway,
            snapshot=snapshot,
            fetch_rows=fetch_rows,
            no_ticket_message=no_ticket_message,
            no_symbol_message=no_symbol_message,
            no_rows_message=no_rows_message,
        )
        if empty_response is not None:
            return empty_response

        if bool(getattr(request, "profit_only", False)) and bool(
            getattr(request, "loss_only", False)
        ):
            return [
                build_error_payload(
                    "profit_only and loss_only cannot both be true.",
                    code="trade_query_error",
                    operation="trade_query",
                )
            ]

        df = _trade_rows_to_dataframe(rows, pd_module=pd_module)
        df = _filter_trade_query_magic(df, request)
        df = _filter_trade_query_profit(df, request, pd_module)
        df = _sort_trade_query_close_priority(df, request, pd_module)
        if len(df) == 0:
            message = _trade_query_empty_filter_message(request)
            if message is not None:
                return [{"message": message}]
        time_utc, time_txt = _build_trade_time_columns(
            df,
            time_source_fields=time_source_fields,
            pd_module=pd_module,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            fmt_time=fmt_time,
        )
        comment_series = _pick_trade_series(df, pd_module, "comment")
        out_df = build_output(
            df=df,
            gateway=gateway,
            request=request,
            time_txt=time_txt,
            pd_module=pd_module,
            fmt_time=fmt_time,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            timezone_label=timezone_label,
        )
        out_df = _filter_trade_query_side_and_type(out_df, request)
        if len(out_df) == 0:
            message = _trade_query_empty_filter_message(request)
            if message is not None:
                return [{"message": message}]
        detail = str(getattr(request, "detail", "compact") or "compact").strip().lower()
        if detail == "full":
            _append_trade_comment_metadata(
                out_df,
                comment_series=comment_series,
                comment_row_metadata=comment_row_metadata,
            )
        total_count = len(out_df)
        limit_value = normalize_limit(request.limit)
        out_df = _apply_trade_query_limit(
            out_df,
            time_utc=time_utc,
            limit=limit_value,
            normalize_limit=normalize_limit,
            preserve_order=bool(getattr(request, "close_priority", None)),
        )
        records = out_df.to_dict(orient="records")
        if limit_value and total_count > len(records):
            return {
                "items": records,
                "total_count": int(total_count),
                "limit": int(limit_value),
                "has_more": True,
                "truncated": True,
                "more_available": int(total_count - len(records)),
            }
        return records
    except Exception as exc:
        return [
            build_error_payload(
                str(exc),
                code="trade_query_error",
                operation="trade_query",
            )
        ]


def _run_trade_get_open_impl(
    *,
    request: TradeGetOpenRequest,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    pd_module: Any,
) -> Any:
    return _run_trade_query_impl(
        request=request,
        gateway=gateway,
        use_client_tz=use_client_tz,
        format_time_minimal=format_time_minimal,
        format_time_minimal_local=format_time_minimal_local,
        mt5_epoch_to_utc=mt5_epoch_to_utc,
        normalize_limit=normalize_limit,
        comment_row_metadata=comment_row_metadata,
        pd_module=pd_module,
        snapshot="positions",
        fetch_rows=gateway.positions_get,
        no_ticket_message=lambda ticket: f"No position found with ID {ticket}",
        no_symbol_message=lambda symbol: f"No open positions for {symbol}",
        no_rows_message="No open positions",
        time_source_fields=("time_update", "time"),
        build_output=_build_trade_get_open_output,
    )


def _run_trade_get_pending_impl(
    *,
    request: TradeGetPendingRequest,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    pd_module: Any,
) -> Any:
    return _run_trade_query_impl(
        request=request,
        gateway=gateway,
        use_client_tz=use_client_tz,
        format_time_minimal=format_time_minimal,
        format_time_minimal_local=format_time_minimal_local,
        mt5_epoch_to_utc=mt5_epoch_to_utc,
        normalize_limit=normalize_limit,
        comment_row_metadata=comment_row_metadata,
        pd_module=pd_module,
        snapshot="orders",
        fetch_rows=gateway.orders_get,
        no_ticket_message=lambda ticket: f"No pending order found with ID {ticket}",
        no_symbol_message=lambda symbol: f"No pending orders for {symbol}",
        no_rows_message="No pending orders",
        time_source_fields=("time_setup", "time"),
        build_output=_build_trade_get_pending_output,
    )
