from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .trading_requests import (
    TradeCloseRequest,
    TradeGetOpenRequest,
    TradeGetPendingRequest,
    TradeHistoryRequest,
    TradeModifyRequest,
    TradePlaceRequest,
    TradeRiskAnalyzeRequest,
)


def run_trade_place(
    request: TradePlaceRequest,
    *,
    normalize_order_type_input: Any,
    normalize_pending_expiration: Any,
    prevalidate_trade_place_market_input: Any,
    place_market_order: Any,
    place_pending_order: Any,
    close_positions: Any,
    safe_int_ticket: Any,
) -> Dict[str, Any]:
    missing: List[str] = []
    symbol_norm = str(request.symbol).strip() if request.symbol is not None else ""
    if not symbol_norm:
        missing.append("symbol")
    if request.volume is None:
        missing.append("volume")
    if request.order_type is None or (
        isinstance(request.order_type, str) and not request.order_type.strip()
    ):
        missing.append("order_type")
    if missing:
        return {
            "error": (
                f"Missing required field(s): {', '.join(missing)}. "
                "Required: symbol, volume, order_type."
            ),
            "required": ["symbol", "volume", "order_type"],
            "hint": (
                "Example: symbol='BTCUSD', volume=0.03, "
                "order_type='BUY_LIMIT' (or ORDER_TYPE_BUY_LIMIT or 2)."
            ),
        }

    order_type_norm, order_type_error = normalize_order_type_input(request.order_type)
    if order_type_error:
        return {"error": order_type_error}
    explicit_pending_types = {"BUY_LIMIT", "BUY_STOP", "SELL_LIMIT", "SELL_STOP"}
    market_side_types = {"BUY", "SELL"}
    supported_order_types = explicit_pending_types.union(market_side_types)
    if order_type_norm not in supported_order_types:
        return {
            "error": (
                f"Unsupported order_type '{request.order_type}'. "
                "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
            )
        }

    price_provided = request.price not in (None, 0)
    try:
        _, expiration_provided = normalize_pending_expiration(request.expiration)
    except (TypeError, ValueError) as ex:
        return {"error": str(ex)}

    is_pending = (
        order_type_norm in explicit_pending_types
        or price_provided
        or expiration_provided
    )
    if bool(request.require_sl_tp) and not is_pending:
        missing_protection: List[str] = []
        if request.stop_loss in (None, 0):
            missing_protection.append("stop_loss")
        if request.take_profit in (None, 0):
            missing_protection.append("take_profit")
        if missing_protection:
            prevalidation_error = prevalidate_trade_place_market_input(
                symbol_norm,
                request.volume,
            )
            if prevalidation_error is not None:
                return prevalidation_error
            return {
                "error": (
                    "require_sl_tp=True requires both stop_loss and take_profit for market orders. "
                    "Refusing to place an unprotected position."
                ),
                "require_sl_tp": True,
                "missing": missing_protection,
                "hint": (
                    "Provide both --stop-loss and --take-profit, "
                    "or explicitly set --require-sl-tp false."
                ),
            }

    if not is_pending:
        result = place_market_order(
            symbol=symbol_norm,
            volume=float(request.volume),
            order_type=order_type_norm,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=request.comment,
            deviation=request.deviation,
        )
        if isinstance(result, dict):
            sl_tp_requested = bool(result.get("sl_tp_requested"))
            sl_tp_failed = str(result.get("sl_tp_apply_status") or "").lower() == "failed"
            if sl_tp_requested and sl_tp_failed:
                warnings_out: List[str] = list(result.get("warnings") or [])
                pos_ticket = result.get("position_ticket")
                if pos_ticket is not None:
                    critical = (
                        "CRITICAL: Order executed without applied TP/SL protection. "
                        f"Run trade_modify {pos_ticket} now, or close the position."
                    )
                else:
                    critical = (
                        "CRITICAL: Order executed without applied TP/SL protection. "
                        "Run trade_modify now, or close the position."
                    )
                if critical not in warnings_out:
                    warnings_out.append(critical)
                if warnings_out:
                    result["warnings"] = warnings_out
                if bool(request.auto_close_on_sl_tp_fail):
                    close_ticket = safe_int_ticket(pos_ticket)
                    if close_ticket is None:
                        auto_close_result: Dict[str, Any] = {
                            "error": "Auto-close skipped: position_ticket unavailable."
                        }
                    else:
                        auto_close_result = close_positions(
                            ticket=close_ticket,
                            comment="AUTO-CLOSE: TP/SL apply failed",
                            deviation=request.deviation,
                        )
                    result["auto_close_on_sl_tp_fail"] = True
                    result["auto_close_result"] = auto_close_result

                    auto_close_ok = False
                    if isinstance(auto_close_result, dict) and "error" not in auto_close_result:
                        if auto_close_result.get("retcode") is not None:
                            auto_close_ok = True
                        else:
                            try:
                                auto_close_ok = int(auto_close_result.get("closed_count", 0)) > 0
                            except Exception:
                                auto_close_ok = False
                    if auto_close_ok:
                        result["protection_status"] = "auto_closed_after_sl_tp_fail"
                    else:
                        warnings_out = list(result.get("warnings") or [])
                        auto_close_warning = (
                            "AUTO-CLOSE FAILED: position remains unprotected; close immediately."
                        )
                        if auto_close_warning not in warnings_out:
                            warnings_out.append(auto_close_warning)
                        result["warnings"] = warnings_out

            if (
                bool(request.require_sl_tp)
                and sl_tp_requested
                and sl_tp_failed
                and "error" not in result
            ):
                result["error"] = "Order was executed, but TP/SL protection could not be applied."
                result["require_sl_tp"] = bool(request.require_sl_tp)
                result["protection_status"] = (
                    result.get("protection_status") or "unprotected_position"
                )
        return result
    if request.price is None:
        return {"error": "price is required for pending orders."}
    return place_pending_order(
        symbol=symbol_norm,
        volume=float(request.volume),
        order_type=order_type_norm,
        price=request.price,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        expiration=request.expiration,
        comment=request.comment,
        deviation=request.deviation,
    )


def run_trade_modify(
    request: TradeModifyRequest,
    *,
    normalize_pending_expiration: Any,
    modify_pending_order: Any,
    modify_position: Any,
) -> Dict[str, Any]:
    price_val = None if request.price in (None, 0) else request.price
    try:
        _, expiration_specified = normalize_pending_expiration(request.expiration)
    except (TypeError, ValueError) as ex:
        return {"error": str(ex)}

    if price_val is not None or expiration_specified:
        result = modify_pending_order(
            ticket=request.ticket,
            price=price_val,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            expiration=request.expiration,
            comment=request.comment,
        )
        if result.get("error") == f"Pending order {request.ticket} not found":
            return {
                "error": (
                    f"Pending order {request.ticket} not found. "
                    "Note: price/expiration only apply to pending orders."
                ),
                "checked_scopes": ["pending_orders"],
            }
        return result

    position_result = modify_position(
        ticket=request.ticket,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        comment=request.comment,
    )
    if position_result.get("success"):
        return position_result
    if position_result.get("error") == f"Position {request.ticket} not found":
        pending_result = modify_pending_order(
            ticket=request.ticket,
            price=None,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            expiration=None,
            comment=request.comment,
        )
        if pending_result.get("error") == f"Pending order {request.ticket} not found":
            return {
                "error": f"Ticket {request.ticket} not found as position or pending order.",
                "checked_scopes": ["positions", "pending_orders"],
            }
        return pending_result
    return position_result


def run_trade_close(
    request: TradeCloseRequest,
    *,
    close_positions: Any,
    cancel_pending: Any,
) -> Dict[str, Any]:
    def _with_no_action(
        payload: Optional[Dict[str, Any]] = None,
        *,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(payload or {})
        if message and not str(out.get("message", "")).strip():
            out["message"] = message
        out["no_action"] = True
        return out

    if request.profit_only or request.loss_only:
        result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            profit_only=request.profit_only,
            loss_only=request.loss_only,
            comment=request.comment,
            deviation=request.deviation,
        )
        if isinstance(result, dict):
            msg = str(result.get("message", "")).strip().lower()
            if msg.startswith("no open positions") or msg == "no positions matched criteria":
                return _with_no_action(result)
        return result

    if request.ticket is not None:
        position_result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            profit_only=False,
            loss_only=False,
            comment=request.comment,
            deviation=request.deviation,
        )
        if (
            isinstance(position_result, dict)
            and position_result.get("error") == f"Position {request.ticket} not found"
        ):
            pending_result = cancel_pending(
                ticket=request.ticket,
                symbol=request.symbol,
                comment=request.comment,
            )
            if (
                isinstance(pending_result, dict)
                and pending_result.get("error") == f"Pending order {request.ticket} not found"
            ):
                return {
                    "error": f"Ticket {request.ticket} not found as position or pending order.",
                    "checked_scopes": ["positions", "pending_orders"],
                }
            return pending_result
        return position_result

    if request.symbol is not None:
        position_result = close_positions(
            symbol=request.symbol,
            profit_only=False,
            loss_only=False,
            comment=request.comment,
            deviation=request.deviation,
        )
        if isinstance(position_result, dict):
            msg = str(position_result.get("message", "")).strip().lower()
            if msg.startswith("no open positions for "):
                pending_result = cancel_pending(
                    symbol=request.symbol,
                    comment=request.comment,
                )
                if isinstance(pending_result, dict):
                    pending_msg = str(pending_result.get("message", "")).strip().lower()
                    if pending_msg.startswith("no pending orders for "):
                        return _with_no_action(
                            message=f"No open positions or pending orders for {request.symbol}"
                        )
                return pending_result
        return position_result

    position_result = close_positions(
        profit_only=False,
        loss_only=False,
        comment=request.comment,
        deviation=request.deviation,
    )
    if isinstance(position_result, dict):
        msg = str(position_result.get("message", "")).strip().lower()
        if msg == "no open positions":
            pending_result = cancel_pending(comment=request.comment)
            if (
                isinstance(pending_result, dict)
                and str(pending_result.get("message", "")).strip().lower() == "no pending orders"
            ):
                return _with_no_action(message="No open positions or pending orders")
            return pending_result
    return position_result


def run_trade_history(
    request: TradeHistoryRequest,
    *,
    mt5: Any,
    auto_connect_wrapper: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    parse_start_datetime: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    normalize_ticket_filter: Any,
    normalize_minutes_back: Any,
    decode_mt5_enum_label: Any,
    mt5_config: Any,
) -> Any:
    import pandas as pd

    @auto_connect_wrapper
    def _get_history():
        try:
            use_client_tz_value = use_client_tz()
            fmt_time = format_time_minimal_local if use_client_tz_value else format_time_minimal
            trigger_pattern = re.compile(r"\[(sl|tp)\s+([+-]?\d+(?:\.\d+)?)\]", re.IGNORECASE)

            def _normalize_time_col(df: "pd.DataFrame", col: str) -> Optional["pd.Series"]:
                if col not in df.columns:
                    return None
                raw = pd.to_numeric(df[col], errors="coerce")
                utc = raw.apply(lambda x: mt5_epoch_to_utc(float(x)) if pd.notna(x) else float("nan"))
                df[col] = utc.apply(lambda x: fmt_time(float(x)) if pd.notna(x) else None)
                return utc

            if request.start and request.minutes_back not in (None, ""):
                return {"error": "Use either start or minutes_back, not both."}

            position_ticket_value, position_ticket_error = normalize_ticket_filter(
                request.position_ticket,
                name="position_ticket",
            )
            if position_ticket_error:
                return {"error": position_ticket_error}
            deal_ticket_value, deal_ticket_error = normalize_ticket_filter(
                request.deal_ticket,
                name="deal_ticket",
            )
            if deal_ticket_error:
                return {"error": deal_ticket_error}
            order_ticket_value, order_ticket_error = normalize_ticket_filter(
                request.order_ticket,
                name="order_ticket",
            )
            if order_ticket_error:
                return {"error": order_ticket_error}
            minutes_back_value, minutes_back_error = normalize_minutes_back(request.minutes_back)
            if minutes_back_error:
                return {"error": minutes_back_error}

            if request.end:
                to_dt = parse_start_datetime(request.end)
                if not to_dt:
                    return {"error": "Invalid end time."}
            else:
                to_dt = datetime.now(timezone.utc).replace(tzinfo=None)

            if minutes_back_value is not None:
                from_dt = to_dt - timedelta(minutes=minutes_back_value)
            elif request.start:
                from_dt = parse_start_datetime(request.start)
                if not from_dt:
                    return {"error": "Invalid start time."}
            else:
                from_dt = datetime(2020, 1, 1)

            if from_dt > to_dt:
                return {"error": "start must be before end."}

            kind = str(request.history_kind or "deals").strip().lower()
            if kind not in ("deals", "orders"):
                return {"error": "history_kind must be 'deals' or 'orders'."}
            if kind == "orders" and deal_ticket_value is not None:
                return {"error": "deal_ticket is only valid when history_kind='deals'."}

            def _decode_enum_column(df: "pd.DataFrame", col: str, prefix: str) -> None:
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

            def _extract_exit_trigger(
                comment: Any,
                reason: Any,
                entry: Any,
            ) -> tuple[Optional[str], Optional[float], Optional[str]]:
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

            def _filter_by_ticket_columns(
                df_in: "pd.DataFrame",
                ticket_value: Optional[int],
                *,
                columns: tuple[str, ...],
            ) -> "pd.DataFrame":
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
                rows = (
                    mt5.history_deals_get(from_dt, to_dt, symbol=request.symbol)
                    if request.symbol
                    else mt5.history_deals_get(from_dt, to_dt)
                )
                if rows is None or len(rows) == 0:
                    return {"message": "No deals found"}
                df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())
                if request.symbol and "symbol" in df.columns:
                    df = df.loc[
                        df["symbol"].astype(str).str.upper() == str(request.symbol).upper()
                    ]
                df = _filter_by_ticket_columns(df, deal_ticket_value, columns=("ticket",))
                df = _filter_by_ticket_columns(df, order_ticket_value, columns=("order",))
                df = _filter_by_ticket_columns(
                    df,
                    position_ticket_value,
                    columns=("position_id", "position_by_id"),
                )
                if len(df) == 0:
                    return {
                        "message": (
                            f"No deals found for {request.symbol}"
                            if request.symbol
                            else "No deals found"
                        )
                    }
                sort_src = _normalize_time_col(df, "time")
                _decode_enum_column(df, "type", "DEAL_TYPE_")
                _decode_enum_column(df, "entry", "DEAL_ENTRY_")
                _decode_enum_column(df, "reason", "DEAL_REASON_")
                if len(df) > 0:
                    triggers = df.apply(
                        lambda row: _extract_exit_trigger(
                            row.get("comment"),
                            row.get("reason"),
                            row.get("entry"),
                        ),
                        axis=1,
                        result_type="expand",
                    )
                    if isinstance(triggers, pd.DataFrame) and triggers.shape[1] == 3:
                        triggers.columns = [
                            "exit_trigger",
                            "exit_trigger_price",
                            "exit_trigger_source",
                        ]
                        for col in triggers.columns:
                            df[col] = triggers[col]
                for noise_col in ("time_msc", "external_id", "fee"):
                    if noise_col in df.columns and _is_non_informative_series(df[noise_col]):
                        df = df.drop(columns=[noise_col])
            else:
                rows = (
                    mt5.history_orders_get(from_dt, to_dt, symbol=request.symbol)
                    if request.symbol
                    else mt5.history_orders_get(from_dt, to_dt)
                )
                if rows is None or len(rows) == 0:
                    return {"message": "No orders found"}
                df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())
                if request.symbol and "symbol" in df.columns:
                    df = df.loc[
                        df["symbol"].astype(str).str.upper() == str(request.symbol).upper()
                    ]
                df = _filter_by_ticket_columns(df, order_ticket_value, columns=("ticket",))
                df = _filter_by_ticket_columns(
                    df,
                    position_ticket_value,
                    columns=("position_id", "position_by_id"),
                )
                if len(df) == 0:
                    return {
                        "message": (
                            f"No orders found for {request.symbol}"
                            if request.symbol
                            else "No orders found"
                        )
                    }
                sort_src = _normalize_time_col(df, "time_setup")
                if sort_src is None:
                    sort_src = _normalize_time_col(df, "time")
                _normalize_time_col(df, "time_done")
                _decode_enum_column(df, "type", "ORDER_TYPE_")
                _decode_enum_column(df, "state", "ORDER_STATE_")
                _decode_enum_column(df, "type_time", "ORDER_TIME_")
                _decode_enum_column(df, "type_filling", "ORDER_FILLING_")
                _decode_enum_column(df, "reason", "ORDER_REASON_")

            df["__sort_utc"] = (
                sort_src if sort_src is not None else pd.Series([float("nan")] * len(df))
            )

            limit_value = normalize_limit(request.limit)
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
            if use_client_tz_value:
                try:
                    tz_obj = mt5_config.get_client_tz()
                    timezone_label = str(
                        getattr(tz_obj, "zone", None) or tz_obj or "client_local"
                    )
                except Exception:
                    timezone_label = "client_local"
            for row in records:
                if isinstance(row, dict):
                    row["timestamp_timezone"] = timezone_label
                    row.update(comment_row_metadata(row.get("comment")))
            return records
        except Exception as exc:
            return {"error": str(exc)}

    return _get_history()


def run_trade_risk_analyze(
    request: TradeRiskAnalyzeRequest,
    *,
    mt5: Any,
    auto_connect_wrapper: Any,
) -> Dict[str, Any]:
    @auto_connect_wrapper
    def _analyze_risk():
        try:
            account = mt5.account_info()
            if account is None:
                return {"error": "Failed to get account info"}

            equity = float(account.equity)
            currency = account.currency
            positions = mt5.positions_get(symbol=request.symbol) if request.symbol else mt5.positions_get()
            if positions is None:
                positions = []

            position_risks: List[Dict[str, Any]] = []
            total_risk_currency = 0.0
            positions_without_sl = 0
            total_notional_exposure = 0.0

            for pos in positions:
                try:
                    sym_info = mt5.symbol_info(pos.symbol)
                    if sym_info is None:
                        continue

                    entry_price = float(pos.price_open)
                    sl_price = float(pos.sl) if pos.sl and pos.sl > 0 else None
                    tp_price = float(pos.tp) if pos.tp and pos.tp > 0 else None
                    volume = float(pos.volume)

                    contract_size = float(sym_info.trade_contract_size)
                    point = float(getattr(sym_info, "point", 0.0) or 0.0)
                    tick_value = float(getattr(sym_info, "trade_tick_value", 0.0) or 0.0)
                    tick_size = float(getattr(sym_info, "trade_tick_size", 0.0) or 0.0)
                    if not math.isfinite(tick_size) or tick_size <= 0:
                        tick_size = point if math.isfinite(point) and point > 0 else 0.0
                    tick_value_valid = math.isfinite(tick_value) and tick_value > 0
                    if not math.isfinite(contract_size) or contract_size <= 0:
                        contract_size = 1.0

                    notional_value = abs(volume) * contract_size * entry_price
                    total_notional_exposure += notional_value

                    risk_currency = None
                    risk_pct = None
                    reward_currency = None
                    rr_ratio = None
                    risk_status = "undefined"

                    if sl_price and tick_size > 0 and tick_value_valid:
                        risk_ticks = (
                            (entry_price - sl_price) / tick_size
                            if pos.type == 0
                            else (sl_price - entry_price) / tick_size
                        )
                        risk_currency = abs(risk_ticks * tick_value * volume)
                        risk_pct = (risk_currency / equity) * 100.0 if equity > 0 else 0.0
                        total_risk_currency += risk_currency
                        risk_status = "defined"

                        if tp_price:
                            reward_ticks = (
                                (tp_price - entry_price) / tick_size
                                if pos.type == 0
                                else (entry_price - tp_price) / tick_size
                            )
                            reward_currency = abs(reward_ticks * tick_value * volume)
                            if risk_currency > 0:
                                rr_ratio = reward_currency / risk_currency
                    elif sl_price:
                        risk_status = "undefined"
                    else:
                        positions_without_sl += 1
                        risk_status = "unlimited"

                    position_risks.append(
                        {
                            "ticket": pos.ticket,
                            "symbol": pos.symbol,
                            "type": "BUY" if pos.type == 0 else "SELL",
                            "volume": volume,
                            "entry": entry_price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "risk_currency": round(risk_currency, 2) if risk_currency else None,
                            "risk_pct": round(risk_pct, 2) if risk_pct else None,
                            "risk_status": risk_status,
                            "notional_value": round(notional_value, 2),
                            "reward_currency": round(reward_currency, 2) if reward_currency else None,
                            "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
                        }
                    )
                except Exception:
                    continue

            total_risk_pct = (total_risk_currency / equity) * 100.0 if equity > 0 else 0.0
            notional_exposure_pct = (
                (total_notional_exposure / equity) * 100.0 if equity > 0 else 0.0
            )

            overall_risk_status = "defined"
            if positions_without_sl > 0:
                overall_risk_status = "unlimited"
            elif total_risk_pct > 10:
                overall_risk_status = "high"
            elif total_risk_pct > 5:
                overall_risk_status = "moderate"
            else:
                overall_risk_status = "low"

            result: Dict[str, Any] = {
                "success": True,
                "account": {"equity": round(equity, 2), "currency": currency},
                "portfolio_risk": {
                    "overall_risk_status": overall_risk_status,
                    "total_risk_currency": round(total_risk_currency, 2),
                    "total_risk_pct": round(total_risk_pct, 2),
                    "positions_count": len(position_risks),
                    "positions_without_sl": positions_without_sl,
                    "notional_exposure": round(total_notional_exposure, 2),
                    "notional_exposure_pct": round(notional_exposure_pct, 2),
                },
                "positions": position_risks,
            }
            if positions_without_sl > 0:
                result["warning"] = (
                    f"{positions_without_sl} position(s) without stop loss - UNLIMITED RISK!"
                )

            if (
                request.desired_risk_pct is not None
                and request.proposed_entry is not None
                and request.proposed_sl is not None
            ):
                if not request.symbol:
                    return {"error": "symbol is required for position sizing"}

                sym_info = mt5.symbol_info(request.symbol)
                if sym_info is None:
                    return {"error": f"Symbol {request.symbol} not found"}

                contract_size = float(sym_info.trade_contract_size)
                point = float(getattr(sym_info, "point", 0.0) or 0.0)
                tick_value = float(getattr(sym_info, "trade_tick_value", 0.0) or 0.0)
                tick_size = float(getattr(sym_info, "trade_tick_size", 0.0) or 0.0)
                if not math.isfinite(tick_size) or tick_size <= 0:
                    tick_size = point if math.isfinite(point) and point > 0 else 0.0
                min_volume = float(sym_info.volume_min)
                max_volume = float(sym_info.volume_max)
                volume_step = float(sym_info.volume_step)
                if not (
                    math.isfinite(tick_value)
                    and tick_value > 0
                    and math.isfinite(tick_size)
                    and tick_size > 0
                ):
                    result["position_sizing_error"] = (
                        "Symbol tick configuration is invalid for risk sizing"
                    )
                    return result
                if not (math.isfinite(volume_step) and volume_step > 0):
                    volume_step = max(min_volume, 0.01)
                if not math.isfinite(contract_size) or contract_size <= 0:
                    contract_size = 1.0

                risk_amount = equity * (request.desired_risk_pct / 100.0)
                sl_distance_ticks = abs(request.proposed_entry - request.proposed_sl) / tick_size
                if sl_distance_ticks > 0:
                    raw_volume = risk_amount / (sl_distance_ticks * tick_value)
                    if not math.isfinite(raw_volume) or raw_volume <= 0:
                        result["position_sizing_error"] = "Calculated volume is invalid"
                        return result

                    volume_steps = math.floor((raw_volume / volume_step) + 1e-12)
                    suggested_volume = volume_steps * volume_step
                    rounding_mode = "rounded_down_to_step"
                    sizing_notes: List[str] = []

                    if suggested_volume < min_volume:
                        suggested_volume = min_volume
                        rounding_mode = "clamped_to_min_volume"
                        sizing_notes.append(
                            "Minimum trade volume forces the size up to the broker minimum."
                        )
                    elif suggested_volume > max_volume:
                        suggested_volume = max_volume
                        rounding_mode = "clamped_to_max_volume"
                        sizing_notes.append(
                            "Maximum trade volume caps the size below the unconstrained target."
                        )
                    elif suggested_volume < raw_volume:
                        sizing_notes.append(
                            "Volume was rounded down to the nearest broker step to avoid exceeding requested risk."
                        )

                    step_txt = f"{volume_step:.10f}".rstrip("0")
                    step_decimals = len(step_txt.split(".")[1]) if "." in step_txt else 0
                    if step_decimals > 0:
                        suggested_volume = float(f"{suggested_volume:.{step_decimals}f}")
                    else:
                        suggested_volume = float(round(suggested_volume))

                    actual_risk = sl_distance_ticks * tick_value * suggested_volume
                    actual_risk_pct = (actual_risk / equity) * 100.0
                    risk_pct_diff = actual_risk_pct - float(request.desired_risk_pct)
                    risk_over_target = actual_risk_pct > (float(request.desired_risk_pct) + 1e-9)
                    overshoot_pct = max(0.0, float(actual_risk_pct) - float(request.desired_risk_pct))
                    overshoot_currency = max(0.0, float(actual_risk) - float(risk_amount))
                    overshoot_reason = None
                    if risk_over_target:
                        if rounding_mode == "clamped_to_min_volume":
                            overshoot_reason = "min_volume_constraint"
                        elif rounding_mode == "clamped_to_max_volume":
                            overshoot_reason = "max_volume_constraint"
                        elif rounding_mode == "rounded_down_to_step":
                            overshoot_reason = "step_rounding_precision"
                        else:
                            overshoot_reason = "broker_volume_constraints"
                        sizing_notes.append(
                            "Actual risk still exceeds the requested level after broker volume constraints."
                        )

                    rr_ratio = None
                    reward_currency = None
                    if request.proposed_tp is not None:
                        tp_distance_ticks = abs(request.proposed_tp - request.proposed_entry) / tick_size
                        reward_currency = tp_distance_ticks * tick_value * suggested_volume
                        if actual_risk > 0:
                            rr_ratio = reward_currency / actual_risk

                    result["position_sizing"] = {
                        "symbol": request.symbol,
                        "suggested_volume": suggested_volume,
                        "requested_risk_currency": round(risk_amount, 2),
                        "requested_risk_pct": float(request.desired_risk_pct),
                        "entry": request.proposed_entry,
                        "sl": request.proposed_sl,
                        "tp": request.proposed_tp,
                        "risk_currency": round(actual_risk, 2),
                        "risk_pct": round(actual_risk_pct, 2),
                        "risk_pct_diff": round(risk_pct_diff, 2),
                        "risk_over_target": risk_over_target,
                        "risk_compliance": (
                            "exceeds_requested_risk"
                            if risk_over_target
                            else "within_requested_risk"
                        ),
                        "risk_overshoot_pct": round(overshoot_pct, 2),
                        "risk_overshoot_currency": round(overshoot_currency, 2),
                        "risk_over_target_reason": overshoot_reason,
                        "raw_volume": round(raw_volume, 8),
                        "volume_step": volume_step,
                        "volume_min": min_volume,
                        "volume_max": max_volume,
                        "volume_rounding": rounding_mode,
                        "reward_currency": round(reward_currency, 2) if reward_currency else None,
                        "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
                        "sizing_notes": sizing_notes,
                    }
                    if risk_over_target:
                        result["position_sizing_warning"] = (
                            f"Requested risk {float(request.desired_risk_pct):.2f}% but actual risk is "
                            f"{float(actual_risk_pct):.2f}% (+{overshoot_pct:.2f}%) after broker volume constraints."
                        )
                        result["risk_alert"] = {
                            "severity": "warning",
                            "code": "risk_overshoot_after_volume_constraints",
                            "reason": overshoot_reason,
                            "requested_risk_pct": float(request.desired_risk_pct),
                            "actual_risk_pct": round(actual_risk_pct, 2),
                            "overshoot_pct": round(overshoot_pct, 2),
                            "requested_risk_currency": round(risk_amount, 2),
                            "actual_risk_currency": round(actual_risk, 2),
                            "overshoot_currency": round(overshoot_currency, 2),
                        }
                else:
                    result["position_sizing_error"] = "SL distance must be greater than 0"

            return result
        except Exception as exc:
            return {"error": str(exc)}

    return _analyze_risk()


def run_trade_get_open(
    request: TradeGetOpenRequest,
    *,
    mt5: Any,
    auto_connect_wrapper: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
) -> List[Dict[str, Any]]:
    import pandas as pd

    @auto_connect_wrapper
    def _get_open():
        try:
            use_client_tz_value = use_client_tz()
            fmt_time = (
                format_time_minimal_local if use_client_tz_value else format_time_minimal
            )

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

            if request.ticket is not None:
                ticket_int = int(request.ticket)
                rows = mt5.positions_get(ticket=ticket_int)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No position found with ID {request.ticket}"}]
            elif request.symbol is not None:
                rows = mt5.positions_get(symbol=request.symbol)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No open positions for {request.symbol}"}]
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
                time_utc = time_src.apply(
                    lambda x: mt5_epoch_to_utc(float(x))
                    if pd.notna(x)
                    else float("nan")
                )
                time_txt = time_utc.apply(
                    lambda x: fmt_time(float(x)) if pd.notna(x) else None
                )

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
            comment_meta = _pick_series(df, "comment").apply(comment_row_metadata)
            out_df["Comment Length"] = comment_meta.apply(
                lambda value: value.get("comment_visible_length")
            )
            out_df["Comment Limit"] = comment_meta.apply(
                lambda value: value.get("comment_max_length")
            )
            out_df["Comment May Be Truncated"] = comment_meta.apply(
                lambda value: value.get("comment_may_be_truncated")
            )
            out_df["__time_utc"] = time_utc

            limit_value = normalize_limit(request.limit)
            if limit_value and len(out_df) > limit_value:
                out_df = out_df.sort_values("__time_utc").tail(limit_value)
            if "__time_utc" in out_df.columns:
                out_df = out_df.drop(columns=["__time_utc"])
            return out_df.to_dict(orient="records")
        except Exception as exc:
            return [{"error": str(exc)}]

    return _get_open()


def run_trade_get_pending(
    request: TradeGetPendingRequest,
    *,
    mt5: Any,
    auto_connect_wrapper: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
) -> List[Dict[str, Any]]:
    import pandas as pd

    @auto_connect_wrapper
    def _get_pending():
        try:
            use_client_tz_value = use_client_tz()
            fmt_time = (
                format_time_minimal_local if use_client_tz_value else format_time_minimal
            )

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

            if request.ticket is not None:
                ticket_int = int(request.ticket)
                rows = mt5.orders_get(ticket=ticket_int)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No pending order found with ID {request.ticket}"}]
            elif request.symbol is not None:
                rows = mt5.orders_get(symbol=request.symbol)
                if rows is None or len(rows) == 0:
                    return [{"message": f"No pending orders for {request.symbol}"}]
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
                time_utc = time_src.apply(
                    lambda x: mt5_epoch_to_utc(float(x))
                    if pd.notna(x)
                    else float("nan")
                )
                time_txt = time_utc.apply(
                    lambda x: fmt_time(float(x)) if pd.notna(x) else None
                )

            if "time_expiration" in df.columns:
                exp_raw = pd.to_numeric(df["time_expiration"], errors="coerce")
                exp_utc = exp_raw.apply(
                    lambda x: mt5_epoch_to_utc(float(x))
                    if pd.notna(x) and float(x) > 0
                    else float("nan")
                )
                expiration = exp_raw.apply(
                    lambda x: "GTC" if pd.notna(x) and float(x) <= 0 else None
                )
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
            comment_meta = _pick_series(df, "comment").apply(comment_row_metadata)
            out_df["Comment Length"] = comment_meta.apply(
                lambda value: value.get("comment_visible_length")
            )
            out_df["Comment Limit"] = comment_meta.apply(
                lambda value: value.get("comment_max_length")
            )
            out_df["Comment May Be Truncated"] = comment_meta.apply(
                lambda value: value.get("comment_may_be_truncated")
            )
            out_df["__time_utc"] = time_utc

            limit_value = normalize_limit(request.limit)
            if limit_value and len(out_df) > limit_value:
                out_df = (
                    out_df.sort_values("__time_utc").tail(limit_value)
                    if "__time_utc" in out_df.columns
                    else out_df.head(limit_value)
                )
            if "__time_utc" in out_df.columns:
                out_df = out_df.drop(columns=["__time_utc"])
            return out_df.to_dict(orient="records")
        except Exception as exc:
            return [{"error": str(exc)}]

    return _get_pending()
