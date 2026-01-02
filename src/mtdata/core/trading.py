"""Trading functions for MetaTrader integration."""


import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Union, List, Dict, Any, Literal

from .server import mcp
from ..utils.mt5 import _auto_connect_wrapper
from .config import mt5_config
from .constants import DEFAULT_ROW_LIMIT
from ..utils.utils import _normalize_limit, _parse_start_datetime


ExpirationValue = Union[int, float, str, datetime]
_GTC_EXPIRATION_TOKENS = {"GTC", "GOOD_TILL_CANCEL", "GOOD_TILL_CANCELLED", "NONE", "NO_EXPIRATION"}

MarketOrderTypeLiteral = Literal["BUY", "SELL"]
OrderTypeLiteral = Literal[
    "BUY",
    "SELL",
    "BUY_LIMIT",
    "BUY_STOP",
    "SELL_LIMIT",
    "SELL_STOP",
]


def _to_server_time_naive(dt: datetime) -> datetime:
    """Convert a datetime (naive or aware) to broker/server local time and drop tzinfo.

    - If client/server tzs are configured (pytz), assume naive input is client tz,
      convert to server tz, and return naive server time.
    - Else, use MT5_TIME_OFFSET_MINUTES relative to UTC as a fallback.
    - If no hints, assume input is UTC.
    """
    try:
        server_tz = mt5_config.get_server_tz()
        client_tz = mt5_config.get_client_tz()
    except Exception:
        server_tz = None
        client_tz = None

    aware = dt
    try:
        if dt.tzinfo is None:
            # Assume client-local when configured; otherwise assume UTC
            if client_tz is not None:
                aware = client_tz.localize(dt) if hasattr(client_tz, 'localize') else dt.replace(tzinfo=client_tz)
            else:
                aware = dt.replace(tzinfo=timezone.utc)
    except Exception:
        aware = dt.replace(tzinfo=timezone.utc)

    if server_tz is not None:
        try:
            server_aware = aware.astimezone(server_tz)
            return server_aware.replace(tzinfo=None)
        except Exception:
            pass

    # Fallback: offset seconds from UTC
    try:
        offset_sec = int(mt5_config.get_time_offset_seconds())
    except Exception:
        offset_sec = 0
    try:
        utc_dt = aware.astimezone(timezone.utc)
    except Exception:
        utc_dt = aware if aware.tzinfo is not None else aware.replace(tzinfo=timezone.utc)
    server_dt = utc_dt + timedelta(seconds=offset_sec)
    return server_dt.replace(tzinfo=None)



def _normalize_pending_expiration(expiration: Optional[ExpirationValue]) -> Tuple[Optional[datetime], bool]:
    """Convert user-supplied expiration data into MetaTrader-friendly datetime objects.

    Returns a tuple ``(normalized_expiration, was_explicitly_provided)``. When
    ``was_explicitly_provided`` is False, callers should preserve the broker's existing
    order setting. When it is True and the normalized expiration is None, callers
    should submit a Good-Till-Cancelled order to clear any previous expiration.
    """
    if expiration is None:
        return None, False

    if isinstance(expiration, datetime):
        return _to_server_time_naive(expiration), True

    if isinstance(expiration, (int, float)):
        if not math.isfinite(expiration) or expiration <= 0:
            return None, True
        try:
            # Treat numeric as epoch seconds in UTC, then convert to server time
            return _to_server_time_naive(datetime.fromtimestamp(expiration, tz=timezone.utc)), True
        except (OverflowError, OSError) as exc:
            raise ValueError(f"Expiration timestamp out of range: {expiration}") from exc

    if isinstance(expiration, str):
        cleaned = expiration.strip().strip('"').strip("'")
        if cleaned == "":
            return None, False

        upper_cleaned = cleaned.upper()
        if upper_cleaned in _GTC_EXPIRATION_TOKENS:
            return None, True

        # Regex for simple relative time like 'in 30 minutes', '1h', '30m'
        import re
        # Pattern for "in X units" or just "X units" or "Xunits"
        # Matches: "in 30 min", "30m", "1 hour", "2h", "10 seconds"
        simple_rel_pattern = re.compile(r'^(?:in\s+)?(\d+(?:\.\d+)?)\s*([a-zA-Z]+)$', re.IGNORECASE)
        match = simple_rel_pattern.match(cleaned)
        if match:
            val = float(match.group(1))
            unit = match.group(2).lower()
            delta = None
            if unit in ('s', 'sec', 'secs', 'second', 'seconds'):
                delta = timedelta(seconds=val)
            elif unit in ('m', 'min', 'mins', 'minute', 'minutes'):
                delta = timedelta(minutes=val)
            elif unit in ('h', 'hr', 'hrs', 'hour', 'hours'):
                delta = timedelta(hours=val)
            elif unit in ('d', 'day', 'days'):
                delta = timedelta(days=val)
            elif unit in ('w', 'wk', 'weeks'):
                delta = timedelta(weeks=val)
            
            if delta is not None:
                # Relative to now (UTC) then converted to server time
                return _to_server_time_naive(datetime.now(timezone.utc) + delta), True

        # Try flexible date parsing first (e.g., 'tomorrow 14:00', 'in 2 hours')
        try:
            import dateparser  # type: ignore
            dt = dateparser.parse(cleaned, settings={
                'RETURN_AS_TIMEZONE_AWARE': False,
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': datetime.now(), # explicit base
            })
            if dt is not None:
                return _to_server_time_naive(dt), True
        except Exception:
            pass

        # Fallbacks: numeric epoch or ISO8601
        try:
            numeric = float(cleaned)
            if not math.isfinite(numeric) or numeric <= 0:
                return None, True
            try:
                return _to_server_time_naive(datetime.fromtimestamp(numeric, tz=timezone.utc)), True
            except (OverflowError, OSError) as exc:
                raise ValueError(f"Expiration timestamp out of range: {expiration}") from exc
        except ValueError:
            try:
                return _to_server_time_naive(datetime.fromisoformat(cleaned)), True
            except ValueError as exc:
                raise ValueError(f"Unsupported expiration format: {expiration}") from exc

    raise TypeError(f"Unsupported expiration type: {type(expiration).__name__}")


def _validate_volume(volume: Union[int, float], symbol_info: Any) -> Tuple[Optional[float], Optional[str]]:
    """Validate lot size against symbol constraints (min/max/step)."""
    try:
        vol = float(volume)
    except (TypeError, ValueError):
        return None, "volume must be numeric"

    if not math.isfinite(vol) or vol <= 0:
        return None, "volume must be positive and finite"

    min_vol = getattr(symbol_info, "volume_min", None)
    max_vol = getattr(symbol_info, "volume_max", None)
    step = getattr(symbol_info, "volume_step", None)

    try:
        min_vol = float(min_vol) if min_vol is not None else None
    except (TypeError, ValueError):
        min_vol = None
    if min_vol is not None and min_vol <= 0:
        min_vol = None

    try:
        max_vol = float(max_vol) if max_vol is not None else None
    except (TypeError, ValueError):
        max_vol = None
    if max_vol is not None and max_vol <= 0:
        max_vol = None

    try:
        step = float(step) if step is not None else None
    except (TypeError, ValueError):
        step = None
    if step is not None and step <= 0:
        step = None

    if min_vol is not None and vol < (min_vol - 1e-12):
        return None, f"volume must be >= {min_vol}"
    if max_vol is not None and vol > (max_vol + 1e-12):
        return None, f"volume must be <= {max_vol}"

    if step is not None:
        normalized = round(vol / step) * step
        normalized = float(f"{normalized:.10f}")
        tol = step * 1e-6
        if abs(normalized - vol) > tol:
            return None, f"volume must align to step {step}. Try {normalized}"
        vol = normalized
        if min_vol is not None and vol < (min_vol - 1e-12):
            return None, f"volume must be >= {min_vol}"
        if max_vol is not None and vol > (max_vol + 1e-12):
            return None, f"volume must be <= {max_vol}"

    return vol, None


def _validate_deviation(deviation: Union[int, float]) -> Tuple[Optional[int], Optional[str]]:
    """Validate/normalize MT5 deviation (slippage tolerance) in points."""
    try:
        dev = int(float(deviation))
    except (TypeError, ValueError):
        return None, "deviation must be numeric"
    if dev < 0:
        return None, "deviation must be >= 0"
    return dev, None


def _normalize_trade_comment(comment: Optional[str], *, default: str, suffix: str = "") -> str:
    """Return an MT5-safe comment string.

    MT5 typically enforces a short comment length; we trim to a conservative
    limit to avoid hard-to-debug order_send failures.
    """
    max_len = 31
    try:
        base = str(comment or "").strip()
    except Exception:
        base = ""
    if not base:
        base = str(default or "").strip() or "MCP"

    full = f"{base}{suffix}" if suffix else base
    try:
        if len(full) > max_len:
            if suffix:
                allowed_base = max_len - len(suffix)
                if allowed_base > 0:
                    full = f"{base[:allowed_base]}{suffix}"
                else:
                    full = base[:max_len]
            else:
                full = base[:max_len]
    except Exception:
        full = str(default or "MCP")[:max_len]
    return full


@mcp.tool()
def trading_account_info() -> dict:
    """Get account information (balance, equity, profit, margin level, free margin, account type, leverage, currency)."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _get_account_info():
        info = mt5.account_info()
        if info is None:
            return {"error": "Failed to get account info"}

        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "currency": info.currency,
            "leverage": info.leverage,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
        }

    return _get_account_info()


@mcp.tool()
def trading_history(
    history_kind: Literal["deals", "orders"] = "deals",  # type: ignore
    start: Optional[str] = None,
    end: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
) -> List[Dict[str, Any]]:
    """Get deal or order history as tabular data."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_history():
        try:
            if start:
                from_dt = _parse_start_datetime(start)
                if not from_dt:
                    return {"error": "Invalid start time."}
            else:
                from_dt = datetime(2020, 1, 1)

            if end:
                to_dt = _parse_start_datetime(end)
                if not to_dt:
                    return {"error": "Invalid end time."}
            else:
                to_dt = datetime.utcnow()

            if from_dt > to_dt:
                return {"error": "start must be before end."}

            kind = str(history_kind or "deals").strip().lower()
            if kind not in ("deals", "orders"):
                return {"error": "history_kind must be 'deals' or 'orders'."}

            if kind == "deals":
                if symbol:
                    rows = mt5.history_deals_get(from_dt, to_dt, symbol=symbol)
                else:
                    rows = mt5.history_deals_get(from_dt, to_dt)
                if rows is None or len(rows) == 0:
                    return {"message": "No deals found"}
                df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                sort_col = 'time' if 'time' in df.columns else None
            else:
                if symbol:
                    rows = mt5.history_orders_get(from_dt, to_dt, symbol=symbol)
                else:
                    rows = mt5.history_orders_get(from_dt, to_dt)
                if rows is None or len(rows) == 0:
                    return {"message": "No orders found"}
                df = pd.DataFrame(list(rows), columns=rows[0]._asdict().keys())
                if 'time_setup' in df.columns:
                    df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
                if 'time_done' in df.columns:
                    df['time_done'] = pd.to_datetime(df['time_done'], unit='s')
                sort_col = 'time_setup' if 'time_setup' in df.columns else None

            limit_value = _normalize_limit(limit)
            if limit_value and len(df) > limit_value:
                if sort_col:
                    df = df.sort_values(sort_col).tail(limit_value)
                else:
                    df = df.tail(limit_value)
            return df.to_dict(orient='records')
        except Exception as e:
            return {"error": str(e)}

    return _get_history()


@mcp.tool()
def trading_open_get(
    open_kind: Literal["positions", "pending"] = "positions",  # type: ignore
    symbol: Optional[str] = None,
    ticket: Optional[Union[int, str]] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
) -> List[Dict[str, Any]]:
    """Get open positions or pending orders."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_open():
        try:
            kind = str(open_kind or "positions").strip().lower()
            if kind not in ("positions", "pending"):
                return {"error": "open_kind must be 'positions' or 'pending'."}

            if kind == "positions":
                if ticket is not None:
                    t_int = int(ticket)
                    rows = mt5.positions_get(ticket=t_int)
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
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                sort_col = 'time' if 'time' in df.columns else None
            else:
                if ticket is not None:
                    t_int = int(ticket)
                    rows = mt5.orders_get(ticket=t_int)
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
                if 'time_setup' in df.columns:
                    df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
                sort_col = 'time_setup' if 'time_setup' in df.columns else None

            limit_value = _normalize_limit(limit)
            if limit_value and len(df) > limit_value:
                if sort_col:
                    df = df.sort_values(sort_col).tail(limit_value)
                else:
                    df = df.head(limit_value)
            return df.to_dict(orient='records')

        except Exception as e:
            return [{"error": str(e)}]

    return _get_open()


def _place_market_order(
    symbol: str,
    volume: float,
    order_type: MarketOrderTypeLiteral,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Internal helper to place a market order."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _place_market_order():
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"error": f"Failed to select symbol {symbol}"}

            volume_validated, volume_error = _validate_volume(volume, symbol_info)
            if volume_error:
                return {"error": volume_error}

            current_tick = mt5.symbol_info_tick(symbol)
            if current_tick is None:
                return {"error": f"Failed to get current price for {symbol}"}

            # Normalize and validate requested order type
            t = (order_type or "").strip().upper()
            if t == "BUY":
                side = "BUY"
            elif t == "SELL":
                side = "SELL"
            else:
                return {"error": f"Unsupported order_type '{order_type}'. Use BUY or SELL."}

            deviation_validated, deviation_error = _validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            price = current_tick.ask if side == "BUY" else current_tick.bid

            # Price normalization helper
            point = float(symbol_info.point or 0.0) if hasattr(symbol_info, "point") else 0.0
            digits = int(symbol_info.digits) if hasattr(symbol_info, "digits") else 5

            def _normalize_price(val: Optional[Union[int, float]]) -> Optional[float]:
                try:
                    if val is None:
                        return None
                    v = float(val)
                    if not math.isfinite(v):
                        return None
                    if point and point > 0:
                        # Align to symbol precision
                        v = round(v / point) * point
                    else:
                        v = round(v, digits)
                    return v
                except Exception:
                    return None

            norm_sl = _normalize_price(stop_loss) if stop_loss not in (None, 0) else None
            norm_tp = _normalize_price(take_profit) if take_profit not in (None, 0) else None

            # SL/TP validation for market orders
            if norm_sl is not None:
                if side == "BUY" and norm_sl >= price:
                    return {"error": f"stop_loss must be below entry for BUY orders. sl={norm_sl}, price={price}"}
                if side == "SELL" and norm_sl <= price:
                    return {"error": f"stop_loss must be above entry for SELL orders. sl={norm_sl}, price={price}"}
            if norm_tp is not None:
                if side == "BUY" and norm_tp <= price:
                    return {"error": f"take_profit must be above entry for BUY orders. tp={norm_tp}, price={price}"}
                if side == "SELL" and norm_tp >= price:
                    return {"error": f"take_profit must be below entry for SELL orders. tp={norm_tp}, price={price}"}

            # Place market order without TP/SL first (TRADE_ACTION_DEAL doesn't
            # reliably support them)
            request_comment = _normalize_trade_comment(comment, default="MCP order")
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_validated,
                "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": deviation_validated,
                "magic": 234000,
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is None:
                # Surface MetaTrader last_error when available for easier debugging
                try:
                    err = mt5.last_error()
                except Exception:
                    err = None
                return {"error": "Failed to send order", "last_error": err}
            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to send order",
                    "retcode": result.retcode,
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            # If TP/SL were specified, modify the position immediately
            position_ticket = result.order
            sl_tp_modified = False
            sl_tp_error = None

            if norm_sl is not None or norm_tp is not None:
                try:
                    # Get the position that was just opened
                    positions = mt5.positions_get(ticket=position_ticket)
                    if positions and len(positions) > 0:
                        # Use TRADE_ACTION_SLTP to set TP/SL on the position
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_ticket,
                            "sl": norm_sl or 0.0,
                            "tp": norm_tp or 0.0,
                            "magic": 234000,
                            "comment": _normalize_trade_comment(
                                comment,
                                default=request_comment,
                                suffix=" - set TP/SL",
                            ),
                        }
                        modify_result = mt5.order_send(modify_request)
                        if modify_result and getattr(modify_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                            sl_tp_modified = True
                        else:
                            sl_tp_error = "Failed to set TP/SL"
                    else:
                        sl_tp_error = "Position not found for TP/SL modification"
                except Exception as e:
                    sl_tp_error = f"Error setting TP/SL: {str(e)}"

            return {
                "retcode": result.retcode,
                "deal": result.deal,
                "order": result.order,
                "volume": result.volume,
                "price": result.price,
                "bid": result.bid,
                "ask": result.ask,
                "comment": result.comment,
                "request_id": result.request_id,
                "sl": norm_sl,
                "tp": norm_tp,
                "sl_tp_modified": sl_tp_modified,
                "sl_tp_error": sl_tp_error,
            }

        except Exception as e:
            return {"error": str(e)}

    return _place_market_order()


def _place_pending_order(
    symbol: str,
    volume: float,
    order_type: OrderTypeLiteral,
    price: Union[int, float],
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Internal helper to place a pending order."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _place_pending_order():
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"error": f"Failed to select symbol {symbol}"}

            volume_validated, volume_error = _validate_volume(volume, symbol_info)
            if volume_error:
                return {"error": volume_error}

            current_price = mt5.symbol_info_tick(symbol)
            if current_price is None:
                return {"error": f"Failed to get current price for {symbol}"}

            deviation_validated, deviation_error = _validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # Normalize and validate requested order type
            t = (order_type or "").strip().upper()
            explicit_map = {
                "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
                "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
                "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
                "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
            }

            # Basic side/price sanity checks for explicit pending types
            bid = float(getattr(current_price, "bid", 0.0) or 0.0)
            ask = float(getattr(current_price, "ask", 0.0) or 0.0)
            point = float(symbol_info.point or 0.0) if hasattr(symbol_info, "point") else 0.0
            digits = int(symbol_info.digits) if hasattr(symbol_info, "digits") else 5

            def _normalize_price(val: Optional[Union[int, float]]) -> Optional[float]:
                try:
                    if val is None:
                        return None
                    v = float(val)
                    if not math.isfinite(v):
                        return None
                    if point and point > 0:
                        # Align to symbol precision
                        v = round(v / point) * point
                    else:
                        v = round(v, digits)
                    return v
                except Exception:
                    return None

            norm_price = _normalize_price(price)
            if norm_price is None:
                return {"error": "price must be a finite number"}

            order_type_value = None
            if t in explicit_map:
                order_type_value = explicit_map[t]
            elif t == "BUY":
                order_type_value = mt5.ORDER_TYPE_BUY_LIMIT if norm_price < ask else mt5.ORDER_TYPE_BUY_STOP
            elif t == "SELL":
                order_type_value = mt5.ORDER_TYPE_SELL_LIMIT if norm_price > bid else mt5.ORDER_TYPE_SELL_STOP
            else:
                return {
                    "error": (
                        f"Unsupported order_type '{order_type}'. "
                        "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
                    )
                }
            norm_sl = _normalize_price(stop_loss) if stop_loss not in (None, 0) else None
            norm_tp = _normalize_price(take_profit) if take_profit not in (None, 0) else None

            if order_type_value == mt5.ORDER_TYPE_BUY_LIMIT and not (norm_price < ask):
                return {"error": f"Price must be below ask for BUY_LIMIT. price={norm_price}, ask={ask}"}
            if order_type_value == mt5.ORDER_TYPE_BUY_STOP and not (norm_price > ask):
                return {"error": f"Price must be above ask for BUY_STOP. price={norm_price}, ask={ask}"}
            if order_type_value == mt5.ORDER_TYPE_SELL_LIMIT and not (norm_price > bid):
                return {"error": f"Price must be above bid for SELL_LIMIT. price={norm_price}, bid={bid}"}
            if order_type_value == mt5.ORDER_TYPE_SELL_STOP and not (norm_price < bid):
                return {"error": f"Price must be below bid for SELL_STOP. price={norm_price}, bid={bid}"}

            normalized_expiration, expiration_specified = _normalize_pending_expiration(expiration)

            # SL/TP sanity relative to entry
            if norm_sl is not None:
                if order_type_value in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP) and norm_sl >= norm_price:
                    return {"error": f"stop_loss must be below entry for BUY orders. sl={norm_sl}, price={norm_price}"}
                if order_type_value in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP) and norm_sl <= norm_price:
                    return {"error": f"stop_loss must be above entry for SELL orders. sl={norm_sl}, price={norm_price}"}
            if norm_tp is not None:
                if order_type_value in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP) and norm_tp <= norm_price:
                    return {"error": f"take_profit must be above entry for BUY orders. tp={norm_tp}, price={norm_price}"}
                if order_type_value in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP) and norm_tp >= norm_price:
                    return {"error": f"take_profit must be below entry for SELL orders. tp={norm_tp}, price={norm_price}"}

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume_validated,
                "type": order_type_value,
                "price": norm_price,
                "sl": norm_sl or 0.0,
                "tp": norm_tp or 0.0,
                "deviation": deviation_validated,
                "magic": 234000,
                "comment": _normalize_trade_comment(comment, default="MCP pending order"),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            if expiration_specified:
                if normalized_expiration is None:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                else:
                    request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    request["expiration"] = normalized_expiration

            result = mt5.order_send(request)
            if result is None:
                # Surface MetaTrader last_error when available for easier debugging
                try:
                    err = mt5.last_error()
                except Exception:
                    err = None
                return {"error": "Failed to send pending order", "last_error": err, "request": request}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to send pending order",
                    "retcode": result.retcode,
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            return {
                "success": True,
                "retcode": result.retcode,
                "deal": result.deal,
                "order": result.order,
                "volume": result.volume,
                "price": result.price,
                "bid": result.bid,
                "ask": result.ask,
                "comment": result.comment,
                "request_id": result.request_id,
            }

        except Exception as e:
            return {"error": str(e)}

    return _place_pending_order()


@mcp.tool()
def trading_place(
    symbol: str,
    volume: float,
    order_type: OrderTypeLiteral,
    price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Place a market or pending order.

    - BUY/SELL: market by default; treated as pending when `price`/`expiration` is provided.
    - BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP: pending (requires `price`).
    """

    t = (order_type or "").strip().upper()
    explicit_pending_types = {
        "BUY_LIMIT",
        "BUY_STOP",
        "SELL_LIMIT",
        "SELL_STOP",
    }
    market_side_types = {"BUY", "SELL"}
    supported_order_types = explicit_pending_types.union(market_side_types)
    if t not in supported_order_types:
        return {
            "error": (
                f"Unsupported order_type '{order_type}'. "
                "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
            )
        }

    price_provided = price not in (None, 0)
    expiration_provided = expiration is not None

    is_pending = (t in explicit_pending_types) or price_provided or expiration_provided
    if not is_pending:
        return _place_market_order(
            symbol=symbol,
            volume=volume,
            order_type=t,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
            deviation=deviation,
        )
    if price is None:
        return {"error": "price is required for pending orders."}
    return _place_pending_order(
        symbol=symbol,
        volume=volume,
        order_type=t,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        expiration=expiration,
        comment=comment,
        deviation=deviation,
    )


def _modify_position(
    ticket: Union[int, str],
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
) -> dict:
    """Internal helper to modify a position by ticket."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _modify_position():
        try:
            ticket_id = int(ticket)
            positions = mt5.positions_get(ticket=ticket_id)
            if positions is None or len(positions) == 0:
                return {"error": f"Position {ticket} not found"}

            position = positions[0]

            # Get symbol info for price normalization
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {position.symbol}"}

            point = float(symbol_info.point or 0.0) if hasattr(symbol_info, "point") else 0.0
            digits = int(symbol_info.digits) if hasattr(symbol_info, "digits") else 5

            def _normalize_price(val: Optional[Union[int, float]]) -> Optional[float]:
                """Normalize price to symbol precision."""
                try:
                    if val is None or val == 0:
                        return None
                    v = float(val)
                    if not math.isfinite(v):
                        return None
                    if point and point > 0:
                        # Align to symbol precision
                        v = round(v / point) * point
                    else:
                        v = round(v, digits)
                    return v
                except Exception:
                    return None

            # Normalize SL/TP values
            norm_sl = _normalize_price(stop_loss) if stop_loss is not None else (position.sl or 0.0)
            norm_tp = _normalize_price(take_profit) if take_profit is not None else (position.tp or 0.0)

            # Ensure SL/TP values are 0.0 if they should be removed
            if norm_sl is None:
                norm_sl = 0.0
            if norm_tp is None:
                norm_tp = 0.0

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket_id,
                "sl": norm_sl,
                "tp": norm_tp,
                "magic": 234000,
                "comment": _normalize_trade_comment(comment, default="MCP modify position"),
            }

            result = mt5.order_send(request)
            if result is None:
                # surface the MT5 terminal error for debugging
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
                return {"error": "Failed to modify position", "request": request, "last_error": last_err}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to modify position",
                    "retcode": result.retcode,
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            return {
                "success": True,
                "retcode": result.retcode,
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "request_id": result.request_id,
            }

        except Exception as e:
            return {"error": str(e)}

    return _modify_position()


def _modify_pending_order(
    ticket: Union[int, str],
    price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
) -> dict:
    """Internal helper to modify a pending order by ticket."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _modify_pending_order():
        try:
            ticket_id = int(ticket)
            orders = mt5.orders_get(ticket=ticket_id)
            if orders is None or len(orders) == 0:
                return {"error": f"Pending order {ticket} not found"}

            order = orders[0]
            normalized_expiration, expiration_specified = _normalize_pending_expiration(expiration)

            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket_id,
                "price": price if price is not None else order.price_open,
                "sl": stop_loss if stop_loss is not None else order.sl,
                "tp": take_profit if take_profit is not None else order.tp,
                "magic": 234000,
                "comment": _normalize_trade_comment(comment, default="MCP modify pending order"),
            }

            if expiration_specified:
                if normalized_expiration is None:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                else:
                    request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    request["expiration"] = normalized_expiration
            else:
                current_type_time = getattr(order, "type_time", None)
                current_expiration = getattr(order, "time_expiration", None)
                if current_type_time is not None:
                    request["type_time"] = current_type_time
                    if current_type_time == mt5.ORDER_TIME_SPECIFIED and current_expiration:
                        request["expiration"] = current_expiration

            result = mt5.order_send(request)
            if result is None:
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
                return {"error": "Failed to modify pending order", "request": request, "last_error": last_err}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to modify pending order",
                    "retcode": result.retcode,
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            return {
                "success": True,
                "retcode": result.retcode,
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "request_id": result.request_id,
            }

        except Exception as e:
            return {"error": str(e)}

    return _modify_pending_order()


@mcp.tool()
def trading_modify(
    ticket: Union[int, str],
    modify_kind: Literal["position", "pending"] = "position",  # type: ignore
    price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
) -> dict:
    """Modify an open position or pending order by ticket."""
    kind = str(modify_kind or "position").strip().lower()
    if kind not in ("position", "pending"):
        return {"error": "modify_kind must be 'position' or 'pending'."}
    if kind == "position":
        if price is not None:
            return {"error": "price is only used for pending orders."}
        if expiration is not None:
            return {"error": "expiration is only used for pending orders."}
        return _modify_position(
            ticket=ticket,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )
    return _modify_pending_order(
        ticket=ticket,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        expiration=expiration,
        comment=comment,
    )


def _close_positions(
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Internal helper to close open positions."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _close_positions():
        try:
            # 1. Fetch positions based on criteria
            if ticket is not None:
                t_int = int(ticket)
                positions = mt5.positions_get(ticket=t_int)
                if positions is None or len(positions) == 0:
                    return {"error": f"Position {ticket} not found"}
            elif symbol is not None:
                positions = mt5.positions_get(symbol=symbol)
                if positions is None or len(positions) == 0:
                    return {"message": f"No open positions for {symbol}"}
            else:
                positions = mt5.positions_get()
                if positions is None or len(positions) == 0:
                    return {"message": "No open positions"}

            # 2. Filter positions
            to_close = []
            for pos in positions:
                if profit_only and pos.profit <= 0:
                    continue
                if loss_only and pos.profit >= 0:
                    continue
                to_close.append(pos)

            if not to_close:
                return {"message": "No positions matched criteria"}

            deviation_validated, deviation_error = _validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # 3. Close positions
            results = []
            for position in to_close:
                tick = mt5.symbol_info_tick(position.symbol)
                if tick is None:
                    results.append({"ticket": position.ticket, "error": f"Failed to get tick data for {position.symbol}"})
                    continue

                close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
                close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": close_type,
                    "price": close_price,
                    "deviation": deviation_validated,
                    "magic": 234000,
                    "comment": _normalize_trade_comment(comment, default="MCP close position"),
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": position.ticket, "error": "Failed to send close order"})
                else:
                    res_dict = {
                        "ticket": position.ticket,
                        "retcode": result.retcode,
                        "deal": result.deal,
                        "order": result.order,
                        "volume": result.volume,
                        "price": result.price,
                        "comment": result.comment,
                    }
                    results.append(res_dict)

            # If only one position was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]

            return {"closed_count": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _close_positions()


def _cancel_pending(
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    comment: Optional[str] = None,
) -> dict:
    """Internal helper to cancel pending orders."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _cancel_pending():
        try:
            # 1. Fetch orders based on criteria
            if ticket is not None:
                t_int = int(ticket)
                orders = mt5.orders_get(ticket=t_int)
                if orders is None or len(orders) == 0:
                    return {"error": f"Pending order {ticket} not found"}
            elif symbol is not None:
                orders = mt5.orders_get(symbol=symbol)
                if orders is None or len(orders) == 0:
                    return {"message": f"No pending orders for {symbol}"}
            else:
                orders = mt5.orders_get()
                if orders is None or len(orders) == 0:
                    return {"message": "No pending orders"}

            # 2. Cancel orders
            results = []
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "magic": 234000,
                    "comment": _normalize_trade_comment(comment, default="MCP cancel pending order"),
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": order.ticket, "error": "Failed to send cancel order"})
                else:
                    results.append({
                        "ticket": order.ticket,
                        "retcode": result.retcode,
                        "deal": result.deal,
                        "order": result.order,
                        "comment": result.comment,
                    })

            # If only one order was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]

            return {"cancelled_count": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _cancel_pending()


@mcp.tool()
def trading_close(
    close_kind: Literal["positions", "pending"] = "positions",  # type: ignore
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Close positions or cancel pending orders."""
    kind = str(close_kind or "positions").strip().lower()
    if kind not in ("positions", "pending"):
        return {"error": "close_kind must be 'positions' or 'pending'."}
    if kind == "pending":
        if profit_only or loss_only:
            return {"error": "profit_only/loss_only only apply to positions."}
        return _cancel_pending(ticket=ticket, symbol=symbol, comment=comment)
    return _close_positions(
        ticket=ticket,
        symbol=symbol,
        profit_only=profit_only,
        loss_only=loss_only,
        comment=comment,
        deviation=deviation,
    )


@mcp.tool()
def trading_risk_analyze(
    symbol: Optional[str] = None,
    desired_risk_pct: Optional[float] = None,
    proposed_entry: Optional[float] = None,
    proposed_sl: Optional[float] = None,
    proposed_tp: Optional[float] = None,
) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades.
    
    Use Cases:
    ----------
    1. Analyze current portfolio risk
    2. Calculate proper position size for a new trade based on risk %
    3. Get R:R ratios for existing positions
    
    Parameters:
    -----------
    symbol : str, optional
        Analyze positions for a specific symbol. If not provided, analyzes all positions.
    
    desired_risk_pct : float, optional
        Desired risk percentage for position sizing (e.g., 2.0 for 2% risk per trade)
        **Required with proposed_entry and proposed_sl for position sizing**
    
    proposed_entry : float, optional
        Entry price for new trade. Used with desired_risk_pct to calculate lot size.
    
    proposed_sl : float, optional
        Stop loss price for new trade.
    
    proposed_tp : float, optional
        Take profit price for new trade. Used to calculate R:R ratio.
    
    Returns:
    --------
    dict
        - success: bool
        - account: {equity, currency}
        - portfolio_risk: {total_risk_currency, total_risk_pct, positions_count}
        - positions: list of position risk details
        - position_sizing: (if desired_risk_pct provided) suggested volume and risk metrics
    
    Examples:
    ---------
    # Analyze current portfolio risk
    trading_risk_analyze()
    
    # Analyze risk for specific symbol
    trading_risk_analyze(symbol="EURUSD")
    
    # Calculate position size for 2% risk
    trading_risk_analyze(
        symbol="EURUSD",
        desired_risk_pct=2.0,
        proposed_entry=1.1000,
        proposed_sl=1.0950,
        proposed_tp=1.1100
    )
    """
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _analyze_risk():
        try:
            # Get account info
            account = mt5.account_info()
            if account is None:
                return {"error": "Failed to get account info"}
            
            equity = float(account.equity)
            currency = account.currency
            
            # Get positions
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                positions = []
            
            # Analyze each position
            position_risks = []
            total_risk_currency = 0.0
            positions_without_sl = 0
            total_notional_exposure = 0.0
            
            for pos in positions:
                try:
                    # Get symbol info for calculations
                    sym_info = mt5.symbol_info(pos.symbol)
                    if sym_info is None:
                        continue
                    
                    # Calculate risk per position
                    entry_price = float(pos.price_open)
                    sl_price = float(pos.sl) if pos.sl and pos.sl > 0 else None
                    tp_price = float(pos.tp) if pos.tp and pos.tp > 0 else None
                    volume = float(pos.volume)
                    
                    # Contract specifications
                    contract_size = float(sym_info.trade_contract_size)
                    point = float(sym_info.point)
                    tick_value = float(sym_info.trade_tick_value)
                    
                    # Calculate notional exposure (position value)
                    notional_value = volume * contract_size * entry_price / (contract_size if contract_size > 1 else 1)
                    total_notional_exposure += notional_value
                    
                    risk_currency = None
                    risk_pct = None
                    reward_currency = None
                    rr_ratio = None
                    risk_status = "undefined"
                    
                    if sl_price:
                        # Calculate risk in price distance
                        if pos.type == 0:  # BUY
                            risk_points = (entry_price - sl_price) / point
                        else:  # SELL
                            risk_points = (sl_price - entry_price) / point
                        
                        # Risk in account currency
                        risk_currency = abs(risk_points * tick_value * volume)
                        risk_pct = (risk_currency / equity) * 100.0 if equity > 0 else 0.0
                        total_risk_currency += risk_currency
                        risk_status = "defined"
                        
                        # Calculate reward if TP is set
                        if tp_price:
                            if pos.type == 0:  # BUY
                                reward_points = (tp_price - entry_price) / point
                            else:  # SELL
                                reward_points = (entry_price - tp_price) / point
                            
                            reward_currency = abs(reward_points * tick_value * volume)
                            
                            if risk_currency > 0:
                                rr_ratio = reward_currency / risk_currency
                    else:
                        # No SL = unlimited risk
                        positions_without_sl += 1
                        risk_status = "unlimited"
                    
                    position_risks.append({
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
                    })
                except Exception as e:
                    continue
            
            # Calculate total portfolio risk
            total_risk_pct = (total_risk_currency / equity) * 100.0 if equity > 0 else 0.0
            notional_exposure_pct = (total_notional_exposure / equity) * 100.0 if equity > 0 else 0.0
            
            # Determine overall risk level
            overall_risk_status = "defined"
            if positions_without_sl > 0:
                overall_risk_status = "unlimited"
            elif total_risk_pct > 10:
                overall_risk_status = "high"
            elif total_risk_pct > 5:
                overall_risk_status = "moderate"
            else:
                overall_risk_status = "low"
            
            result = {
                "success": True,
                "account": {
                    "equity": round(equity, 2),
                    "currency": currency,
                },
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
            
            # Add warning if positions lack SL
            if positions_without_sl > 0:
                result["warning"] = f"{positions_without_sl} position(s) without stop loss - UNLIMITED RISK!"
            
            # Calculate position sizing if desired_risk_pct is provided
            if desired_risk_pct is not None and proposed_entry is not None and proposed_sl is not None:
                if not symbol:
                    return {"error": "symbol is required for position sizing"}
                
                sym_info = mt5.symbol_info(symbol)
                if sym_info is None:
                    return {"error": f"Symbol {symbol} not found"}
                
                # Calculate position size
                contract_size = float(sym_info.trade_contract_size)
                point = float(sym_info.point)
                tick_value = float(sym_info.trade_tick_value)
                min_volume = float(sym_info.volume_min)
                max_volume = float(sym_info.volume_max)
                volume_step = float(sym_info.volume_step)
                
                # Risk amount in account currency
                risk_amount = equity * (desired_risk_pct / 100.0)
                
                # SL distance in points
                sl_distance_points = abs(proposed_entry - proposed_sl) / point
                
                if sl_distance_points > 0:
                    # Calculate volume
                    suggested_volume = risk_amount / (sl_distance_points * tick_value)
                    
                    # Round to volume step
                    suggested_volume = round(suggested_volume / volume_step) * volume_step
                    
                    # Clamp to min/max
                    suggested_volume = max(min_volume, min(suggested_volume, max_volume))
                    
                    # Calculate actual risk with suggested volume
                    actual_risk = sl_distance_points * tick_value * suggested_volume
                    actual_risk_pct = (actual_risk / equity) * 100.0
                    
                    # Calculate R:R if TP is provided
                    rr_ratio = None
                    reward_currency = None
                    if proposed_tp is not None:
                        tp_distance_points = abs(proposed_tp - proposed_entry) / point
                        reward_currency = tp_distance_points * tick_value * suggested_volume
                        if actual_risk > 0:
                            rr_ratio = reward_currency / actual_risk
                    
                    result["position_sizing"] = {
                        "symbol": symbol,
                        "suggested_volume": round(suggested_volume, 2),
                        "entry": proposed_entry,
                        "sl": proposed_sl,
                        "tp": proposed_tp,
                        "risk_currency": round(actual_risk, 2),
                        "risk_pct": round(actual_risk_pct, 2),
                        "reward_currency": round(reward_currency, 2) if reward_currency else None,
                        "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
                    }
                else:
                    result["position_sizing_error"] = "SL distance must be greater than 0"
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    return _analyze_risk()
