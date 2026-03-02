
from typing import Any, Dict
import math

from ..utils.mt5 import _mt5_epoch_to_utc
from ..utils.utils import _format_time_minimal, _format_time_minimal_local, _use_client_tz
from .server import mcp, _auto_connect_wrapper
import MetaTrader5 as mt5

@mcp.tool()
@_auto_connect_wrapper
def market_depth_fetch(symbol: str, spread: bool = False) -> Dict[str, Any]:
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`.

    Parameters: symbol
    """
    try:
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            return {"error": f"Failed to select symbol {symbol}: {mt5.last_error()}"}

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"error": f"Symbol {symbol} not found"}

        digits = max(0, int(getattr(symbol_info, "digits", 0) or 0))
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)

        def _compute_spread_metrics(bid: Any, ask: Any) -> Dict[str, Any] | None:
            try:
                bid_f = float(bid)
                ask_f = float(ask)
            except Exception:
                return None
            if not (math.isfinite(bid_f) and math.isfinite(ask_f)):
                return None
            spread_abs = float(ask_f - bid_f)
            if spread_abs < 0:
                return None
            mid = (ask_f + bid_f) / 2.0
            spread_points = (spread_abs / point) if point > 0 else None
            spread_pct = ((spread_abs / mid) * 100.0) if mid > 0 else None
            spread_usd = None
            if tick_size > 0 and tick_value > 0:
                spread_usd = (spread_abs / tick_size) * tick_value
            return {
                "spread": spread_abs,
                "spread_display": _price_display(spread_abs),
                "spread_points": spread_points,
                "spread_pct": spread_pct,
                "spread_usd": spread_usd,
                "pricing_basis": "per_1_lot_estimate" if spread_usd is not None else "quote_only",
            }

        def _price_display(value: Any) -> Any:
            if value is None:
                return None
            try:
                val = float(value)
                if digits > 0:
                    return f"{val:.{digits}f}"
                return str(val)
            except Exception:
                return None
        
        # Try to get market depth first
        depth = mt5.market_book_get(symbol)
        
        if depth is not None and len(depth) > 0:
            # Process DOM levels
            buy_orders = []
            sell_orders = []
            
            for level in depth:
                order_data = {
                    "price": float(level["price"]),
                    "price_display": _price_display(level["price"]),
                    "volume": float(level["volume"]),
                    "volume_real": float(level["volume_real"])
                }
                
                if int(level["type"]) == 0:  # Buy order
                    buy_orders.append(order_data)
                else:  # Sell order
                    sell_orders.append(order_data)
            
            out = {
                "success": True,
                "symbol": symbol,
                "type": "full_depth",
                "price_precision": digits,
                "data": {
                    "buy_orders": buy_orders,
                    "sell_orders": sell_orders
                }
            }
            if spread and buy_orders and sell_orders:
                best_bid = max(float(row.get("price")) for row in buy_orders if row.get("price") is not None)
                best_ask = min(float(row.get("price")) for row in sell_orders if row.get("price") is not None)
                spread_metrics = _compute_spread_metrics(best_bid, best_ask)
                if spread_metrics is not None:
                    out["data"]["best_bid"] = best_bid
                    out["data"]["best_ask"] = best_ask
                    out["data"].update(spread_metrics)
            return out
        else:
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}
            
            out = {
                "success": True,
                "symbol": symbol,
                "type": "tick_data",
                "price_precision": digits,
                "data": {
                    "bid": float(tick.bid) if tick.bid else None,
                    "ask": float(tick.ask) if tick.ask else None,
                    "last": float(tick.last) if tick.last else None,
                    "bid_display": _price_display(tick.bid),
                    "ask_display": _price_display(tick.ask),
                    "last_display": _price_display(tick.last),
                    "volume": int(tick.volume) if tick.volume else None,
                    "time": int(_mt5_epoch_to_utc(float(tick.time))) if tick.time else None,
                    "note": "Full market depth not available, showing current bid/ask"
                }
            }
            if spread:
                spread_metrics = _compute_spread_metrics(
                    out["data"].get("bid"),
                    out["data"].get("ask"),
                )
                if spread_metrics is not None:
                    out["data"].update(spread_metrics)
            _use_ctz = _use_client_tz()
            if tick.time and _use_ctz:
                out["data"]["time_display"] = _format_time_minimal_local(_mt5_epoch_to_utc(float(tick.time)))
            elif tick.time:
                out["data"]["time_display"] = _format_time_minimal(_mt5_epoch_to_utc(float(tick.time)))
            if not _use_ctz:
                out["timezone"] = "UTC"
            return out
    except Exception as e:
        return {"error": f"Error getting market depth: {str(e)}"}
