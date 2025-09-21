
from typing import Any, Dict

from ..utils.mt5 import _mt5_epoch_to_utc
from ..utils.utils import _format_time_minimal_util, _format_time_minimal_local_util, _use_client_tz_util
from .server import mcp, _auto_connect_wrapper
import MetaTrader5 as mt5

@mcp.tool()
@_auto_connect_wrapper
def market_depth_fetch(symbol: str) -> Dict[str, Any]:
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`.

    Parameters: symbol
    """
    try:
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            return {"error": f"Failed to select symbol {symbol}: {mt5.last_error()}"}
        
        # Try to get market depth first
        depth = mt5.market_book_get(symbol)
        
        if depth is not None and len(depth) > 0:
            # Process DOM levels
            buy_orders = []
            sell_orders = []
            
            for level in depth:
                order_data = {
                    "price": float(level["price"]),
                    "volume": float(level["volume"]),
                    "volume_real": float(level["volume_real"])
                }
                
                if int(level["type"]) == 0:  # Buy order
                    buy_orders.append(order_data)
                else:  # Sell order
                    sell_orders.append(order_data)
            
            return {
                "success": True,
                "symbol": symbol,
                "type": "full_depth",
                "data": {
                    "buy_orders": buy_orders,
                    "sell_orders": sell_orders
                }
            }
        else:
            # DOM not available, fall back to symbol tick info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}
            
            out = {
                "success": True,
                "symbol": symbol,
                "type": "tick_data",
                "data": {
                    "bid": float(tick.bid) if tick.bid else None,
                    "ask": float(tick.ask) if tick.ask else None,
                    "last": float(tick.last) if tick.last else None,
                    "volume": int(tick.volume) if tick.volume else None,
                    "time": int(_mt5_epoch_to_utc(float(tick.time))) if tick.time else None,
                    # spread removed from outputs by request
                    "note": "Full market depth not available, showing current bid/ask"
                }
            }
            _use_ctz = _use_client_tz_util()
            if tick.time and _use_ctz:
                out["data"]["time_display"] = _format_time_minimal_local_util(_mt5_epoch_to_utc(float(tick.time)))
            elif tick.time:
                out["data"]["time_display"] = _format_time_minimal_util(_mt5_epoch_to_utc(float(tick.time)))
            if not _use_ctz:
                out["timezone"] = "UTC"
            return out
    except Exception as e:
        return {"error": f"Error getting market depth: {str(e)}"}
