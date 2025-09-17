"""Trading functions for MetaTrader integration."""

from typing import Optional, Union
from .server import mcp
from ..utils.mt5 import _auto_connect_wrapper


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
def trading_deals_history(from_date: Optional[str] = None, to_date: Optional[str] = None, symbol: Optional[str] = None) -> str:
    """Get historical deals as CSV. Date input in format: 'YYYY-MM-DD'."""
    import MetaTrader5 as mt5
    import pandas as pd
    from datetime import datetime

    @_auto_connect_wrapper
    def _get_deals():
        try:
            if from_date:
                from_dt = datetime.strptime(from_date, '%Y-%m-%d')
            else:
                from_dt = datetime(2020, 1, 1)

            if to_date:
                to_dt = datetime.strptime(to_date, '%Y-%m-%d')
            else:
                to_dt = datetime.now()

            if symbol:
                deals = mt5.history_deals_get(from_dt, to_dt, symbol=symbol)
            else:
                deals = mt5.history_deals_get(from_dt, to_dt)

            if deals is None or len(deals) == 0:
                return "No deals found"

            df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_deals()


@mcp.tool()
def trading_orders_active(from_date: Optional[str] = None, to_date: Optional[str] = None, symbol: Optional[str] = None) -> str:
    """Get historical orders as CSV. Date input in format: 'YYYY-MM-DD'"""
    import MetaTrader5 as mt5
    import pandas as pd
    from datetime import datetime

    @_auto_connect_wrapper
    def _get_orders():
        try:
            if from_date:
                from_dt = datetime.strptime(from_date, '%Y-%m-%d')
            else:
                from_dt = datetime(2020, 1, 1)

            if to_date:
                to_dt = datetime.strptime(to_date, '%Y-%m-%d')
            else:
                to_dt = datetime.now()

            if symbol:
                orders = mt5.history_orders_get(from_dt, to_dt, symbol=symbol)
            else:
                orders = mt5.history_orders_get(from_dt, to_dt)

            if orders is None or len(orders) == 0:
                return "No orders found"

            df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
            df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
            if 'time_done' in df.columns:
                df['time_done'] = pd.to_datetime(df['time_done'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_orders()


@mcp.tool()
def trading_positions_list() -> str:
    """Get all open positions."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_all_positions():
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return "No open positions"

            df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_all_positions()


@mcp.tool()
def trading_positions_by_symbol(symbol: str) -> str:
    """Get open positions for a specific symbol."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_positions_by_symbol():
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                return f"No open positions for {symbol}"

            df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_positions_by_symbol()


@mcp.tool()
def trading_positions_by_id(id: Union[int, str]) -> str:
    """Get open positions by ID."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_positions_by_id():
        try:
            ticket = int(id)
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                return f"No position found with ID {id}"

            df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_positions_by_id()


@mcp.tool()
def trading_pending_list() -> str:
    """Get all pending orders."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_all_pending_orders():
        try:
            orders = mt5.orders_get()
            if orders is None or len(orders) == 0:
                return "No pending orders"

            df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
            df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_all_pending_orders()


@mcp.tool()
def trading_pending_by_symbol(symbol: str) -> str:
    """Get pending orders for a specific symbol."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_pending_orders_by_symbol():
        try:
            orders = mt5.orders_get(symbol=symbol)
            if orders is None or len(orders) == 0:
                return f"No pending orders for {symbol}"

            df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
            df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_pending_orders_by_symbol()


@mcp.tool()
def trading_pending_by_id(id: Union[int, str]) -> str:
    """Get pending orders by id."""
    import MetaTrader5 as mt5
    import pandas as pd

    @_auto_connect_wrapper
    def _get_pending_orders_by_id():
        try:
            ticket = int(id)
            orders = mt5.orders_get(ticket=ticket)
            if orders is None or len(orders) == 0:
                return f"No pending order found with ID {id}"

            df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
            df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
            return df.to_csv(index=False)

        except Exception as e:
            return f"Error: {str(e)}"

    return _get_pending_orders_by_id()


@mcp.tool()
def trading_orders_place_market(symbol: str, volume: float, type: str) -> dict:
    """
    Place a market order. Parameters:
        symbol: Symbol name (e.g., 'EURUSD')
        volume: Lot size. (e.g. 1.5)
        type: Order type ('BUY' or 'SELL')
    """
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

            price = mt5.symbol_info_tick(symbol).ask if type.upper() == "BUY" else mt5.symbol_info_tick(symbol).bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if type.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "MCP order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                return {"error": "Failed to send order"}

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
            }

        except Exception as e:
            return {"error": str(e)}

    return _place_market_order()


@mcp.tool()
def trading_pending_place(symbol: str, volume: float, type: str, price: float, stop_loss: Optional[Union[int, float]] = 0, take_profit: Optional[Union[int, float]] = 0) -> dict:
    """
    Place a pending order. Parameters:
        symbol: Symbol name (e.g., 'EURUSD')
        volume: Lot size. (e.g. 1.5)
        type: Order type ('BUY', 'SELL').
        price: Pending order price.
        stop_loss (optional): Stop loss price.
        take_profit (optional): Take profit price.
    """
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

            current_price = mt5.symbol_info_tick(symbol)
            if current_price is None:
                return {"error": f"Failed to get current price for {symbol}"}

            # Determine order type based on current price and desired direction
            if type.upper() == "BUY":
                order_type = mt5.ORDER_TYPE_BUY_LIMIT if price < current_price.ask else mt5.ORDER_TYPE_BUY_STOP
            else:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT if price > current_price.bid else mt5.ORDER_TYPE_SELL_STOP

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss if stop_loss and stop_loss > 0 else 0.0,
                "tp": take_profit if take_profit and take_profit > 0 else 0.0,
                "deviation": 20,
                "magic": 234000,
                "comment": "MCP pending order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                return {"error": "Failed to send pending order"}

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
            }

        except Exception as e:
            return {"error": str(e)}

    return _place_pending_order()


@mcp.tool()
def trading_positions_modify(id: Union[int, str], stop_loss: Optional[Union[int, float]] = None, take_profit: Optional[Union[int, float]] = None) -> dict:
    """Modify an open position by ID."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _modify_position():
        try:
            ticket = int(id)
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                return {"error": f"Position {id} not found"}

            position = positions[0]

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": stop_loss if stop_loss is not None else position.sl,
                "tp": take_profit if take_profit is not None else position.tp,
                "magic": 234000,
                "comment": "MCP modify position",
            }

            result = mt5.order_send(request)
            if result is None:
                return {"error": "Failed to modify position"}

            return {
                "retcode": result.retcode,
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "request_id": result.request_id,
            }

        except Exception as e:
            return {"error": str(e)}

    return _modify_position()


@mcp.tool()
def trading_pending_modify(id: Union[int, str], price: Optional[Union[int, float]] = None, stop_loss: Optional[Union[int, float]] = None, take_profit: Optional[Union[int, float]] = None) -> dict:
    """Modify a pending order by ID."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _modify_pending_order():
        try:
            ticket = int(id)
            orders = mt5.orders_get(ticket=ticket)
            if orders is None or len(orders) == 0:
                return {"error": f"Pending order {id} not found"}

            order = orders[0]

            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket,
                "price": price if price is not None else order.price_open,
                "sl": stop_loss if stop_loss is not None else order.sl,
                "tp": take_profit if take_profit is not None else order.tp,
                "magic": 234000,
                "comment": "MCP modify pending order",
            }

            result = mt5.order_send(request)
            if result is None:
                return {"error": "Failed to modify pending order"}

            return {
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
def trading_positions_close(id: Union[int, str]) -> dict:
    """Close an open position by ID."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _close_position():
        try:
            ticket = int(id)
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                return {"error": f"Position {id} not found"}

            position = positions[0]

            # Determine close price and type
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {position.symbol}"}

            close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "MCP close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                return {"error": "Failed to close position"}

            return {
                "retcode": result.retcode,
                "deal": result.deal,
                "order": result.order,
                "volume": result.volume,
                "price": result.price,
                "comment": result.comment,
                "request_id": result.request_id,
            }

        except Exception as e:
            return {"error": str(e)}

    return _close_position()


@mcp.tool()
def trading_pending_cancel(id: Union[int, str]) -> dict:
    """Cancel a pending order by ID."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _cancel_pending_order():
        try:
            ticket = int(id)
            orders = mt5.orders_get(ticket=ticket)
            if orders is None or len(orders) == 0:
                return {"error": f"Pending order {id} not found"}

            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
                "magic": 234000,
                "comment": "MCP cancel pending order",
            }

            result = mt5.order_send(request)
            if result is None:
                return {"error": "Failed to cancel pending order"}

            return {
                "retcode": result.retcode,
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "request_id": result.request_id,
            }

        except Exception as e:
            return {"error": str(e)}

    return _cancel_pending_order()


@mcp.tool()
def trading_positions_close_all() -> dict:
    """Close all open positions."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _close_all_positions():
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return {"message": "No open positions to close"}

            results = []
            for position in positions:
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
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "MCP close all positions",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": position.ticket, "error": "Failed to send close order"})
                else:
                    results.append({
                        "ticket": position.ticket,
                        "retcode": result.retcode,
                        "deal": result.deal,
                        "order": result.order,
                    })

            return {"closed_positions": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _close_all_positions()


@mcp.tool()
def trading_positions_close_symbol(symbol: str) -> dict:
    """Close all open positions for a specific symbol."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _close_all_positions_by_symbol():
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                return {"message": f"No open positions for {symbol} to close"}

            results = []
            for position in positions:
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
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"MCP close {symbol} positions",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": position.ticket, "error": "Failed to send close order"})
                else:
                    results.append({
                        "ticket": position.ticket,
                        "retcode": result.retcode,
                        "deal": result.deal,
                        "order": result.order,
                    })

            return {"symbol": symbol, "closed_positions": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _close_all_positions_by_symbol()


@mcp.tool()
def trading_positions_close_profitable() -> dict:
    """Close all profitable positions."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _close_all_profitable_positions():
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return {"message": "No open positions to close"}

            profitable_positions = [pos for pos in positions if pos.profit > 0]
            if len(profitable_positions) == 0:
                return {"message": "No profitable positions to close"}

            results = []
            for position in profitable_positions:
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
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "MCP close profitable positions",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": position.ticket, "error": "Failed to send close order"})
                else:
                    results.append({
                        "ticket": position.ticket,
                        "profit": position.profit,
                        "retcode": result.retcode,
                        "deal": result.deal,
                        "order": result.order,
                    })

            return {"closed_positions": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _close_all_profitable_positions()


@mcp.tool()
def trading_positions_close_losing() -> dict:
    """Close all losing positions."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _close_all_losing_positions():
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return {"message": "No open positions to close"}

            losing_positions = [pos for pos in positions if pos.profit < 0]
            if len(losing_positions) == 0:
                return {"message": "No losing positions to close"}

            results = []
            for position in losing_positions:
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
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "MCP close losing positions",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": position.ticket, "error": "Failed to send close order"})
                else:
                    results.append({
                        "ticket": position.ticket,
                        "profit": position.profit,
                        "retcode": result.retcode,
                        "deal": result.deal,
                        "order": result.order,
                    })

            return {"closed_positions": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _close_all_losing_positions()


@mcp.tool()
def trading_pending_cancel_all() -> dict:
    """Cancel all pending orders."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _cancel_all_pending_orders():
        try:
            orders = mt5.orders_get()
            if orders is None or len(orders) == 0:
                return {"message": "No pending orders to cancel"}

            results = []
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "magic": 234000,
                    "comment": "MCP cancel all pending orders",
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": order.ticket, "error": "Failed to send cancel order"})
                else:
                    results.append({
                        "ticket": order.ticket,
                        "retcode": result.retcode,
                        "order": result.order,
                    })

            return {"cancelled_orders": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _cancel_all_pending_orders()


@mcp.tool()
def trading_pending_cancel_symbol(symbol: str) -> dict:
    """Cancel all pending orders for a specific symbol."""
    import MetaTrader5 as mt5

    @_auto_connect_wrapper
    def _cancel_pending_orders_by_symbol():
        try:
            orders = mt5.orders_get(symbol=symbol)
            if orders is None or len(orders) == 0:
                return {"message": f"No pending orders for {symbol} to cancel"}

            results = []
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "magic": 234000,
                    "comment": f"MCP cancel {symbol} orders",
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": order.ticket, "error": "Failed to send cancel order"})
                else:
                    results.append({
                        "ticket": order.ticket,
                        "retcode": result.retcode,
                        "order": result.order,
                    })

            return {"symbol": symbol, "cancelled_orders": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _cancel_pending_orders_by_symbol()
