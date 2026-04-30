"""Shared MT5 gateway helpers for core entrypoints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5_adapter
from .error_envelope import build_error_payload
from .execution_logging import (
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)

logger = logging.getLogger(__name__)


@dataclass
class MT5Gateway:
    """Thin adapter boundary around MT5 access plus connection enforcement."""

    adapter: Any
    ensure_connection_impl: Callable[[], None]

    def ensure_connection(self) -> None:
        started_at = time.perf_counter()
        adapter_type = type(self.adapter).__name__
        log_operation_start(
            logger,
            operation="mt5_ensure_connection",
            adapter_type=adapter_type,
        )
        try:
            self.ensure_connection_impl()
        except Exception as exc:
            log_operation_exception(
                logger,
                operation="mt5_ensure_connection",
                started_at=started_at,
                exc=exc,
                adapter_type=adapter_type,
            )
            raise
        log_operation_finish(
            logger,
            operation="mt5_ensure_connection",
            started_at=started_at,
            success=True,
            adapter_type=adapter_type,
        )

    def _adapter_attr(self, name: str) -> Any:
        return getattr(self.adapter, name)

    @property
    def ORDER_TIME_GTC(self) -> Any:
        return self._adapter_attr("ORDER_TIME_GTC")

    @property
    def ORDER_TIME_SPECIFIED(self) -> Any:
        return self._adapter_attr("ORDER_TIME_SPECIFIED")

    @property
    def ORDER_FILLING_FOK(self) -> Any:
        return self._adapter_attr("ORDER_FILLING_FOK")

    @property
    def ORDER_FILLING_IOC(self) -> Any:
        return self._adapter_attr("ORDER_FILLING_IOC")

    @property
    def ORDER_FILLING_RETURN(self) -> Any:
        return self._adapter_attr("ORDER_FILLING_RETURN")

    @property
    def ORDER_TYPE_BUY(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_BUY")

    @property
    def ORDER_TYPE_BUY_LIMIT(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_BUY_LIMIT")

    @property
    def ORDER_TYPE_BUY_STOP(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_BUY_STOP")

    @property
    def ORDER_TYPE_BUY_STOP_LIMIT(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_BUY_STOP_LIMIT")

    @property
    def ORDER_TYPE_SELL(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_SELL")

    @property
    def ORDER_TYPE_SELL_LIMIT(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_SELL_LIMIT")

    @property
    def ORDER_TYPE_SELL_STOP(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_SELL_STOP")

    @property
    def ORDER_TYPE_SELL_STOP_LIMIT(self) -> Any:
        return self._adapter_attr("ORDER_TYPE_SELL_STOP_LIMIT")

    @property
    def POSITION_TYPE_BUY(self) -> Any:
        return self._adapter_attr("POSITION_TYPE_BUY")

    @property
    def POSITION_TYPE_SELL(self) -> Any:
        return self._adapter_attr("POSITION_TYPE_SELL")

    @property
    def DEAL_ENTRY_IN(self) -> Any:
        return self._adapter_attr("DEAL_ENTRY_IN")

    @property
    def DEAL_ENTRY_INOUT(self) -> Any:
        return self._adapter_attr("DEAL_ENTRY_INOUT")

    @property
    def DEAL_ENTRY_OUT(self) -> Any:
        return self._adapter_attr("DEAL_ENTRY_OUT")

    @property
    def DEAL_ENTRY_OUT_BY(self) -> Any:
        return self._adapter_attr("DEAL_ENTRY_OUT_BY")

    @property
    def DEAL_REASON_CLIENT(self) -> Any:
        return self._adapter_attr("DEAL_REASON_CLIENT")

    @property
    def DEAL_REASON_SL(self) -> Any:
        return self._adapter_attr("DEAL_REASON_SL")

    @property
    def DEAL_REASON_TP(self) -> Any:
        return self._adapter_attr("DEAL_REASON_TP")

    @property
    def DEAL_TYPE_BUY(self) -> Any:
        return self._adapter_attr("DEAL_TYPE_BUY")

    @property
    def DEAL_TYPE_SELL(self) -> Any:
        return self._adapter_attr("DEAL_TYPE_SELL")

    @property
    def SYMBOL_FILLING_FOK(self) -> Any:
        return self._adapter_attr("SYMBOL_FILLING_FOK")

    @property
    def SYMBOL_FILLING_IOC(self) -> Any:
        return self._adapter_attr("SYMBOL_FILLING_IOC")

    @property
    def SYMBOL_FILLING_RETURN(self) -> Any:
        return self._adapter_attr("SYMBOL_FILLING_RETURN")

    @property
    def SYMBOL_TRADE_MODE_CLOSEONLY(self) -> Any:
        return self._adapter_attr("SYMBOL_TRADE_MODE_CLOSEONLY")

    @property
    def SYMBOL_TRADE_MODE_DISABLED(self) -> Any:
        return self._adapter_attr("SYMBOL_TRADE_MODE_DISABLED")

    @property
    def SYMBOL_TRADE_MODE_FULL(self) -> Any:
        return self._adapter_attr("SYMBOL_TRADE_MODE_FULL")

    @property
    def SYMBOL_TRADE_MODE_LONGONLY(self) -> Any:
        return self._adapter_attr("SYMBOL_TRADE_MODE_LONGONLY")

    @property
    def SYMBOL_TRADE_MODE_SHORTONLY(self) -> Any:
        return self._adapter_attr("SYMBOL_TRADE_MODE_SHORTONLY")

    @property
    def TRADE_ACTION_DEAL(self) -> Any:
        return self._adapter_attr("TRADE_ACTION_DEAL")

    @property
    def TRADE_ACTION_MODIFY(self) -> Any:
        return self._adapter_attr("TRADE_ACTION_MODIFY")

    @property
    def TRADE_ACTION_PENDING(self) -> Any:
        return self._adapter_attr("TRADE_ACTION_PENDING")

    @property
    def TRADE_ACTION_REMOVE(self) -> Any:
        return self._adapter_attr("TRADE_ACTION_REMOVE")

    @property
    def TRADE_ACTION_SLTP(self) -> Any:
        return self._adapter_attr("TRADE_ACTION_SLTP")

    @property
    def TRADE_RETCODE_DONE(self) -> Any:
        return self._adapter_attr("TRADE_RETCODE_DONE")

    @property
    def TRADE_RETCODE_PRICE_CHANGED(self) -> Any:
        return self._adapter_attr("TRADE_RETCODE_PRICE_CHANGED")

    def account_info(self) -> Any:
        return self.adapter.account_info()

    def copy_rates_from(self, symbol: str, timeframe: Any, dt_from: Any, count: int) -> Any:
        return self.adapter.copy_rates_from(symbol, timeframe, dt_from, count)

    def copy_rates_from_pos(self, symbol: str, timeframe: Any, start_pos: int, count: int) -> Any:
        return self.adapter.copy_rates_from_pos(symbol, timeframe, start_pos, count)

    def copy_rates_range(self, symbol: str, timeframe: Any, dt_from: Any, dt_to: Any) -> Any:
        return self.adapter.copy_rates_range(symbol, timeframe, dt_from, dt_to)

    def copy_ticks_from(self, symbol: str, dt_from: Any, count: int, flags: Any) -> Any:
        return self.adapter.copy_ticks_from(symbol, dt_from, count, flags)

    def copy_ticks_range(self, symbol: str, dt_from: Any, dt_to: Any, flags: Any) -> Any:
        return self.adapter.copy_ticks_range(symbol, dt_from, dt_to, flags)

    def history_deals_get(self, dt_from: Any, dt_to: Any, **kwargs: Any) -> Any:
        return self.adapter.history_deals_get(dt_from, dt_to, **kwargs)

    def history_orders_get(self, dt_from: Any, dt_to: Any, **kwargs: Any) -> Any:
        return self.adapter.history_orders_get(dt_from, dt_to, **kwargs)

    def last_error(self) -> Any:
        return self.adapter.last_error()

    def market_book_add(self, symbol: str) -> Any:
        return self.adapter.market_book_add(symbol)

    def market_book_get(self, symbol: str) -> Any:
        return self.adapter.market_book_get(symbol)

    def market_book_release(self, symbol: str) -> Any:
        return self.adapter.market_book_release(symbol)

    def order_send(self, request: Any) -> Any:
        return self.adapter.order_send(request)

    def orders_get(self, **kwargs: Any) -> Any:
        return self.adapter.orders_get(**kwargs)

    def positions_get(self, **kwargs: Any) -> Any:
        return self.adapter.positions_get(**kwargs)

    def retcode_name(self, code: Any) -> Any:
        return self.adapter.retcode_name(code)

    def symbol_info(self, symbol: str) -> Any:
        return self.adapter.symbol_info(symbol)

    def symbol_info_tick(self, symbol: str) -> Any:
        return self.adapter.symbol_info_tick(symbol)

    def symbol_select(self, symbol: str, visible: bool = True) -> Any:
        return self.adapter.symbol_select(symbol, visible)

    def symbols_get(self, *args: Any, **kwargs: Any) -> Any:
        return self.adapter.symbols_get(*args, **kwargs)

    def terminal_info(self) -> Any:
        return self.adapter.terminal_info()


def create_mt5_gateway(
    *,
    adapter: Any = mt5_adapter,
    ensure_connection_impl: Callable[[], None] = ensure_mt5_connection_or_raise,
) -> MT5Gateway:
    return MT5Gateway(
        adapter=adapter,
        ensure_connection_impl=ensure_connection_impl,
    )


def mt5_connection_error(
    gateway: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    try:
        mt5_gateway = gateway if gateway is not None else create_mt5_gateway()
        mt5_gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return build_error_payload(
            str(exc),
            code="mt5_connection_error",
            operation="mt5_ensure_connection",
        )
    return None
