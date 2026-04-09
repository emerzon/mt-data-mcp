"""Trading-specific MT5 gateway helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..mt5_gateway import MT5Gateway
from ...utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5_adapter


class MT5TradingGateway(MT5Gateway):
    """Small adapter boundary for trading execution helpers."""

    def __init__(
        self,
        *,
        adapter: Any,
        ensure_connection_impl: Callable[[], None],
        build_trade_preflight_impl: Optional[Callable[..., Dict[str, Any]]] = None,
        retcode_name_impl: Optional[Callable[[Any, Any], Optional[str]]] = None,
    ) -> None:
        super().__init__(
            adapter=adapter,
            ensure_connection_impl=ensure_connection_impl,
        )
        self.build_trade_preflight_impl = build_trade_preflight_impl
        self.retcode_name_impl = retcode_name_impl

    def build_trade_preflight(
        self,
        *,
        account_info: Any = None,
        terminal_info: Any = None,
    ) -> Dict[str, Any]:
        if self.build_trade_preflight_impl is None:
            raise RuntimeError("build_trade_preflight is not configured for this gateway")
        return self.build_trade_preflight_impl(
            self.adapter,
            account_info=account_info,
            terminal_info=terminal_info,
        )

    def retcode_name(self, retcode: Any) -> Optional[str]:
        if self.retcode_name_impl is None:
            return None
        return self.retcode_name_impl(self.adapter, retcode)


def create_trading_gateway(
    gateway: Optional["MT5TradingGateway"] = None,
    *,
    include_trade_preflight: bool = False,
    include_retcode_name: bool = False,
    adapter: Any = mt5_adapter,
    ensure_connection_impl: Optional[Callable[[], None]] = None,
) -> "MT5TradingGateway":
    if ensure_connection_impl is None:
        ensure_connection_impl = ensure_mt5_connection_or_raise

    if gateway is not None:
        return gateway

    gateway_kwargs: Dict[str, Any] = {}
    if include_trade_preflight or include_retcode_name:
        from .common import _build_trade_preflight, _retcode_name

        if include_trade_preflight:
            gateway_kwargs["build_trade_preflight_impl"] = _build_trade_preflight
        if include_retcode_name:
            gateway_kwargs["retcode_name_impl"] = _retcode_name

    return MT5TradingGateway(
        adapter=adapter,
        ensure_connection_impl=ensure_connection_impl,
        **gateway_kwargs,
    )


def trading_connection_error(
    gateway: Optional["MT5TradingGateway"] = None,
) -> Optional[Dict[str, Any]]:
    try:
        create_trading_gateway(gateway).ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return None
