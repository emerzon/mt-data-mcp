"""Trading-specific MT5 gateway helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .mt5_gateway import MT5Gateway


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
