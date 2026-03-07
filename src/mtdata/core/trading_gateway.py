"""Trading-specific MT5 gateway helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class MT5TradingGateway:
    """Small adapter boundary for trading execution helpers."""

    adapter: Any
    ensure_connection_impl: Callable[[], None]
    build_trade_preflight_impl: Optional[Callable[..., Dict[str, Any]]] = None
    retcode_name_impl: Optional[Callable[[Any, Any], Optional[str]]] = None

    def ensure_connection(self) -> None:
        self.ensure_connection_impl()

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

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        try:
            names.update(dir(self.adapter))
        except Exception:
            pass
        return sorted(names)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.adapter, name)
