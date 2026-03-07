"""Shared MT5 gateway helpers for core entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter


@dataclass
class MT5Gateway:
    """Thin adapter boundary around MT5 access plus connection enforcement."""

    adapter: Any
    ensure_connection_impl: Callable[[], None]

    def ensure_connection(self) -> None:
        self.ensure_connection_impl()

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        try:
            names.update(dir(self.adapter))
        except Exception:
            pass
        return sorted(names)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.adapter, name)


def create_mt5_gateway(
    *,
    adapter: Any = mt5_adapter,
    ensure_connection_impl: Callable[[], None] = ensure_mt5_connection_or_raise,
) -> MT5Gateway:
    return MT5Gateway(
        adapter=adapter,
        ensure_connection_impl=ensure_connection_impl,
    )


def get_default_mt5_gateway() -> MT5Gateway:
    return create_mt5_gateway()
