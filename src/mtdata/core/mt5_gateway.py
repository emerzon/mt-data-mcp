"""Shared MT5 gateway helpers for core entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Dict, Optional

from .execution_logging import (
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5_adapter

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


def get_mt5_gateway(
    *,
    adapter: Any | None = None,
    ensure_connection_impl: Callable[[], None] | None = None,
) -> MT5Gateway:
    return create_mt5_gateway(
        adapter=mt5_adapter if adapter is None else adapter,
        ensure_connection_impl=(
            ensure_mt5_connection_or_raise
            if ensure_connection_impl is None
            else ensure_connection_impl
        ),
    )


def get_default_mt5_gateway() -> MT5Gateway:
    return get_mt5_gateway()


def get_web_api_mt5_gateway(
    *,
    adapter: Any = None,
    ensure_connection_impl: Callable[[], None] | None = None,
) -> MT5Gateway:
    """Build the MT5 gateway used by the web API transport."""
    return get_mt5_gateway(
        adapter=adapter,
        ensure_connection_impl=ensure_connection_impl,
    )


def mt5_connection_error(
    gateway: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    try:
        mt5_gateway = gateway if gateway is not None else get_default_mt5_gateway()
        mt5_gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return None
