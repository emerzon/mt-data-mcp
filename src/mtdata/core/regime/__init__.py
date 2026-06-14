"""Public regime detection package."""

from . import methods
from .api import MT5ConnectionError, ensure_mt5_connection_or_raise, regime_detect

__all__ = [
    "MT5ConnectionError",
    "ensure_mt5_connection_or_raise",
    "methods",
    "regime_detect",
]
