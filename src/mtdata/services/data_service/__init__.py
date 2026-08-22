"""MT5 history gateway for candles and ticks."""

from . import errors as _errors
from .candles import fetch_candles
from .ticks import fetch_ticks

__all__ = [
    "fetch_candles",
    "fetch_ticks",
]
