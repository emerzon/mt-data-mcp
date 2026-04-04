"""Crypto detection utilities for regime detection."""
from typing import Any

from ...shared.symbols import CRYPTO_SYMBOL_HINTS as _CRYPTO_SYMBOL_HINTS
from ...shared.symbols import is_probably_crypto_symbol


def _is_probably_crypto_symbol(symbol: Any) -> bool:
    """Detect if symbol is likely a cryptocurrency based on common tickers."""
    return is_probably_crypto_symbol(symbol)


__all__ = ["_is_probably_crypto_symbol", "_CRYPTO_SYMBOL_HINTS"]
