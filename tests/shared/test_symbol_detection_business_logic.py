from __future__ import annotations

from mtdata.core.regime.crypto import _is_probably_crypto_symbol as regime_is_crypto_symbol
from mtdata.forecast.barriers_shared import _is_crypto_symbol as barrier_is_crypto_symbol
from mtdata.patterns.common import _is_probably_crypto_symbol as pattern_is_crypto_symbol
from mtdata.shared.symbols import CRYPTO_SYMBOL_HINTS, is_probably_crypto_symbol


def test_shared_crypto_symbol_hints_include_extended_tokens() -> None:
    assert {"BNB", "TRX", "NEAR", "FIL"}.issubset(set(CRYPTO_SYMBOL_HINTS))


def test_crypto_symbol_detection_stays_consistent_across_modules() -> None:
    for symbol in ("BNBUSDT", "TRXUSD", "NEARUSD", "FILUSD"):
        assert is_probably_crypto_symbol(symbol) is True
        assert pattern_is_crypto_symbol(symbol) is True
        assert barrier_is_crypto_symbol(symbol) is True
        assert regime_is_crypto_symbol(symbol) is True

    for symbol in ("EURUSD", "", None):
        assert is_probably_crypto_symbol(symbol) is False
        assert pattern_is_crypto_symbol(symbol) is False
        assert barrier_is_crypto_symbol(symbol) is False
        assert regime_is_crypto_symbol(symbol) is False
