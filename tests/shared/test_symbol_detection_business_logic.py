from __future__ import annotations

from mtdata.shared.symbols import CRYPTO_SYMBOL_HINTS, is_probably_crypto_symbol


def test_shared_crypto_symbol_hints_include_extended_tokens() -> None:
        assert {"BNB", "TRX", "NEAR", "FIL"}.issubset(set(CRYPTO_SYMBOL_HINTS))


def test_crypto_symbol_detection_stays_consistent_across_modules() -> None:
    for symbol in ("BNBUSDT", "TRXUSD", "NEARUSD", "FILUSD"):
        assert is_probably_crypto_symbol(symbol) is True

    for symbol in ("EURUSD", "", None):
        assert is_probably_crypto_symbol(symbol) is False
