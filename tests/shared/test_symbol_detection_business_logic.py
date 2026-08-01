from __future__ import annotations

from mtdata.forecast.common import bars_per_year
from mtdata.shared.symbols import (
    CRYPTO_SYMBOL_HINTS,
    is_probably_crypto_symbol,
    is_probably_forex_symbol,
)


def test_shared_crypto_symbol_hints_include_extended_tokens() -> None:
        assert {"BNB", "TRX", "NEAR", "FIL"}.issubset(set(CRYPTO_SYMBOL_HINTS))


def test_crypto_symbol_detection_stays_consistent_across_modules() -> None:
    for symbol in ("BNBUSDT", "TRXUSD", "NEARUSD", "FILUSD"):
        assert is_probably_crypto_symbol(symbol) is True

    for symbol in (
        "EURUSD",
        "SOLV",
        "ATOM",
        "UNIT",
        "LINKEDIN",
        "",
        None,
    ):
        assert is_probably_crypto_symbol(symbol) is False


def test_forex_detection_covers_extended_codes_and_broker_prefixes() -> None:
    for symbol in (
        "USDSGD",
        "USDZAR.pro",
        "FX_EURUSD",
        "mEURUSD",
        "broker_EURNOK.a",
    ):
        assert is_probably_forex_symbol(symbol) is True

    for symbol in ("BTCUSD", "SOLV", "NASDAQ", "", None):
        assert is_probably_forex_symbol(symbol) is False

    assert bars_per_year("H1", "USDSGD") == 260.0 * 24.0
