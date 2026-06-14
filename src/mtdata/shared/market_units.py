"""Shared market quote unit helpers."""

from typing import Optional

_FOREX_CURRENCY_CODES = frozenset(
    {
        "AUD",
        "CAD",
        "CHF",
        "CNH",
        "CNY",
        "EUR",
        "GBP",
        "HKD",
        "JPY",
        "MXN",
        "NOK",
        "NZD",
        "SEK",
        "SGD",
        "USD",
        "ZAR",
    }
)


def forex_points_per_pip(
    symbol: str,
    *,
    path: str = "",
    point: float,
    digits: int,
) -> Optional[float]:
    """Return broker points per pip when the symbol is identifiable as FX."""
    name_letters = "".join(char for char in str(symbol).upper() if "A" <= char <= "Z")
    pair_prefix = name_letters[:6]
    is_currency_pair = (
        len(pair_prefix) == 6
        and pair_prefix[:3] in _FOREX_CURRENCY_CODES
        and pair_prefix[3:] in _FOREX_CURRENCY_CODES
    )
    path_folded = str(path or "").casefold()
    if (
        not is_currency_pair
        and "forex" not in path_folded
        and "\\fx" not in path_folded
        and "/fx" not in path_folded
    ):
        return None
    if digits in {3, 5}:
        return 10.0
    if digits in {2, 4}:
        return 1.0
    if point in {0.00001, 0.001}:
        return 10.0
    if point in {0.0001, 0.01}:
        return 1.0
    return None
