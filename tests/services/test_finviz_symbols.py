import pytest

from mtdata.services.finviz.symbols import normalize_finviz_equity_symbol


@pytest.mark.parametrize(
    ("broker_symbol", "expected"),
    (
        ("AAPL.OTC", "AAPL"),
        ("AAPL_US", "AAPL"),
        ("AAPL-NYQ", "AAPL"),
        ("AAPL.NAS-24", "AAPL"),
        ("msft.o", "MSFT"),
        ("BRK.B", "BRK.B"),
        ("EURUSD", "EURUSD"),
        ("BTC/USD", "BTC/USD"),
    ),
)
def test_normalize_finviz_equity_symbol(broker_symbol: str, expected: str) -> None:
    assert normalize_finviz_equity_symbol(broker_symbol) == expected
