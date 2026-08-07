from types import SimpleNamespace

import pytest

from mtdata.utils.mt5 import account_currency_from_gateway
from mtdata.utils.quote import tick_epoch, tick_value


class _IndexedTick:
    def __getitem__(self, field: str):
        return {"time": 100.0, "time_msc": 100_250, "bid": 1.25}[field]


@pytest.mark.parametrize(
    ("tick", "expected"),
    [
        ({"time": 100.0, "time_msc": 100_250}, 100.25),
        (SimpleNamespace(time=100.0, time_msc=0), 100.0),
        (_IndexedTick(), 100.25),
        ({"time": float("nan"), "time_msc": None}, None),
    ],
)
def test_tick_epoch_normalizes_supported_tick_shapes(tick, expected) -> None:
    assert tick_epoch(tick) == expected


def test_tick_value_normalizes_supported_tick_shapes() -> None:
    assert tick_value({"bid": 1.1}, "bid") == 1.1
    assert tick_value(SimpleNamespace(bid=1.2), "bid") == 1.2
    assert tick_value(_IndexedTick(), "bid") == 1.25


@pytest.mark.parametrize(
    ("currency", "expected"),
    [
        (" USD ", "USD"),
        ("", None),
        ("<MagicMock name='currency'>", None),
        ("X" * 17, None),
        (object(), None),
    ],
)
def test_account_currency_from_gateway_rejects_non_currency_values(
    currency, expected
) -> None:
    gateway = SimpleNamespace(
        account_info=lambda: SimpleNamespace(currency=currency)
    )

    assert account_currency_from_gateway(gateway) == expected


def test_account_currency_from_gateway_handles_unavailable_account() -> None:
    def unavailable():
        raise RuntimeError("terminal disconnected")

    assert account_currency_from_gateway(SimpleNamespace(account_info=unavailable)) is None
