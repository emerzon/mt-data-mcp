from types import SimpleNamespace

import pytest

from mtdata.utils.mt5 import account_currency_from_gateway
from mtdata.utils.quote import compute_spread_metrics, tick_epoch, tick_value


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


def test_compute_spread_metrics_returns_raw_measurements() -> None:
    result = compute_spread_metrics(
        1.1,
        1.1002,
        point=0.00001,
        points_per_pip=10,
        tick_size=0.00001,
        tick_value_money=1.0,
        account_currency="USD",
    )

    assert result["spread_quality"] == "two_sided"
    assert result["spread_valid"] is True
    assert result["mid"] == pytest.approx(1.1001)
    assert result["spread"] == pytest.approx(0.0002)
    assert result["spread_points"] == pytest.approx(20.0)
    assert result["spread_pips"] == pytest.approx(2.0)
    assert result["spread_cost_per_lot"] == pytest.approx(20.0)
    assert result["pricing_basis"] == "per_1_lot_estimate"


@pytest.mark.parametrize(
    ("bid", "ask", "quality", "spread", "valid"),
    [
        (1.1, None, "one_sided", None, False),
        (1.2, 1.1, "inverted", None, False),
        (1.1, 1.1, "locked", 0.0, False),
    ],
)
def test_compute_spread_metrics_classifies_quote_boundaries(
    bid, ask, quality, spread, valid
) -> None:
    result = compute_spread_metrics(bid, ask, point=0.0001)

    assert result["spread_quality"] == quality
    assert result["spread"] == spread
    assert result["spread_valid"] is valid


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
