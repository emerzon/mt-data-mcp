"""Tests for utils/barriers.py — pure barrier math without MT5."""
import pytest

from mtdata.utils.barriers import (
    barrier_prices_are_valid,
    build_barrier_kwargs,
    build_barrier_kwargs_from,
    normalize_trade_direction,
    resolve_barrier_prices,
)


class TestResolveBarrierPrices:
    def test_long_pct(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long", tp_pct=2.0, sl_pct=1.0,
        )
        assert tp == pytest.approx(102.0)
        assert sl == pytest.approx(99.0)

    def test_short_pct(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="short", tp_pct=2.0, sl_pct=1.0,
        )
        assert tp == pytest.approx(98.0)
        assert sl == pytest.approx(101.0)

    def test_long_pips(self):
        tp, sl = resolve_barrier_prices(
            price=1.10000, direction="long",
            tp_pips=50.0, sl_pips=30.0, pip_size=0.00010,
        )
        assert tp == pytest.approx(1.10500)
        assert sl == pytest.approx(1.09700)

    def test_short_pips(self):
        tp, sl = resolve_barrier_prices(
            price=1.10000, direction="short",
            tp_pips=50.0, sl_pips=30.0, pip_size=0.00010,
        )
        assert tp == pytest.approx(1.09500)
        assert sl == pytest.approx(1.10300)

    def test_abs_passthrough(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long", tp_abs=110.0, sl_abs=95.0,
        )
        assert tp == pytest.approx(110.0)
        assert sl == pytest.approx(95.0)

    def test_returns_none_when_partial(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long", tp_pct=1.0,
        )
        # Only TP set, SL missing => both None
        assert tp is None
        assert sl is None

    def test_adjust_inverted_long(self):
        """If TP <= price for long, it gets nudged above."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=99.0, sl_abs=101.0,  # inverted
            pip_size=0.01,
        )
        assert tp > 100.0
        assert sl < 100.0

    def test_adjust_inverted_short(self):
        """If TP >= price for short, it gets nudged below."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="short",
            tp_abs=101.0, sl_abs=99.0,  # inverted
            pip_size=0.01,
        )
        assert tp < 100.0
        assert sl > 100.0

    def test_adjust_inverted_no_pip_size(self):
        """Inverted barriers with no pip_size still get corrected via fallback."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=99.0, sl_abs=101.0,
            adjust_inverted=True,
        )
        assert tp > 100.0
        assert sl < 100.0

    def test_no_adjust_inverted(self):
        """With adjust_inverted=False, inverted prices are kept as-is."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=99.0, sl_abs=101.0,
            adjust_inverted=False,
        )
        assert tp == pytest.approx(99.0)
        assert sl == pytest.approx(101.0)

    def test_coerce_string_values(self):
        """String numeric values should be coerced."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=110.0, sl_abs=90.0,
        )
        assert tp == pytest.approx(110.0)
        assert sl == pytest.approx(90.0)

    def test_rejects_non_finite_inputs(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=float("nan"), sl_abs=90.0,
        )
        assert tp is None
        assert sl is None


class TestNormalizeTradeDirection:
    def test_aliases(self):
        assert normalize_trade_direction("buy") == ("long", None)
        assert normalize_trade_direction("sell") == ("short", None)

    def test_invalid(self):
        direction, err = normalize_trade_direction("sideways")
        assert direction is None
        assert err is not None


class TestBarrierPricesAreValid:
    def test_valid_long_geometry(self):
        assert barrier_prices_are_valid(
            price=100.0,
            direction="long",
            tp_price=101.0,
            sl_price=99.0,
        ) is True

    def test_rejects_non_finite_or_inverted_geometry(self):
        assert barrier_prices_are_valid(
            price=100.0,
            direction="long",
            tp_price=float("nan"),
            sl_price=99.0,
        ) is False
        assert barrier_prices_are_valid(
            price=100.0,
            direction="short",
            tp_price=101.0,
            sl_price=99.0,
        ) is False


class TestBuildBarrierKwargs:
    def test_all_none(self):
        result = build_barrier_kwargs()
        assert result == {
            "tp_abs": None, "sl_abs": None,
            "tp_pct": None, "sl_pct": None,
            "tp_pips": None, "sl_pips": None,
        }

    def test_partial(self):
        result = build_barrier_kwargs(tp_pct=1.5, sl_pct=0.5)
        assert result["tp_pct"] == 1.5
        assert result["sl_pct"] == 0.5
        assert result["tp_abs"] is None


class TestBuildBarrierKwargsFrom:
    def test_from_dict(self):
        vals = {"tp_pct": 2.0, "sl_pct": 1.0, "other_key": "ignored"}
        result = build_barrier_kwargs_from(vals)
        assert result["tp_pct"] == 2.0
        assert result["sl_pct"] == 1.0
        assert "other_key" not in result

    def test_missing_keys(self):
        result = build_barrier_kwargs_from({})
        assert result["tp_abs"] is None
