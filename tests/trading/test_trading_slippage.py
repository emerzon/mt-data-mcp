"""Tests for slippage-to-deviation conversion."""

from types import SimpleNamespace

import pytest

from src.mtdata.core.trading.validation import _resolve_slippage_to_deviation


# ---------------------------------------------------------------------------
# Default fallback
# ---------------------------------------------------------------------------

def test_no_inputs_returns_default():
    dev, meta, err = _resolve_slippage_to_deviation()
    assert err is None
    assert dev == 20
    assert meta["source"] == "default"


# ---------------------------------------------------------------------------
# Explicit deviation precedence
# ---------------------------------------------------------------------------

def test_explicit_deviation_wins():
    dev, meta, err = _resolve_slippage_to_deviation(deviation=10)
    assert err is None
    assert dev == 10
    assert meta["source"] == "deviation"


def test_explicit_deviation_over_pips():
    """deviation takes precedence even when slippage_pips is also provided."""
    dev, meta, err = _resolve_slippage_to_deviation(
        deviation=5, slippage_pips=3.0, symbol_info=SimpleNamespace(point=0.00001, digits=5),
    )
    assert dev == 5
    assert meta["source"] == "deviation"


def test_invalid_deviation():
    dev, meta, err = _resolve_slippage_to_deviation(deviation="abc")
    assert err is not None
    assert "numeric" in err.lower()


# ---------------------------------------------------------------------------
# slippage_pips conversion
# ---------------------------------------------------------------------------

def test_pips_5digit_broker():
    """5-digit broker: 1 pip = 10 points."""
    sym = SimpleNamespace(point=0.00001, digits=5)
    dev, meta, err = _resolve_slippage_to_deviation(slippage_pips=2.0, symbol_info=sym)
    assert err is None
    assert dev == 20  # 2 pips * 10
    assert meta["source"] == "slippage_pips"
    assert meta["points_per_pip"] == 10


def test_pips_4digit_broker():
    """4-digit broker: 1 pip = 1 point."""
    sym = SimpleNamespace(point=0.0001, digits=4)
    dev, meta, err = _resolve_slippage_to_deviation(slippage_pips=3.0, symbol_info=sym)
    assert err is None
    assert dev == 3  # 3 pips * 1
    assert meta["points_per_pip"] == 1


def test_pips_3digit_broker():
    """3-digit broker (JPY pairs): 1 pip = 10 points."""
    sym = SimpleNamespace(point=0.001, digits=3)
    dev, meta, err = _resolve_slippage_to_deviation(slippage_pips=1.5, symbol_info=sym)
    assert err is None
    assert dev == 15  # 1.5 pips * 10


def test_pips_2digit_broker():
    """2-digit (gold, indices): 1 pip = 1 point."""
    sym = SimpleNamespace(point=0.01, digits=2)
    dev, meta, err = _resolve_slippage_to_deviation(slippage_pips=5.0, symbol_info=sym)
    assert err is None
    assert dev == 5


def test_pips_zero_rejects():
    dev, meta, err = _resolve_slippage_to_deviation(slippage_pips=2.0, symbol_info=None)
    assert err is not None
    assert "unavailable" in err.lower()


def test_pips_negative_rejects():
    sym = SimpleNamespace(point=0.00001, digits=5)
    dev, meta, err = _resolve_slippage_to_deviation(slippage_pips=-1.0, symbol_info=sym)
    assert err is not None
    assert ">= 0" in err


def test_pips_nan_rejects():
    sym = SimpleNamespace(point=0.00001, digits=5)
    dev, meta, err = _resolve_slippage_to_deviation(slippage_pips=float("nan"), symbol_info=sym)
    assert err is not None
