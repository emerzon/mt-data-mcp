"""Tests for trading safety rails policy."""

from types import SimpleNamespace

import pytest

from src.mtdata.core.trading.safety import (
    TradeSafetyPolicy,
    _estimate_order_risk_currency,
    _evaluate_safety_policy,
    assess_margin_stress,
)


def test_order_risk_uses_integer_tick_distance() -> None:
    risk, error = _estimate_order_risk_currency(
        symbol_info=SimpleNamespace(
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            trade_tick_value_loss=1.0,
        ),
        volume=0.3,
        entry_price=1.1,
        stop_loss=1.095,
        side="BUY",
    )

    assert error is None
    assert risk == 150.0


def test_breached_stop_overrun_is_counted_by_explicit_policy() -> None:
    risk, error = _estimate_order_risk_currency(
        symbol_info=SimpleNamespace(
            trade_tick_size=1.0,
            trade_tick_value=1.0,
            trade_tick_value_loss=2.0,
        ),
        volume=1.0,
        entry_price=100.0,
        stop_loss=110.0,
        side="BUY",
        wrong_side_policy="overrun",
    )

    assert error is None
    assert risk == 20.0


def test_stressed_margin_reports_triggering_thresholds() -> None:
    result = assess_margin_stress(
        SimpleNamespace(
            equity=100.0,
            margin=76.0,
            margin_free=24.0,
            margin_level=140.0,
        )
    )

    assert result["status"] == "stressed"
    assert result["reasons"] == [
        "margin_level_at_or_below_150_pct",
        "margin_utilization_at_or_above_75_pct",
        "free_margin_at_or_below_25_pct_of_equity",
    ]


@pytest.mark.parametrize("equity", [0.0, -100.0])
def test_nonpositive_equity_is_critical(equity: float) -> None:
    result = assess_margin_stress(
        SimpleNamespace(
            equity=equity,
            margin=0.0,
            margin_free=-25.0,
            margin_level=0.0,
        )
    )

    assert result["status"] == "critical"
    assert result["reasons"] == ["equity_at_or_below_zero"]


# ---------------------------------------------------------------------------
# No-policy pass-through
# ---------------------------------------------------------------------------

def test_no_policy_returns_none():
    """None policy always passes."""
    assert _evaluate_safety_policy(None, volume=100.0) is None


def test_empty_policy_returns_none():
    """Default policy (all fields disabled) always passes."""
    policy = TradeSafetyPolicy()
    assert _evaluate_safety_policy(policy, volume=100.0, side="BUY") is None


# ---------------------------------------------------------------------------
# max_volume
# ---------------------------------------------------------------------------

def test_max_volume_blocks():
    policy = TradeSafetyPolicy(max_volume=1.0)
    result = _evaluate_safety_policy(policy, volume=2.0)
    assert result is not None
    assert "exceeds" in result["violations"][0].lower()


def test_max_volume_allows():
    policy = TradeSafetyPolicy(max_volume=5.0)
    assert _evaluate_safety_policy(policy, volume=4.5) is None


def test_max_volume_boundary():
    policy = TradeSafetyPolicy(max_volume=1.0)
    assert _evaluate_safety_policy(policy, volume=1.0) is None


# ---------------------------------------------------------------------------
# require_stop_loss
# ---------------------------------------------------------------------------

def test_require_sl_blocks_missing():
    policy = TradeSafetyPolicy(require_stop_loss=True)
    result = _evaluate_safety_policy(policy, stop_loss=None)
    assert result is not None
    assert "stop-loss" in result["violations"][0].lower()


def test_require_sl_allows_present():
    policy = TradeSafetyPolicy(require_stop_loss=True)
    assert _evaluate_safety_policy(policy, stop_loss=1.1200) is None


def test_require_sl_blocks_nan():
    policy = TradeSafetyPolicy(require_stop_loss=True)
    result = _evaluate_safety_policy(policy, stop_loss=float("nan"))
    assert result is not None


def test_require_sl_blocks_zero():
    # stop_loss=0 (MT5 "no stop") is not a real stop-loss and must not satisfy
    # the require_stop_loss policy.
    policy = TradeSafetyPolicy(require_stop_loss=True)
    result = _evaluate_safety_policy(policy, stop_loss=0.0)
    assert result is not None
    assert "stop-loss" in result["violations"][0].lower()


def test_require_sl_blocks_negative():
    policy = TradeSafetyPolicy(require_stop_loss=True)
    result = _evaluate_safety_policy(policy, stop_loss=-1.0)
    assert result is not None


# ---------------------------------------------------------------------------
# max_deviation
# ---------------------------------------------------------------------------

def test_max_deviation_blocks():
    policy = TradeSafetyPolicy(max_deviation=10)
    result = _evaluate_safety_policy(policy, deviation=50)
    assert result is not None
    assert "deviation" in result["violations"][0].lower()


def test_max_deviation_allows():
    policy = TradeSafetyPolicy(max_deviation=20)
    assert _evaluate_safety_policy(policy, deviation=15) is None


# ---------------------------------------------------------------------------
# reduce_only
# ---------------------------------------------------------------------------

def test_reduce_only_allows_close_direction():
    policy = TradeSafetyPolicy(reduce_only=True)
    assert _evaluate_safety_policy(policy, side="SELL", existing_side="BUY") is None


def test_reduce_only_blocks_same_direction():
    policy = TradeSafetyPolicy(reduce_only=True)
    result = _evaluate_safety_policy(policy, side="BUY", existing_side="BUY")
    assert result is not None
    assert "reduce-only" in result["violations"][0].lower()


def test_reduce_only_blocks_no_position():
    policy = TradeSafetyPolicy(reduce_only=True)
    result = _evaluate_safety_policy(policy, side="BUY", existing_side=None)
    assert result is not None
    assert "no existing position" in result["violations"][0].lower()


def test_reduce_only_blocks_volume_that_would_flip_position():
    policy = TradeSafetyPolicy(reduce_only=True)
    result = _evaluate_safety_policy(
        policy,
        side="SELL",
        existing_side="BUY",
        volume=1.0,
        existing_net_volume=0.10,
    )
    assert result is not None
    assert any("exceeds" in v.lower() for v in result["violations"])


def test_reduce_only_allows_volume_within_net_position():
    policy = TradeSafetyPolicy(reduce_only=True)
    assert (
        _evaluate_safety_policy(
            policy,
            side="SELL",
            existing_side="BUY",
            volume=0.10,
            existing_net_volume=0.10,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Multiple violations
# ---------------------------------------------------------------------------

def test_multiple_violations():
    policy = TradeSafetyPolicy(max_volume=0.5, require_stop_loss=True, max_deviation=5)
    result = _evaluate_safety_policy(
        policy, volume=2.0, stop_loss=None, deviation=20,
    )
    assert result is not None
    assert len(result["violations"]) == 3


def test_error_key_present():
    policy = TradeSafetyPolicy(require_stop_loss=True)
    result = _evaluate_safety_policy(policy, stop_loss=None)
    assert "error" in result
    assert "safety policy" in result["error"].lower()
