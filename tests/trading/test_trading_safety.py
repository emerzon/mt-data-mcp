"""Tests for trading safety rails policy."""

import pytest

from src.mtdata.core.trading.safety import (
    TradeSafetyPolicy,
    _evaluate_safety_policy,
)


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
