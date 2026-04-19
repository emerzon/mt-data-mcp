"""Tests for risk-based position sizing."""

import math

import pytest

from src.mtdata.core.trading.sizing import compute_risk_based_volume


def _base_params(**overrides):
    """Standard EURUSD 5-digit broker parameters."""
    defaults = dict(
        equity=10000.0,
        risk_pct=1.0,
        entry_price=1.10000,
        stop_loss_price=1.09500,
        direction="long",
        tick_value=1.0,
        tick_size=0.00001,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Basic sizing
# ---------------------------------------------------------------------------

def test_basic_long_sizing():
    vol, meta = compute_risk_based_volume(**_base_params())
    assert vol is not None
    assert vol > 0
    assert meta["suggested_volume"] == vol
    assert "error" not in meta
    assert meta["requested_risk_pct"] == 1.0
    assert meta["risk_over_target"] is False
    assert meta["risk_compliance"] == "within_requested_risk"


def test_basic_short_sizing():
    vol, meta = compute_risk_based_volume(**_base_params(
        direction="short",
        entry_price=1.10000,
        stop_loss_price=1.10500,
    ))
    assert vol is not None
    assert vol > 0


def test_risk_amount_scales_with_pct():
    vol1, _ = compute_risk_based_volume(**_base_params(risk_pct=1.0))
    vol2, _ = compute_risk_based_volume(**_base_params(risk_pct=2.0))
    assert vol2 > vol1


def test_uses_loss_tick_value_when_available():
    vol, meta = compute_risk_based_volume(**_base_params(
        equity=1000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss_price=90.0,
        tick_value=1.0,
        tick_value_loss=2.0,
        tick_size=1.0,
        volume_min=0.1,
        volume_max=10.0,
        volume_step=0.1,
    ))
    assert vol == 0.5
    assert meta["risk_tick_value"] == 2.0
    assert meta["actual_risk"] == 10.0


# ---------------------------------------------------------------------------
# Volume clamping
# ---------------------------------------------------------------------------

def test_clamp_to_min_volume():
    vol, meta = compute_risk_based_volume(**_base_params(
        equity=1000.0,
        risk_pct=0.1,
        entry_price=100.0,
        stop_loss_price=80.0,
        tick_value=1.0,
        tick_size=1.0,
        volume_min=0.1,
        volume_max=10.0,
        volume_step=0.1,
    ))
    assert vol is not None
    assert vol == 0.1
    assert meta["volume_rounding"] == "clamped_to_min_volume"
    assert meta["risk_over_target"] is True
    assert meta["risk_compliance"] == "exceeds_requested_risk"
    assert meta["risk_over_target_reason"] == "min_volume_constraint"
    assert meta["risk_overshoot_pct"] > 0.0
    assert meta["risk_overshoot_currency"] > 0.0
    assert any("minimum" in n.lower() for n in meta["notes"])
    assert any("exceeds the requested level" in n.lower() for n in meta["notes"])


def test_clamp_to_max_volume():
    vol, meta = compute_risk_based_volume(**_base_params(
        equity=10_000_000.0,
        risk_pct=50.0,
        volume_max=10.0,
    ))
    assert vol is not None
    assert vol <= 10.0
    assert meta["volume_rounding"] == "clamped_to_max_volume"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_wrong_sl_side_long():
    """SL above entry for a long → negative distance → error."""
    vol, meta = compute_risk_based_volume(**_base_params(
        stop_loss_price=1.11000,  # Above entry for long
    ))
    assert vol is None
    assert "error" in meta


def test_wrong_sl_side_short():
    """SL below entry for a short → negative distance → error."""
    vol, meta = compute_risk_based_volume(**_base_params(
        direction="short",
        stop_loss_price=1.09000,  # Below entry for short
    ))
    assert vol is None
    assert "error" in meta


def test_zero_equity_rejects():
    vol, meta = compute_risk_based_volume(**_base_params(equity=0))
    assert vol is None
    assert "error" in meta


def test_nan_entry_rejects():
    vol, meta = compute_risk_based_volume(**_base_params(entry_price=float("nan")))
    assert vol is None
    assert "error" in meta


def test_zero_tick_value_rejects():
    vol, meta = compute_risk_based_volume(**_base_params(tick_value=0))
    assert vol is None
    assert "error" in meta


# ---------------------------------------------------------------------------
# Volume step rounding
# ---------------------------------------------------------------------------

def test_volume_step_rounding():
    vol, meta = compute_risk_based_volume(**_base_params(volume_step=0.1))
    assert vol is not None
    # Volume should be a multiple of 0.1
    remainder = round(vol % 0.1, 10)
    assert remainder == 0.0 or remainder == 0.1


def test_exact_step_volume_is_not_rounded_down_by_float_artifact():
    vol, meta = compute_risk_based_volume(**_base_params(
        equity=1000.0,
        risk_pct=0.3,
        entry_price=100.0,
        stop_loss_price=90.0,
        tick_value=1.0,
        tick_size=1.0,
        volume_min=0.1,
        volume_max=10.0,
        volume_step=0.1,
    ))
    assert vol == 0.3
    assert meta["suggested_volume"] == 0.3
    assert meta["raw_volume"] == 0.3
    assert meta["risk_over_target"] is False
