"""Tests for account-level risk gate."""

from types import SimpleNamespace

from src.mtdata.core.trading.safety import (
    AccountRiskLimits,
    _evaluate_account_risk_gate,
)


# ---------------------------------------------------------------------------
# No-limits pass-through
# ---------------------------------------------------------------------------

def test_no_limits_returns_none():
    assert _evaluate_account_risk_gate(None) is None


def test_empty_limits_returns_none():
    limits = AccountRiskLimits()
    assert _evaluate_account_risk_gate(limits) is None


# ---------------------------------------------------------------------------
# Margin level
# ---------------------------------------------------------------------------

def test_margin_level_blocks_low():
    limits = AccountRiskLimits(min_margin_level_pct=200.0)
    acct = SimpleNamespace(margin_level=150.0, profit=0.0)
    result = _evaluate_account_risk_gate(limits, account_info=acct)
    assert result is not None
    assert "margin level" in result["violations"][0].lower()


def test_margin_level_allows_sufficient():
    limits = AccountRiskLimits(min_margin_level_pct=200.0)
    acct = SimpleNamespace(margin_level=300.0, profit=0.0)
    assert _evaluate_account_risk_gate(limits, account_info=acct) is None


def test_margin_level_no_account_info_passes():
    limits = AccountRiskLimits(min_margin_level_pct=200.0)
    assert _evaluate_account_risk_gate(limits, account_info=None) is None


# ---------------------------------------------------------------------------
# Floating loss
# ---------------------------------------------------------------------------

def test_floating_loss_blocks_excessive():
    limits = AccountRiskLimits(max_floating_loss=500.0)
    acct = SimpleNamespace(margin_level=500.0, profit=-750.0)
    result = _evaluate_account_risk_gate(limits, account_info=acct)
    assert result is not None
    assert "floating loss" in result["violations"][0].lower()


def test_floating_loss_allows_within_limit():
    limits = AccountRiskLimits(max_floating_loss=500.0)
    acct = SimpleNamespace(margin_level=500.0, profit=-200.0)
    assert _evaluate_account_risk_gate(limits, account_info=acct) is None


def test_floating_loss_ignores_positive_profit():
    limits = AccountRiskLimits(max_floating_loss=500.0)
    acct = SimpleNamespace(margin_level=500.0, profit=10000.0)
    assert _evaluate_account_risk_gate(limits, account_info=acct) is None


# ---------------------------------------------------------------------------
# Total exposure
# ---------------------------------------------------------------------------

def test_exposure_blocks_over_limit():
    limits = AccountRiskLimits(max_total_exposure_lots=5.0)
    result = _evaluate_account_risk_gate(
        limits, existing_volume=4.0, new_volume=2.0,
    )
    assert result is not None
    assert "exposure" in result["violations"][0].lower()


def test_exposure_allows_within_limit():
    limits = AccountRiskLimits(max_total_exposure_lots=5.0)
    assert _evaluate_account_risk_gate(
        limits, existing_volume=2.0, new_volume=2.5,
    ) is None


# ---------------------------------------------------------------------------
# Multiple violations
# ---------------------------------------------------------------------------

def test_multiple_violations():
    limits = AccountRiskLimits(
        min_margin_level_pct=300.0,
        max_floating_loss=100.0,
        max_total_exposure_lots=1.0,
    )
    acct = SimpleNamespace(margin_level=100.0, profit=-500.0)
    result = _evaluate_account_risk_gate(
        limits, account_info=acct, existing_volume=2.0, new_volume=1.0,
    )
    assert result is not None
    assert len(result["violations"]) == 3
    assert "account risk gate" in result["error"].lower()
