from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

from mtdata.core.trading import trade_risk_analyze


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _make_symbol_info(*, volume_min: float = 0.1, volume_step: float = 0.1, volume_max: float = 10.0):
    return SimpleNamespace(
        trade_contract_size=1.0,
        point=1.0,
        trade_tick_value=1.0,
        trade_tick_size=1.0,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
    )


def test_trade_risk_analyze_rounds_down_to_step_to_avoid_overshoot() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    raw = _unwrap(trade_risk_analyze)
    with patch("mtdata.core.trading_risk._auto_connect_wrapper", lambda f: f):
        out = raw(
            symbol="EURUSD",
            desired_risk_pct=1.0,
            proposed_entry=100.0,
            proposed_sl=92.06,
        )

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    sizing = out["position_sizing"]
    assert sizing["suggested_volume"] == 1.2
    assert sizing["volume_rounding"] == "rounded_down_to_step"
    assert sizing["risk_over_target"] is False
    assert sizing["risk_compliance"] == "within_requested_risk"
    assert sizing["risk_overshoot_pct"] == 0.0
    assert sizing["risk_pct"] <= 1.0
    assert any("rounded down" in note.lower() for note in sizing["sizing_notes"])


def test_trade_risk_analyze_warns_when_min_volume_forces_overshoot() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(volume_min=0.1, volume_step=0.1, volume_max=10.0)

    raw = _unwrap(trade_risk_analyze)
    with patch("mtdata.core.trading_risk._auto_connect_wrapper", lambda f: f):
        out = raw(
            symbol="EURUSD",
            desired_risk_pct=0.1,
            proposed_entry=100.0,
            proposed_sl=80.0,
        )

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    sizing = out["position_sizing"]
    assert sizing["suggested_volume"] == 0.1
    assert sizing["volume_rounding"] == "clamped_to_min_volume"
    assert sizing["risk_over_target"] is True
    assert sizing["risk_compliance"] == "exceeds_requested_risk"
    assert sizing["risk_overshoot_pct"] > 0.0
    assert sizing["risk_over_target_reason"] == "min_volume_constraint"
    assert "position_sizing_warning" in out
    assert "risk_alert" in out
    assert out["risk_alert"]["code"] == "risk_overshoot_after_volume_constraints"
    assert any("minimum trade volume" in note.lower() for note in sizing["sizing_notes"])
