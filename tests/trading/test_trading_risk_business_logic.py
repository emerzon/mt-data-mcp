from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

from mtdata.core import trading_risk as core_trading_risk
from mtdata.core.trading import trade_risk_analyze as _trade_risk_analyze_tool
from mtdata.core.trading_requests import TradeRiskAnalyzeRequest
from mtdata.core.trading_use_cases import run_trade_risk_analyze
from mtdata.utils.mt5 import MT5ConnectionError


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def trade_risk_analyze(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", True))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeRiskAnalyzeRequest(**kwargs)
    with patch("mtdata.core.trading_risk.ensure_mt5_connection_or_raise", return_value=None):
        return _trade_risk_analyze_tool(request=request, __cli_raw=raw_output)


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

    out = trade_risk_analyze(
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

    out = trade_risk_analyze(
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


def test_trade_risk_analyze_accepts_explicit_short_direction() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    out = trade_risk_analyze(
        symbol="EURUSD",
        direction="short",
        desired_risk_pct=1.0,
        proposed_entry=100.0,
        proposed_sl=108.0,
        proposed_tp=92.0,
    )

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    sizing = out["position_sizing"]
    assert sizing["direction"] == "short"
    assert sizing["direction_source"] == "explicit"
    assert sizing["risk_pct"] <= 1.0
    assert sizing["risk_compliance"] == "within_requested_risk"
    assert sizing["rr_ratio"] == 1.0


def test_trade_risk_analyze_rejects_wrong_side_stop_for_short_trade() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    out = trade_risk_analyze(
        symbol="EURUSD",
        direction="short",
        desired_risk_pct=1.0,
        proposed_entry=100.0,
        proposed_sl=95.0,
    )

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["position_sizing_error"] == "For short trades, proposed_sl must be above proposed_entry."
    assert "position_sizing" not in out


def test_trade_risk_analyze_rejects_wrong_side_take_profit_for_long_trade() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    out = trade_risk_analyze(
        symbol="EURUSD",
        direction="long",
        desired_risk_pct=1.0,
        proposed_entry=100.0,
        proposed_sl=92.0,
        proposed_tp=95.0,
    )

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["position_sizing_error"] == "For long trades, proposed_tp must be above proposed_entry."
    assert "position_sizing" not in out


def test_trade_risk_analyze_returns_connection_error_payload() -> None:
    with patch(
        "mtdata.core.trading_risk.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running."),
    ):
        out = _trade_risk_analyze_tool(
            request=TradeRiskAnalyzeRequest(),
            __cli_raw=True,
        )

    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["operation"] == "trade_risk_analyze"
    assert out["success"] is False


def test_run_trade_risk_analyze_logs_finish_event(caplog) -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [],
    )

    with caplog.at_level("INFO", logger="mtdata.core.trading_use_cases"):
        out = run_trade_risk_analyze(
            TradeRiskAnalyzeRequest(),
            gateway=gateway,
        )

    assert out["success"] is True
    assert any(
        "event=finish operation=trade_risk_analyze success=True" in record.message
        for record in caplog.records
    )


def test_trade_risk_analyze_logs_finish_event(caplog) -> None:
    raw = _unwrap(_trade_risk_analyze_tool)

    with patch.object(core_trading_risk, "create_trading_gateway", return_value=object()), patch.object(
        core_trading_risk,
        "run_trade_risk_analyze",
        return_value={"success": True, "positions": []},
    ), caplog.at_level(logging.INFO, logger=core_trading_risk.logger.name):
        out = raw(TradeRiskAnalyzeRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=trade_risk_analyze success=True" in record.message
        for record in caplog.records
    )


def test_trade_risk_analyze_reports_calculation_failures() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=7,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            price_open=100.0,
            sl=90.0,
            tp=110.0,
        )
    ]
    mt5.symbol_info.return_value = SimpleNamespace(
        trade_contract_size="bad",
        point=1.0,
        trade_tick_value=1.0,
        trade_tick_size=1.0,
    )

    out = trade_risk_analyze(__cli_raw=True)

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["portfolio_risk"]["overall_risk_status"] == "incomplete"
    assert out["portfolio_risk"]["positions_with_risk_calculation_failures"] == 1
    assert len(out["risk_calculation_failures"]) == 1
    assert out["risk_calculation_failures"][0]["ticket"] == 7


def test_trade_risk_analyze_flags_invalid_tick_configuration_with_existing_stop_loss() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=8,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            price_open=100.0,
            sl=90.0,
            tp=110.0,
        )
    ]
    mt5.symbol_info.return_value = SimpleNamespace(
        trade_contract_size=1.0,
        point=1.0,
        trade_tick_value=0.0,
        trade_tick_size=0.0,
    )

    raw = _unwrap(_trade_risk_analyze_tool)
    with patch("mtdata.core.trading_risk.ensure_mt5_connection_or_raise", return_value=None):
        out = raw(request=TradeRiskAnalyzeRequest())

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["portfolio_risk"]["overall_risk_status"] == "incomplete"
    assert out["portfolio_risk"]["positions_without_sl"] == 0
    assert out["portfolio_risk"]["positions_with_risk_calculation_failures"] == 1
    assert out["positions"][0]["risk_status"] == "undefined"
    assert out["risk_calculation_failures"][0]["ticket"] == 8
    assert out["risk_calculation_failures"][0]["error_type"] == "InvalidTickConfiguration"


def test_trade_risk_analyze_preserves_quantified_risk_level_with_unlimited_positions() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=100.0, currency="USD"),
        positions_get=lambda symbol=None: [
            SimpleNamespace(
                ticket=9,
                symbol="EURUSD",
                type=0,
                volume=1.0,
                price_open=100.0,
                sl=80.0,
                tp=120.0,
            ),
            SimpleNamespace(
                ticket=10,
                symbol="EURUSD",
                type=0,
                volume=1.0,
                price_open=100.0,
                sl=0.0,
                tp=0.0,
            ),
        ],
        symbol_info=lambda symbol: _make_symbol_info(),
    )

    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(),
        gateway=gateway,
    )

    assert out["success"] is True
    assert out["portfolio_risk"]["overall_risk_status"] == "unlimited"
    assert out["portfolio_risk"]["quantified_risk_level"] == "high"
    assert out["portfolio_risk"]["total_risk_pct"] == 20.0
    assert out["portfolio_risk"]["positions_without_sl"] == 1
