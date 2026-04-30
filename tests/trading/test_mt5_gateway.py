from types import SimpleNamespace

import pytest

from mtdata.core.mt5_gateway import create_mt5_gateway, mt5_connection_error
from mtdata.utils.mt5 import MT5ConnectionError


def test_mt5_gateway_ensure_connection_logs_finish_event(caplog):
    gateway = create_mt5_gateway(
        adapter=SimpleNamespace(),
        ensure_connection_impl=lambda: None,
    )

    with caplog.at_level("DEBUG", logger="mtdata.core.mt5_gateway"):
        gateway.ensure_connection()

    assert any(
        "event=finish operation=mt5_ensure_connection success=True" in record.message
        for record in caplog.records
    )


def test_mt5_gateway_ensure_connection_logs_exception(caplog):
    gateway = create_mt5_gateway(
        adapter=SimpleNamespace(),
        ensure_connection_impl=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with caplog.at_level("ERROR", logger="mtdata.core.mt5_gateway"):
        with pytest.raises(RuntimeError, match="boom"):
            gateway.ensure_connection()

    assert any(
        "event=error operation=mt5_ensure_connection" in record.message
        for record in caplog.records
    )


def test_mt5_connection_error_returns_stage1_envelope():
    gateway = create_mt5_gateway(
        adapter=SimpleNamespace(),
        ensure_connection_impl=lambda: (_ for _ in ()).throw(
            MT5ConnectionError("MT5 unavailable")
        ),
    )

    out = mt5_connection_error(gateway)

    assert out is not None
    assert out["success"] is False
    assert out["error"] == "MT5 unavailable"
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)
    assert out["request_id"]


def test_mt5_gateway_exposes_explicit_adapter_methods():
    calls = []

    class Adapter:
        ORDER_TYPE_BUY = 0

        def symbol_info(self, symbol):
            calls.append(("symbol_info", symbol))
            return {"symbol": symbol}

        def positions_get(self, **kwargs):
            calls.append(("positions_get", kwargs))
            return []

    gateway = create_mt5_gateway(
        adapter=Adapter(),
        ensure_connection_impl=lambda: None,
    )

    assert gateway.symbol_info("EURUSD") == {"symbol": "EURUSD"}
    assert gateway.ORDER_TYPE_BUY == 0
    assert gateway.positions_get(symbol="EURUSD") == []
    assert calls == [
        ("symbol_info", "EURUSD"),
        ("positions_get", {"symbol": "EURUSD"}),
    ]
