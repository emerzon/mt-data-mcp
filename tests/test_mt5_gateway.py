from types import SimpleNamespace

import pytest

from mtdata.core.mt5_gateway import create_mt5_gateway


def test_mt5_gateway_ensure_connection_logs_finish_event(caplog):
    gateway = create_mt5_gateway(
        adapter=SimpleNamespace(),
        ensure_connection_impl=lambda: None,
    )

    with caplog.at_level("INFO", logger="mtdata.core.mt5_gateway"):
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
