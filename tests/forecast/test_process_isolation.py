from __future__ import annotations

from mtdata.core import forecast as core_forecast
from mtdata.forecast.gpu_runtime import forecast_method_may_use_gpu


class _DummyQueue:
    def __init__(self) -> None:
        self.messages = []

    def put(self, message):
        self.messages.append(message)


def test_process_isolation_mode_aliases(monkeypatch):
    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_ISOLATION", "off")
    assert core_forecast._forecast_process_isolation_mode() == "off"

    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_ISOLATION", "all")
    assert core_forecast._forecast_process_isolation_mode() == "all"

    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_ISOLATION", "1")
    assert core_forecast._forecast_process_isolation_mode() == "all"

    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_ISOLATION", "gpu")
    assert core_forecast._forecast_process_isolation_mode() == "gpu"

    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_ISOLATION", "not-a-mode")
    assert core_forecast._forecast_process_isolation_mode() == "gpu"


def test_should_isolate_gpu_forecast_by_default(monkeypatch):
    monkeypatch.delenv("MTDATA_FORECAST_PROCESS_ISOLATION", raising=False)
    monkeypatch.delenv("MTDATA_FORECAST_PROCESS_CHILD", raising=False)

    assert core_forecast._should_isolate_forecast_operation(
        "forecast_generate",
        {"library": "pretrained", "method": "chronos2", "params": {}},
    )
    assert not core_forecast._should_isolate_forecast_operation(
        "forecast_generate",
        {"library": "native", "method": "theta", "params": {}},
    )


def test_should_isolate_all_forecast_operations(monkeypatch):
    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_ISOLATION", "all")
    monkeypatch.delenv("MTDATA_FORECAST_PROCESS_CHILD", raising=False)

    assert core_forecast._should_isolate_forecast_operation(
        "forecast_list_methods",
        {"detail": "compact"},
    )


def test_should_not_isolate_inside_child(monkeypatch):
    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_ISOLATION", "all")
    monkeypatch.setenv("MTDATA_FORECAST_PROCESS_CHILD", "1")

    assert not core_forecast._should_isolate_forecast_operation(
        "forecast_generate",
        {"library": "pretrained", "method": "chronos2"},
    )


def test_capability_style_method_names_count_as_gpu_methods():
    assert forecast_method_may_use_gpu("pretrained:chronos2")
    assert forecast_method_may_use_gpu(
        "ensemble",
        {"methods": ["theta", "pretrained:timesfm"]},
    )


def test_direct_payload_dispatch_rebuilds_forecast_generate_request(monkeypatch):
    def fake_run_forecast_generate(request, **kwargs):
        return {
            "success": True,
            "symbol": request.symbol,
            "method": request.method,
            "has_forecast_impl": kwargs.get("forecast_impl") is not None,
        }

    monkeypatch.setattr(core_forecast, "run_forecast_generate", fake_run_forecast_generate)
    monkeypatch.setattr(core_forecast, "ensure_mt5_connection_or_raise", lambda: None)

    result = core_forecast._run_forecast_payload_direct(
        "forecast_generate",
        {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "library": "native",
            "method": "theta",
            "horizon": 3,
        },
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "method": "theta",
        "has_forecast_impl": True,
    }


def test_child_entry_returns_ok_message(monkeypatch):
    monkeypatch.setattr(
        core_forecast,
        "_run_forecast_payload_direct",
        lambda operation, payload: {"success": True, "operation": operation},
    )
    queue = _DummyQueue()

    core_forecast._forecast_process_entry("forecast_generate", {}, queue)

    assert queue.messages == [
        {
            "status": "ok",
            "result": {"success": True, "operation": "forecast_generate"},
        }
    ]


def test_child_entry_returns_exception_message(monkeypatch):
    def fail(operation, payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(core_forecast, "_run_forecast_payload_direct", fail)
    queue = _DummyQueue()

    core_forecast._forecast_process_entry("forecast_generate", {}, queue)

    assert queue.messages[0]["status"] == "exception"
    assert queue.messages[0]["type"] == "RuntimeError"
    assert queue.messages[0]["message"] == "boom"
