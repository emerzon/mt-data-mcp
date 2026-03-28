from __future__ import annotations

from types import ModuleType, SimpleNamespace
import importlib
import logging
import pkgutil
import sys

import pytest

from mtdata.core import forecast as cf
from mtdata.forecast.exceptions import ForecastError
from mtdata.forecast import use_cases as forecast_use_cases
from mtdata.forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
)
from mtdata.utils.mt5 import MT5ConnectionError


def _unwrap(fn):
    current = fn
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", lambda: None)


def test_normalize_forecaster_name_and_resolve_variants(monkeypatch):
    monkeypatch.setattr(
        forecast_use_cases,
        "_discover_sktime_forecasters",
        lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
            "naiveforecaster": ("NaiveForecaster", "sktime.forecasting.naive.NaiveForecaster"),
        },
    )

    assert forecast_use_cases._normalize_forecaster_name("Theta-Forecaster v2") == "thetaforecasterv2"
    assert cf._resolve_sktime_forecaster("theta") == (
        "ThetaForecaster",
        "sktime.forecasting.theta.ThetaForecaster",
    )
    assert cf._resolve_sktime_forecaster("naive_fore") == (
        "NaiveForecaster",
        "sktime.forecasting.naive.NaiveForecaster",
    )
    assert cf._resolve_sktime_forecaster("") is None


def test_discover_sktime_forecasters_filters_test_and_non_forecaster_modules(monkeypatch):
    cf._discover_sktime_forecasters.cache_clear()

    class BaseForecaster:
        pass

    theta_mod = ModuleType("sktime.forecasting.theta")
    theta_mod.ThetaForecaster = type(
        "ThetaForecaster",
        (BaseForecaster,),
        {"__module__": "sktime.forecasting.theta"},
    )
    theta_mod.NotAForecaster = type(
        "NotAForecaster",
        (),
        {"__module__": "sktime.forecasting.theta"},
    )
    theta_mod._PrivateForecaster = type(
        "_PrivateForecaster",
        (BaseForecaster,),
        {"__module__": "sktime.forecasting.theta"},
    )

    fake_sktime = ModuleType("sktime")
    fake_forecasting = ModuleType("sktime.forecasting")
    fake_forecasting.__path__ = ["fake"]
    fake_base = ModuleType("sktime.forecasting.base")
    fake_base.BaseForecaster = BaseForecaster

    modules = {
        "sktime.forecasting.theta": theta_mod,
        "sktime.forecasting.tests.something": ModuleType("sktime.forecasting.tests.something"),
    }

    def fake_import_module(name):
        if name in modules:
            return modules[name]
        return importlib.import_module(name)

    monkeypatch.setitem(sys.modules, "sktime", fake_sktime)
    monkeypatch.setitem(sys.modules, "sktime.forecasting", fake_forecasting)
    monkeypatch.setitem(sys.modules, "sktime.forecasting.base", fake_base)
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        lambda _path, _prefix: [
            SimpleNamespace(name="sktime.forecasting.tests.something"),
            SimpleNamespace(name="sktime.forecasting.theta"),
        ],
    )
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    mapping = cf._discover_sktime_forecasters()

    assert "thetaforecaster" in mapping
    assert mapping["thetaforecaster"][0] == "ThetaForecaster"
    assert "privateforecaster" not in mapping
    cf._discover_sktime_forecasters.cache_clear()


def test_forecast_generate_routes_by_library_and_validates_inputs(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "method": kwargs["method"], "params": kwargs["params"]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)
    monkeypatch.setattr(
        cf,
        "_resolve_sktime_forecaster",
        lambda q: ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster") if q == "theta" else None,
    )

    with pytest.raises(Exception):
        ForecastGenerateRequest(symbol="EURUSD", horizon=0)

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method=""))
    assert out["error"] == "method is required for library=statsforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="unknown"))
    assert "Unknown sktime forecaster" in out["error"]

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="", params={"x": 1}))
    assert out["ok"] is True
    assert captured["method"] == "theta"
    assert captured["params"] == {"x": 1}

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method="AutoARIMA", params={}))
    assert out["ok"] is True
    assert captured["method"] == "statsforecast"
    assert captured["params"]["model_name"] == "AutoARIMA"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="theta", params={}))
    assert out["ok"] is True
    assert captured["method"] == "sktime"
    assert captured["params"]["estimator"] == "sktime.forecasting.theta.ThetaForecaster"

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="sktime",
            method="sktime.forecasting.naive.NaiveForecaster",
            params={},
        )
    )
    assert out["ok"] is True
    assert captured["params"]["estimator"] == "sktime.forecasting.naive.NaiveForecaster"

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="mlforecast",
            method="sklearn.linear_model.LinearRegression",
            params={},
        )
    )
    assert out["ok"] is True
    assert captured["method"] == "mlforecast"
    assert captured["params"]["model"] == "sklearn.linear_model.LinearRegression"

    with pytest.raises(Exception):
        ForecastGenerateRequest(symbol="EURUSD", library="unsupported", method="x")


def test_forecast_generate_native_theta_adds_disambiguation_warning(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fake_forecast_impl(**kwargs):
        return {"ok": True, "method": kwargs["method"]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)

    out = raw(request=ForecastGenerateRequest(symbol="BTCUSD", timeframe="H1", library="native", method="theta", horizon=12))

    assert out["ok"] is True
    assert any("StatsForecast theta is available" in str(w) for w in out.get("warnings", []))


def test_run_forecast_generate_logs_finish_event(caplog):
    with caplog.at_level("INFO", logger="mtdata.forecast.use_cases"):
        result = forecast_use_cases.run_forecast_generate(
            ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", library="native", method="theta"),
            forecast_impl=lambda **kwargs: {"ok": True, "method": kwargs["method"]},
            resolve_sktime_forecaster=lambda query: None,
        )
    assert result["ok"] is True
    assert any(
        "event=finish operation=forecast_generate success=True" in record.message
        for record in caplog.records
    )


def test_run_forecast_backtest_derives_target_from_quantity():
    captured = {}

    def fake_backtest_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", quantity="return"),
        backtest_impl=fake_backtest_impl,
    )

    assert result["success"] is True
    assert captured["quantity"] == "return"
    assert "target" not in captured


def test_forecast_generate_converts_typed_forecast_errors(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: (_ for _ in ()).throw(ForecastError("engine exploded")))

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["error"] == "engine exploded"


def test_forecast_generate_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: {"success": True, "method": kwargs["method"]})

    with caplog.at_level(logging.INFO, logger=cf.logger.name):
        out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_generate success=True" in record.message
        for record in caplog.records
    )


def test_forecast_generate_wrapper_emits_single_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: {"success": True, "method": kwargs["method"]})

    with caplog.at_level(logging.INFO):
        out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=forecast_generate success=True" in record.message
    ]
    assert len(finish_records) == 1
    assert finish_records[0].name == cf.logger.name


def test_forecast_generate_returns_connection_error_payload(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: pytest.fail("forecast implementation should not run"))

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out == {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}


def test_forecast_tune_genetic_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_tune_genetic)
    monkeypatch.setattr(cf, "run_forecast_tune_genetic", lambda request, genetic_search_impl: {"success": True, "best": {}})

    with caplog.at_level(logging.INFO, logger=cf.logger.name):
        out = raw(request=ForecastTuneGeneticRequest(symbol="EURUSD", method="theta"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_tune_genetic success=True" in record.message
        for record in caplog.records
    )


def test_forecast_barrier_optimize_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_barrier_optimize)
    monkeypatch.setattr(cf, "run_forecast_barrier_optimize", lambda request, parse_kv_or_json, barrier_optimize_impl: {"success": True, "best": {}})

    fake_barriers = ModuleType("mtdata.forecast.barriers")
    fake_barriers.forecast_barrier_optimize = lambda **kwargs: {"unused": True}
    monkeypatch.setitem(sys.modules, "mtdata.forecast.barriers", fake_barriers)

    with caplog.at_level(logging.INFO, logger=cf.logger.name):
        out = raw(request=ForecastBarrierOptimizeRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_barrier_optimize success=True" in record.message
        for record in caplog.records
    )


def test_forecast_barrier_optimize_request_defaults_to_summary_output():
    request = ForecastBarrierOptimizeRequest(symbol="EURUSD")
    assert request.output == "summary"


def test_forecast_list_library_models_and_list_methods(monkeypatch):
    stats_mod = ModuleType("statsforecast")
    models_mod = ModuleType("statsforecast.models")
    models_mod.__name__ = "statsforecast.models"
    models_mod.AutoARIMA = type(
        "AutoARIMA",
        (),
        {
            "__module__": "statsforecast.models",
            "fit": lambda self: None,
        },
    )
    models_mod.OtherHelper = 123
    stats_mod.models = models_mod

    monkeypatch.setitem(sys.modules, "statsforecast", stats_mod)
    monkeypatch.setattr(
        cf,
        "_discover_sktime_forecasters",
        lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
            "naiveforecaster": ("NaiveForecaster", "sktime.forecasting.naive.NaiveForecaster"),
        },
    )

    raw_list_models = _unwrap(cf.forecast_list_library_models)
    out_native = raw_list_models("native")
    assert out_native["library"] == "native"
    assert isinstance(out_native["models"], list)

    out_stats = raw_list_models("statsforecast")
    assert out_stats["library"] == "statsforecast"
    assert "AutoARIMA" in out_stats["models"]

    out_sktime = raw_list_models("sktime")
    assert out_sktime["models"] == ["NaiveForecaster", "ThetaForecaster"]

    out_ml = raw_list_models("mlforecast")
    assert out_ml["library"] == "mlforecast"
    assert "note" in out_ml

    out_bad = raw_list_models("other")
    assert "Unsupported library" in out_bad["error"]

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 2,
            "categories": {"classical": ["theta"], "ml": ["mlf_rf"]},
            "methods": [
                {
                    "method": "theta",
                    "available": True,
                    "description": "Theta model.",
                    "params": [{"name": "window"}],
                    "requires": [],
                },
                {
                    "method": "mlf_rf",
                    "available": False,
                    "description": "RF model.",
                    "params": [{"name": "n_estimators"}, {"name": "max_depth"}],
                    "requires": ["mlforecast", "sklearn"],
                },
            ],
        },
    )
    compact = _unwrap(cf.forecast_list_methods)()
    assert compact["detail"] == "compact"
    assert compact["total"] == 2
    assert compact["available"] == 1
    assert compact["unavailable"] == 1
    assert compact["methods"][0]["method"] == "theta"
    assert "category_summary" in compact
    assert "params_count" in compact["methods"][0]
    assert compact["methods"][0]["namespace"] == "native"
    assert compact["methods"][0]["method_id"] == "native:theta"
    assert compact["methods"][0]["concept"] == "theta"
    assert "params" not in compact["methods"][0]
    assert all("requires" not in row for row in compact["methods"])
    assert compact["methods_shown"] == 2
    assert compact["methods_hidden"] == 0
    assert compact["note"].endswith("Use --detail full to see all methods.")

    full = _unwrap(cf.forecast_list_methods)(detail="full")
    assert isinstance(full.get("methods"), list)
    assert "params" in full["methods"][0]
    assert "method_id" in full["methods"][0]

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 5,
            "categories": {
                "classical": ["theta"],
                "statsforecast": ["sf_autoarima", "sf_theta", "sf_ets", "sf_naive"],
            },
            "methods": [
                {"method": "theta", "available": True, "description": "Theta.", "params": [], "requires": []},
                {"method": "sf_autoarima", "available": True, "description": "A", "params": [], "requires": []},
                {"method": "sf_theta", "available": True, "description": "B", "params": [], "requires": []},
                {"method": "sf_ets", "available": False, "description": "C", "params": [], "requires": []},
                {"method": "sf_naive", "available": True, "description": "D", "params": [], "requires": []},
            ],
        },
    )
    grouped = _unwrap(cf.forecast_list_methods)()
    sf_rows = [r for r in grouped["methods"] if r.get("category") == "statsforecast"]
    assert len(sf_rows) <= 3
    if sf_rows:
        assert all(str(r.get("namespace")) == "statsforecast" for r in sf_rows)
    assert grouped["methods_hidden"] >= 1
    filtered = _unwrap(cf.forecast_list_methods)(search="theta", limit=1)
    assert filtered["filters"]["search"] == "theta"
    assert filtered["filters"]["limit"] == 1
    assert len(filtered["methods"]) == 1
    assert "theta" in str(filtered["methods"][0]["method"]).lower()

    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: {"methods": [1]})
    assert _unwrap(cf.forecast_list_methods)() == {"methods": [1]}
    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "Error listing forecast methods" in _unwrap(cf.forecast_list_methods)()["error"]


def test_forecast_list_methods_does_not_require_mt5_connection(monkeypatch):
    def fail_connection():
        raise MT5ConnectionError("should not be called")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {"total": 1, "categories": {}, "methods": [{"method": "theta", "available": True}]},
    )

    out = _unwrap(cf.forecast_list_methods)()

    assert out["methods"][0]["method"] == "theta"


def test_forecast_list_library_models_logs_finish_event(caplog):
    raw_list_models = _unwrap(cf.forecast_list_library_models)

    with caplog.at_level(logging.INFO, logger=cf.logger.name):
        out = raw_list_models("mlforecast")

    assert out["library"] == "mlforecast"
    assert any(
        "event=finish operation=forecast_list_library_models success=True" in record.message
        for record in caplog.records
    )


def test_forecast_conformal_intervals_success_and_errors(monkeypatch):
    raw = _unwrap(cf.forecast_conformal_intervals)

    monkeypatch.setattr(
        cf,
        "_forecast_backtest_impl",
        lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0, 11.0], "actual": [9.0, 12.0]},
                        {"forecast": [13.0, 14.0], "actual": [12.0, 15.0]},
                    ]
                }
            }
        },
    )
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: {"forecast_price": [100.0, 101.0]})

    out = raw(
        request=ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=2,
            ci_alpha=0.1,
            steps=2,
        )
    )

    assert out["ci_alpha"] == 0.1
    assert out["conformal"]["ci_alpha"] == 0.1
    assert len(out["lower_price"]) == 2
    assert len(out["upper_price"]) == 2
    assert out["lower_price"][0] <= 100.0 <= out["upper_price"][0]

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"error": "backtest failed"})
    assert raw(request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2))["error"] == "backtest failed"

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"results": {"theta": {"details": []}}})
    assert "Conformal calibration failed" in raw(
        request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2)
    )["error"]

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"results": {"theta": {"details": [{}]}}})
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: (_ for _ in ()).throw(ForecastError("engine exploded")))
    assert raw(request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2))["error"] == "engine exploded"


def test_forecast_tune_genetic_and_barrier_prob_routing(monkeypatch):
    raw_tune = _unwrap(cf.forecast_tune_genetic)
    raw_barrier = _unwrap(cf.forecast_barrier_prob)

    captured = {}
    ss_calls = {}

    def fake_genetic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cf, "_genetic_search_impl", fake_genetic)

    import mtdata.forecast.tune as tune_mod

    def fake_default_search_space(method=None, methods=None):
        ss_calls["method"] = method
        ss_calls["methods"] = methods
        return {"theta": {"window": {"min": 1, "max": 3}}}

    monkeypatch.setattr(tune_mod, "default_search_space", fake_default_search_space)
    out = raw_tune(request=ForecastTuneGeneticRequest(symbol="EURUSD", method="theta", search_space=None))
    assert out == {"ok": True}
    assert captured["method"] == "theta"
    assert ss_calls["method"] == "theta"
    assert ss_calls["methods"] is None
    assert "theta" in captured["search_space"]

    out = raw_tune(
        request=ForecastTuneGeneticRequest(
            symbol="EURUSD",
            method="theta",
            methods=["theta", "naive"],
            search_space={"x": {"type": "int"}},
        )
    )
    assert out == {"ok": True}
    assert captured["method"] is None

    monkeypatch.setattr(cf, "_genetic_search_impl", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert "Error in genetic tuning" in raw_tune(request=ForecastTuneGeneticRequest(symbol="EURUSD"))["error"]

    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers as barriers_mod

    called = {}

    def fake_mc(**kwargs):
        called.update(kwargs)
        return {"kind": "mc", "direction": kwargs["direction"], "method": kwargs["method"]}

    def fake_cf(**kwargs):
        called.update(kwargs)
        return {"kind": "cf", "direction": kwargs["direction"]}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_hit_probabilities", fake_mc)
    monkeypatch.setattr(barriers_mod, "forecast_barrier_closed_form", fake_cf)

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="auto",
            direction="down",
        )
    )
    assert out["kind"] == "mc"
    assert out["method"] == "auto"
    assert out["direction"] == "short"

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            direction="weird",
        )
    )
    assert "error" in out
    assert "Invalid direction" in out["error"]

    out = raw_barrier(request=ForecastBarrierProbRequest(symbol="EURUSD", method="mystery"))
    assert out["error"] == "Unknown method: mystery"


def test_forecast_barrier_prob_wrapper_emits_single_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_barrier_prob)
    monkeypatch.setattr(cf, "_forecast_connection_error", lambda: None)
    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers as barriers_mod

    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_hit_probabilities",
        lambda **kwargs: {"success": True, "kind": "mc", "direction": kwargs["direction"]},
    )
    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_closed_form",
        lambda **kwargs: {"success": True, "kind": "closed_form", "direction": kwargs["direction"]},
    )

    with caplog.at_level(logging.INFO):
        out = raw(request=ForecastBarrierProbRequest(symbol="EURUSD", timeframe="H1"))

    assert out["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=forecast_barrier_prob success=True" in record.message
    ]
    assert len(finish_records) == 1


def test_forecast_tune_optuna_routing(monkeypatch):
    raw_tune = _unwrap(cf.forecast_tune_optuna)
    captured = {}
    ss_calls = {}

    def fake_optuna(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cf, "_optuna_search_impl", fake_optuna)

    import mtdata.forecast.tune as tune_mod

    def fake_default_search_space(method=None, methods=None):
        ss_calls["method"] = method
        ss_calls["methods"] = methods
        return {"theta": {"window": {"min": 1, "max": 3}}}

    monkeypatch.setattr(tune_mod, "default_search_space", fake_default_search_space)
    out = raw_tune(request=ForecastTuneOptunaRequest(symbol="EURUSD", method="theta", search_space=None))
    assert out == {"ok": True}
    assert captured["method"] == "theta"
    assert ss_calls["method"] == "theta"
    assert ss_calls["methods"] is None
    assert "theta" in captured["search_space"]

    out = raw_tune(
        request=ForecastTuneOptunaRequest(
            symbol="EURUSD",
            method="theta",
            methods=["theta", "naive"],
            search_space={"x": {"type": "int"}},
        )
    )
    assert out == {"ok": True}
    assert captured["method"] is None

    monkeypatch.setattr(cf, "_optuna_search_impl", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert "Error in optuna tuning" in raw_tune(request=ForecastTuneOptunaRequest(symbol="EURUSD"))["error"]


def test_forecast_options_and_quantlib_tool_routing(monkeypatch):
    raw_exp = _unwrap(cf.forecast_options_expirations)
    raw_chain = _unwrap(cf.forecast_options_chain)
    raw_price = _unwrap(cf.forecast_quantlib_barrier_price)
    raw_cal = _unwrap(cf.forecast_quantlib_heston_calibrate)

    import mtdata.services.options_service as options_service
    import mtdata.forecast.quantlib_tools as quantlib_tools

    monkeypatch.setattr(options_service, "get_options_expirations", lambda **kwargs: {"kind": "exp", **kwargs})
    monkeypatch.setattr(options_service, "get_options_chain", lambda **kwargs: {"kind": "chain", **kwargs})
    monkeypatch.setattr(quantlib_tools, "price_barrier_option_quantlib", lambda **kwargs: {"kind": "price", **kwargs})
    monkeypatch.setattr(quantlib_tools, "calibrate_heston_quantlib_from_options", lambda **kwargs: {"kind": "cal", **kwargs})

    out = raw_exp(symbol="AAPL")
    assert out["kind"] == "exp"
    assert out["symbol"] == "AAPL"

    out = raw_chain(symbol="AAPL", expiration="2026-06-19", option_type="call", min_open_interest=10, min_volume=5, limit=20)
    assert out["kind"] == "chain"
    assert out["symbol"] == "AAPL"
    assert out["option_type"] == "call"
    assert out["limit"] == 20

    out = raw_price(
        spot=100.0,
        strike=105.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        volatility=0.25,
        rebate=0.0,
    )
    assert out["kind"] == "price"
    assert out["spot"] == 100.0

    out = raw_cal(
        symbol="AAPL",
        expiration="2026-06-19",
        option_type="put",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        min_open_interest=10,
        min_volume=2,
        max_contracts=15,
    )
    assert out["kind"] == "cal"
    assert out["symbol"] == "AAPL"
    assert out["option_type"] == "put"


def test_forecast_options_chain_logs_finish_event(caplog, monkeypatch):
    raw_chain = _unwrap(cf.forecast_options_chain)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(options_service, "get_options_chain", lambda **kwargs: {"success": True, **kwargs})

    with caplog.at_level(logging.INFO, logger=cf.logger.name):
        out = raw_chain(symbol="AAPL", expiration="2026-06-19", option_type="call", limit=25)

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_options_chain success=True" in record.message
        for record in caplog.records
    )


def test_forecast_barrier_optimize_routes_profile_args(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True, "search_profile": kwargs.get("search_profile"), "fast_defaults": kwargs.get("fast_defaults")}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(
        request=ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            fast_defaults=True,
            search_profile="long",
        )
    )
    assert out["ok"] is True
    assert out["fast_defaults"] is True
    assert out["search_profile"] == "long"
    assert called["fast_defaults"] is True
    assert called["search_profile"] == "long"


def test_forecast_barrier_optimize_applies_default_optuna_config(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD"))
    assert out["ok"] is True
    assert called["method"] == "auto"
    assert called["search_profile"] == "long"
    assert called["params"]["optimizer"] == "optuna"
    assert called["params"]["sampler"] == "tpe"
    assert called["params"]["pruner"] == "median"
    assert int(called["params"]["n_jobs"]) >= 1
    assert called["params"]["seed"] == 42


def test_forecast_barrier_optimize_returns_connection_error_payload(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)

    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="EURUSD"))

    assert out == {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}
