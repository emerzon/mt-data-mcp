from __future__ import annotations

from types import ModuleType, SimpleNamespace
import importlib
import pkgutil
import sys

from mtdata.core import forecast as cf


def _unwrap(fn):
    current = fn
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


def test_normalize_forecaster_name_and_resolve_variants(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_discover_sktime_forecasters",
        lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
            "naiveforecaster": ("NaiveForecaster", "sktime.forecasting.naive.NaiveForecaster"),
        },
    )

    assert cf._normalize_forecaster_name("Theta-Forecaster v2") == "thetaforecasterv2"
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
    monkeypatch.setattr(cf, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(
        cf,
        "_resolve_sktime_forecaster",
        lambda q: ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster") if q == "theta" else None,
    )

    out = raw(symbol="EURUSD", horizon=0)
    assert out["error"] == "horizon must be a positive integer"

    out = raw(symbol="EURUSD", library="statsforecast", model="")
    assert out["error"] == "model is required for library=statsforecast"

    out = raw(symbol="EURUSD", library="sktime", model="unknown")
    assert "Unknown sktime forecaster" in out["error"]

    out = raw(symbol="EURUSD", library="native", model="", model_params={"x": 1})
    assert out["ok"] is True
    assert captured["method"] == "theta"
    assert captured["params"] == {"x": 1}

    out = raw(symbol="EURUSD", library="statsforecast", model="AutoARIMA", model_params={})
    assert out["ok"] is True
    assert captured["method"] == "statsforecast"
    assert captured["params"]["model_name"] == "AutoARIMA"

    out = raw(symbol="EURUSD", library="sktime", model="theta", model_params={})
    assert out["ok"] is True
    assert captured["method"] == "sktime"
    assert captured["params"]["estimator"] == "sktime.forecasting.theta.ThetaForecaster"

    out = raw(symbol="EURUSD", library="sktime", model="sktime.forecasting.naive.NaiveForecaster", model_params={})
    assert out["ok"] is True
    assert captured["params"]["estimator"] == "sktime.forecasting.naive.NaiveForecaster"

    out = raw(symbol="EURUSD", library="mlforecast", model="sklearn.linear_model.LinearRegression", model_params={})
    assert out["ok"] is True
    assert captured["method"] == "mlforecast"
    assert captured["params"]["model"] == "sklearn.linear_model.LinearRegression"

    out = raw(symbol="EURUSD", library="unsupported", model="x")
    assert out["error"] == "Unsupported library: unsupported"


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

    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: {"methods": [1]})
    assert _unwrap(cf.forecast_list_methods)() == {"methods": [1]}
    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "Error listing forecast methods" in _unwrap(cf.forecast_list_methods)()["error"]


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

    out = raw(symbol="EURUSD", method="theta", horizon=2, alpha=0.1, steps=2)

    assert out["ci_alpha"] == 0.1
    assert out["conformal"]["alpha"] == 0.1
    assert len(out["lower_price"]) == 2
    assert len(out["upper_price"]) == 2
    assert out["lower_price"][0] <= 100.0 <= out["upper_price"][0]

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"error": "backtest failed"})
    assert raw(symbol="EURUSD", method="theta", horizon=2)["error"] == "backtest failed"

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"results": {"theta": {"details": []}}})
    assert "Conformal calibration failed" in raw(symbol="EURUSD", method="theta", horizon=2)["error"]


def test_forecast_tune_genetic_and_barrier_prob_routing(monkeypatch):
    raw_tune = _unwrap(cf.forecast_tune_genetic)
    raw_barrier = _unwrap(cf.forecast_barrier_prob)

    captured = {}
    ss_calls = {}

    def fake_genetic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cf, "_genetic_search_impl", fake_genetic)
    monkeypatch.setattr(cf, "_parse_kv_or_json", lambda v: dict(v or {}))

    import mtdata.forecast.tune as tune_mod

    def fake_default_search_space(method=None, methods=None):
        ss_calls["method"] = method
        ss_calls["methods"] = methods
        return {"theta": {"window": {"min": 1, "max": 3}}}

    monkeypatch.setattr(tune_mod, "default_search_space", fake_default_search_space)
    out = raw_tune(symbol="EURUSD", method="theta", search_space=None)
    assert out == {"ok": True}
    assert captured["method"] == "theta"
    assert ss_calls["method"] == "theta"
    assert ss_calls["methods"] is None
    assert "theta" in captured["search_space"]

    out = raw_tune(symbol="EURUSD", method="theta", methods=["theta", "naive"], search_space={"x": {"type": "int"}})
    assert out == {"ok": True}
    assert captured["method"] is None

    monkeypatch.setattr(cf, "_genetic_search_impl", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert "Error in genetic tuning" in raw_tune(symbol="EURUSD")["error"]

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

    out = raw_barrier(symbol="EURUSD", method="auto", mc_method="hmm_mc", direction="down")
    assert out["kind"] == "mc"
    assert out["method"] == "auto"
    assert out["direction"] == "short"

    out = raw_barrier(symbol="EURUSD", method="closed_form", direction="weird")
    assert out["kind"] == "cf"
    assert out["direction"] == "long"

    out = raw_barrier(symbol="EURUSD", method="mystery")
    assert out["error"] == "Unknown method: mystery"
