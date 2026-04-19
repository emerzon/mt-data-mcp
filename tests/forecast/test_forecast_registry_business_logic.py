import pytest

from mtdata.forecast import forecast_registry as fr


def test_check_requirements_marks_mlf_rf_unavailable_and_normalizes_sklearn(monkeypatch):
    checked_names = []

    def fake_find_spec(name):
        checked_names.append(name)
        return object() if name == "sklearn" else None

    monkeypatch.setattr(fr, "_MLF_AVAILABLE", False)
    monkeypatch.setattr(fr._importlib_util, "find_spec", fake_find_spec)

    available, reqs = fr._check_requirements("mlf_rf", ["scikit-learn>=1.6"])

    assert available is False
    assert "mlforecast, scikit-learn" in reqs
    assert "sklearn" in checked_names


def test_check_requirements_strips_versions_and_maps_python_dotenv(monkeypatch):
    checked_names = []

    def fake_find_spec(name):
        checked_names.append(name)
        return object()

    monkeypatch.setattr(fr._importlib_util, "find_spec", fake_find_spec)

    available, reqs = fr._check_requirements(
        "custom_method",
        ["python-dotenv>=1.0.0", "numpy==2.0.0"],
    )

    assert available is True
    assert reqs == ["python-dotenv>=1.0.0", "numpy==2.0.0"]
    assert "dotenv" in checked_names
    assert "numpy" in checked_names


def test_check_requirements_marks_chronos_unavailable_on_runtime_mismatch(monkeypatch):
    monkeypatch.setattr(fr, "_CHRONOS_AVAILABLE", True)
    monkeypatch.setattr(
        fr,
        "_check_chronos_runtime_support",
        lambda: (False, ["chronos-forecasting>=2.0.0"]),
    )
    monkeypatch.setattr(fr._importlib_util, "find_spec", lambda _name: object())

    available, reqs = fr._check_requirements("chronos2", ["chronos"])

    assert available is False
    assert "chronos-forecasting>=2.0.0" in reqs


def test_check_requirements_marks_gluonts_extra_methods_unsupported_on_python_314(monkeypatch):
    monkeypatch.setattr(fr, "_PYTHON_314_PLUS", True)
    monkeypatch.setattr(fr._importlib_util, "find_spec", lambda _name: object())

    available, reqs = fr._check_requirements("gt_deepar", ["gluonts", "torch"])

    assert available is False
    assert fr._GLUONTS_PYTHON_RUNTIME_REQUIREMENT in reqs


def test_ensure_registry_loaded_continues_after_one_module_import_failure(monkeypatch):
    imported = []

    def fake_import(name):
        imported.append(name)
        if name.endswith(".gluonts_extra"):
            raise ImportError("unsupported runtime")
        return object()

    monkeypatch.setattr(fr, "_FORECAST_METHOD_MODULES", ("classical", "gluonts_extra", "analog"))
    monkeypatch.setattr(fr, "_OPTIONAL_FORECAST_METHOD_MODULES", frozenset({"gluonts_extra"}))
    monkeypatch.setattr(fr, "_LOADED_FORECAST_METHOD_MODULES", set())
    monkeypatch.setattr(fr, "_FAILED_OPTIONAL_FORECAST_MODULES", {})
    monkeypatch.setattr(fr._importlib, "import_module", fake_import)

    fr._ensure_registry_loaded()

    assert imported == [
        "mtdata.forecast.methods.classical",
        "mtdata.forecast.methods.gluonts_extra",
        "mtdata.forecast.methods.analog",
    ]
    assert fr._LOADED_FORECAST_METHOD_MODULES == {"classical", "analog"}
    assert fr._FAILED_OPTIONAL_FORECAST_MODULES == {"gluonts_extra": "unsupported runtime"}


def test_ensure_registry_loaded_memoizes_optional_import_failures(monkeypatch):
    imported = []

    def fake_import(name):
        imported.append(name)
        if name.endswith(".gluonts_extra"):
            raise ImportError("unsupported runtime")
        return object()

    monkeypatch.setattr(fr, "_FORECAST_METHOD_MODULES", ("classical", "gluonts_extra", "analog"))
    monkeypatch.setattr(fr, "_OPTIONAL_FORECAST_METHOD_MODULES", frozenset({"gluonts_extra"}))
    monkeypatch.setattr(fr, "_LOADED_FORECAST_METHOD_MODULES", set())
    monkeypatch.setattr(fr, "_FAILED_OPTIONAL_FORECAST_MODULES", {})
    monkeypatch.setattr(fr._importlib, "import_module", fake_import)

    fr._ensure_registry_loaded()
    fr._ensure_registry_loaded()

    assert imported == [
        "mtdata.forecast.methods.classical",
        "mtdata.forecast.methods.gluonts_extra",
        "mtdata.forecast.methods.analog",
    ]


def test_ensure_registry_loaded_reraises_non_optional_import_errors(monkeypatch):
    def fake_import(name):
        if name.endswith(".classical"):
            raise ImportError("core import broke")
        return object()

    monkeypatch.setattr(fr, "_FORECAST_METHOD_MODULES", ("classical",))
    monkeypatch.setattr(fr, "_OPTIONAL_FORECAST_METHOD_MODULES", frozenset({"gluonts_extra"}))
    monkeypatch.setattr(fr, "_LOADED_FORECAST_METHOD_MODULES", set())
    monkeypatch.setattr(fr, "_FAILED_OPTIONAL_FORECAST_MODULES", {})
    monkeypatch.setattr(fr._importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match="core import broke"):
        fr._ensure_registry_loaded()


def test_get_forecast_methods_data_assembles_categories_and_skips_broken(monkeypatch):
    class GoodMethod:
        """Good method summary."""

        supports_features = {"price": True, "return": True, "volatility": False, "ci": True}
        required_packages = ["numpy>=1.0"]
        PARAMS = [{"name": "window", "type": "int"}]
        category = "Classic"

    class MissingParamsMethod:
        supports_features = None
        required_packages = []
        PARAMS = {"invalid": "shape"}
        category = None

    class BrokenMethod:
        def __init__(self):
            raise RuntimeError("construction failed")

    class FakeRegistry:
        @staticmethod
        def get_all_method_names():
            return ["good", "broken", "ensemble", "missing_params"]

        @staticmethod
        def get_class(name):
            mapping = {
                "good": GoodMethod,
                "broken": BrokenMethod,
                "missing_params": MissingParamsMethod,
            }
            return mapping[name]

    def fake_check_requirements(method, requires):
        if method == "missing_params":
            return False, ["some_pkg"]
        return True, list(requires)

    monkeypatch.setattr(fr, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fr, "_ensure_registry_loaded", lambda: None)
    monkeypatch.setattr(fr, "_check_requirements", fake_check_requirements)

    data = fr.get_forecast_methods_data()
    methods = {m["method"]: m for m in data["methods"]}

    assert data["total"] == 3
    assert "broken" not in methods

    assert methods["good"]["category"] == "classic"
    assert methods["good"]["description"] == "Good method summary."
    assert methods["good"]["params"] == [{"name": "window", "type": "int"}]
    assert methods["good"]["available"] is True

    assert methods["missing_params"]["available"] is False
    assert methods["missing_params"]["category"] == "unknown"
    assert methods["missing_params"]["params"] == []
    assert methods["missing_params"]["supports"] == {
        "price": True,
        "return": True,
        "volatility": False,
        "ci": False,
    }

    assert "good" in data["categories"]["classic"]
    assert "missing_params" in data["categories"]["unknown"]
    assert "ensemble" in data["categories"]["ensemble"]


def test_find_method_definition_returns_match_and_none():
    method_data = {
        "methods": [
            {"method": "theta", "available": True},
            {"method": "mlf_rf", "available": False},
        ]
    }

    assert fr._find_method_definition("theta", method_data) == {"method": "theta", "available": True}
    assert fr._find_method_definition("missing", method_data) is None


def test_get_forecast_method_availability_snapshot_reuses_shared_snapshot_builder(monkeypatch):
    monkeypatch.setattr(fr, "_ensure_registry_loaded", lambda: None)
    monkeypatch.setattr(
        fr,
        "_build_forecast_methods_snapshot",
        lambda: (
            [
                {"method": "theta", "available": True},
                {"method": "timesfm", "available": False},
                {"method": "", "available": True},
                "invalid",
            ],
            {"classic": ["theta"]},
        ),
    )

    assert fr.get_forecast_method_availability_snapshot() == {
        "theta": True,
        "timesfm": False,
    }
