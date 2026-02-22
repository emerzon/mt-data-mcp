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

    assert methods["good"]["description"] == "Good method summary."
    assert methods["good"]["params"] == [{"name": "window", "type": "int"}]
    assert methods["good"]["available"] is True

    assert methods["missing_params"]["available"] is False
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
