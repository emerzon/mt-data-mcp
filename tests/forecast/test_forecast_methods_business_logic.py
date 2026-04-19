from __future__ import annotations

from mtdata.forecast import forecast_methods as fm


def test_method_metadata_lookup_helpers(monkeypatch):
    methods_data = {
        "methods": [
            {
                "method": "theta",
                "requires": ["numpy", "statsmodels"],
                "supports": {"price": True, "return": True, "volatility": False, "ci": False},
                "params": [{"name": "seasonality", "type": "int"}],
            },
            {
                "method": "mlf_rf",
                "requires": ["mlforecast", "scikit-learn"],
                "supports": {"price": True, "return": False, "volatility": False, "ci": False},
                "params": [{"name": "n_estimators", "type": "int"}],
            },
        ],
        "categories": {
            "classical": ["theta"],
            "ml": ["mlf_rf"],
        },
    }
    monkeypatch.setattr(fm, "_registry_methods_data", lambda: methods_data)
    monkeypatch.setattr(fm, "_get_registered_capabilities", lambda: [])

    assert fm.get_forecast_methods_data() == methods_data
    assert fm.get_method_category("theta") == "classical"
    assert fm.get_method_category("unknown") == "unknown"
    assert fm.get_method_requirements("mlf_rf") == ["mlforecast", "scikit-learn"]
    assert fm.get_method_requirements("none") == []
    assert fm.get_method_supports("theta")["price"] is True
    assert fm.get_method_supports("none") == {"price": False, "return": False, "volatility": False, "ci": False}
    assert fm.get_forecast_method_names() == ("theta", "mlf_rf")


def test_validate_method_params_type_rules(monkeypatch):
    methods_data = {
        "methods": [
            {
                "method": "m",
                "requires": [],
                "supports": {},
                "params": [
                    {"name": "int_p", "type": "int"},
                    {"name": "float_p", "type": "float"},
                    {"name": "bool_p", "type": "bool"},
                    {"name": "tuple_p", "type": "tuple"},
                ],
            }
        ],
        "categories": {},
    }
    monkeypatch.setattr(fm, "_registry_methods_data", lambda: methods_data)
    monkeypatch.setattr(fm, "_get_registered_capabilities", lambda: [])

    errors = fm.validate_method_params(
        "m",
        {
            "int_p": "abc",
            "float_p": "abc",
            "bool_p": "true",
            "tuple_p": [1, 2],
        },
    )
    assert "Parameter 'int_p' should be an integer" in errors
    assert "Parameter 'float_p' should be a float" in errors
    assert "Parameter 'bool_p' should be a boolean" in errors
    assert "Parameter 'tuple_p' should have 3 elements" in errors

    ok = fm.validate_method_params(
        "m",
        {
            "int_p": "2",
            "float_p": "2.5",
            "bool_p": True,
            "tuple_p": (1, 1, 1),
        },
    )
    assert ok == []

    unknown = fm.validate_method_params("nope", {})
    assert unknown == ["Unknown method: nope"]


def test_forecast_methods_snapshot_enriches_rows_with_capabilities(monkeypatch):
    methods_data = {
        "methods": [
            {
                "method": "theta",
                "available": True,
                "description": "Theta model.",
                "requires": [],
                "supports": {"price": True, "return": True, "volatility": False, "ci": False},
                "params": [{"name": "window", "type": "int"}],
            },
            {
                "method": "sf_theta",
                "available": True,
                "description": "StatsForecast theta.",
                "requires": ["statsforecast"],
                "supports": {"price": True, "return": True, "volatility": False, "ci": False},
                "params": [],
            },
        ],
        "categories": {
            "classical": ["theta"],
            "statsforecast": ["sf_theta"],
        },
    }
    monkeypatch.setattr(fm, "_registry_methods_data", lambda: methods_data)
    monkeypatch.setattr(
        fm,
        "_get_registered_capabilities",
        lambda: [
            {
                "method": "theta",
                "namespace": "native",
                "concept": "theta",
                "capability_id": "native:theta",
                "adapter_method": "theta",
                "selector": {"mode": "method"},
                "execution": {"library": "native", "method": "theta"},
                "display_name": "Theta",
                "aliases": ["theta-model"],
                "source": "registry",
                "supports": {"price": True, "return": True, "volatility": False, "ci": True},
            },
            {
                "method": "sf_theta",
                "namespace": "statsforecast",
                "concept": "theta",
                "capability_id": "statsforecast:theta",
                "adapter_method": "statsforecast",
                "selector": {"mode": "class_name", "key": "model_name", "value": "Theta"},
                "execution": {
                    "library": "statsforecast",
                    "method": "statsforecast",
                    "params": {"model_name": "Theta"},
                },
                "display_name": "Theta",
                "source": "registry",
                "supports": {"price": True, "return": True, "volatility": False, "ci": True},
            },
        ],
    )

    snapshot = fm.get_forecast_methods_snapshot()

    assert snapshot["methods_valid"] is True
    theta = next(row for row in snapshot["methods"] if row["method"] == "theta")
    assert theta["category"] == "classical"
    assert theta["method_id"] == "native:theta"
    assert theta["capability_id"] == "native:theta"
    assert theta["display_name"] == "Theta"
    assert theta["aliases"] == ["theta-model"]
    assert theta["supports_ci"] is True

    sf_theta = next(row for row in snapshot["methods"] if row["method"] == "sf_theta")
    assert sf_theta["category"] == "statsforecast"
    assert sf_theta["namespace"] == "statsforecast"
    assert sf_theta["execution"]["method"] == "statsforecast"
    assert sf_theta["selector"]["key"] == "model_name"
