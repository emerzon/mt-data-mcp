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

    assert fm.get_forecast_methods_data() == methods_data
    assert fm.get_method_category("theta") == "classical"
    assert fm.get_method_category("unknown") == "unknown"
    assert fm.get_method_requirements("mlf_rf") == ["mlforecast", "scikit-learn"]
    assert fm.get_method_requirements("none") == []
    assert fm.get_method_supports("theta")["price"] is True
    assert fm.get_method_supports("none") == {"price": False, "return": False, "volatility": False, "ci": False}


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
