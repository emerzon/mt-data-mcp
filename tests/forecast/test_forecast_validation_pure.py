"""Tests for src/mtdata/forecast/forecast_validation.py"""
import pytest
import pandas as pd
import numpy as np

from mtdata.forecast.forecast_validation import (
    validate_horizon,
    validate_lookback,
    validate_ci_alpha,
    validate_quantity_target_combination,
    validate_denoise_spec,
    validate_features_spec,
    validate_dimred_spec,
    validate_target_spec,
    validate_data_sufficiency,
    validate_seasonality_for_method,
    create_error_response,
    safe_cast_numeric,
    sanitize_params,
    ForecastValidationError,
)


class TestValidateHorizon:
    def test_valid(self):
        assert validate_horizon(10) == []

    def test_zero(self):
        errors = validate_horizon(0)
        assert len(errors) > 0

    def test_negative(self):
        errors = validate_horizon(-5)
        assert len(errors) > 0

    def test_too_large(self):
        errors = validate_horizon(1001)
        assert len(errors) > 0

    def test_non_int(self):
        errors = validate_horizon(3.5)
        assert len(errors) > 0


class TestValidateLookback:
    def test_none_is_valid(self):
        assert validate_lookback(None) == []

    def test_valid(self):
        assert validate_lookback(100) == []

    def test_zero(self):
        errors = validate_lookback(0)
        assert len(errors) > 0

    def test_too_large(self):
        errors = validate_lookback(10001)
        assert len(errors) > 0


class TestValidateCiAlpha:
    def test_none_is_valid(self):
        assert validate_ci_alpha(None) == []

    def test_valid(self):
        assert validate_ci_alpha(0.05) == []

    def test_zero(self):
        errors = validate_ci_alpha(0.0)
        assert len(errors) > 0

    def test_one(self):
        errors = validate_ci_alpha(1.0)
        assert len(errors) > 0

    def test_negative(self):
        errors = validate_ci_alpha(-0.1)
        assert len(errors) > 0

    def test_non_numeric(self):
        errors = validate_ci_alpha("abc")
        assert len(errors) > 0


class TestValidateQuantityTargetCombination:
    def test_price_price(self):
        errors = validate_quantity_target_combination("price", "price", "linear")
        assert not any("Invalid quantity" in e for e in errors)

    def test_return_return(self):
        errors = validate_quantity_target_combination("return", "return", "linear")
        assert not any("Invalid quantity" in e for e in errors)

    def test_invalid_quantity(self):
        errors = validate_quantity_target_combination("foo", "price", "linear")
        assert any("Invalid quantity" in e for e in errors)

    def test_invalid_target(self):
        errors = validate_quantity_target_combination("price", "foo", "linear")
        assert any("Invalid target" in e for e in errors)

    def test_volatility_method_warns(self):
        errors = validate_quantity_target_combination("price", "price", "vol_garch")
        assert any("forecast_volatility" in e for e in errors)

    def test_volatility_quantity_warns(self):
        errors = validate_quantity_target_combination("volatility", "price", "linear")
        assert any("forecast_volatility" in e for e in errors)


class TestValidateDenoiseSpec:
    def test_none_is_valid(self):
        assert validate_denoise_spec(None) == []

    def test_not_dict(self):
        errors = validate_denoise_spec("ema")
        assert len(errors) > 0

    def test_missing_method(self):
        errors = validate_denoise_spec({})
        assert any("method" in e for e in errors)

    def test_valid_method(self):
        errors = validate_denoise_spec({"method": "ema"})
        assert errors == []

    def test_invalid_method(self):
        errors = validate_denoise_spec({"method": "unknown"})
        assert len(errors) > 0

    def test_valid_causality(self):
        errors = validate_denoise_spec({"method": "ema", "causality": "causal"})
        assert errors == []

    def test_invalid_causality(self):
        errors = validate_denoise_spec({"method": "ema", "causality": "bogus"})
        assert any("causality" in e for e in errors)

    def test_valid_when(self):
        errors = validate_denoise_spec({"method": "ema", "when": "pre_ti"})
        assert errors == []

    def test_invalid_when(self):
        errors = validate_denoise_spec({"method": "ema", "when": "bogus"})
        assert any("when" in e for e in errors)


class TestValidateFeaturesSpec:
    def test_none_is_valid(self):
        assert validate_features_spec(None) == []

    def test_not_dict(self):
        errors = validate_features_spec("features")
        assert len(errors) > 0

    def test_valid_ti(self):
        assert validate_features_spec({"ti": "rsi"}) == []

    def test_invalid_ti_type(self):
        errors = validate_features_spec({"ti": 42})
        assert len(errors) > 0

    def test_valid_exog_string(self):
        assert validate_features_spec({"exog": "symbol2"}) == []

    def test_empty_exog_string(self):
        errors = validate_features_spec({"exog": ""})
        assert len(errors) > 0

    def test_invalid_exog_type(self):
        errors = validate_features_spec({"exog": 42})
        assert len(errors) > 0


class TestValidateDimredSpec:
    def test_none_is_valid(self):
        assert validate_dimred_spec(None, None) == []

    def test_valid_pca(self):
        assert validate_dimred_spec("pca", {"n_components": 3}) == []

    def test_invalid_method(self):
        errors = validate_dimred_spec("invalid", None)
        assert len(errors) > 0

    def test_non_dict_params(self):
        errors = validate_dimred_spec("pca", "bad")
        assert len(errors) > 0

    def test_pca_invalid_n_components(self):
        errors = validate_dimred_spec("pca", {"n_components": -1})
        assert len(errors) > 0

    def test_tsne_invalid_n_components(self):
        errors = validate_dimred_spec("tsne", {"n_components": 5})
        assert len(errors) > 0

    def test_tsne_valid_n_components(self):
        assert validate_dimred_spec("tsne", {"n_components": 2}) == []

    def test_selectkbest_invalid_k(self):
        errors = validate_dimred_spec("selectkbest", {"k": 0})
        assert len(errors) > 0


class TestValidateTargetSpec:
    def test_none_is_valid(self):
        assert validate_target_spec(None) == []

    def test_not_dict(self):
        errors = validate_target_spec("bad")
        assert len(errors) > 0

    def test_valid_column(self):
        assert validate_target_spec({"column": "close"}) == []

    def test_empty_column(self):
        errors = validate_target_spec({"column": ""})
        assert len(errors) > 0

    def test_valid_transform(self):
        assert validate_target_spec({"transform": "log_return"}) == []

    def test_invalid_transform(self):
        errors = validate_target_spec({"transform": "bogus"})
        assert len(errors) > 0

    def test_valid_base(self):
        assert validate_target_spec({"base": "open"}) == []

    def test_empty_base(self):
        errors = validate_target_spec({"base": ""})
        assert len(errors) > 0


class TestValidateDataSufficiency:
    def test_enough_data(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
        assert validate_data_sufficiency(df, "close") == []

    def test_too_few_rows(self):
        df = pd.DataFrame({"close": [1.0]})
        errors = validate_data_sufficiency(df, "close")
        assert len(errors) > 0

    def test_missing_column(self):
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        errors = validate_data_sufficiency(df, "close")
        assert any("not found" in e for e in errors)

    def test_too_many_nans(self):
        df = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})
        errors = validate_data_sufficiency(df, "close")
        assert len(errors) > 0


class TestValidateSeasonalityForMethod:
    def test_seasonal_naive_no_seasonality(self):
        errors = validate_seasonality_for_method("seasonal_naive", None)
        assert len(errors) > 0

    def test_seasonal_naive_valid(self):
        assert validate_seasonality_for_method("seasonal_naive", 12) == []

    def test_holt_winters_negative(self):
        errors = validate_seasonality_for_method("holt_winters_add", -1)
        assert len(errors) > 0

    def test_non_seasonal_method(self):
        assert validate_seasonality_for_method("linear", None) == []


class TestCreateErrorResponse:
    def test_no_errors(self):
        result = create_error_response([])
        assert result["success"] is True

    def test_with_errors(self):
        result = create_error_response(["err1", "err2"])
        assert "error" in result
        assert result["error_count"] == 2


class TestSafeCastNumeric:
    def test_int(self):
        assert safe_cast_numeric("42", "x") == 42

    def test_float(self):
        assert safe_cast_numeric("3.14", "x") == 3.14  # int() fails, falls through to float

    def test_none(self):
        assert safe_cast_numeric(None, "x") is None

    def test_string(self):
        assert safe_cast_numeric("abc", "x") == "abc"


class TestSanitizeParams:
    def test_none(self):
        assert sanitize_params(None) == {}

    def test_casts_values(self):
        result = sanitize_params({"a": "5", "b": "hello"})
        assert result["a"] == 5
        assert result["b"] == "hello"
