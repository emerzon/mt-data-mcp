import numpy as np

from mtdata.forecast.methods.pretrained_helpers import validate_and_clean_data


def test_validate_and_clean_data_interpolates_internal_gaps():
    cleaned, error = validate_and_clean_data(np.array([1.0, np.nan, np.nan, 4.0]))

    assert error is None
    np.testing.assert_allclose(cleaned, np.array([1.0, 2.0, 3.0, 4.0]))


def test_validate_and_clean_data_edge_fills_missing_values():
    cleaned, error = validate_and_clean_data(np.array([np.nan, 2.0, np.nan, 6.0, np.nan]))

    assert error is None
    np.testing.assert_allclose(cleaned, np.array([2.0, 2.0, 4.0, 6.0, 6.0]))
