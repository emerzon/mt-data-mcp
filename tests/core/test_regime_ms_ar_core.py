import numpy as np

from mtdata.core.regime.methods.ms_ar.core import _ms_ar_reliability_from_smoothed


def test_ms_ar_reliability_counts_absorbing_rows_not_matrix_elements():
    smoothed = np.array(
        [
            [1.0, 0.0],
            [0.99995, 0.00005],
            [0.80, 0.20],
            [0.70, 0.30],
        ],
        dtype=float,
    )

    result = _ms_ar_reliability_from_smoothed(smoothed, {"n_states": 2, "order": 1})

    assert result["notes"] == "ok"
    assert result["n_states"] == 2
