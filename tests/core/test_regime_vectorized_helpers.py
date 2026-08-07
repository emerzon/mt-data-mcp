from __future__ import annotations

import numpy as np

from mtdata.core.regime.api import _rolling_band_energy, _rolling_prefix_std


def test_rolling_prefix_std_matches_inclusive_legacy_windows() -> None:
    values = np.array([1.0, 3.0, 2.0, 8.0, 5.0, 9.0])
    lookback = 3
    expected = np.array(
        [
            np.std(values[max(0, index - lookback) : index + 1])
            for index in range(len(values))
        ]
    )

    np.testing.assert_allclose(_rolling_prefix_std(values, lookback), expected)


def test_rolling_band_energy_matches_leading_partial_windows() -> None:
    bands = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.0, 0.0, -2.0, 1.0]),
    ]
    window = 3
    expected = np.column_stack(
        [
            [
                np.mean(np.square(band[max(0, index - window + 1) : index + 1]))
                for index in range(len(band))
            ]
            for band in bands
        ]
    )

    np.testing.assert_allclose(_rolling_band_energy(bands, window), expected)


def test_rolling_band_energy_handles_no_bands() -> None:
    assert _rolling_band_energy([], 5).shape == (0, 0)
