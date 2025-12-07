import numpy as np

from mtdata.utils.patterns import _mass_distance_profile


def _brute_zdist(query: np.ndarray, window: np.ndarray) -> float:
    q = np.asarray(query, dtype=float)
    w = np.asarray(window, dtype=float)
    qz = (q - q.mean()) / q.std()
    wz = (w - w.mean()) / w.std()
    return float(np.linalg.norm(qz - wz))


def test_mass_distance_profile_matches_bruteforce():
    series = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 7.0])
    query = np.array([2.0, 4.0, 3.0])

    dist_profile = _mass_distance_profile(query, series)
    assert dist_profile.shape[0] == len(series) - len(query) + 1

    expected = []
    for i in range(len(series) - len(query) + 1):
        window = series[i : i + len(query)]
        expected.append(_brute_zdist(query, window))

    assert np.allclose(dist_profile, expected, atol=1e-6)


def test_mass_distance_profile_handles_constant_query():
    series = np.arange(10, dtype=float)
    query = np.ones(4, dtype=float)

    dist_profile = _mass_distance_profile(query, series)
    assert dist_profile.shape[0] == len(series) - len(query) + 1
    assert np.all(np.isinf(dist_profile))
