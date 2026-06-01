import numpy as np

from mtdata.utils import dtw as dtw_mod


def test_dtw_distance_fallback_prefers_tslearn_when_available(monkeypatch):
    captured = {}

    def fake_dtw(x, y, **kwargs):
        captured["x"] = x.copy()
        captured["y"] = y.copy()
        captured["kwargs"] = kwargs
        return 1.25

    monkeypatch.setattr(dtw_mod, "_get_tslearn_dtw", lambda: fake_dtw)

    result = dtw_mod.dtw_distance_fallback(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
        sakoe_chiba_radius=1,
    )

    assert result == 1.25
    np.testing.assert_array_equal(captured["x"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(captured["y"], np.array([2.0, 3.0, 4.0]))
    assert captured["kwargs"] == {
        "global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius": 1,
    }


def test_dtw_distance_fallback_keeps_python_fallback_on_backend_error(monkeypatch):
    def raising_backend(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(dtw_mod, "_get_tslearn_dtw", lambda: raising_backend)

    result = dtw_mod.dtw_distance_fallback(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert result == 0.0
