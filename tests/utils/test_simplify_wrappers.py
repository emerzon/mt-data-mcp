"""Tests for canonical simplify helpers in utils.simplify."""

from mtdata.utils.simplify import (
    _point_line_distance,
    _max_line_error,
    _rdp_select_indices,
    _pla_select_indices,
    _apca_select_indices,
    _choose_simplify_points,
    _default_target_points,
    _rdp_autotune_epsilon,
    _pla_autotune_max_error,
    _apca_autotune_max_error,
    _select_indices_for_timeseries,
)


class TestPointLineDistance:
    def test_on_line(self):
        d = _point_line_distance(1.0, 1.0, 0.0, 0.0, 2.0, 2.0)
        assert abs(d) < 1e-10

    def test_perpendicular(self):
        d = _point_line_distance(1.0, 1.0, 0.0, 0.0, 2.0, 0.0)
        assert abs(d - 1.0) < 1e-10


class TestMaxLineError:
    def test_linear(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 2.0, 3.0]
        err = _max_line_error(x, y, 0, 3)
        assert err < 1e-10

    def test_with_deviation(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 5.0, 0.0, 3.0]
        err = _max_line_error(x, y, 0, 3)
        assert err > 0


class TestRdpSelectIndices:
    def test_simple_line(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 0.0, 0.0, 0.0, 0.0]  # flat line
        indices = _rdp_select_indices(x, y, epsilon=0.1)
        assert 0 in indices
        assert 4 in indices
        assert len(indices) == 2  # just endpoints for flat line

    def test_preserves_spike(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 0.0, 10.0, 0.0, 0.0]  # big spike at index 2
        indices = _rdp_select_indices(x, y, epsilon=0.1)
        assert 2 in indices


class TestPlaSelectIndices:
    def test_basic(self):
        x = list(range(20))
        y = [float(i) for i in range(20)]
        indices = _pla_select_indices(x, y, segments=3)
        assert 0 in indices
        assert 19 in indices
        assert len(indices) >= 2

    def test_with_max_error(self):
        x = list(range(10))
        y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        indices = _pla_select_indices(x, y, max_error=0.01)
        assert len(indices) >= 2


class TestApcaSelectIndices:
    def test_basic(self):
        y = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0]
        indices = _apca_select_indices(y, segments=3)
        assert 0 in indices
        assert len(indices) >= 2


class TestChooseSimplifyPoints:
    def test_with_points(self):
        assert _choose_simplify_points(100, {"points": 50}) == 50

    def test_with_ratio(self):
        result = _choose_simplify_points(100, {"ratio": 0.5})
        assert result == 50

    def test_default(self):
        result = _choose_simplify_points(100, {})
        assert result > 0


class TestDefaultTargetPoints:
    def test_small(self):
        result = _default_target_points(10)
        assert 1 <= result <= 10

    def test_large(self):
        result = _default_target_points(10000)
        assert result < 10000


class TestRdpAutotuneEpsilon:
    def test_basic(self):
        x = list(range(50))
        y = [float(i**2) for i in range(50)]
        indices, eps = _rdp_autotune_epsilon(x, y, target_points=10)
        assert len(indices) > 0
        assert eps > 0


class TestPlaAutotuneMaxError:
    def test_basic(self):
        x = list(range(50))
        y = [float(i**2) for i in range(50)]
        indices, err = _pla_autotune_max_error(x, y, target_points=10)
        assert len(indices) > 0
        assert err >= 0


class TestApcaAutotuneMaxError:
    def test_basic(self):
        y = [float(i**2) for i in range(50)]
        indices, err = _apca_autotune_max_error(y, target_points=10)
        assert len(indices) > 0
        assert err >= 0


class TestSelectIndicesForTimeseries:
    def test_rdp_method(self):
        x = list(range(50))
        y = [float(i**2) for i in range(50)]
        indices, method, meta = _select_indices_for_timeseries(
            x, y, {"method": "rdp", "points": 10}
        )
        assert len(indices) > 0
        assert method == "rdp"

    def test_default_method(self):
        x = list(range(50))
        y = [float(i**2) for i in range(50)]
        indices, method, meta = _select_indices_for_timeseries(x, y, {"points": 10})
        assert len(indices) > 0

    def test_none_spec(self):
        x = list(range(20))
        y = [float(i) for i in range(20)]
        indices, method, meta = _select_indices_for_timeseries(x, y, None)
        assert len(indices) > 0
