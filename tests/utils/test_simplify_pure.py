"""Tests for utils/simplify.py — pure selection/simplification helpers."""
import numpy as np
import pytest

from mtdata.utils.simplify import (
    _apca_select_indices,
    _choose_simplify_points,
    _default_target_points,
    _fallback_lttb_indices,
    _finalize_indices,
    _lttb_select_indices,
    _max_line_error,
    _n_bkps_from_segments_points,
    _pla_select_indices,
    _point_line_distance,
    _rdp_keep_mask,
    _rdp_select_indices,
    _segment_endpoints_to_indices,
)


class TestDefaultTargetPoints:
    def test_small(self):
        result = _default_target_points(10)
        assert 3 <= result <= 10

    def test_large(self):
        result = _default_target_points(10000)
        assert result >= 3
        assert result <= 10000

    def test_min_3(self):
        result = _default_target_points(2)
        assert result >= 2  # can't exceed total


class TestChooseSimplifyPoints:
    def test_empty_spec(self):
        assert _choose_simplify_points(100, {}) == 100

    def test_points_key(self):
        assert _choose_simplify_points(100, {"points": 50}) == 50

    def test_max_points_key(self):
        assert _choose_simplify_points(100, {"max_points": 30}) == 30

    def test_target_points_key(self):
        assert _choose_simplify_points(100, {"target_points": 20}) == 20

    def test_ratio(self):
        result = _choose_simplify_points(100, {"ratio": 0.5})
        assert result == 50

    def test_ratio_below_3(self):
        result = _choose_simplify_points(100, {"ratio": 0.01})
        assert result >= 3

    def test_clamp_to_total(self):
        assert _choose_simplify_points(50, {"points": 200}) == 50

    def test_method_only_uses_default(self):
        result = _choose_simplify_points(200, {"method": "rdp"})
        assert 3 <= result <= 200


class TestFinalizeIndices:
    def test_basic(self):
        result = _finalize_indices(10, [0, 5, 9])
        assert result == [0, 5, 9]

    def test_adds_first_last(self):
        result = _finalize_indices(10, [3, 6])
        assert result[0] == 0
        assert result[-1] == 9

    def test_dedup(self):
        result = _finalize_indices(10, [0, 0, 5, 5, 9])
        assert result == [0, 5, 9]

    def test_empty_n(self):
        assert _finalize_indices(0, []) == []

    def test_empty_idxs(self):
        result = _finalize_indices(5, [])
        assert len(result) == 5


class TestSegmentEndpointsToIndices:
    def test_basic(self):
        result = _segment_endpoints_to_indices(10, [3, 7, 10])
        assert result[0] == 0
        assert result[-1] == 9

    def test_empty_bkps(self):
        result = _segment_endpoints_to_indices(5, [])
        assert result == [0, 4]


class TestNBkpsFromSegmentsPoints:
    def test_from_segments(self):
        assert _n_bkps_from_segments_points(100, 5, None) == 4

    def test_from_points(self):
        assert _n_bkps_from_segments_points(100, None, 10) == 8

    def test_none(self):
        assert _n_bkps_from_segments_points(100, None, None) is None


class TestFallbackLttbIndices:
    def test_basic(self):
        y = np.random.RandomState(42).randn(100)
        result = _fallback_lttb_indices(y, 20)
        assert result[0] == 0
        assert result[-1] == 99
        assert len(result) <= 25  # approximately target

    def test_nout_exceeds(self):
        y = np.arange(10, dtype=float)
        result = _fallback_lttb_indices(y, 20)
        assert result == list(range(10))

    def test_small(self):
        y = np.array([1.0, 2.0])
        result = _fallback_lttb_indices(y, 2)
        assert result == [0, 1]


class TestLttbSelectIndices:
    def test_basic(self):
        x = list(range(100))
        y = [float(i) for i in range(100)]
        result = _lttb_select_indices(x, y, 20)
        assert result[0] == 0
        assert result[-1] == 99

    def test_nout_exceeds(self):
        x = list(range(5))
        y = [float(i) for i in range(5)]
        assert _lttb_select_indices(x, y, 10) == list(range(5))


class TestPointLineDistance:
    def test_on_line(self):
        d = _point_line_distance(1.0, 1.0, 0.0, 0.0, 2.0, 2.0)
        assert abs(d) < 1e-6

    def test_off_line(self):
        d = _point_line_distance(1.0, 2.0, 0.0, 0.0, 2.0, 0.0)
        assert abs(d - 2.0) < 1e-6

    def test_vertical(self):
        d = _point_line_distance(7.0, 10.0, 5.0, 0.0, 5.0, 20.0)
        assert abs(d - 2.0) < 1e-6


class TestRdpKeepMask:
    def test_straight_line(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        mask = _rdp_keep_mask(x, y, 0.1)
        assert mask[0] and mask[-1]
        assert mask.sum() == 2  # only endpoints for perfect line

    def test_with_peak(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 0.0, 5.0, 0.0, 0.0]
        mask = _rdp_keep_mask(x, y, 0.1)
        assert mask[2]  # peak must be kept


class TestRdpSelectIndices:
    def test_basic(self):
        x = list(range(50))
        y = [float(i) for i in range(50)]
        result = _rdp_select_indices(x, y, 0.01)
        assert result[0] == 0
        assert result[-1] == 49
        assert len(result) == 2  # straight line: only endpoints

    def test_zero_epsilon(self):
        x = list(range(10))
        y = [0.0] * 10
        result = _rdp_select_indices(x, y, 0)
        assert result == list(range(10))


class TestMaxLineError:
    def test_straight(self):
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        assert _max_line_error(x, y, 0, 2) < 1e-6

    def test_with_deviation(self):
        x = [0.0, 1.0, 2.0]
        y = [0.0, 5.0, 0.0]
        assert _max_line_error(x, y, 0, 2) == 5.0


class TestPlaSelectIndices:
    def test_basic(self):
        rng = np.random.RandomState(42)
        x = list(range(100))
        y = list(np.cumsum(rng.randn(100)))
        result = _pla_select_indices(x, y, segments=5)
        assert result[0] == 0
        assert result[-1] == 99
        assert len(result) <= 10

    def test_short(self):
        assert _pla_select_indices([0.0, 1.0], [0.0, 1.0]) == [0, 1]


class TestApcaSelectIndices:
    def test_basic(self):
        rng = np.random.RandomState(42)
        y = list(np.cumsum(rng.randn(100)))
        result = _apca_select_indices(y, segments=5)
        assert result[0] == 0
        assert result[-1] == 99

    def test_short(self):
        assert _apca_select_indices([1.0, 2.0]) == [0, 1]
