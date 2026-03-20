"""Extended tests for mtdata.utils.simplify covering uncovered lines.

Targets: _select_indices_for_timeseries (rdp/pla/apca dispatch, fallback),
         _handle_select_mode, _simplify_dataframe_rows_ext, _simplify_dataframe_rows,
         _handle_encode_mode, _handle_segment_mode, _handle_symbolic_mode,
         _rdp_autotune_epsilon, _pla_autotune_max_error, _apca_autotune_max_error,
         _choose_simplify_points, _default_target_points, _fallback_lttb_indices,
         _finalize_indices, _point_line_distance, _pla_select_indices,
         _apca_select_indices.
"""

import numpy as np
import pandas as pd

from mtdata.utils.simplify import (
    _choose_simplify_points,
    _default_target_points,
    _rdp_autotune_epsilon,
    _pla_select_indices,
    _pla_autotune_max_error,
    _apca_select_indices,
    _apca_autotune_max_error,
    _select_indices_for_timeseries,
    _handle_select_mode,
    _handle_encode_mode,
    _handle_segment_mode,
    _handle_symbolic_mode,
    _simplify_dataframe_rows_ext,
    _simplify_dataframe_rows,
    _segment_endpoints_to_indices,
    _n_bkps_from_segments_points,
    _max_line_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
np.random.seed(42)
_N = 200
_X = np.linspace(0, 10, _N).tolist()
_Y = (np.sin(np.linspace(0, 4 * np.pi, _N)) * 50 + 100).tolist()


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    epochs = np.arange(n, dtype=float) * 60
    close = np.cumsum(rng.randn(n)) + 100.0
    return pd.DataFrame(
        {
            "time": pd.to_datetime(epochs, unit="s").astype(str),
            "__epoch": epochs,
            "close": close,
            "open": close + rng.randn(n) * 0.1,
            "high": close + abs(rng.randn(n)),
            "low": close - abs(rng.randn(n)),
        }
    )


# ===== _select_indices_for_timeseries: RDP branch =====
class TestSelectIndicesRDP:
    def test_rdp_with_epsilon(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "epsilon": 5.0}
        )
        assert method == "rdp"
        assert 0 in idxs and (_N - 1) in idxs
        assert len(idxs) < _N

    def test_rdp_with_target_points(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "points": 30}
        )
        assert method == "rdp"
        assert meta.get("auto_tuned") is True

    def test_rdp_with_ratio(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "ratio": 0.2}
        )
        assert method == "rdp"
        assert len(idxs) <= _N

    def test_rdp_fallback_to_lttb(self):
        """When epsilon is missing and target >= len, falls through to lttb fallback."""
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "points": _N + 10}
        )
        assert method == "lttb"
        assert "fallback" in meta


# ===== _select_indices_for_timeseries: PLA branch =====
class TestSelectIndicesPLA:
    def test_pla_with_max_error(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "max_error": 10.0}
        )
        assert method == "pla"
        assert 0 in idxs

    def test_pla_with_segments(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "segments": 10}
        )
        assert method == "pla"

    def test_pla_with_target_points(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "points": 20}
        )
        assert method == "pla"
        assert meta.get("auto_tuned") is True

    def test_pla_fallback_to_lttb(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "points": _N + 5}
        )
        assert method == "lttb"
        assert "fallback" in meta


# ===== _select_indices_for_timeseries: APCA branch =====
class TestSelectIndicesAPCA:
    def test_apca_with_max_error(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "max_error": 15.0}
        )
        assert method == "apca"

    def test_apca_with_segments(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "segments": 8}
        )
        assert method == "apca"

    def test_apca_with_target_points(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "points": 25}
        )
        assert method == "apca"
        assert meta.get("auto_tuned") is True

    def test_apca_fallback_to_lttb(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "points": _N + 5}
        )
        assert method == "lttb"
        assert "fallback" in meta


# ===== Unknown method fallback =====
class TestSelectIndicesUnknown:
    def test_unknown_method_fallback_lttb(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "bogus_method", "points": 30}
        )
        assert method == "lttb"
        assert "fallback" in meta
        assert "bogus_method" in meta["fallback"]

    def test_none_spec(self):
        idxs, method, meta = _select_indices_for_timeseries(_X, _Y, None)
        assert method == "none"
        assert len(idxs) == _N


# ===== _handle_select_mode =====
class TestHandleSelectMode:
    def test_basic(self):
        df = _make_df(100)
        out_df, meta = _handle_select_mode(df, ["time", "close"], {"points": 30})
        assert meta is not None
        assert meta["mode"] == "select"
        assert len(out_df) <= 100

    def test_short_df_passthrough(self):
        df = _make_df(2)
        out_df, meta = _handle_select_mode(df, ["time", "close"], {"points": 1})
        assert meta is None
        assert len(out_df) == 2

    def test_no_close_column(self):
        df = pd.DataFrame(
            {
                "time": ["a", "b", "c", "d"],
                "__epoch": [0, 1, 2, 3],
                "price": [1, 2, 3, 4],
            }
        )
        out_df, meta = _handle_select_mode(df, ["time", "price"], {"points": 2})
        assert meta is not None or len(out_df) <= 4

    def test_no_numeric_columns_returns_original(self):
        df = pd.DataFrame({"time": ["a", "b", "c", "d"], "__epoch": [0, 1, 2, 3]})
        out_df, meta = _handle_select_mode(df, ["time"], {"points": 2})
        assert meta is None


# ===== _simplify_dataframe_rows_ext =====
class TestSimplifyDataframeRowsExt:
    def test_empty_df(self):
        df = pd.DataFrame()
        out, meta = _simplify_dataframe_rows_ext(df, [], None)
        assert meta is None

    def test_select_mode(self):
        df = _make_df(80)
        out, meta = _simplify_dataframe_rows_ext(
            df, ["time", "close"], {"mode": "select", "points": 20}
        )
        assert meta is not None
        assert meta["mode"] == "select"

    def test_encode_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(
            df, ["time", "close"], {"mode": "encode"}
        )
        assert meta is not None
        assert meta["mode"] == "encode"

    def test_segment_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(
            df, ["time", "close"], {"mode": "segment"}
        )
        assert meta is not None
        assert meta["mode"] == "segment"

    def test_symbolic_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(
            df, ["time", "close"], {"mode": "symbolic"}
        )
        assert meta is not None
        assert meta["mode"] == "symbolic"

    def test_resample_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(
            df, ["time", "close"], {"mode": "resample", "rule": "5min"}
        )
        assert meta is not None


# ===== _simplify_dataframe_rows (main dispatcher) =====
class TestSimplifyDataframeRows:
    def test_none_simplify(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows(df, ["time", "close"], None)
        assert meta is None
        assert len(out) == 50

    def test_small_df(self):
        df = _make_df(3)
        out, meta = _simplify_dataframe_rows(df, ["time", "close"], {"method": "lttb"})
        assert meta is None

    def test_select_mode_dispatch(self):
        df = _make_df(100)
        out, meta = _simplify_dataframe_rows(
            df, ["time", "close"], {"method": "lttb", "points": 20}
        )
        assert meta is not None

    def test_encode_method_promotes_to_mode(self):
        df = _make_df(100)
        out, meta = _simplify_dataframe_rows(
            df, ["time", "close"], {"method": "encode"}
        )
        assert meta is not None
        assert meta.get("mode") == "encode"

    def test_symbolic_method_promotes_to_mode(self):
        df = _make_df(100)
        out, meta = _simplify_dataframe_rows(
            df, ["time", "close"], {"method": "symbolic"}
        )
        assert meta is not None
        assert meta.get("mode") == "symbolic"

    def test_segment_method_promotes_to_mode(self):
        df = _make_df(100)
        out, meta = _simplify_dataframe_rows(
            df, ["time", "close"], {"method": "segment"}
        )
        assert meta is not None
        assert meta.get("mode") == "segment"

    def test_resample_mode_with_epoch(self):
        df = _make_df(100)
        out, meta = _simplify_dataframe_rows(
            df,
            ["time", "close", "open", "high", "low"],
            {"mode": "resample", "method": "lttb", "bucket_seconds": 600},
        )
        assert meta is not None
        assert meta.get("mode") == "resample"
        assert len(out) < 100


# ===== _handle_encode_mode =====
class TestHandleEncodeMode:
    def test_delta_encoding(self):
        df = _make_df(50)
        out, meta = _handle_encode_mode(df, ["time", "close"], {"schema": "delta"})
        assert "encoding" in out.columns
        assert meta["schema"] == "delta"

    def test_envelope_encoding(self):
        df = _make_df(50)
        out, meta = _handle_encode_mode(df, ["time", "close"], {"schema": "envelope"})
        assert "start=" in out["encoding"].iloc[0]

    def test_delta_as_chars(self):
        df = _make_df(50)
        out, meta = _handle_encode_mode(
            df, ["time", "close"], {"schema": "delta", "as_chars": True}
        )
        enc = out["encoding"].iloc[0]
        assert all(c in "+-0" for c in enc)

    def test_no_numeric_column(self):
        df = pd.DataFrame({"time": ["a", "b", "c", "d"], "text": ["x", "y", "z", "w"]})
        out, meta = _handle_encode_mode(df, ["time", "text"], {})
        assert "error" in meta


# ===== _handle_segment_mode =====
class TestHandleSegmentMode:
    def test_basic_zigzag(self):
        df = _make_df(100)
        out, meta = _handle_segment_mode(df, ["time", "close"], {"threshold_pct": 0.01})
        assert meta["algo"] == "zigzag"
        assert len(out) < 100

    def test_no_value_col(self):
        df = pd.DataFrame({"time": ["a", "b"], "x": [1, 2]})
        out, meta = _handle_segment_mode(df, ["time", "x"], {})
        # Falls back to select mode
        assert out is not None

    def test_small_vals(self):
        df = pd.DataFrame({"time": ["a", "b"], "close": [1.0, 2.0]})
        out, meta = _handle_segment_mode(df, ["time", "close"], {})
        assert meta["mode"] == "segment"


# ===== _handle_symbolic_mode =====
class TestHandleSymbolicMode:
    def test_basic_sax(self):
        df = _make_df(100)
        out, meta = _handle_symbolic_mode(df, ["time", "close"], {"paa": 10})
        assert "symbolic" in out.columns
        assert meta["mode"] == "symbolic"

    def test_no_znorm(self):
        df = _make_df(50)
        out, meta = _handle_symbolic_mode(
            df, ["time", "close"], {"paa": 5, "znorm": False}
        )
        assert len(out["symbolic"].iloc[0]) == 5

    def test_no_numeric_col(self):
        df = pd.DataFrame({"time": ["a", "b"], "text": ["x", "y"]})
        out, meta = _handle_symbolic_mode(df, ["time", "text"], {})
        assert "error" in meta


# ===== Autotune helpers =====
class TestAutotune:
    def test_rdp_autotune(self):
        idxs, eps = _rdp_autotune_epsilon(_X, _Y, 30)
        assert 0 in idxs and (_N - 1) in idxs
        assert eps >= 0

    def test_rdp_autotune_target_too_large(self):
        idxs, eps = _rdp_autotune_epsilon(_X, _Y, _N + 5)
        assert len(idxs) == _N
        assert eps == 0.0

    def test_pla_autotune(self):
        idxs, me = _pla_autotune_max_error(_X, _Y, 20)
        assert 0 in idxs and (_N - 1) in idxs

    def test_pla_autotune_target_too_large(self):
        idxs, me = _pla_autotune_max_error(_X, _Y, _N + 5)
        assert len(idxs) == _N

    def test_apca_autotune(self):
        idxs, me = _apca_autotune_max_error(_Y, 20)
        assert 0 in idxs

    def test_apca_autotune_target_too_large(self):
        idxs, me = _apca_autotune_max_error(_Y, _N + 5)
        assert len(idxs) == _N


# ===== PLA / APCA direct =====
class TestPlaApca:
    def test_pla_max_error(self):
        idxs = _pla_select_indices(_X, _Y, max_error=5.0)
        assert 0 in idxs

    def test_pla_segments(self):
        idxs = _pla_select_indices(_X, _Y, segments=10)
        assert len(idxs) >= 2

    def test_pla_short(self):
        assert _pla_select_indices([0], [1]) == [0]

    def test_apca_max_error(self):
        idxs = _apca_select_indices(_Y, max_error=10.0)
        assert 0 in idxs

    def test_apca_segments(self):
        idxs = _apca_select_indices(_Y, segments=8)
        assert len(idxs) >= 2

    def test_apca_short(self):
        assert _apca_select_indices([0]) == [0]

    def test_apca_no_params(self):
        idxs = _apca_select_indices(_Y)
        assert len(idxs) == _N


# ===== Misc helpers =====
class TestMiscHelpers:
    def test_n_bkps_from_segments(self):
        assert _n_bkps_from_segments_points(100, 5, None) == 4
        assert _n_bkps_from_segments_points(100, None, 10) == 8

    def test_n_bkps_none(self):
        assert _n_bkps_from_segments_points(100, None, None) is None

    def test_segment_endpoints(self):
        idxs = _segment_endpoints_to_indices(10, [3, 7, 10])
        assert 0 in idxs and 9 in idxs

    def test_max_line_error_adjacent(self):
        assert _max_line_error([0, 1], [0, 1], 0, 1) == 0.0

    def test_max_line_error_nonzero(self):
        x = [0, 1, 2, 3]
        y = [0, 5, 0, 0]
        err = _max_line_error(x, y, 0, 3)
        assert err > 0

    def test_choose_simplify_points_ratio(self):
        n = _choose_simplify_points(1000, {"ratio": 0.1})
        assert n == 100

    def test_choose_simplify_points_empty_spec(self):
        assert _choose_simplify_points(1000, {}) == 1000

    def test_default_target_points_basic(self):
        t = _default_target_points(1000)
        assert 3 <= t <= 1000
