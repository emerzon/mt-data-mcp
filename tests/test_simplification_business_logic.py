import pandas as pd

from mtdata.services import simplification as simp


def _sample_ohlc_df():
    epochs = [0, 1800, 3600, 5400, 7200]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(epochs, unit="s"),
            "__epoch": epochs,
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [3.0, 4.0, 5.0, 6.0, 7.0],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [2.0, 3.0, 4.0, 5.0, 6.0],
            "volume": [10, 5, 7, 8, 9],
        }
    )


def test_simplify_dataframe_rows_ext_empty_dataframe_returns_none_meta():
    empty = pd.DataFrame(columns=["time", "close"])

    result_df, meta = simp._simplify_dataframe_rows_ext(empty, ["time", "close"], {"mode": "select"})

    assert result_df.empty
    assert meta is None


def test_simplify_dataframe_rows_ext_approximate_falls_back_to_select(monkeypatch):
    called = {"count": 0}

    def fake_select(df, headers, spec):
        called["count"] += 1
        return df.iloc[:1].copy(), {"mode": "select"}

    monkeypatch.setattr(simp, "_handle_select", fake_select)
    df = _sample_ohlc_df()

    result_df, meta = simp._simplify_dataframe_rows_ext(df, list(df.columns), {"mode": "approximate"})

    assert called["count"] == 1
    assert len(result_df) == 1
    assert meta == {"mode": "select"}


def test_handle_resample_missing_rule_returns_error_metadata():
    df = _sample_ohlc_df()

    result_df, meta = simp._handle_resample(df, list(df.columns), {})

    assert result_df.equals(df)
    assert meta == {"error": "Missing rule for resample"}


def test_handle_resample_aggregates_ohlc_and_volume():
    df = _sample_ohlc_df()
    headers = ["open", "high", "low", "close", "volume"]

    result_df, meta = simp._handle_resample(df, headers, {"rule": "1h"})

    assert len(result_df) == 3
    assert result_df.iloc[0]["open"] == 1.0
    assert result_df.iloc[0]["high"] == 4.0
    assert result_df.iloc[0]["low"] == 0.5
    assert result_df.iloc[0]["close"] == 3.0
    assert result_df.iloc[0]["volume"] == 15
    assert meta == {"mode": "resample", "rule": "1h", "rows": 3}


def test_handle_select_returns_subset_and_metadata(monkeypatch):
    df = _sample_ohlc_df()

    monkeypatch.setattr(simp, "_choose_simplify_points", lambda original_count, spec: 3)
    monkeypatch.setattr(
        simp,
        "_select_indices_for_timeseries",
        lambda epochs, series, spec: ([0, 2, 4], "lttb", {"bucket_size": 2}),
    )

    result_df, meta = simp._handle_select(df, list(df.columns), {"mode": "select", "points": 3})

    assert list(result_df.index) == [0, 2, 4]
    assert meta == {
        "mode": "select",
        "method": "lttb",
        "original_rows": 5,
        "returned_rows": 3,
        "points": 3,
        "bucket_size": 2,
    }


def test_handle_select_returns_original_when_series_unavailable(monkeypatch):
    df = pd.DataFrame({"time": ["t1", "t2", "t3"], "__epoch": [1, 2, 3]})

    monkeypatch.setattr(simp, "_choose_simplify_points", lambda original_count, spec: 2)

    result_df, meta = simp._handle_select(df, ["time"], {"points": 2})

    assert result_df.equals(df)
    assert meta is None
