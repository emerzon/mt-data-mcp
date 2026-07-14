from __future__ import annotations

import numpy as np
import pandas as pd

import mtdata.core.diagnostics as diagnostics


class _Gateway:
    def ensure_connection(self) -> None:
        return None


def _bars(close: np.ndarray, *, volume: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "time": np.arange(1_700_000_000, 1_700_000_000 + n * 3600, 3600),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "tick_volume": volume if volume is not None else np.full(n, 100.0),
            "real_volume": np.zeros(n),
        }
    )


def _session_bars(close: np.ndarray, *, bars_per_day: int) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=int(np.ceil(len(close) / bars_per_day)))
    timestamps = [
        day + pd.Timedelta(hours=14 + offset)
        for day in days
        for offset in range(bars_per_day)
    ][: len(close)]
    frame = _bars(close)
    frame["time"] = np.asarray([stamp.timestamp() for stamp in timestamps], dtype=float)
    return frame


def _raw(tool):
    return getattr(tool, "__wrapped__", tool)


def test_stationarity_test_combines_adf_and_kpss(monkeypatch):
    rng = np.random.default_rng(7)
    frame = _bars(100.0 + rng.normal(0.0, 1.0, 500))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.stationarity_test)(
        symbol="TEST",
        target="close",
        tests="adf,kpss",
    )

    assert result["success"] is True
    assert result["conclusion"] == "stationary"
    assert {row["test"] for row in result["items"]} == {"adf", "kpss"}


def test_clean_stationarity_warning_translates_kpss_lookup_warning():
    raw = (
        "The test statistic is outside of the range of p-values available in the "
        "look-up table. The actual p-value is smaller than the p-value returned."
    )
    cleaned = diagnostics._clean_stationarity_warning(raw)
    assert "KPSS p-value is approximate" in cleaned
    assert "smaller than the reported value" in cleaned
    # Raw statsmodels jargon should not leak.
    assert "look-up table" not in cleaned


def test_clean_stationarity_warning_passes_through_other_text():
    assert diagnostics._clean_stationarity_warning("some other note") == "some other note"


def test_seasonality_detect_finds_known_period(monkeypatch):
    x = np.arange(480, dtype=float)
    frame = _bars(100.0 + np.sin(2.0 * np.pi * x / 12.0))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.seasonality_detect)(
        symbol="TEST",
        target="close",
        min_period=4,
        max_period=30,
    )

    assert result["success"] is True
    assert result["dominant_period_bars"] == 12
    assert result["signal_quality"] in {"moderate", "strong"}
    assert "signal_quality" in result["items"][0]


def test_seasonality_detect_does_not_inflate_noise_spectral_score(monkeypatch):
    values = 100.0 + np.random.default_rng(42).normal(size=1000)
    frame = _bars(values)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.seasonality_detect)(
        symbol="TEST",
        target="close",
        min_period=4,
        max_period=50,
    )

    assert result["success"] is True
    assert max(row["spectral_strength"] for row in result["items"]) < 0.05
    assert max(row["score"] for row in result["items"]) < 0.15
    assert all(
        row["signal_quality"] in {"very_weak", "weak", "moderate"}
        for row in result["items"]
    )


def test_outliers_detect_flags_price_and_volume_spike(monkeypatch):
    close = np.linspace(100.0, 101.0, 120)
    close[80] = 130.0
    volume = np.full(120, 100.0)
    volume[80] = 5000.0
    frame = _bars(close, volume=volume)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.outliers_detect)(
        symbol="TEST",
        score_fields="return,volume",
        detail="full",
    )

    assert result["success"] is True
    assert result["outliers_total"] >= 1
    assert any("volume" in row["fields"] for row in result["items"])


def test_volatility_term_structure_returns_requested_horizons(monkeypatch):
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400)))
    frame = _bars(close)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.volatility_term_structure)(
        symbol="TEST",
        horizons="1,5,20",
    )

    assert result["success"] is True
    assert [row["horizon_bars"] for row in result["items"]] == [1, 5, 20]
    assert all("p50" in row["cone"] for row in result["items"])
    assert result["items"][0]["stability"] == "very_low"
    assert all("per_bar_volatility" in row for row in result["items"])
    assert result["comparable_to_options_iv"] is False
    assert result["analysis_kind"] == "historical_realized_volatility_cones"
    assert result["unit"] == "annualized_decimal_volatility"
    assert "0.01 means 1%" in result["unit_note"]
    assert result["units"]["current_volatility"] == "decimal_return_fraction"
    assert result["units"]["cone"] == "decimal_return_fraction"
    assert result["units"]["percentile_rank"] == "percentage_points (0-100)"
    assert result["bars_per_year"] == 6048.0
    assert result["bars_per_session"] == 24.0
    assert result["annualization_basis"] == "observed_median_bars_per_utc_session_x_252_sessions"


def test_volatility_term_structure_uses_observed_session_density(monkeypatch):
    rng = np.random.default_rng(12)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 420)))
    frame = _session_bars(close, bars_per_day=7)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.volatility_term_structure)(
        symbol="US500",
        timeframe="H1",
        horizons="1,5,20",
    )

    assert result["success"] is True
    assert result["bars_per_session"] == 7.0
    assert result["sessions_per_year"] == 252
    assert result["bars_per_year"] == 1764.0
    assert result["annualization_basis"] == "observed_median_bars_per_utc_session_x_252_sessions"
