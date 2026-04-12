import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.mtdata.core.patterns_support as patterns_support_mod
import src.mtdata.patterns.candlestick as candlestick_mod
import src.mtdata.patterns.classic as classic_mod
import src.mtdata.services.data_service as data_service_mod
from src.mtdata.core import patterns as core_patterns
from src.mtdata.core.patterns import _apply_config_to_obj, _build_pattern_response
from src.mtdata.core.patterns_requests import PatternsDetectRequest
from src.mtdata.patterns.candlestick import (
    _extract_candlestick_rows,
    _get_candlestick_pattern_methods,
    _is_candlestick_allowed,
    _normalize_candlestick_name,
)
from src.mtdata.patterns.classic import (
    ClassicDetectorConfig,
    ClassicPatternResult,
    _count_recent_touches,
    _fit_lines_and_arrays,
    detect_classic_patterns,
)
from src.mtdata.utils.mt5 import MT5ConnectionError


def patterns_detect(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", True))
    request = kwargs.pop("request", None)
    if request is None:
        request = PatternsDetectRequest(**kwargs)
    return core_patterns.patterns_detect(request=request, __cli_raw=raw_output)


def test_patterns_detect_returns_connection_error_payload(monkeypatch):
    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(core_patterns, "ensure_mt5_connection_or_raise", fail_connection)
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda *args, **kwargs: pytest.fail("fetch should not run"))

    out = patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")

    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["success"] is False
    assert out["error_code"] == "tool_error"
    assert out["operation"] == "patterns_detect"
    assert isinstance(out.get("request_id"), str)


def test_fit_lines_and_arrays_uses_cfg_for_robust_fit(monkeypatch):
    calls = []

    def _fake_robust(x, y, cfg):
        calls.append(bool(cfg.use_robust_fit))
        # slope, intercept, r2
        return 1.0, 2.0, 0.9

    # Patch the implementation module since we refactored logic into classic_impl
    from src.mtdata.patterns.classic_impl import utils as impl_utils
    monkeypatch.setattr(impl_utils, "_fit_line_robust", _fake_robust)

    ih = np.array([1, 5, 9], dtype=int)
    il = np.array([2, 6, 10], dtype=int)
    c = np.linspace(10.0, 20.0, 12)
    cfg = ClassicDetectorConfig(use_robust_fit=True)

    sh, bh, r2h, sl, bl, r2l, upper, lower = _fit_lines_and_arrays(ih, il, c, len(c), cfg)

    assert calls == [True, True]
    assert (sh, bh, r2h) == (1.0, 2.0, 0.9)
    assert (sl, bl, r2l) == (1.0, 2.0, 0.9)
    assert upper.shape[0] == len(c)
    assert lower.shape[0] == len(c)


def test_candlestick_allowed_respects_whitelist_when_not_robust_only():
    robust_set = {"engulfing", "harami"}
    whitelist = {"engulfing"}

    assert _is_candlestick_allowed(
        "engulfing",
        robust_only=False,
        robust_set=robust_set,
        whitelist_set=whitelist,
    )
    assert not _is_candlestick_allowed(
        "doji",
        robust_only=False,
        robust_set=robust_set,
        whitelist_set=whitelist,
    )


def test_candlestick_allowed_requires_robust_when_enabled():
    robust_set = {"engulfing", "harami"}

    assert _is_candlestick_allowed(
        "engulfing",
        robust_only=True,
        robust_set=robust_set,
        whitelist_set=None,
    )
    assert not _is_candlestick_allowed(
        "doji",
        robust_only=True,
        robust_set=robust_set,
        whitelist_set=None,
    )


def test_candlestick_name_normalization_canonicalizes_common_variants():
    assert _normalize_candlestick_name("cdl_closing_marubozu") == "closingmarubozu"
    assert _is_candlestick_allowed(
        "closing_marubozu",
        robust_only=False,
        robust_set=set(),
        whitelist_set={"closingmarubozu"},
    )


def test_get_candlestick_pattern_methods_caches_discovery(monkeypatch):
    class _FakeTA:
        def __init__(self):
            self.dir_calls = 0

        def __dir__(self):
            self.dir_calls += 1
            return ["cdl_alpha", "not_a_pattern"]

        def __getattr__(self, name):
            if name == "cdl_alpha":
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    fake_temp = SimpleNamespace(ta=_FakeTA())
    monkeypatch.setattr(candlestick_mod, "_CANDLESTICK_PATTERN_METHOD_CACHE", None)

    out_first = _get_candlestick_pattern_methods(fake_temp)
    out_second = _get_candlestick_pattern_methods(fake_temp)

    assert out_first == ["cdl_alpha"]
    assert out_second == ["cdl_alpha"]
    assert fake_temp.ta.dir_calls == 1


def test_get_candlestick_pattern_methods_invalidates_cache_on_accessor_type_change(monkeypatch):
    class _AlphaTA:
        def __init__(self):
            self.dir_calls = 0

        def __dir__(self):
            self.dir_calls += 1
            return ["cdl_alpha"]

        def __getattr__(self, name):
            if name == "cdl_alpha":
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    class _BetaTA:
        def __init__(self):
            self.dir_calls = 0

        def __dir__(self):
            self.dir_calls += 1
            return ["cdl_beta"]

        def __getattr__(self, name):
            if name == "cdl_beta":
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    alpha_temp = SimpleNamespace(ta=_AlphaTA())
    beta_temp = SimpleNamespace(ta=_BetaTA())
    monkeypatch.setattr(candlestick_mod, "_CANDLESTICK_PATTERN_METHOD_CACHE", None)
    monkeypatch.setattr(candlestick_mod, "_CANDLESTICK_PATTERN_METHOD_CACHE_KEY", None)

    out_alpha = _get_candlestick_pattern_methods(alpha_temp)
    out_beta = _get_candlestick_pattern_methods(beta_temp)

    assert out_alpha == ["cdl_alpha"]
    assert out_beta == ["cdl_beta"]
    assert alpha_temp.ta.dir_calls == 1
    assert beta_temp.ta.dir_calls == 1


def test_extract_candlestick_rows_prefers_non_deprioritized_hits():
    df_tail = pd.DataFrame({"time": ["T0", "T1"]})
    temp_tail = pd.DataFrame(
        {
            "cdl_doji": [0.0, 100.0],
            "cdl_engulfing": [0.0, 100.0],
            "cdl_longline": [0.0, 80.0],
        }
    )

    rows = _extract_candlestick_rows(
        df_tail,
        temp_tail,
        ["cdl_doji", "cdl_engulfing", "cdl_longline"],
        threshold=0.95,
        robust_only=False,
        robust_set={"engulfing"},
        whitelist_set=None,
        min_gap=0,
        top_k=1,
        deprioritize={"doji", "longline"},
    )

    assert rows == [["T1", "Bullish ENGULFING"]]


def test_extract_candlestick_rows_includes_metrics_when_enabled():
    df_tail = pd.DataFrame({"time": ["T0", "T1"], "close": [100.0, 101.5]})
    temp_tail = pd.DataFrame({"cdl_engulfing": [0.0, 100.0]})

    rows = _extract_candlestick_rows(
        df_tail,
        temp_tail,
        ["cdl_engulfing"],
        threshold=0.95,
        robust_only=False,
        robust_set={"engulfing"},
        whitelist_set=None,
        min_gap=0,
        top_k=1,
        deprioritize=set(),
        include_metrics=True,
    )

    assert len(rows) == 1
    assert rows[0][0] == "T1"
    assert rows[0][1] == "Bullish ENGULFING"
    assert rows[0][2] == "bullish"
    assert rows[0][3] == pytest.approx(0.95)
    assert rows[0][4:] == [100, 101.5, "T0", "T1", 2, 0, 1]


@contextmanager
def _always_ready_guard(*_args, **_kwargs):
    yield None, None


def test_detect_candlestick_patterns_rejects_out_of_range_min_strength():
    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=1.5,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
    )

    assert res == {"error": "min_strength must be between 0.0 and 1.0."}


def test_detect_candlestick_patterns_does_not_flatten_unexpected_errors(monkeypatch):
    monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", _always_ready_guard)
    monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", lambda *_a, **_k: [object()])
    monkeypatch.setattr(candlestick_mod, "_rates_to_df", lambda _rates: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})

    with pytest.raises(RuntimeError, match="boom"):
        candlestick_mod.detect_candlestick_patterns(
            symbol="EURUSD",
            timeframe="H1",
            limit=10,
            min_strength=0.95,
            min_gap=0,
            robust_only=False,
            whitelist=None,
            top_k=1,
        )


def test_detect_candlestick_patterns_prefilters_methods_by_whitelist(monkeypatch):
    calls = []

    class _FakeFrame(pd.DataFrame):
        _metadata = ["_calls"]

        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    frame._calls.append("cdl_alpha")
                    frame["cdl_alpha"] = [0.0, 100.0]

                def cdl_beta(self, append=True):
                    _ = append
                    frame._calls.append("cdl_beta")
                    frame["cdl_beta"] = [0.0, 100.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", _always_ready_guard)
    monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", lambda *_a, **_k: [object(), object()])
    monkeypatch.setattr(candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha", "cdl_beta"])

    def _fake_rates_to_df(_rates):
        frame = _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )
        frame._calls = calls
        return frame

    monkeypatch.setattr(candlestick_mod, "_rates_to_df", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist="alpha",
        top_k=1,
    )

    assert res["success"] is True
    assert calls == ["cdl_alpha"]
    assert res["min_strength"] == pytest.approx(0.95)
    assert res["strength_scale"] == "semantic_pattern_conviction_v2"
    assert res["signal_scale"] == "pandas_ta_signal_x100"


def test_detect_candlestick_patterns_top_k_uses_semantic_strength(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_doji(self, append=True):
                    _ = append
                    frame["cdl_doji"] = [0.0, 100.0]

                def cdl_engulfing(self, append=True):
                    _ = append
                    frame["cdl_engulfing"] = [0.0, 100.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", _always_ready_guard)
    monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", lambda *_a, **_k: [object(), object()])
    monkeypatch.setattr(candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_doji", "cdl_engulfing"])

    def _fake_rates_to_df(_rates):
        return _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )

    monkeypatch.setattr(candlestick_mod, "_rates_to_df", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.90,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
    )

    assert res["success"] is True
    assert res["data"][0]["pattern"] == "Bullish ENGULFING"
    assert res["data"][0]["confidence"] == pytest.approx(0.95)


def test_detect_candlestick_patterns_drops_still_forming_last_bar(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    values = [0.0] * len(frame)
                    values[-1] = 200.0
                    frame["cdl_alpha"] = values

            return _Accessor()

    now_ts = float(datetime.now(timezone.utc).timestamp())

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", _always_ready_guard)
    monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", lambda *_a, **_k: [object(), object(), object()])
    monkeypatch.setattr(candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"])

    def _fake_rates_to_df(_rates):
        return _FakeFrame(
            {
                "time": [now_ts - 7200.0, now_ts - 3600.0, now_ts - 1800.0],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
            }
        )

    monkeypatch.setattr(candlestick_mod, "_rates_to_df", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=2,
    )

    assert res["success"] is True
    assert len(res["data"]) == 1
    assert res["data"][0]["end_index"] == 1


def test_detect_candlestick_patterns_uses_broker_tick_reference_for_live_bar_trim(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    values = [0.0] * len(frame)
                    values[-1] = 200.0
                    frame["cdl_alpha"] = values

            return _Accessor()

    now_ts = float(datetime.now(timezone.utc).timestamp())
    last_bar_open = now_ts - 3660.0

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", _always_ready_guard)
    monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", lambda *_a, **_k: [object(), object(), object()])
    monkeypatch.setattr(candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"])

    def _fake_rates_to_df(_rates):
        return _FakeFrame(
            {
                "time": [last_bar_open - 7200.0, last_bar_open - 3600.0, last_bar_open],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
            }
        )

    monkeypatch.setattr(candlestick_mod, "_rates_to_df", _fake_rates_to_df)
    monkeypatch.setattr(
        data_service_mod,
        "_resolve_live_bar_reference_epoch",
        lambda *_a, **_k: last_bar_open + 120.0,
    )

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=2,
    )

    assert res["success"] is True
    assert len(res["data"]) == 1
    assert res["data"][0]["end_index"] == 1


def test_detect_candlestick_patterns_exposes_raw_signal_and_quality_warning(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    frame["cdl_alpha"] = [0.0, 200.0, 0.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", _always_ready_guard)
    monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", lambda *_a, **_k: [object(), object(), object()])
    monkeypatch.setattr(candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"])

    def _fake_rates_to_df(_rates):
        return _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0, 1_700_007_200.0],
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
            }
        )

    monkeypatch.setattr(candlestick_mod, "_rates_to_df", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
    )

    assert res["success"] is True
    assert res["data"][0]["raw_signal"] == 200
    assert "warnings" in res
    assert any("repeated close prices" in warning for warning in res["warnings"])


def test_detect_candlestick_patterns_adds_volume_and_regime_enrichment(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    frame["cdl_alpha"] = [0.0] * (len(frame) - 1) + [200.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", _always_ready_guard)
    monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", lambda *_a, **_k: [object()] * 5)
    monkeypatch.setattr(candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"])

    def _fake_rates_to_df(_rates):
        rows = 25
        return _FakeFrame(
            {
                "time": [1_700_000_000.0 + (3600.0 * i) for i in range(rows)],
                "open": [100.0 + (0.3 * i) for i in range(rows)],
                "high": [100.5 + (0.3 * i) for i in range(rows)],
                "low": [99.8 + (0.3 * i) for i in range(rows)],
                "close": [100.0 + (0.35 * i) for i in range(rows)],
                "tick_volume": [100 + (3 * i) for i in range(rows - 1)] + [260],
            }
        )

    monkeypatch.setattr(candlestick_mod, "_rates_to_df", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=20,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
        config={
            "use_volume_confirmation": True,
            "use_regime_context": True,
            "volume_confirm_min_ratio": 1.1,
        },
    )

    row = res["data"][0]
    assert row["volume_confirmation"]["status"] == "confirmed"
    assert row["regime_context"]["status"] == "aligned"
    assert row["end_index"] == 24


def test_attach_candlestick_volume_confirmation_uses_full_multibar_signal_window():
    row = {"start_index": 20, "end_index": 22, "confidence": 0.4}
    volume = np.full(30, 100.0, dtype=float)
    volume[20:23] = np.array([220.0, 90.0, 90.0], dtype=float)

    candlestick_mod._attach_candlestick_volume_confirmation(
        row,
        volume,
        "tick_volume",
        {"use_volume_confirmation": True, "volume_confirm_min_ratio": 1.1},
    )

    assert row["volume_confirmation"]["status"] == "confirmed"
    assert row["volume_confirmation"]["signal_avg_volume"] > 130.0
    assert row["volume_confirmation"]["signal_to_baseline_ratio"] > 1.1


def test_extract_candlestick_rows_respects_start_index():
    df = pd.DataFrame({"time": ["T0", "T1", "T2"], "close": [100.0, 101.0, 102.0]})
    temp = pd.DataFrame({"cdl_engulfing": [100.0, 100.0, 100.0]})
    rows = _extract_candlestick_rows(
        df,
        temp,
        ["cdl_engulfing"],
        threshold=0.95,
        robust_only=True,
        robust_set={"engulfing"},
        whitelist_set=None,
        min_gap=0,
        top_k=1,
        deprioritize=set(),
        include_metrics=True,
        start_index=2,
    )
    assert len(rows) == 1
    assert rows[0][0] == "T2"


def test_patterns_detect_candlestick_passes_last_n_bars(monkeypatch):
    captured = {}

    def _fake_detect(**kwargs):
        captured.update(kwargs)
        return {"success": True, "patterns": []}

    monkeypatch.setattr(core_patterns, "_detect_candlestick_patterns", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        timeframe="H1",
        mode="candlestick",
        detail="full",
        last_n_bars=8,
    )

    assert res.get("success") is True
    assert captured.get("last_n_bars") == 8


def test_patterns_detect_candlestick_passes_config(monkeypatch):
    captured = {}

    def _fake_detect(**kwargs):
        captured.update(kwargs)
        return {"success": True, "patterns": []}

    monkeypatch.setattr(core_patterns, "_detect_candlestick_patterns", _fake_detect)

    patterns_detect(
        symbol="EURUSD",
        timeframe="H1",
        mode="candlestick",
        detail="full",
        config={"use_volume_confirmation": False},
        __cli_raw=True,
    )

    assert captured["config"] == {"use_volume_confirmation": False}


def test_patterns_support_config_helpers_read_dict_values():
    config = {
        "use_volume_confirmation": False,
        "volume_confirm_breakout_bars": 4,
        "volume_confirm_min_ratio": 1.35,
    }

    assert patterns_support_mod._config_bool(config, "use_volume_confirmation", True) is False
    assert patterns_support_mod._config_int(config, "volume_confirm_breakout_bars", 2) == 4
    assert patterns_support_mod._config_float(config, "volume_confirm_min_ratio", 1.1) == pytest.approx(1.35)


def test_candlestick_volume_confirmation_respects_dict_disable():
    row = {"start_index": 0, "end_index": 1, "confidence": 0.5}

    candlestick_mod._attach_candlestick_volume_confirmation(
        row,
        np.array([100.0, 110.0, 120.0], dtype=float),
        "volume",
        {"use_volume_confirmation": False},
    )

    assert row["volume_confirmation"]["status"] == "disabled"


def test_candlestick_bar_spans_keep_three_bar_patterns_aligned():
    assert candlestick_mod._CANDLESTICK_PATTERN_BAR_SPANS["2crows"] == 3
    assert candlestick_mod._CANDLESTICK_PATTERN_BAR_SPANS["hikkake"] == 3
    assert candlestick_mod._CANDLESTICK_PATTERN_BAR_SPANS["hikkakemod"] == 3


def test_patterns_detect_candlestick_rejects_non_positive_last_n_bars():
    res = patterns_detect(
        symbol="EURUSD",
        timeframe="H1",
        mode="candlestick",
        detail="full",
        last_n_bars=0,
    )
    assert "error" in res
    assert "last_n_bars" in str(res["error"])


def test_detect_classic_uses_singular_pennant_name(monkeypatch):
    n = 110
    close = np.linspace(100.0, 130.0, n)
    window = 30
    seg = close[-window:].copy()
    seg_peaks = np.array([6, 13, 20, 27], dtype=int)
    seg_troughs = np.array([4, 11, 18, 25], dtype=int)

    top = np.linspace(150.0, 148.0, window)
    bot = np.linspace(144.0, 145.5, window)
    seg[:] = (top + bot) / 2.0
    seg[seg_peaks] = top[seg_peaks]
    seg[seg_troughs] = bot[seg_troughs]
    close[-window:] = seg

    df = pd.DataFrame(
        {"time": np.arange(n, dtype=float), "close": close, "high": close + 0.2, "low": close - 0.2}
    )

    def _fake_pivots(c, cfg, *args):
        n_local = len(c)
        if n_local >= 20:
            return np.array([2, 7, 13, n_local - 4], dtype=int), np.array([4, 10, 16, n_local - 2], dtype=int)
        return np.array([], dtype=int), np.array([], dtype=int)

    from src.mtdata.patterns.classic_impl import continuation
    monkeypatch.setattr(continuation, "_detect_pivots_close", _fake_pivots)

    def _fake_fit_lines(ih, il, c, n, cfg):
        if n >= 20:
            return (
                -0.07,
                150.0,
                0.9,
                0.05,
                144.0,
                0.9,
                np.linspace(150.0, 148.0, n),
                np.linspace(144.0, 145.5, n),
            )
        x = np.arange(n, dtype=float)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x.copy(), x.copy()

    monkeypatch.setattr(continuation, "_fit_lines_and_arrays", _fake_fit_lines)

    out = detect_classic_patterns(df, ClassicDetectorConfig(min_pole_return_pct=1.0, max_consolidation_bars=window))
    names = {p.name for p in out}
    assert "Bull Pennant" in names
    assert "Bull Pennants" not in names


def test_detect_flags_pennants_measure_pole_from_tip_not_last_bar(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 120
    window = 30
    close = np.full(n, 102.0, dtype=float)
    close[75:90] = np.linspace(100.0, 106.0, 15)
    seg = np.linspace(104.0, 103.0, window)
    seg[0] = 106.0
    seg[-1] = 101.8
    close[-window:] = seg

    peaks = np.array([4, 11, 18, 25], dtype=int)
    troughs = np.array([2, 9, 16, 23], dtype=int)
    top = np.linspace(106.0, 104.0, window)
    bot = np.linspace(102.0, 103.0, window)

    monkeypatch.setattr(continuation, "_detect_pivots_close", lambda *_args, **_kwargs: (peaks, troughs))
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (-0.05, 106.0, 0.9, 0.03, 102.0, 0.9, top.copy(), bot.copy()),
    )

    out = continuation.detect_flags_pennants(
        close,
        close + 0.2,
        close - 0.2,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(max_consolidation_bars=window, min_pole_return_pct=4.0),
    )

    assert out
    assert out[0].name == "Bull Pennant"
    assert out[0].details["pole_return_pct"] > 4.0
    assert out[0].details["pole_tip_price"] == pytest.approx(106.0)


def test_detect_flags_pennants_reject_protrend_consolidation(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 120
    window = 30
    close = np.full(n, 102.0, dtype=float)
    close[75:90] = np.linspace(100.0, 106.0, 15)
    seg = np.linspace(103.5, 105.8, window)
    seg[0] = 106.0
    close[-window:] = seg

    peaks = np.array([4, 11, 18, 25], dtype=int)
    troughs = np.array([2, 9, 16, 23], dtype=int)
    top = np.linspace(104.0, 105.6, window)
    bot = np.linspace(102.0, 103.6, window)

    monkeypatch.setattr(continuation, "_detect_pivots_close", lambda *_args, **_kwargs: (peaks, troughs))
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.06, 104.0, 0.9, 0.06, 102.0, 0.9, top.copy(), bot.copy()),
    )

    out = continuation.detect_flags_pennants(
        close,
        close + 0.2,
        close - 0.2,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(max_consolidation_bars=window, min_pole_return_pct=4.0),
    )

    assert out == []


def test_detect_classic_channel_parallel_ratio_uses_config(monkeypatch):
    n = 150
    close = np.linspace(100.0, 120.0, n)
    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    upper = 1.18 * np.arange(n, dtype=float) + 150.0
    lower = 1.00 * np.arange(n, dtype=float) + 120.0
    close[peaks] = upper[peaks]
    close[troughs] = lower[troughs]
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close, "high": close + 0.2, "low": close - 0.2})

    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs))
    monkeypatch.setattr(classic_mod, "_is_converging", lambda *args, **kwargs: False)

    def _fake_fit_lines(ih, il, c, n, cfg):
        return 1.18, 150.0, 0.95, 1.00, 120.0, 0.95, upper, lower

    monkeypatch.setattr(classic_mod, "_fit_lines_and_arrays", _fake_fit_lines)

    out_default = detect_classic_patterns(df, ClassicDetectorConfig(min_channel_touches=2, max_consolidation_bars=5))
    out_relaxed = detect_classic_patterns(
        df,
        ClassicDetectorConfig(min_channel_touches=2, max_consolidation_bars=5, channel_parallel_slope_ratio=0.2),
    )

    assert not any("Channel" in p.name for p in out_default)
    assert any("Channel" in p.name for p in out_relaxed)


def test_select_classic_engines_uses_registry(monkeypatch):
    monkeypatch.setitem(
        core_patterns._CLASSIC_ENGINE_REGISTRY,
        "unit_test",
        lambda symbol, df, cfg, config: ([{"name": "Unit Test"}], None),
    )

    engines, invalid = core_patterns._select_classic_engines("unit_test", ensemble=False)

    assert engines == ["unit_test"]
    assert invalid == []


def test_apply_config_to_obj_coerces_bool_strings():
    cfg = ClassicDetectorConfig()
    assert cfg.use_robust_fit is True

    _apply_config_to_obj(cfg, {"use_robust_fit": "false", "use_dtw_check": "0"})
    assert cfg.use_robust_fit is False
    assert cfg.use_dtw_check is False

    _apply_config_to_obj(cfg, {"use_robust_fit": "true", "use_dtw_check": "yes"})
    assert cfg.use_robust_fit is True
    assert cfg.use_dtw_check is True


def test_apply_config_to_obj_rejects_invalid_bool_strings():
    cfg = ClassicDetectorConfig()
    original = cfg.use_robust_fit

    invalid = _apply_config_to_obj(cfg, {"use_robust_fit": "maybe"})

    assert cfg.use_robust_fit is original
    assert invalid == ["use_robust_fit"]


def test_apply_config_to_obj_rejects_invalid_type_coercion():
    cfg = ClassicDetectorConfig()
    original = cfg.min_distance

    invalid = _apply_config_to_obj(cfg, {"min_distance": "not-an-int"})

    assert cfg.min_distance == original
    assert invalid == ["min_distance"]


def test_build_pattern_response_include_completed_filter_behavior():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    patterns = [
        {"name": "A", "status": "forming", "confidence": 0.5},
        {"name": "B", "status": "completed", "confidence": 0.6},
    ]

    forming_only = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
    )
    with_completed = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=True,
        include_series=False,
        series_time="string",
        df=df,
    )

    assert forming_only["n_patterns"] == 1
    assert forming_only["patterns"][0]["status"] == "forming"
    assert forming_only["completed_patterns_hidden"] == 1
    assert "include_completed=true" in forming_only["note"]
    assert with_completed["n_patterns"] == 2


def test_build_pattern_response_elliott_hidden_completed_preview_is_truthful():
    df = pd.DataFrame({"time": [1, 2, 3, 4], "close": [10.0, 11.0, 12.0, 13.0]})
    patterns = [
        {
            "wave_type": "Correction",
            "status": "completed",
            "confidence": 0.71,
            "start_index": 0,
            "end_index": 1,
            "start_date": "2026-03-01 00:00",
            "end_date": "2026-03-02 00:00",
            "details": {"trend": "bear", "pattern_confirmed": True, "has_unconfirmed_terminal_pivot": False},
        },
        {
            "wave_type": "Impulse",
            "status": "completed",
            "confidence": 0.82,
            "start_index": 1,
            "end_index": 3,
            "start_date": "2026-03-02 00:00",
            "end_date": "2026-03-04 00:00",
            "details": {"trend": "bull", "pattern_confirmed": True, "has_unconfirmed_terminal_pivot": False},
        },
    ]

    res = _build_pattern_response(
        "EURUSD",
        "H4",
        100,
        "elliott",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
    )

    assert res["n_patterns"] == 0
    assert res["completed_patterns_hidden"] == 2
    assert "No forming Elliott Wave structures detected" in res["diagnostic"]
    assert "No valid Elliott Wave structures detected" not in res["diagnostic"]
    assert res["completed_patterns_preview"][0]["pattern"] == "Impulse"
    assert res["completed_patterns_preview"][0]["timeframe"] == "H4"
    assert "strongest hidden count" in res["note"]
    assert "include_completed=true" in res["note"]


def test_build_pattern_response_elliott_compact_keeps_hidden_completed_preview():
    df = pd.DataFrame({"time": [1, 2, 3, 4], "close": [10.0, 11.0, 12.0, 13.0]})
    patterns = [
        {
            "wave_type": "Impulse",
            "status": "forming",
            "confidence": 0.55,
            "start_index": 1,
            "end_index": 3,
            "start_date": "2026-03-02 00:00",
            "end_date": "2026-03-04 00:00",
            "details": {"trend": "bull"},
        },
        {
            "wave_type": "Correction",
            "status": "completed",
            "confidence": 0.77,
            "start_index": 0,
            "end_index": 1,
            "start_date": "2026-03-01 00:00",
            "end_date": "2026-03-02 00:00",
            "details": {"trend": "bear", "pattern_confirmed": True},
        },
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H4",
        100,
        "elliott",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["n_patterns"] == 1
    assert compact["completed_patterns_hidden"] == 1
    assert compact["completed_patterns_preview"][0]["pattern"] == "Correction"
    assert compact["completed_patterns_preview"][0]["timeframe"] == "H4"


def test_build_pattern_response_compact_detail_returns_summary():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    patterns = [
        {"name": "A", "status": "forming", "confidence": 0.5, "end_index": 1},
        {"name": "B", "status": "forming", "confidence": 0.7, "end_index": 2},
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["n_patterns"] == 2
    assert "recent_patterns" in compact
    assert "summary" in compact
    assert "patterns" not in compact


def test_build_pattern_response_compact_keeps_actionable_fields():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    patterns = [
        {
            "name": "Double Bottom",
            "status": "forming",
            "confidence": 0.85,
            "end_index": 2,
            "bias": "bullish",
            "reference_price": 12.0,
            "target_price": 13.2,
            "invalidation_price": 11.4,
            "price": 12.0,
        }
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    recent = compact["recent_patterns"][0]
    assert recent["bias"] == "bullish"
    assert "target_price" in recent
    assert "invalidation_price" in recent
    assert compact["summary"]["signal_bias"]["net_bias"] == "bullish"


def test_build_pattern_response_compact_adds_hint_when_rows_are_truncated():
    df = pd.DataFrame({"time": list(range(12)), "close": [100.0 + i for i in range(12)]})
    patterns = [
        {
            "name": f"Pattern {i}",
            "status": "forming",
            "confidence": 0.9 - (i * 0.01),
            "end_index": min(i, len(df) - 1),
        }
        for i in range(9)
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["summary"]["more_patterns"] == 1
    assert compact["show_all_hint"] == "Use --detail full to show all detected patterns."


def test_patterns_detect_elliott_without_timeframe_scans_default_subset(monkeypatch):
    monkeypatch.setattr(core_patterns, "TIMEFRAME_MAP", {"M1": 1, "H1": 2, "H4": 3, "D1": 4})

    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    def _fake_fetch(symbol, timeframe, limit, denoise):
        return df.copy(), None

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", _fake_fetch)

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Impulse",
                confidence=0.91,
                start_index=0,
                end_index=5,
                start_time=1.0,
                end_time=6.0,
                details={"score": 0.8},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe=None,
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["timeframe"] == "ALL"
    assert res["scanned_timeframes"] == ["H1", "H4", "D1"]
    assert res["n_patterns"] == 3
    assert len(res["findings"]) == 3
    assert {p["timeframe"] for p in res["patterns"]} == {"H1", "H4", "D1"}


def test_patterns_detect_elliott_scan_timeframes_can_override_default(monkeypatch):
    monkeypatch.setattr(core_patterns, "TIMEFRAME_MAP", {"M15": 1, "H1": 2, "H4": 3, "D1": 4})

    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    def _fake_fetch(symbol, timeframe, limit, denoise):
        return df.copy(), None

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", _fake_fetch)

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Impulse",
                confidence=0.88,
                start_index=0,
                end_index=5,
                start_time=1.0,
                end_time=6.0,
                details={"score": 0.75},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe=None,
        include_completed=True,
        config={"scan_timeframes": ["M15", "H1"], "max_scan_timeframes": 2},
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["scanned_timeframes"] == ["M15", "H1"]
    assert res["n_patterns"] == 2


def test_patterns_detect_classic_applies_regime_context(monkeypatch):
    n = 180
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "close": np.linspace(100.0, 140.0, n),
            "tick_volume": np.full(n, 100.0),
        }
    )

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

    def _fake_run_classic_engine(engine, symbol, source_df, cfg, config):
        _ = (engine, symbol, source_df, cfg, config)
        return (
            [
                {
                    "name": "Ascending Triangle",
                    "status": "forming",
                    "confidence": 0.60,
                    "start_index": 20,
                    "end_index": n - 2,
                    "details": {"bias": "bullish", "support": 132.0, "resistance": 141.0},
                }
            ],
            None,
        )

    monkeypatch.setattr(core_patterns, "_run_classic_engine", _fake_run_classic_engine)

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        detail="full",
        timeframe="H1",
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["n_patterns"] == 1
    row = res["patterns"][0]
    regime = row["details"]["regime_context"]
    assert regime["state"] == "trending"
    assert regime["direction"] == "bullish"
    assert regime["status"] == "aligned"
    assert float(row["confidence"]) > 0.60


def test_patterns_detect_elliott_with_explicit_timeframe_uses_single_output(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    def _fake_fetch(symbol, timeframe, limit, denoise):
        return df.copy(), None

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", _fake_fetch)

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Correction",
                confidence=0.75,
                start_index=1,
                end_index=5,
                start_time=2.0,
                end_time=6.0,
                details={"score": 0.7},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe="H1",
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["timeframe"] == "H1"
    assert "findings" not in res
    assert res["n_patterns"] == 1


def test_patterns_detect_elliott_with_explicit_timeframe_hidden_completed_is_truthful(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Impulse",
                confidence=0.81,
                start_index=0,
                end_index=1,
                start_time=1.0,
                end_time=2.0,
                details={"trend": "bull", "pattern_confirmed": True, "has_unconfirmed_terminal_pivot": False},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe="H1",
        limit=100,
        include_completed=False,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["n_patterns"] == 0
    assert res["completed_patterns_hidden"] == 1
    assert "No forming Elliott Wave structures detected" in res["diagnostic"]
    assert "completed_patterns_preview" in res
    assert res["completed_patterns_preview"][0]["pattern"] == "Impulse"


def test_patterns_detect_elliott_scan_hidden_completed_is_truthful(monkeypatch):
    monkeypatch.setattr(core_patterns, "TIMEFRAME_MAP", {"M1": 1, "H1": 2, "H4": 3, "D1": 4})

    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Correction",
                confidence=0.79,
                start_index=0,
                end_index=1,
                start_time=1.0,
                end_time=2.0,
                details={"trend": "bear", "pattern_confirmed": True, "has_unconfirmed_terminal_pivot": False},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe=None,
        include_completed=False,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["n_patterns"] == 0
    assert res["completed_patterns_hidden"] == 3
    assert "No forming Elliott Wave structures were detected across scanned timeframes." in res["diagnostic"]
    assert len(res["completed_patterns_preview"]) == 3
    assert {item["timeframe"] for item in res["completed_patterns_preview"]} == {"H1", "H4", "D1"}
    assert all("No forming Elliott Wave structures detected" in row["diagnostic"] for row in res["findings"])


def test_count_recent_touches_respects_lookback():
    series = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    close = np.array([9.7, 10.3, 9.9, 10.0, 10.1])
    # lookback=3 => compare [9.9, 10.0, 10.1] against 10.0 with tol 0.15 => 3 touches
    assert _count_recent_touches(series, close, tol_abs=0.15, lookback_bars=3) == 3


def test_detect_classic_patterns_historical_scan_finds_older_prefix_pattern(monkeypatch):
    n = 220
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": np.linspace(100.0, 120.0, n)})

    def _fake_rounding(c, t, cfg):
        _ = cfg
        if len(c) != 140:
            return []
        return [
            ClassicPatternResult(
                name="Rounding Bottom",
                status="forming",
                confidence=0.82,
                start_index=100,
                end_index=139,
                start_time=float(t[100]),
                end_time=float(t[139]),
                details={},
            )
        ]

    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (np.array([], dtype=int), np.array([], dtype=int)))
    monkeypatch.setattr(classic_mod, "detect_rounding", _fake_rounding)

    out_default = detect_classic_patterns(df, ClassicDetectorConfig(max_consolidation_bars=5))
    out_scan = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            max_consolidation_bars=5,
            scan_historical=True,
            scan_step_bars=10,
            scan_min_prefix_bars=120,
        ),
    )

    assert not any(p.name == "Rounding Bottom" for p in out_default)
    match = next(p for p in out_scan if p.name == "Rounding Bottom")
    assert match.status == "forming"
    assert match.end_index == 139

    out_scan_completed = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            max_consolidation_bars=5,
            scan_historical=True,
            scan_step_bars=10,
            scan_min_prefix_bars=120,
            auto_complete_stale_forming=True,
        ),
    )
    assert next(p for p in out_scan_completed if p.name == "Rounding Bottom").status == "completed"


def test_detect_classic_patterns_historical_scan_reuses_global_pivots(monkeypatch):
    n = 220
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": np.linspace(100.0, 120.0, n)})
    calls = {"count": 0}

    def _fake_pivots(c, cfg, *args):
        _ = c
        _ = cfg
        _ = args
        calls["count"] += 1
        return np.array([], dtype=int), np.array([], dtype=int)

    monkeypatch.setattr(classic_mod, "_detect_pivots_close", _fake_pivots)
    monkeypatch.setattr(classic_mod, "detect_diamonds", lambda *args, **kwargs: [])
    monkeypatch.setattr(classic_mod, "detect_flags_pennants", lambda *args, **kwargs: [])

    out = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            max_consolidation_bars=5,
            scan_historical=True,
            scan_step_bars=10,
            scan_min_prefix_bars=120,
        ),
    )

    assert out == []
    assert calls["count"] == 1


def test_detect_classic_patterns_surfaces_confidence_calibration_errors(monkeypatch):
    n = 150
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": np.linspace(100.0, 110.0, n)})

    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (np.array([], dtype=int), np.array([], dtype=int)))
    monkeypatch.setattr(
        classic_mod,
        "detect_rounding",
        lambda c, t, cfg: [
            ClassicPatternResult(
                name="Rounding Bottom",
                status="forming",
                confidence=0.7,
                start_index=40,
                end_index=149,
                start_time=float(t[40]),
                end_time=float(t[149]),
                details={},
            )
        ],
    )
    monkeypatch.setattr(classic_mod, "_calibrate_confidence", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        detect_classic_patterns(df, ClassicDetectorConfig(max_consolidation_bars=5))


def test_detect_cup_handle_respects_configurable_handle_pullback():
    from src.mtdata.patterns.classic_impl.continuation import detect_cup_handle

    n = 180
    anchors = [(0, 100.0), (25, 100.0), (90, 82.0), (135, 100.0), (150, 98.0), (165, 95.0), (179, 101.0)]
    close = np.full(n, 100.0, dtype=float)
    for (a_i, a_v), (b_i, b_v) in zip(anchors, anchors[1:]):
        close[a_i : b_i + 1] = np.linspace(a_v, b_v, b_i - a_i + 1)

    strict_cfg = ClassicDetectorConfig(
        cup_handle_max_handle_pullback_pct=4.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
    )
    relaxed_cfg = ClassicDetectorConfig(
        cup_handle_max_handle_pullback_pct=6.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
    )

    out_strict = detect_cup_handle(close, np.arange(n, dtype=float), strict_cfg)
    out_relaxed = detect_cup_handle(close, np.arange(n, dtype=float), relaxed_cfg)

    assert out_strict == []
    assert out_relaxed
    assert out_relaxed[0].status == "completed"
    assert out_relaxed[0].details["handle_pullback_pct"] == pytest.approx(5.0)


def test_detect_cup_handle_scores_rim_mismatch_instead_of_hard_reject():
    from src.mtdata.patterns.classic_impl.continuation import detect_cup_handle

    n = 180
    anchors = [(0, 100.0), (25, 100.0), (90, 82.0), (135, 104.0), (150, 102.0), (165, 99.0), (179, 106.0)]
    close = np.full(n, 100.0, dtype=float)
    for (a_i, a_v), (b_i, b_v) in zip(anchors, anchors[1:]):
        close[a_i : b_i + 1] = np.linspace(a_v, b_v, b_i - a_i + 1)

    strict_cfg = ClassicDetectorConfig(
        cup_handle_max_rim_mismatch_pct=2.0,
        cup_handle_max_handle_pullback_pct=6.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
        same_level_tol_pct=0.5,
    )
    relaxed_cfg = ClassicDetectorConfig(
        cup_handle_max_rim_mismatch_pct=6.0,
        cup_handle_max_handle_pullback_pct=6.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
        same_level_tol_pct=0.5,
    )

    out_strict = detect_cup_handle(close, np.arange(n, dtype=float), strict_cfg)
    out_relaxed = detect_cup_handle(close, np.arange(n, dtype=float), relaxed_cfg)

    assert out_strict == []
    assert out_relaxed
    assert out_relaxed[0].status == "completed"
    assert out_relaxed[0].details["near_equal_rim"] == "no"
    assert out_relaxed[0].details["rim_mismatch_pct"] == pytest.approx(3.8461538461538463)
    assert 0.0 < out_relaxed[0].details["rim_symmetry"] < 1.0


def test_detect_inverted_cup_handle_detects_bearish_variant():
    from src.mtdata.patterns.classic_impl.continuation import detect_cup_handle

    n = 180
    anchors = [(0, 100.0), (25, 100.0), (90, 118.0), (135, 100.0), (150, 102.0), (165, 105.0), (179, 99.0)]
    close = np.full(n, 100.0, dtype=float)
    for (a_i, a_v), (b_i, b_v) in zip(anchors, anchors[1:]):
        close[a_i : b_i + 1] = np.linspace(a_v, b_v, b_i - a_i + 1)

        out = detect_cup_handle(
            close,
            np.arange(n, dtype=float),
            ClassicDetectorConfig(
                breakout_lookahead=40,
                completion_lookback_bars=40,
                cup_handle_max_depth_pct=100.0,
                cup_handle_max_handle_pullback_pct=30.0,
            ),
        )

    inverted = next(pattern for pattern in out if pattern.name == "Inverted Cup and Handle")
    assert inverted.details["breakout_direction"] == "down"
    assert inverted.details["bias"] == "bearish"
    assert inverted.status == "completed"


def test_detect_triangles_skip_same_sign_converging_shapes(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 118.0, n)
    bot = np.linspace(102.0, 106.0, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(shapes, "_fit_lines_and_arrays", lambda *_args, **_kwargs: (0.04, 112.0, 0.9, 0.02, 102.0, 0.9, top, bot))
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    tri = shapes.detect_triangles(close, peaks, troughs, np.arange(n, dtype=float), ClassicDetectorConfig(min_channel_touches=2))
    wedge = shapes.detect_wedges(close, peaks, troughs, np.arange(n, dtype=float), ClassicDetectorConfig(min_channel_touches=2))

    assert tri == []
    assert wedge
    assert wedge[0].name == "Rising Wedge"


def test_detect_triangles_allows_near_flat_boundary_with_same_sign_slopes(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 114.0, n)
    bot = np.linspace(102.0, 112.0, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.01, 112.0, 0.9, 0.08, 102.0, 0.9, top, bot),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    tri = shapes.detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, max_flat_slope=0.02),
    )

    assert tri
    assert tri[0].name == "Ascending Triangle"


def test_detect_triangles_reject_crossed_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 98.0, n)
    bot = np.linspace(102.0, 104.0, n)

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (-0.09, 112.0, 0.9, 0.013, 102.0, 0.9, top.copy(), bot.copy()),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    out = shapes.detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )

    assert out == []


def test_detect_triangles_reject_both_flat_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 112.8, n)
    bot = np.linspace(102.0, 101.2, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.005, 112.0, 0.9, -0.005, 102.0, 0.9, top, bot),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    out = shapes.detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, max_flat_slope=0.02),
    )

    assert out == []


def test_detect_wedges_reject_near_flat_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 113.0, n)
    bot = np.linspace(102.0, 102.9, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.01, 112.0, 0.9, 0.008, 102.0, 0.9, top, bot),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    out = shapes.detect_wedges(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, max_flat_slope=0.02),
    )

    assert out == []


def test_detect_flags_prefers_flag_when_convergence_is_only_noise(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 160
    window = 30
    close = np.full(n, 100.0, dtype=float)
    close[-window:] = np.linspace(104.0, 103.0, window)
    high = close + 0.1
    low = close - 0.1
    peaks = np.array([5, 12, 19, 26], dtype=int)
    troughs = np.array([3, 10, 17, 24], dtype=int)
    top = np.linspace(104.0, 103.01, window)
    bot = np.linspace(102.0, 101.02, window)

    monkeypatch.setattr(continuation, "_detect_pivots_close", lambda *_args, **_kwargs: (peaks, troughs))
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (-0.03, 104.0, 0.9, -0.029, 102.0, 0.9, top.copy(), bot.copy()),
    )

    out = continuation.detect_flags_pennants(
        close,
        high,
        low,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(
            max_consolidation_bars=window,
            min_pole_return_pct=2.0,
            pennant_min_convergence_ratio=0.05,
        ),
    )

    assert out
    assert out[0].name == "Bull Flag"


def test_detect_flags_pennants_excludes_pole_from_consolidation_fit(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 130
    window = 30
    close = np.full(n, 98.0, dtype=float)
    close[-window:] = np.concatenate(
        (
            np.linspace(100.0, 110.0, 10),
            np.linspace(109.5, 106.0, 20),
        )
    )
    high = close + 0.1
    low = close - 0.1
    captured = {}

    def _fake_pivots(seg, cfg, *args):
        _ = cfg
        _ = args
        captured["seg"] = seg.copy()
        return np.array([2, 7, 12, 17], dtype=int), np.array([4, 9, 14, 19], dtype=int)

    monkeypatch.setattr(continuation, "_detect_pivots_close", _fake_pivots)
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            -0.08,
            110.0,
            0.9,
            -0.05,
            107.0,
            0.9,
            np.linspace(110.0, 106.5, 21),
            np.linspace(107.5, 104.5, 21),
        ),
    )

    out = continuation.detect_flags_pennants(
        close,
        high,
        low,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(max_consolidation_bars=window, min_pole_return_pct=5.0),
    )

    assert out
    assert captured["seg"].size == 21
    assert captured["seg"][0] == pytest.approx(110.0)
    assert out[0].details["consolidation_start_index"] == (n - window + 9)


def test_detect_rectangles_mark_completed_on_breakout():
    from src.mtdata.patterns.classic_impl.shapes import detect_rectangles

    n = 120
    close = np.full(n, 100.0, dtype=float)
    peaks = np.array([20, 40, 60], dtype=int)
    troughs = np.array([30, 50, 70], dtype=int)
    close[peaks] = 105.0
    close[troughs] = 95.0
    close[-1] = 106.0

    out = detect_rectangles(close, peaks, troughs, np.arange(n, dtype=float), ClassicDetectorConfig(min_channel_touches=2))

    assert out
    assert out[0].status == "completed"
    assert out[0].details["breakout_direction"] == "up"


def test_detect_rectangles_without_time_values_return_none_timestamps():
    from src.mtdata.patterns.classic_impl.shapes import detect_rectangles

    n = 120
    close = np.full(n, 100.0, dtype=float)
    peaks = np.array([20, 40, 60], dtype=int)
    troughs = np.array([30, 50, 70], dtype=int)
    close[peaks] = 105.0
    close[troughs] = 95.0
    close[-1] = 106.0

    out = detect_rectangles(close, peaks, troughs, np.asarray([], dtype=float), ClassicDetectorConfig(min_channel_touches=2))

    assert out
    assert out[0].start_time is None
    assert out[0].end_time is None


def test_detect_trend_lines_extend_to_current_bar():
    from src.mtdata.patterns.classic_impl.trend import detect_trend_lines

    n = 140
    close = np.linspace(100.0, 120.0, n)
    peaks = np.array([20, 50, 80, 110], dtype=int)
    troughs = np.array([10, 40, 70, 100], dtype=int)
    close[peaks] = np.linspace(104.0, 118.0, peaks.size)
    close[troughs] = np.linspace(98.0, 112.0, troughs.size)

    out = detect_trend_lines(close, peaks, troughs, np.arange(n, dtype=float), ClassicDetectorConfig())

    assert out
    assert all(p.end_index == (n - 1) for p in out)


def test_detect_channels_allow_small_absolute_slope_spread(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 160
    close = np.linspace(100.0, 101.0, n)
    peaks = np.array([30, 60, 90, 120, 150], dtype=int)
    troughs = np.array([20, 50, 80, 110, 140], dtype=int)
    upper = 110.0 + (2e-5 * np.arange(n, dtype=float))
    lower = 100.0 + (9e-5 * np.arange(n, dtype=float))

    monkeypatch.setattr(trend, "_fit_lines_and_arrays", lambda *_args, **_kwargs: (2e-5, 110.0, 0.95, 9e-5, 100.0, 0.95, upper, lower))
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)

    out = trend.detect_channels(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, channel_parallel_slope_ratio=0.15),
    )

    assert out
    assert out[0].name == "Horizontal Channel"


def test_detect_channels_reject_widening_parallel_structure(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 60
    x = np.arange(n, dtype=float)
    upper = 110.0 + (0.470 * x)
    lower = 100.0 + (0.400 * x)
    close = (upper + lower) / 2.0
    peaks = np.array([10, 20, 30, 40, 50], dtype=int)
    troughs = np.array([5, 15, 25, 35, 45], dtype=int)

    monkeypatch.setattr(
        trend,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.470, 110.0, 0.95, 0.400, 100.0, 0.95, upper.copy(), lower.copy()),
    )
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trend, "_count_touches", lambda *_args, **_kwargs: 6)

    out = trend.detect_channels(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )

    assert out == []


def test_detect_channels_reject_crossed_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 60
    x = np.arange(n, dtype=float)
    upper = 110.0 - (0.08 * x)
    lower = 100.0 + (0.10 * x)
    close = (upper + lower) / 2.0
    peaks = np.array([10, 20, 30, 40, 50], dtype=int)
    troughs = np.array([5, 15, 25, 35, 45], dtype=int)

    monkeypatch.setattr(
        trend,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (-0.08, 110.0, 0.95, 0.10, 100.0, 0.95, upper.copy(), lower.copy()),
    )
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trend, "_count_touches", lambda *_args, **_kwargs: 6)

    out = trend.detect_channels(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, channel_parallel_slope_ratio=2.0),
    )

    assert out == []


def test_detect_trend_lines_require_breakout_for_completed_status(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 40
    peaks = np.array([5, 15, 25], dtype=int)
    troughs = np.array([10, 20, 30], dtype=int)

    def _fake_fit(x, _y):
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks.tolist()):
            return 0.0, 110.0, 0.95
        return 0.0, 100.0, 0.95

    monkeypatch.setattr(trend, "_fit_line", _fake_fit)
    monkeypatch.setattr(trend, "_last_touch_indexes", lambda _line, idxs, _c, _tol: idxs.tolist())

    cfg = ClassicDetectorConfig(
        use_robust_fit=False,
        same_level_tol_pct=0.1,
        completion_lookback_bars=4,
        completion_confirm_bars=2,
    )

    close_forming = np.full(n, 100.0, dtype=float)
    close_forming[-4:] = np.array([100.02, 99.98, 100.01, 100.0], dtype=float)
    out_forming = trend.detect_trend_lines(
        close_forming,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )
    support_forming = next(pattern for pattern in out_forming if pattern.details["side"] == "low")

    assert support_forming.status == "forming"
    assert support_forming.details["touches_recent"] == 4
    assert support_forming.details["breakout_index"] is None

    close_break = close_forming.copy()
    close_break[-1] = 99.0
    out_break = trend.detect_trend_lines(
        close_break,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )
    support_break = next(pattern for pattern in out_break if pattern.details["side"] == "low")

    assert support_break.status == "completed"
    assert support_break.end_index == n - 1
    assert support_break.details["breakout_direction"] == "down"
    assert support_break.details["breakout_index"] == n - 1


def test_detect_channels_require_breakout_for_completed_status(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 60
    peaks = np.array([10, 20, 30, 40, 50], dtype=int)
    troughs = np.array([5, 15, 25, 35, 45], dtype=int)
    upper = np.full(n, 105.0, dtype=float)
    lower = np.full(n, 95.0, dtype=float)

    monkeypatch.setattr(
        trend,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.0, 105.0, 0.95, 0.0, 95.0, 0.95, upper.copy(), lower.copy()),
    )
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trend, "_count_touches", lambda *_args, **_kwargs: 6)

    cfg = ClassicDetectorConfig(
        min_channel_touches=2,
        same_level_tol_pct=0.1,
        completion_lookback_bars=4,
        completion_confirm_bars=2,
    )

    close_forming = np.full(n, 100.0, dtype=float)
    close_forming[-4:] = np.array([105.0, 95.0, 100.0, 104.95], dtype=float)
    out_forming = trend.detect_channels(
        close_forming,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )

    assert out_forming
    assert out_forming[0].status == "forming"
    assert out_forming[0].details["touches_recent"] == 3
    assert out_forming[0].details["breakout_index"] is None

    close_break = close_forming.copy()
    close_break[-1] = 106.0
    out_break = trend.detect_channels(
        close_break,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )

    assert out_break[0].status == "completed"
    assert out_break[0].end_index == n - 1
    assert out_break[0].details["breakout_direction"] == "up"
    assert out_break[0].details["breakout_index"] == n - 1


def test_detect_diamonds_respects_geometry_threshold(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 200
    close = np.linspace(100.0, 101.0, n)
    close[-1] = 111.0
    peaks = np.array([30, 60, 90, 130, 160], dtype=int)
    troughs = np.array([20, 50, 100, 140, 170], dtype=int)

    def _fake_fit(x, y):
        _ = y
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks[peaks < 100].tolist()):
            return 0.05, 102.0, 0.9
        if xs == tuple(troughs[troughs < 100].tolist()):
            return -0.05, 98.0, 0.9
        if xs == tuple(peaks[peaks >= 100].tolist()):
            return -0.05, 115.0, 0.9
        if xs == tuple(troughs[troughs >= 100].tolist()):
            return 0.05, 85.0, 0.9
        return 0.0, 100.0, 0.0

    monkeypatch.setattr(shapes, "_detect_pivots_close", lambda seg, cfg, *args: (peaks, troughs))
    monkeypatch.setattr(shapes, "_fit_line", _fake_fit)

    out_strict = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            diamond_min_boundary_r2=0.0,
            diamond_min_width_ratio=1.5,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )
    out_relaxed = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            diamond_min_boundary_r2=0.0,
            diamond_min_width_ratio=1.2,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )

    assert out_strict == []
    assert out_relaxed
    assert out_relaxed[0].status == "completed"
    assert out_relaxed[0].details["diamond_split_index"] == 100
    assert out_relaxed[0].details["prior_pole_span_bars"] > 0
    assert 0.0 < out_relaxed[0].details["geometry_score"] < 1.0


def test_detect_diamonds_forward_high_low_arrays_to_pivot_detection(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    close = np.linspace(100.0, 105.0, n)
    high = close + 1.0
    low = close - 1.0
    captured = {}

    def _fake_pivots(seg, cfg, seg_h, seg_l):
        _ = seg
        _ = cfg
        captured["high"] = seg_h.copy()
        captured["low"] = seg_l.copy()
        return np.array([], dtype=int), np.array([], dtype=int)

    monkeypatch.setattr(shapes, "_detect_pivots_close", _fake_pivots)

    out = shapes.detect_diamonds(close, np.arange(n, dtype=float), ClassicDetectorConfig(), high, low)

    assert out == []
    assert np.array_equal(captured["high"], high[-n:])
    assert np.array_equal(captured["low"], low[-n:])


def test_detect_diamonds_accepts_asymmetric_split_with_stricter_default_r2(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 200
    close = np.linspace(100.0, 101.0, n)
    close[-1] = 111.0
    peaks = np.array([10, 30, 50, 160, 180], dtype=int)
    troughs = np.array([5, 25, 49, 170, 190], dtype=int)

    def _fake_fit(x, y):
        _ = y
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks[peaks < 50].tolist()):
            return 0.05, 102.0, 0.75
        if xs == tuple(troughs[troughs < 50].tolist()):
            return -0.05, 98.0, 0.75
        if xs == tuple(peaks[peaks >= 50].tolist()):
            return -0.02, 105.5, 0.75
        if xs == tuple(troughs[troughs >= 50].tolist()):
            return 0.02, 94.5, 0.75
        return 0.0, 100.0, 0.0

    monkeypatch.setattr(shapes, "_detect_pivots_close", lambda seg, cfg, *args: (peaks, troughs))
    monkeypatch.setattr(shapes, "_fit_line", _fake_fit)

    out = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )

    assert out
    assert out[0].details["diamond_split_index"] in {49, 50}


def test_detect_diamonds_reject_disjoint_split_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 200
    close = np.linspace(100.0, 101.0, n)
    peaks = np.array([30, 60, 90, 130, 160], dtype=int)
    troughs = np.array([20, 50, 100, 140, 170], dtype=int)

    def _fake_fit(x, y):
        _ = y
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks[peaks < 100].tolist()):
            return 0.05, 102.0, 0.9
        if xs == tuple(troughs[troughs < 100].tolist()):
            return -0.05, 98.0, 0.9
        if xs == tuple(peaks[peaks >= 100].tolist()):
            return -0.05, 130.0, 0.9
        if xs == tuple(troughs[troughs >= 100].tolist()):
            return 0.05, 70.0, 0.9
        return 0.0, 100.0, 0.0

    monkeypatch.setattr(shapes, "_detect_pivots_close", lambda seg, cfg, *args: (peaks, troughs))
    monkeypatch.setattr(shapes, "_fit_line", _fake_fit)

    out = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            diamond_min_boundary_r2=0.0,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )

    assert out == []


def test_detect_tops_bottoms_merges_connected_same_level_cluster():
    from src.mtdata.patterns.classic_impl.reversal import detect_tops_bottoms

    close = np.array(
        [98.0, 100.1, 95.0, 100.0, 94.8, 99.9, 95.2, 100.2, 96.0],
        dtype=float,
    )
    peaks = np.array([1, 3, 5, 7], dtype=int)
    troughs = np.array([2, 4, 6], dtype=int)

    out = detect_tops_bottoms(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(same_level_tol_pct=0.5),
    )

    assert out
    triple_top = next(pattern for pattern in out if pattern.name == "Triple Top")
    assert triple_top.details["touches"] == 4


def test_level_components_preserves_transitive_clusters_in_input_order():
    from src.mtdata.patterns.classic_impl.reversal import _level_components

    vals = np.array([100.0, 112.0, 103.0, 115.0, 106.0], dtype=float)

    assert _level_components(vals, 5.0) == [[0, 2, 4], [1, 3]]


def test_detect_head_shoulders_fits_neckline_with_reaction_troughs(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    captured = {}

    def _fake_fit(x, y):
        captured["x"] = x.tolist()
        captured["y"] = y.tolist()
        return 0.0, 95.0, 0.8

    monkeypatch.setattr(reversal, "_fit_line", _fake_fit)

    close = np.array([96.0, 100.0, 95.0, 103.0, 110.0, 96.0, 95.5, 99.0, 100.5, 97.0], dtype=float)
    peaks = np.array([1, 4, 8], dtype=int)
    troughs = np.array([2, 5, 6], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(same_level_tol_pct=1.0, use_dtw_check=False, use_robust_fit=False),
    )

    assert out
    assert captured["x"] == [2.0, 5.0]
    assert out[0].details["neckline_source"] == "troughs"
    assert out[0].details["neck_points"] == 2
    assert out[0].details["neck_validation_points"] == 1


def test_detect_inverse_head_shoulders_uses_reaction_peaks_for_neckline(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    captured = {}

    def _fake_fit(x, y):
        captured["x"] = x.tolist()
        captured["y"] = y.tolist()
        return 0.0, 105.0, 0.85

    monkeypatch.setattr(reversal, "_fit_line", _fake_fit)

    close = np.array([104.0, 95.0, 103.0, 90.0, 102.0, 96.0, 108.0], dtype=float)
    peaks = np.array([0, 2, 4, 6], dtype=int)
    troughs = np.array([1, 3, 5], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(same_level_tol_pct=2.0, use_dtw_check=False, use_robust_fit=False),
    )

    assert out
    inverse = next(pattern for pattern in out if pattern.name == "Inverse Head and Shoulders")
    assert captured["x"] == [2.0, 4.0]
    assert inverse.details["neckline_source"] == "peaks"
    assert inverse.details["neck_points"] == 2


def test_detect_head_shoulders_rejects_neckline_above_shoulders():
    from src.mtdata.patterns.classic_impl import reversal

    close = np.array([96.0, 100.0, 101.0, 103.0, 110.0, 101.5, 101.2, 99.0, 100.5, 97.0], dtype=float)
    peaks = np.array([1, 4, 8], dtype=int)
    troughs = np.array([2, 5, 6], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(same_level_tol_pct=1.0, use_dtw_check=False, use_robust_fit=False),
    )

    assert out == []


def test_detect_inverse_head_shoulders_rejects_neckline_below_shoulders():
    from src.mtdata.patterns.classic_impl import reversal

    close = np.array([104.0, 95.0, 94.0, 90.0, 94.5, 96.0, 108.0], dtype=float)
    peaks = np.array([0, 2, 4, 6], dtype=int)
    troughs = np.array([1, 3, 5], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(same_level_tol_pct=2.0, use_dtw_check=False, use_robust_fit=False),
    )

    assert out == []


def test_head_shoulders_two_point_neckline_does_not_get_free_r2_boost():
    from src.mtdata.patterns.classic_impl import reversal

    cfg = ClassicDetectorConfig(max_flat_slope=1e-4)
    quality = reversal._neckline_quality_score(
        slope=cfg.max_flat_slope * 2.5,
        r2=1.0,
        point_count=2,
        cfg=cfg,
    )

    assert quality == pytest.approx(0.5)


def test_detect_rounding_tries_multiple_windows(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    n = 260
    close = np.linspace(100.0, 110.0, n)
    called = []

    def _fake_polyfit(x, y, deg):
        _ = y
        _ = deg
        called.append(len(x))
        if len(x) == 100:
            return np.array([0.6, 0.0, 95.0], dtype=float)
        raise np.linalg.LinAlgError("skip other windows")

    monkeypatch.setattr(reversal.np, "polyfit", _fake_polyfit)
    monkeypatch.setattr(reversal, "_level_close", lambda *_args, **_kwargs: True)

    out = reversal.detect_rounding(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(rounding_window_bars=220, rounding_window_sizes=[100, 220]),
    )

    assert called == [100, 220] or called == [220, 100]
    assert out
    assert out[0].details["window_bars"] == 100


def test_detect_rounding_returns_multiple_non_overlapping_windows(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    n = 320
    close = np.linspace(100.0, 110.0, n)
    fits = {
        100: np.array([0.5, 0.0, 95.0], dtype=float),
        220: np.array([0.55, 0.0, 95.0], dtype=float),
    }

    def _fake_polyfit(x, y, deg):
        _ = (y, deg)
        value = fits.get(len(x))
        if value is None:
            raise np.linalg.LinAlgError("skip")
        return value

    monkeypatch.setattr(reversal.np, "polyfit", _fake_polyfit)
    monkeypatch.setattr(reversal, "_level_close", lambda *_args, **_kwargs: True)

    out = reversal.detect_rounding(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(rounding_window_bars=220, rounding_window_sizes=[100, 220]),
    )

    assert len(out) == 2
    assert {pattern.details["window_bars"] for pattern in out} == {100, 220}


def test_dedupe_overlapping_head_shoulders_results():
    from src.mtdata.patterns.classic_impl.reversal import _dedupe_overlapping_patterns

    patterns = [
        ClassicPatternResult(
            name="Head and Shoulders",
            status="forming",
            confidence=0.82,
            start_index=10,
            end_index=40,
            start_time=None,
            end_time=None,
            details={},
        ),
        ClassicPatternResult(
            name="Head and Shoulders",
            status="forming",
            confidence=0.75,
            start_index=14,
            end_index=38,
            start_time=None,
            end_time=None,
            details={},
        ),
        ClassicPatternResult(
            name="Head and Shoulders",
            status="forming",
            confidence=0.9,
            start_index=70,
            end_index=100,
            start_time=None,
            end_time=None,
            details={},
        ),
    ]

    out = _dedupe_overlapping_patterns(patterns, overlap_threshold=0.6)

    assert len(out) == 2
    assert any(p.start_index == 10 and p.end_index == 40 for p in out)
    assert any(p.start_index == 70 and p.end_index == 100 for p in out)


def test_detect_classic_patterns_disables_aliases_by_default(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.3 * np.arange(n) + 4.0 * np.sin(x)
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close})

    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg: (peaks, troughs))

    out_default = detect_classic_patterns(df, ClassicDetectorConfig(max_consolidation_bars=5))
    names_default = {p.name for p in out_default}
    assert "Ascending Trend Line" in names_default
    assert "Trend Line" not in names_default
    assert "Trend Channel" not in names_default

    out_alias = detect_classic_patterns(
        df, ClassicDetectorConfig(include_aliases=True, max_consolidation_bars=5)
    )
    names_alias = {p.name for p in out_alias}
    assert "Trend Line" in names_alias or "Trend Channel" in names_alias


def test_detect_classic_patterns_reraises_internal_pivot_type_errors(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.3 * np.arange(n) + 4.0 * np.sin(x)
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close})

    def _bad_pivots(c, cfg, *args):
        raise TypeError("internal pivot failure")

    monkeypatch.setattr(classic_mod, "_detect_pivots_close", _bad_pivots)

    with pytest.raises(TypeError, match="internal pivot failure"):
        detect_classic_patterns(df, ClassicDetectorConfig(max_consolidation_bars=5))


def test_patterns_detect_classic_ensemble_merges_engine_outputs(monkeypatch, caplog):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

    def _fake_engine(engine, symbol, df_in, cfg, config):
        _ = symbol
        _ = df_in
        _ = cfg
        _ = config
        if engine == "native":
            return [
                {
                    "name": "Double Top",
                    "status": "forming",
                    "confidence": 0.8,
                    "start_index": 10,
                    "end_index": 20,
                    "start_date": "A",
                    "end_date": "B",
                    "details": {"x": 1},
                }
            ], None
        if engine == "stock_pattern":
            return [
                {
                    "name": "Double Top",
                    "status": "forming",
                    "confidence": 0.6,
                    "start_index": 12,
                    "end_index": 22,
                    "start_date": "C",
                    "end_date": "D",
                    "details": {"y": 2},
                }
            ], None
        return [], "unexpected"

    monkeypatch.setattr(core_patterns, "_run_classic_engine", _fake_engine)

    with caplog.at_level(logging.INFO, logger=core_patterns.logger.name):
        res = patterns_detect(
            symbol="EURUSD",
            mode="classic",
            detail="full",
            timeframe="H1",
            engine="native,stock_pattern",
            ensemble=True,
            include_completed=True,
            __cli_raw=True,
        )

    assert res["success"] is True
    assert res["engine"] == "ensemble"
    assert res["n_patterns"] == 1
    assert res["patterns"][0]["support_count"] == 2
    assert set(res["patterns"][0]["source_engines"]) == {"native", "stock_pattern"}
    assert "event=finish operation=patterns_detect success=True" in caplog.text


def test_patterns_detect_rejects_hidden_precise_engine(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
        }
    )
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda *_args, **_kwargs: (df.copy(), None))

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        timeframe="H1",
        engine="precise_patterns",
        __cli_raw=True,
    )

    assert "error" in res
    assert "Invalid classic engine" in str(res["error"])


def test_patterns_detect_classic_adds_signal_summary_and_levels(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [100.0, 101.0, 102.0, 101.0, 100.5, 100.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

    def _fake_engine(engine, symbol, df_in, cfg, config):
        _ = engine
        _ = symbol
        _ = df_in
        _ = cfg
        _ = config
        return [
            {
                "name": "Double Top",
                "status": "forming",
                "confidence": 0.9,
                "start_index": 1,
                "end_index": 5,
                "details": {"support": 98.5, "resistance": 102.5},
            },
            {
                "name": "Bull Pennant",
                "status": "forming",
                "confidence": 0.7,
                "start_index": 1,
                "end_index": 5,
                "details": {"support": 98.8, "resistance": 102.2},
            },
        ], None

    monkeypatch.setattr(core_patterns, "_run_classic_engine", _fake_engine)

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        detail="full",
        timeframe="H1",
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert "signal_summary" in res
    assert res["signal_summary"]["conflict"] is True
    assert res["signal_summary"]["net_bias"] == "mixed"
    rows = res["patterns"]
    assert all("bias" in row for row in rows)
    assert all("reference_price" in row for row in rows)
    assert all("volume_confirmation" in row["details"] for row in rows if isinstance(row.get("details"), dict))
    assert all("target_price" in row for row in rows)
    assert all("invalidation_price" in row for row in rows)


def test_patterns_detect_engine_findings_report_hidden_completed(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))
    monkeypatch.setattr(
        core_patterns,
        "_run_classic_engine",
        lambda engine, symbol, df_in, cfg, config: (
            [
                {"name": "Triangle", "status": "forming", "confidence": 0.8, "start_index": 1, "end_index": 4},
                {"name": "Triangle", "status": "completed", "confidence": 0.7, "start_index": 0, "end_index": 3},
            ],
            None,
        ),
    )

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        detail="full",
        timeframe="H1",
        include_completed=False,
        __cli_raw=True,
    )

    assert res["success"] is True
    finding = res["engine_findings"][0]
    assert finding["n_patterns"] == 1
    assert finding["n_completed"] == 0
    assert finding["n_completed_hidden"] == 1
    assert finding["n_patterns_total"] == 2


def test_patterns_detect_classic_invalid_engine_returns_error(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        timeframe="H1",
        engine="bad_engine",
        __cli_raw=True,
    )

    assert "error" in res
    assert "Invalid classic engine" in str(res["error"])


def test_detect_pivots_close_prefers_high_low_when_enabled():
    n = 160
    close = np.full(n, 100.0, dtype=float)
    high = close.copy()
    low = close.copy()
    high[[30, 75, 120]] = [106.0, 107.0, 106.5]
    low[[45, 95, 140]] = [94.0, 93.5, 94.2]

    cfg_hl = ClassicDetectorConfig(
        pivot_use_hl=True,
        pivot_enable_fallback=False,
        pivot_use_atr_adaptive_prominence=False,
        pivot_use_atr_adaptive_distance=False,
        min_prominence_pct=1.0,
        min_distance=5,
    )
    cfg_close = ClassicDetectorConfig(
        pivot_use_hl=False,
        pivot_enable_fallback=False,
        pivot_use_atr_adaptive_prominence=False,
        pivot_use_atr_adaptive_distance=False,
        min_prominence_pct=1.0,
        min_distance=5,
    )

    peaks_hl, troughs_hl = classic_mod._detect_pivots_close(close, cfg_hl, high, low)
    peaks_close, troughs_close = classic_mod._detect_pivots_close(close, cfg_close, high, low)

    assert peaks_hl.size >= 2
    assert troughs_hl.size >= 2
    assert peaks_close.size == 0
    assert troughs_close.size == 0


def test_detect_classic_triangle_marks_completed_on_breakout(monkeypatch):
    n = 170
    close = np.full(n, 100.0, dtype=float)
    top_line = np.linspace(106.0, 100.0, n)
    bot_line = np.linspace(94.0, 99.0, n)
    close[:-3] = (top_line[:-3] + bot_line[:-3]) / 2.0
    close[-3:] = top_line[-3:] + 1.0

    peaks = np.array([30, 60, 90, 120, 150], dtype=int)
    troughs = np.array([20, 50, 80, 110, 140], dtype=int)
    close[peaks] = top_line[peaks]
    close[troughs] = bot_line[troughs]
    close[-3:] = top_line[-3:] + 1.0
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close, "high": close + 0.2, "low": close - 0.2})
    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs))

    def _fake_fit_lines(ih, il, c, n, cfg):
        return -0.03, 106.0, 0.9, 0.03, 94.0, 0.9, top_line.copy(), bot_line.copy()

    monkeypatch.setattr(classic_mod, "_fit_lines_and_arrays", _fake_fit_lines)

    out = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            min_channel_touches=2,
            breakout_lookahead=6,
            completion_lookback_bars=6,
            completion_confirm_bars=1,
            max_consolidation_bars=5,
        ),
    )
    tri = [p for p in out if "Triangle" in p.name]
    assert tri
    assert any(p.status == "completed" for p in tri)


def test_detect_classic_converging_parallel_shape_excludes_channel(monkeypatch):
    n = 170
    x = np.arange(n, dtype=float)
    top_line = 110.0 - 0.020 * x
    bot_line = 100.0 - 0.019 * x
    close = (top_line + bot_line) / 2.0

    peaks = np.array([30, 60, 90, 120, 150], dtype=int)
    troughs = np.array([20, 50, 80, 110, 140], dtype=int)
    close[peaks] = top_line[peaks]
    close[troughs] = bot_line[troughs]

    df = pd.DataFrame(
        {"time": np.arange(n, dtype=float), "close": close, "high": close + 0.2, "low": close - 0.2}
    )
    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs))

    def _fake_fit_lines(ih, il, c, n, cfg):
        return -0.020, 110.0, 0.95, -0.019, 100.0, 0.95, top_line.copy(), bot_line.copy()

    monkeypatch.setattr(classic_mod, "_fit_lines_and_arrays", _fake_fit_lines)

    out = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            min_channel_touches=2,
            max_consolidation_bars=5,
        ),
    )
    names = {p.name for p in out}
    assert "Falling Wedge" in names
    assert "Symmetrical Triangle" not in names
    assert not any("Channel" in name for name in names)


def test_detect_classic_confidence_calibration_and_lifecycle(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.25 * np.arange(n) + 3.0 * np.sin(x)
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close, "high": close + 0.3, "low": close - 0.3})

    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs))

    cfg = ClassicDetectorConfig(
        calibrate_confidence=True,
        confidence_calibration_map={"default": {"0.0": 0.0, "1.0": 0.5}},
        confidence_calibration_blend=1.0,
        max_consolidation_bars=5,
    )
    out = detect_classic_patterns(df, cfg)
    assert out
    sample = out[0]
    assert isinstance(sample.details, dict)
    assert "raw_confidence" in sample.details
    assert "calibrated_confidence" in sample.details
    assert sample.confidence <= float(sample.details["raw_confidence"])
    assert sample.details.get("lifecycle_state") in {"forming", "confirmed"}


def test_run_classic_engine_native_multiscale_merges(monkeypatch):
    cfg = ClassicDetectorConfig(min_distance=6, min_prominence_pct=0.6)
    df = pd.DataFrame({"close": np.linspace(100.0, 120.0, 200), "time": np.arange(200, dtype=float)})

    def _fake_format(_df, cfg_in):
        md = int(cfg_in.min_distance)
        return [
            {
                "name": "Double Top",
                "status": "forming",
                "confidence": 0.4 + 0.03 * md,
                "start_index": 40 + (md % 2),
                "end_index": 90 + (md % 3),
                "start_date": None,
                "end_date": None,
                "details": {"min_distance": md},
            }
        ]

    monkeypatch.setattr(core_patterns, "_format_classic_native_patterns", _fake_format)

    out, err = core_patterns._run_classic_engine_native(
        symbol="EURUSD",
        df=df,
        cfg=cfg,
        config={"native_multiscale": True, "native_scale_factors": [0.8, 1.0, 1.25]},
    )
    assert err is None
    assert out
    assert out[0]["details"].get("native_multiscale") is True
    assert int(out[0].get("support_count", 1)) >= 2
