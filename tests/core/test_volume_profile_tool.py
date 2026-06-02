from types import SimpleNamespace

from mtdata.core import volume_profile as vp


def test_compute_volume_profile_payload_uses_m1_fallback_for_large_auto_window(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01 00:00:00",
                    "open": 1.1000,
                    "high": 1.1010,
                    "low": 1.0990,
                    "close": 1.1005,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )
    monkeypatch.setattr(vp, "fetch_ticks", lambda **_: (_ for _ in ()).throw(AssertionError("no tick fetch")))

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-02-01",
        source="auto",
        bucket_size=0.0005,
        detail="full",
    )

    assert result["success"] is True
    assert result["source"] == "m1_bars"
    assert result["volume_kind"] == "tick_volume"
    assert result["diagnostics"]["auto_fallback_reason"] == "requested window exceeds bounded tick window"
    assert result["buckets"]


def test_compute_volume_profile_payload_uses_tick_rows(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "data": [
                {"time": "2026-01-01 00:00:00.000", "bid": 1.0999, "ask": 1.1001, "mid": 1.1000, "tick_volume": 1, "real_volume": 0},
                {"time": "2026-01-01 00:00:01.000", "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "tick_volume": 2, "real_volume": 0},
                {"time": "2026-01-01 00:00:02.000", "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "tick_volume": 2, "real_volume": 0},
            ]
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01 00:00",
        end="2026-01-01 00:01",
        source="ticks",
        bucket_size=0.0001,
        detail="compact",
    )

    assert result["success"] is True
    assert result["source"] == "ticks"
    assert result["poc"]["level"] == "POC"
    assert "buckets" not in result


def test_compute_volume_profile_payload_exposes_fetch_freshness_and_units(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "as_of": "2026-06-02T12:00:00Z",
            "timezone": "UTC",
            "data_freshness_seconds": 12.5,
            "data_stale": False,
            "data": [
                {"time": "2026-06-02 12:00:00.000", "bid": 1.0999, "ask": 1.1001, "mid": 1.1000, "tick_volume": 2, "real_volume": 0},
                {"time": "2026-06-02 12:00:01.000", "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "tick_volume": 3, "real_volume": 0},
            ],
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        source="ticks",
        bucket_size=0.0001,
    )

    assert result["as_of"] == "2026-06-02T12:00:00Z"
    assert result["timezone"] == "UTC"
    assert result["data_age_seconds"] == 12.5
    assert result["data_stale"] is False
    assert result["units"]["price"] == "absolute_price"
    assert result["units"]["volume"] == "tick_volume"


def test_compute_volume_profile_payload_uses_limit_as_tick_cap(monkeypatch):
    captured = {}
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )

    def fake_fetch_ticks(**kwargs):
        captured.update(kwargs)
        return {
            "data": [
                {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "mid": 1.1000,
                    "tick_volume": 1,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        source="ticks",
        limit=5000,
        bucket_size=0.0001,
    )

    assert result["success"] is True
    assert result["source"] == "ticks"
    assert result["window"] == {"start": None, "end": None}
    assert captured["limit"] == 5000
    assert captured["start"] is None
    assert captured["end"] is None


def test_compute_volume_profile_payload_auto_falls_back_on_low_tick_mid_coverage(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "data": [
                {"bid": None, "ask": 1.1001, "tick_volume": 1, "real_volume": 0},
                {"bid": 1.1000, "ask": None, "tick_volume": 1, "real_volume": 0},
            ]
        },
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01 00:00:00",
                    "open": 1.1000,
                    "high": 1.1010,
                    "low": 1.0990,
                    "close": 1.1005,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        source="auto",
        bucket_size=0.0005,
    )

    assert result["success"] is True
    assert result["source"] == "m1_bars"
    assert result["diagnostics"]["auto_fallback_reason"] == "tick price coverage below threshold"
    assert result["diagnostics"]["tick_price_quality"] == {
        "price_source": "mid",
        "input_rows": 2,
        "valid_price_rows": 0,
        "dropped_price_rows": 2,
        "valid_price_ratio": 0.0,
    }


def test_compute_volume_profile_payload_derives_window_from_timeframe_limit(monkeypatch):
    captured = {}
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )

    def fake_fetch_ticks(**kwargs):
        captured.update(kwargs)
        return {
            "data": [
                {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "mid": 1.1000,
                    "tick_volume": 1,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        end="2026-01-02 00:00:00",
        timeframe="H1",
        limit=24,
        source="ticks",
        bucket_size=0.0001,
    )

    assert result["success"] is True
    assert result["window"] == {
        "start": "2026-01-01 00:00:00",
        "end": "2026-01-02 00:00:00",
    }
    assert captured["start"] == "2026-01-01 00:00:00"
    assert captured["end"] == "2026-01-02 00:00:00"


def test_compute_volume_profile_payload_defaults_timeframe_limit(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )

    def fake_fetch_ticks(**kwargs):
        captured.update(kwargs)
        return {
            "data": [
                {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "mid": 1.1000,
                    "tick_volume": 1,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        end="2026-01-02 00:00:00",
        timeframe="H1",
        source="ticks",
        bucket_size=0.0001,
    )

    assert result["success"] is True
    assert result["window"] == {
        "start": "2025-12-24 16:00:00",
        "end": "2026-01-02 00:00:00",
    }
    assert captured["start"] == "2025-12-24 16:00:00"
    assert captured["end"] == "2026-01-02 00:00:00"


def test_compute_volume_profile_payload_invalid_limit_suggests_default() -> None:
    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        timeframe="H1",
        limit=0,
    )

    assert result == {
        "error": (
            "limit must be a positive integer when timeframe is provided; "
            "omit limit to use the default 200 bars."
        )
    }


class _Guard:
    def __init__(self, err, info):
        self.err = err
        self.info = info

    def __enter__(self):
        return self.err, self.info

    def __exit__(self, *args):
        return False
