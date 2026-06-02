from __future__ import annotations

from mtdata.core import denoise as denoise_core


def _raw_list_methods():
    raw = denoise_core.denoise_list_methods
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


def test_denoise_list_methods_compact_returns_capped_quick_catalog(monkeypatch):
    rows = [
        {
            "method": f"m{idx}",
            "available": idx % 2 == 0,
            "requires": "pkg",
            "params": ["window", "alpha"],
            "description": "verbose method description",
        }
        for idx in range(12)
    ]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()(limit=10)

    assert result["count"] == 10
    assert result["total"] == 12
    assert result["has_more"] is True
    assert result["methods_hidden"] == 2
    assert result["columns"] == ["method", "available"]
    assert set(result["methods"][0]) == {"method", "available"}


def test_denoise_list_methods_full_keeps_complete_catalog(monkeypatch):
    rows = [
        {
            "method": "kalman",
            "available": True,
            "requires": "",
            "params": ["process_var"],
            "description": "Kalman smoothing",
        }
    ]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()(detail="full", limit=1)

    assert result["count"] == 1
    assert result["methods"] == rows
    assert "has_more" not in result
