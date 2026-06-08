from __future__ import annotations

from mtdata.core import denoise as denoise_core


def _raw_list_methods():
    raw = denoise_core.denoise_list_methods
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


def test_denoise_list_methods_compact_lists_small_catalog_by_default(monkeypatch):
    rows = [
        {
            "method": f"m{idx}",
            "available": idx % 2 == 0,
            "requires": "pkg",
            "params": ["window", "alpha"],
            "supports": {"causality": ["causal"]},
            "description": "verbose method description",
        }
        for idx in range(12)
    ]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()()

    assert result["count"] == 12
    assert result["total"] == 12
    assert result["has_more"] is False
    assert result["methods_hidden"] == 0
    assert result["columns"] == ["method", "available", "causality"]
    assert set(result["methods"][0]) == {
        "method",
        "available",
        "causality",
    }
    assert result["methods"][0]["causality"] == ["causal"]
    assert "list_all_hint" not in result

    standard = _raw_list_methods()(detail="standard")
    assert standard["detail"] == "standard"
    assert standard["columns"] == [
        "method",
        "available",
        "causality",
        "requires",
        "params",
    ]
    assert standard["methods"][0]["requires"] == "pkg"
    assert standard["methods"][0]["params"] == ["window", "alpha"]


def test_denoise_list_methods_compact_reports_hidden_catalog_hint(monkeypatch):
    rows = [{"method": f"m{idx}", "available": True} for idx in range(35)]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()()

    assert result["count"] == 30
    assert result["total"] == 35
    assert result["has_more"] is True
    assert result["methods_hidden"] == 5
    assert result["list_all_hint"] == "Pass limit=35 to list every method."


def test_denoise_list_methods_full_keeps_complete_catalog(monkeypatch):
    rows = [
        {
            "method": "kalman",
            "available": True,
            "requires": "",
            "params": ["process_var"],
            "supports": {"causality": ["causal"]},
            "description": "Kalman smoothing",
        }
    ]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()(detail="full", limit=1)

    assert result["count"] == 1
    assert result["methods"] == rows
    assert "has_more" not in result


def test_denoise_list_methods_filters_for_causal_available_no_extras(monkeypatch):
    rows = [
        {
            "method": "ema",
            "available": True,
            "requires": "",
            "params": ["span"],
            "supports": {"causality": ["causal", "zero_phase"]},
        },
        {
            "method": "wavelet",
            "available": True,
            "requires": "PyWavelets",
            "params": ["wavelet"],
            "supports": {"causality": ["zero_phase"]},
        },
        {
            "method": "vmd",
            "available": False,
            "requires": "vmdpy",
            "params": ["k"],
            "supports": {"causality": ["zero_phase"]},
        },
    ]
    monkeypatch.setattr(
        denoise_core,
        "_denoise_methods",
        lambda available_only=False: [row for row in rows if row["available"]]
        if available_only
        else rows,
    )

    result = _raw_list_methods()(available_only=True, causality="causal", no_extras=True)

    assert result["count"] == 1
    assert result["total"] == 1
    assert result["available_only"] is True
    assert result["causality"] == "causal"
    assert result["no_extras"] is True
    assert result["methods"][0]["method"] == "ema"
