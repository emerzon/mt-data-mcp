from __future__ import annotations

import random

from mtdata.forecast import tune


def test_default_search_space_modes():
    multi = tune.default_search_space(methods=["theta", "fourier_ols"])
    assert "_shared" in multi
    assert "theta" in multi and "seasonality" in multi["theta"]
    assert "fourier_ols" in multi and "K" in multi["fourier_ols"]

    single_known = tune.default_search_space(method="theta")
    assert "seasonality" in single_known

    single_unknown = tune.default_search_space(method="unknown")
    assert single_unknown == {"seasonality": {"type": "int", "min": 8, "max": 48}}

    none_given = tune.default_search_space()
    assert "_shared" in none_given
    assert "theta" in none_given


def test_sample_and_mutate_param_helpers():
    rng = random.Random(7)

    assert tune._sample_param({"type": "categorical", "choices": ["a", "b"]}, rng) in {"a", "b"}
    assert tune._sample_param({"type": "categorical", "choices": []}, rng) is None

    assert tune._sample_param({"type": "int", "min": 5, "max": 3}, rng) in {3, 4, 5}
    assert isinstance(tune._sample_param({"type": "float", "min": 0.1, "max": 0.2, "log": True}, rng), float)

    assert tune._mutate_value("a", {"type": "categorical", "choices": ["a", "b"]}, rng) == "b"
    assert tune._mutate_value("a", {"type": "categorical", "choices": ["a"]}, rng) == "a"
    assert tune._mutate_value(5, {"type": "int", "min": 0, "max": 10}, rng) >= 0
    assert 0.0 <= tune._mutate_value(0.5, {"type": "float", "min": 0.0, "max": 1.0}, rng) <= 1.0


def test_crossover_for_method_blends_and_fills_none():
    rng = random.Random(13)
    a = {"x": 1.0, "cat": "a", "i": None}
    b = {"x": 3.0, "cat": "b", "i": None}
    spaces = {
        "x": {"type": "float", "min": 0.0, "max": 10.0},
        "cat": {"type": "categorical", "choices": ["a", "b"]},
        "i": {"type": "int", "min": 1, "max": 2},
    }

    child = tune._crossover_for_method(a, b, spaces, rng)

    assert child["x"] == 2.0
    assert child["cat"] in {"a", "b"}
    assert child["i"] in {1, 2}


def test_eval_candidate_handles_method_selection_and_failures(monkeypatch):
    def fake_backtest(**kwargs):
        m = kwargs["methods"][0]
        if m == "bad":
            return {"results": {m: {"success": False}}}
        if m == "missing_metric":
            return {"results": {m: {"success": True, "avg_mae": 2.5}}}
        return {"results": {m: {"success": True, "avg_rmse": 1.2}}}

    monkeypatch.setattr(tune, "_forecast_backtest", fake_backtest)

    score, result = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={},
        metric="avg_rmse",
        mode="min",
    )
    assert score == 1.2
    assert result["_sel_method"] == "theta"

    score, _ = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={"method": "bad"},
        metric="avg_rmse",
        mode="min",
    )
    assert score == float("inf")

    score, result = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={"method": "missing_metric"},
        metric="avg_rmse",
        mode="max",
    )
    assert score == -2.5
    assert result["_sel_method"] == "missing_metric"

    score, result = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={},
        metric="avg_rmse",
        mode="min",
    )
    assert score == float("inf")
    assert result["error"] == "No method provided"


def test_genetic_search_method_scoped_and_flat_spaces(monkeypatch):
    def fake_eval_candidate(**kwargs):
        cand = kwargs["candidate_params"]
        m = cand.get("method") or kwargs.get("method") or "theta"
        score_val = float(cand.get("x", 1.0))
        return (
            score_val if kwargs["mode"] == "min" else -score_val,
            {"_sel_method": m, "results": {m: {"horizon": kwargs["horizon"], "success": True}}},
        )

    monkeypatch.setattr(tune, "_eval_candidate", fake_eval_candidate)

    method_scoped_space = {
        "_shared": {"x": {"type": "float", "min": 0.1, "max": 1.0}},
        "theta": {"k": {"type": "int", "min": 1, "max": 3}},
        "naive": {"k": {"type": "int", "min": 4, "max": 6}},
    }
    out = tune.genetic_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        methods=["theta", "naive"],
        horizon=3,
        steps=2,
        spacing=1,
        search_space=method_scoped_space,
        population=4,
        generations=2,
        seed=11,
    )
    assert out["success"] is True
    assert out["history_count"] == 8
    assert out["best_method"] in {"theta", "naive"}
    assert "best_result_summary" in out
    assert len(out["history_tail"]) <= 50

    flat_space = {
        "method": {"type": "categorical", "choices": ["theta", "naive"]},
        "x": {"type": "float", "min": 0.2, "max": 0.9},
    }
    out = tune.genetic_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        methods=None,
        horizon=2,
        steps=2,
        spacing=1,
        search_space=flat_space,
        mode="max",
        population=3,
        generations=2,
        seed=17,
    )
    assert out["success"] is True
    assert out["mode"] == "max"
    assert out["history_count"] == 6
