"""Tests for forecast optimization hints feature."""

from unittest.mock import patch

import pytest

from mtdata.forecast.optimize import (
    build_comprehensive_search_space,
    composite_fitness_score,
    extract_method_params_from_genotype,
    scale_metric_to_01,
)
from mtdata.forecast.requests import ForecastOptimizeHintsRequest
from mtdata.forecast.tune import genetic_search_optimize_hints


class TestScaleMetricTo01:
    """Test metric scaling to [0, 1]."""

    def test_none_value_returns_zero(self):
        assert scale_metric_to_01(None) == 0.0

    def test_normal_scaling(self):
        # Value at min returns 0
        assert scale_metric_to_01(0.0, vmin=0.0, vmax=1.0) == 0.0
        # Value at max returns 1
        assert scale_metric_to_01(1.0, vmin=0.0, vmax=1.0) == 1.0
        # Value in middle returns 0.5
        assert abs(scale_metric_to_01(0.5, vmin=0.0, vmax=1.0) - 0.5) < 1e-6

    def test_clamps_to_01(self):
        assert scale_metric_to_01(-10.0, vmin=0.0, vmax=1.0) == 0.0
        assert scale_metric_to_01(10.0, vmin=0.0, vmax=1.0) == 1.0

    def test_invalid_vmin_vmax(self):
        # When vmin >= vmax, returns 0.5
        assert scale_metric_to_01(5.0, vmin=10.0, vmax=10.0) == 0.5

    def test_nan_and_inf(self):
        assert scale_metric_to_01(float('nan')) == 0.0
        assert scale_metric_to_01(float('inf')) == 0.0


class TestCompositeFitnessScore:
    """Test composite fitness score calculation."""

    def test_empty_metrics_returns_zero(self):
        # Empty metrics should still accumulate some score from 0.5 defaults
        score = composite_fitness_score({})
        assert 0.0 <= score <= 1.0

    def test_missing_metrics_handled_gracefully(self):
        # Should not crash with missing metrics
        metrics = {}
        score = composite_fitness_score(metrics)
        assert isinstance(score, float)

    def test_default_weights(self):
        metrics = {
            'sharpe_ratio': 1.0,
            'win_rate': 0.6,
            'max_drawdown': 0.1,
            'avg_return_per_trade': 0.01,
        }
        score = composite_fitness_score(metrics)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        metrics = {
            'sharpe_ratio': 1.0,
            'win_rate': 0.6,
            'max_drawdown': 0.1,
            'avg_return_per_trade': 0.01,
        }
        custom_weights = {
            'sharpe_ratio': 1.0,  # 100% on Sharpe
            'win_rate': 0.0,
            'inverse_max_drawdown': 0.0,
            'avg_return': 0.0,
        }
        score = composite_fitness_score(metrics, weights=custom_weights)
        # Score should be dominated by sharpe scaling
        assert 0.0 <= score <= 1.0

    def test_none_metric_values_handled(self):
        metrics = {
            'sharpe_ratio': None,
            'win_rate': 0.6,
        }
        score = composite_fitness_score(metrics)
        assert 0.0 <= score <= 1.0


class TestBuildComprehensiveSearchSpace:
    """Test search space builder."""

    def test_default_space(self):
        space = build_comprehensive_search_space()
        assert 'timeframe' in space
        assert 'method' in space
        assert '_method_spaces' in space
        assert space['timeframe']['type'] == 'categorical'
        assert space['method']['type'] == 'categorical'

    def test_custom_timeframes(self):
        space = build_comprehensive_search_space(timeframes=['H1', 'D1'])
        assert set(space['timeframe']['choices']) == {'H1', 'D1'}

    def test_custom_methods(self):
        space = build_comprehensive_search_space(methods=['theta', 'naive'])
        assert set(space['method']['choices']) == {'theta', 'naive'}

    def test_include_features(self):
        space = build_comprehensive_search_space(include_features=True)
        assert 'features' in space
        assert 'rsi' in space['features']
        assert 'macd' in space['features']


class TestExtractMethodParamsFromGenotype:
    """Test genotype extraction."""

    def test_extract_params_from_genotype(self):
        search_space = build_comprehensive_search_space(
            methods=['theta', 'naive'],
        )
        genotype = {
            'timeframe': 'H4',
            'method': 'theta',
            'seasonality': 24,
        }
        tf, method, params = extract_method_params_from_genotype(genotype, search_space)
        assert tf == 'H4'
        assert method == 'theta'
        assert params.get('seasonality') == 24

    def test_handles_missing_keys(self):
        search_space = build_comprehensive_search_space()
        genotype = {}
        tf, method, params = extract_method_params_from_genotype(genotype, search_space)
        assert isinstance(tf, str)
        assert isinstance(method, str)
        assert isinstance(params, dict)


class TestForecastOptimizeHintsRequest:
    """Test request model."""

    def test_valid_request(self):
        req = ForecastOptimizeHintsRequest(
            symbol='EURUSD',
            population=20,
            generations=10,
        )
        assert req.symbol == 'EURUSD'
        assert req.population == 20
        assert req.generations == 10

    def test_default_values(self):
        req = ForecastOptimizeHintsRequest(symbol='EURUSD')
        assert req.fitness_metric == 'composite'
        assert req.population == 20
        assert req.generations == 15
        assert req.top_n == 5


def test_genetic_search_optimize_hints_uses_nested_backtest_metrics():
    backtest_result = {
        "results": {
            "theta": {
                "success": True,
                "avg_rmse": 0.12,
                "metrics": {
                    "sharpe_ratio": 1.5,
                    "win_rate": 0.61,
                    "max_drawdown": 0.11,
                    "avg_return_per_trade": 0.008,
                    "calmar_ratio": 1.8,
                    "annual_return": 0.14,
                },
            }
        }
    }

    with patch("mtdata.forecast.tune._eval_candidate", return_value=(0.12, backtest_result)):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta"],
            horizon=6,
            steps=5,
            spacing=12,
            population=2,
            generations=1,
            top_n=1,
            fitness_metric="composite",
            seed=42,
        )

    hint = result["hints"][0]
    metrics = hint["backtest_metrics"]
    assert metrics["sharpe_ratio"] == 1.5
    assert metrics["win_rate"] == 0.61
    assert metrics["avg_return_per_trade"] == 0.008
    assert hint["fitness_score"] > 0.1


def test_genetic_search_optimize_hints_deduplicates_identical_configs():
    backtest_result = {
        "results": {
            "naive": {
                "success": True,
                "avg_rmse": 0.12,
                "metrics": {"win_rate": 0.5},
            }
        }
    }

    with patch("mtdata.forecast.tune._eval_candidate", return_value=(0.12, backtest_result)):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive"],
            horizon=6,
            steps=5,
            spacing=12,
            population=4,
            generations=1,
            top_n=3,
            fitness_metric="avg_rmse",
            seed=42,
        )

    assert len(result["hints"]) == 1
    assert result["hints"][0]["rank"] == 1
    assert result["search_summary"]["unique_configs_returned"] == 1
    assert result["search_summary"]["duplicate_results_filtered"] > 0


@pytest.mark.skip(reason="Long-running integration test; run manually")
class TestGeneticSearchOptimizeHints:
    """Integration test for genetic search (skipped by default)."""

    def test_basic_search(self):
        """Basic search should complete and return hints."""
        result = genetic_search_optimize_hints(
            symbol='EURUSD',
            timeframes=['H1'],
            methods=['theta', 'naive'],
            horizon=12,
            steps=3,
            spacing=10,
            population=4,
            generations=2,
            fitness_metric='composite',
            top_n=2,
        )
        assert result['success'] is True
        assert 'hints' in result
        assert 'search_summary' in result
        assert len(result['hints']) <= 2
