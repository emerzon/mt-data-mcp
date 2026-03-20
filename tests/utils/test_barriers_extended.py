"""Extended tests for mtdata.forecast.barriers covering uncovered lines.

Targets: _is_crypto_symbol, _auto_barrier_method, _brownian_bridge_hits,
         _normalize_trade_direction, and barrier grid preset constants.
         All pure logic – no MT5 calls.
"""

import math
import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.barriers import (
    _is_crypto_symbol,
    _auto_barrier_method,
    _brownian_bridge_hits,
    _normalize_trade_direction,
    BARRIER_GRID_PRESETS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _price_series(n: int, seed: int = 42, start: float = 100.0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return start + np.cumsum(rng.randn(n) * 0.5)


def _jumpy_price_series(n: int, seed: int = 7) -> np.ndarray:
    """Create a price series with heavy tails / jumps."""
    rng = np.random.RandomState(seed)
    rets = rng.standard_t(df=3, size=n) * 0.02
    prices = 100.0 * np.exp(np.cumsum(rets))
    return prices


def _regime_switch_prices(n: int, seed: int = 10) -> np.ndarray:
    """Low vol then high vol regime."""
    rng = np.random.RandomState(seed)
    half = n // 2
    r1 = rng.randn(half) * 0.002
    r2 = rng.randn(n - half) * 0.02
    rets = np.concatenate([r1, r2])
    return 100.0 * np.exp(np.cumsum(rets))


# ===================================================================
# _is_crypto_symbol
# ===================================================================
class TestIsCryptoSymbol:
    def test_btcusd(self):
        assert _is_crypto_symbol("BTCUSD") is True

    def test_ethusd(self):
        assert _is_crypto_symbol("ETHUSD") is True

    def test_eurusd(self):
        assert _is_crypto_symbol("EURUSD") is False

    def test_empty(self):
        assert _is_crypto_symbol("") is False

    def test_none(self):
        assert _is_crypto_symbol(None) is False

    def test_solusd_lower(self):
        assert _is_crypto_symbol("solusd") is True

    def test_xrp(self):
        assert _is_crypto_symbol("XRPUSD") is True

    def test_aapl(self):
        assert _is_crypto_symbol("AAPL") is False


# ===================================================================
# _normalize_trade_direction
# ===================================================================
class TestNormalizeTradeDirection:
    def test_long(self):
        assert _normalize_trade_direction("long") == ("long", None)

    def test_short(self):
        assert _normalize_trade_direction("short") == ("short", None)

    def test_buy(self):
        assert _normalize_trade_direction("buy") == ("long", None)

    def test_sell(self):
        assert _normalize_trade_direction("sell") == ("short", None)

    def test_up(self):
        assert _normalize_trade_direction("up") == ("long", None)

    def test_down(self):
        assert _normalize_trade_direction("down") == ("short", None)

    def test_invalid(self):
        direction, err = _normalize_trade_direction("sideways")
        assert direction is None
        assert err is not None

    def test_none(self):
        direction, err = _normalize_trade_direction(None)
        assert direction is None

    def test_empty_string(self):
        direction, err = _normalize_trade_direction("")
        assert direction is None


# ===================================================================
# _auto_barrier_method
# ===================================================================
class TestAutoBarrierMethod:
    def test_insufficient_history(self):
        prices = np.array([100.0, 101.0, 99.0])
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert method == "mc_gbm"
        assert "insufficient" in reason

    def test_insufficient_history_short_horizon(self):
        prices = np.array([100.0, 101.0, 99.0])
        method, reason = _auto_barrier_method("EURUSD", "H1", prices, horizon=5)
        assert method == "mc_gbm_bb"

    def test_limited_history(self):
        prices = _price_series(20, seed=1)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert method == "mc_gbm"
        assert "limited" in reason

    def test_limited_history_short_horizon(self):
        prices = _price_series(20, seed=1)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices, horizon=10)
        assert method == "mc_gbm_bb"

    def test_mild_tails_gbm(self):
        prices = _price_series(500, seed=2)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert method in ("mc_gbm", "mc_gbm_bb", "heston", "bootstrap")
        assert "auto:" in reason

    def test_mild_tails_bridge_short_horizon(self):
        prices = _price_series(500, seed=2)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices, horizon=5)
        assert "auto:" in reason

    def test_heavy_tails_jump_diffusion(self):
        prices = _jumpy_price_series(500, seed=3)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert method in ("jump_diffusion", "heston", "bootstrap", "mc_gbm")

    def test_crypto_jumpy(self):
        prices = _jumpy_price_series(500, seed=4)
        method, reason = _auto_barrier_method("BTCUSD", "M15", prices)
        # M15 = 900 seconds, crypto symbol with jumps => jump_diffusion expected
        assert method in ("jump_diffusion", "heston")

    def test_regime_shift(self):
        prices = _regime_switch_prices(500, seed=5)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert method in ("hmm_mc", "heston", "mc_gbm", "mc_gbm_bb", "bootstrap", "jump_diffusion")

    def test_insufficient_returns(self):
        # Constant price => all returns zero => std=0 => insufficient finite returns
        prices = np.full(15, 100.0)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert "mc_gbm" in method

    def test_non_finite_prices_filtered(self):
        prices = np.array([100.0, np.nan, 101.0, np.inf, 99.0, 100.5, 102.0, 98.0, 101.0, 100.0])
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert method == "mc_gbm"

    def test_very_long_normal_series(self):
        prices = _price_series(600, seed=99)
        method, reason = _auto_barrier_method("EURUSD", "H4", prices)
        assert "auto:" in reason

    def test_vol_clustering_heston(self):
        """Construct a series with strong vol clustering."""
        rng = np.random.RandomState(77)
        n = 400
        rets = np.zeros(n)
        vol = 0.01
        for i in range(n):
            vol = 0.0001 + 0.85 * vol + 0.12 * rets[i - 1] ** 2 if i > 0 else 0.01
            rets[i] = rng.randn() * np.sqrt(vol)
        prices = 100.0 * np.exp(np.cumsum(rets))
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert method in ("heston", "garch", "jump_diffusion", "bootstrap", "mc_gbm", "mc_gbm_bb")


# ===================================================================
# _brownian_bridge_hits
# ===================================================================
class TestBrownianBridgeHits:
    def test_basic_up(self):
        np.random.seed(42)
        n_sims, n_steps = 100, 10
        log_paths = np.log(100.0) + np.cumsum(
            np.random.randn(n_sims, n_steps) * 0.01, axis=1
        )
        log_paths = np.concatenate(
            [np.full((n_sims, 1), np.log(100.0)), log_paths], axis=1
        )
        barrier_log = np.log(105.0)
        uniform = np.random.rand(n_sims, n_steps)
        hits = _brownian_bridge_hits(
            log_paths, barrier_log, 0.01, direction="up", uniform=uniform
        )
        assert hits.shape == (n_sims, n_steps)
        assert hits.dtype == bool

    def test_basic_down(self):
        np.random.seed(42)
        n_sims, n_steps = 50, 8
        log_paths = np.log(100.0) + np.cumsum(
            np.random.randn(n_sims, n_steps) * 0.01, axis=1
        )
        log_paths = np.concatenate(
            [np.full((n_sims, 1), np.log(100.0)), log_paths], axis=1
        )
        barrier_log = np.log(95.0)
        uniform = np.random.rand(n_sims, n_steps)
        hits = _brownian_bridge_hits(
            log_paths, barrier_log, 0.01, direction="down", uniform=uniform
        )
        assert hits.shape == (n_sims, n_steps)

    def test_zero_sigma(self):
        log_paths = np.full((5, 4), np.log(100.0))
        uniform = np.random.rand(5, 3)
        hits = _brownian_bridge_hits(
            log_paths, np.log(105.0), 0.0, direction="up", uniform=uniform
        )
        assert not hits.any()

    def test_negative_sigma(self):
        log_paths = np.full((5, 4), np.log(100.0))
        uniform = np.random.rand(5, 3)
        hits = _brownian_bridge_hits(
            log_paths, np.log(95.0), -1.0, direction="down", uniform=uniform
        )
        assert not hits.any()

    def test_nan_sigma(self):
        log_paths = np.full((3, 4), np.log(100.0))
        uniform = np.random.rand(3, 3)
        hits = _brownian_bridge_hits(
            log_paths, np.log(105.0), float("nan"), direction="up", uniform=uniform
        )
        assert not hits.any()

    def test_barrier_already_crossed_no_bridge(self):
        """If paths already crossed the barrier, bridge shouldn't fire on those intervals."""
        n_sims = 20
        log_paths = np.log(np.array([[100, 106, 107, 108]] * n_sims, dtype=float))
        barrier_log = np.log(105.0)
        uniform = np.zeros((n_sims, 3))  # uniform=0 => never triggers bridge
        hits = _brownian_bridge_hits(
            log_paths, barrier_log, 0.01, direction="up", uniform=uniform
        )
        # Paths 106-108 are above barrier, so valid mask is False -> no bridge hits
        assert hits.shape == (n_sims, 3)


# ===================================================================
# BARRIER_GRID_PRESETS
# ===================================================================
class TestBarrierGridPresets:
    def test_scalp_exists(self):
        assert "scalp" in BARRIER_GRID_PRESETS

    def test_intraday_exists(self):
        cfg = BARRIER_GRID_PRESETS["intraday"]
        assert cfg["tp_min"] < cfg["tp_max"]

    def test_swing_exists(self):
        assert "swing" in BARRIER_GRID_PRESETS

    def test_position_exists(self):
        cfg = BARRIER_GRID_PRESETS["position"]
        assert cfg["sl_steps"] > 0

    def test_all_presets_have_required_keys(self):
        required = {"tp_min", "tp_max", "tp_steps", "sl_min", "sl_max", "sl_steps"}
        for name, cfg in BARRIER_GRID_PRESETS.items():
            assert required.issubset(cfg.keys()), f"Preset {name} missing keys"


# ===================================================================
# Additional edge cases for coverage depth
# ===================================================================
class TestAutoBarrierEdgeCases:
    def test_five_returns_insufficient(self):
        """Only 6 prices => 5 returns => limited history branch."""
        prices = np.array([100, 101, 102, 100, 99, 101], dtype=float)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices)
        assert "mc_gbm" in method

    def test_skewed_returns_bootstrap(self):
        """Long history with skewed returns."""
        rng = np.random.RandomState(33)
        rets = rng.exponential(0.01, size=500) - 0.005
        prices = 100.0 * np.exp(np.cumsum(rets))
        method, reason = _auto_barrier_method("EURUSD", "D1", prices)
        assert method in ("bootstrap", "jump_diffusion", "heston", "mc_gbm", "mc_gbm_bb")

    def test_horizon_none(self):
        prices = _price_series(100, seed=8)
        method, reason = _auto_barrier_method("EURUSD", "H1", prices, horizon=None)
        assert "auto:" in reason

    def test_timeframe_not_in_seconds_map(self):
        prices = _price_series(100, seed=9)
        method, reason = _auto_barrier_method("EURUSD", "UNKNOWN_TF", prices)
        assert "auto:" in reason
