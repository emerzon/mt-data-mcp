from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..monte_carlo import simulate_gbm_mc, simulate_hmm_mc
from ..registry import ForecastRegistry


def _ci_from_sims(paths: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    lo_q = float(alpha / 2.0)
    hi_q = float(1.0 - alpha / 2.0)
    lo = np.quantile(paths, lo_q, axis=0)
    hi = np.quantile(paths, hi_q, axis=0)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _series_looks_like_prices(series: pd.Series) -> bool:
    x = np.asarray(series.values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return False
    if np.any(x <= 0):
        return False
    # Heuristic: prices are typically much larger in magnitude than returns.
    return float(np.nanmedian(np.abs(x))) > 1.0


@ForecastRegistry.register("mc_gbm")
class MonteCarloGBMMethod(ForecastMethod):
    @property
    def name(self) -> str:
        return "mc_gbm"

    @property
    def category(self) -> str:
        return "monte_carlo"

    @property
    def required_packages(self) -> List[str]:
        return ["numpy"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": False, "ci": True}

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> ForecastResult:
        fh = int(horizon)
        n_sims = int(params.get("n_sims", 500))
        seed = int(params.get("seed", 42))
        ci_alpha = params.get("ci_alpha", None)

        x = np.asarray(series.values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            raise ValueError("Not enough history for Monte Carlo simulation")

        if _series_looks_like_prices(series):
            prices = x
            mu_override = params.get("mu", None)
            sigma_override = params.get("sigma", None)
            if mu_override is None and sigma_override is None:
                sim = simulate_gbm_mc(prices=prices, horizon=fh, n_sims=n_sims, seed=seed)
                paths = np.asarray(sim["price_paths"], dtype=float)
                point = np.median(paths, axis=0)
                ci = None
                if isinstance(ci_alpha, (float, int)) and 0.0 < float(ci_alpha) < 1.0:
                    ci = _ci_from_sims(paths, float(ci_alpha))
                meta = {"n_sims": n_sims, "seed": seed}
                return ForecastResult(forecast=point, ci_values=ci, params_used=params, metadata=meta)

            rets = np.diff(np.log(np.clip(prices, 1e-12, None)))
            mu = float(mu_override) if mu_override is not None else float(np.mean(rets))
            sigma = float(sigma_override) if sigma_override is not None else float(np.std(rets) + 1e-12)
            rng = np.random.RandomState(seed)
            ret_paths = rng.normal(loc=mu, scale=max(sigma, 1e-12), size=(n_sims, fh))
            price_paths = np.zeros_like(ret_paths, dtype=float)
            cur = np.full(n_sims, float(prices[-1]), dtype=float)
            for t in range(fh):
                cur = cur * np.exp(ret_paths[:, t])
                price_paths[:, t] = cur
            point = np.median(price_paths, axis=0)
            ci = None
            if isinstance(ci_alpha, (float, int)) and 0.0 < float(ci_alpha) < 1.0:
                ci = _ci_from_sims(price_paths, float(ci_alpha))
            meta = {"n_sims": n_sims, "seed": seed, "mu": mu, "sigma": sigma}
            return ForecastResult(forecast=point, ci_values=ci, params_used=params, metadata=meta)

        # Return-series target: simulate Gaussian log-returns directly.
        rets = x
        mu = float(params.get("mu", float(np.mean(rets))))
        sigma = float(params.get("sigma", float(np.std(rets) + 1e-12)))
        rng = np.random.RandomState(seed)
        ret_paths = rng.normal(loc=mu, scale=max(sigma, 1e-12), size=(n_sims, fh))
        point = np.median(ret_paths, axis=0)
        ci = None
        if isinstance(ci_alpha, (float, int)) and 0.0 < float(ci_alpha) < 1.0:
            ci = _ci_from_sims(ret_paths, float(ci_alpha))
        meta = {"n_sims": n_sims, "seed": seed, "mu": mu, "sigma": sigma, "target": "return"}
        return ForecastResult(forecast=point, ci_values=ci, params_used=params, metadata=meta)


@ForecastRegistry.register("hmm_mc")
class MonteCarloHMMMethod(ForecastMethod):
    @property
    def name(self) -> str:
        return "hmm_mc"

    @property
    def category(self) -> str:
        return "monte_carlo"

    @property
    def required_packages(self) -> List[str]:
        return ["numpy"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": False, "ci": True}

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> ForecastResult:
        fh = int(horizon)
        n_sims = int(params.get("n_sims", 500))
        seed = int(params.get("seed", 42))
        n_states = int(params.get("n_states", 2))
        ci_alpha = params.get("ci_alpha", None)

        x = np.asarray(series.values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            raise ValueError("Not enough history for Monte Carlo simulation")

        if _series_looks_like_prices(series):
            sim = simulate_hmm_mc(prices=x, horizon=fh, n_states=n_states, n_sims=n_sims, seed=seed)
            paths = np.asarray(sim["price_paths"], dtype=float)
            point = np.median(paths, axis=0)
            ci = None
            if isinstance(ci_alpha, (float, int)) and 0.0 < float(ci_alpha) < 1.0:
                ci = _ci_from_sims(paths, float(ci_alpha))
            meta = {
                "n_sims": n_sims,
                "seed": seed,
                "n_states": n_states,
                "mu": [float(v) for v in np.asarray(sim.get("mu", []), dtype=float).tolist()],
                "sigma": [float(v) for v in np.asarray(sim.get("sigma", []), dtype=float).tolist()],
            }
            return ForecastResult(forecast=point, ci_values=ci, params_used=params, metadata=meta)

        # Return-series target: treat inputs as log-returns and build a pseudo price series.
        rets = x
        prices = np.exp(np.cumsum(np.concatenate(([0.0], rets))))
        sim = simulate_hmm_mc(prices=prices, horizon=fh, n_states=n_states, n_sims=n_sims, seed=seed)
        ret_paths = np.asarray(sim["return_paths"], dtype=float)
        point = np.median(ret_paths, axis=0)
        ci = None
        if isinstance(ci_alpha, (float, int)) and 0.0 < float(ci_alpha) < 1.0:
            ci = _ci_from_sims(ret_paths, float(ci_alpha))
        meta = {
            "n_sims": n_sims,
            "seed": seed,
            "n_states": n_states,
            "target": "return",
            "mu": [float(v) for v in np.asarray(sim.get("mu", []), dtype=float).tolist()],
            "sigma": [float(v) for v in np.asarray(sim.get("sigma", []), dtype=float).tolist()],
        }
        return ForecastResult(forecast=point, ci_values=ci, params_used=params, metadata=meta)

