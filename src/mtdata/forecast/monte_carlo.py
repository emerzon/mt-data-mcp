from __future__ import annotations

"""Monte Carlo simulation utilities, including a simple HMM-based regime model.

This module provides two primary helpers:
- simulate_gbm_mc: Calibrate GBM from recent log-returns and simulate paths
- simulate_hmm_mc: Fit a light-weight Gaussian HMM (2–3 states) to log-returns
  using an EM-like procedure and simulate regime-switching returns forward.

No external dependencies beyond numpy/pandas are required. The HMM fitting is a
minimal, robust EM tailored for 1D Gaussian emissions and a small number of
states, with sensible initialization and numerical stability.
"""

from typing import Dict, Tuple, Optional
import numpy as np


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-12))
    z = (x - mu) / sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z * z)


def _normal_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-12))
    z = (x - mu) / sigma
    return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + z * z)


def fit_gaussian_mixture_1d(
    x: np.ndarray, n_states: int = 2, max_iter: int = 50, tol: float = 1e-6, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a 1D Gaussian mixture via EM and return (weights, means, sigmas, gamma, ll).

    - x: 1D array of observations (e.g., log-returns)
    - n_states: number of Gaussian components
    - returns:
        w: shape (K,)
        mu: shape (K,)
        sigma: shape (K,)
        gamma: responsibilities, shape (N, K)
        ll: final log-likelihood
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    K = max(1, int(n_states))
    if N < K:
        # Degenerate, fall back to single Gaussian
        mu = np.array([float(np.mean(x))]) if N else np.array([0.0])
        sigma = np.array([float(np.std(x)) + 1e-6])
        w = np.array([1.0])
        gamma = np.ones((N, 1), dtype=float)
        return w, mu, sigma, gamma, float(np.sum(_normal_logpdf(x, mu[0], sigma[0])))

    # Initialization: split by quantiles for stability
    qs = np.linspace(0.1, 0.9, K)
    mu = np.array([np.quantile(x, q) for q in qs], dtype=float)
    sigma = np.full(K, float(np.std(x) + 1e-6), dtype=float)
    w = np.full(K, 1.0 / K, dtype=float)

    prev_ll = -np.inf
    gamma = np.zeros((N, K), dtype=float)
    for _ in range(int(max_iter)):
        # E-step: responsibilities
        for k in range(K):
            gamma[:, k] = w[k] * _normal_pdf(x, mu[k], sigma[k])
        total = np.sum(gamma, axis=1, keepdims=True)
        total[total <= 0.0] = 1e-18
        gamma /= total

        # M-step
        Nk = np.sum(gamma, axis=0) + 1e-18
        w = Nk / float(N)
        mu = (gamma.T @ x) / Nk
        # unbiased variance estimate guarded from below
        var = np.zeros(K, dtype=float)
        for k in range(K):
            diff = x - mu[k]
            var[k] = np.sum(gamma[:, k] * diff * diff) / Nk[k]
        sigma = np.sqrt(np.clip(var, 1e-12, None))

        # Compute log-likelihood
        log_probs = np.zeros((N, K), dtype=float)
        for k in range(K):
            log_probs[:, k] = np.log(max(w[k], 1e-18)) + _normal_logpdf(x, mu[k], sigma[k])
        ll = float(np.sum(np.log(np.sum(np.exp(log_probs - log_probs.max(axis=1, keepdims=True)), axis=1) + 1e-18) + log_probs.max(axis=1)))
        if ll - prev_ll < tol:
            prev_ll = ll
            break
        prev_ll = ll

    return w, mu, sigma, gamma, prev_ll


def estimate_transition_matrix_from_gamma(gamma: np.ndarray) -> np.ndarray:
    """Estimate a Markov transition matrix from soft assignments gamma.

    Uses expected pairwise transitions xi_t(i->j) ≈ gamma[t,i] * gamma[t+1,j].
    Returns a KxK row-stochastic matrix.
    """
    gamma = np.asarray(gamma, dtype=float)
    N, K = gamma.shape
    if N <= 1 or K <= 0:
        return np.eye(max(1, K), dtype=float)
    counts = np.zeros((K, K), dtype=float)
    for t in range(N - 1):
        gi = gamma[t].reshape(-1, 1)
        gj = gamma[t + 1].reshape(1, -1)
        counts += gi @ gj
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    A = counts / row_sum
    # Guard against numerical issues
    A = np.where(np.isfinite(A), A, 0.0)
    # Ensure rows sum to 1
    A /= np.sum(A, axis=1, keepdims=True)
    return A


def simulate_markov_chain(A: np.ndarray, init: np.ndarray, steps: int, sims: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Simulate discrete Markov states.

    Returns array of shape (sims, steps) with state indices [0..K-1].
    """
    if rng is None:
        rng = np.random.RandomState(123)
    A = np.asarray(A, dtype=float)
    init = np.asarray(init, dtype=float)
    K = A.shape[0]
    out = np.zeros((int(sims), int(steps)), dtype=int)
    for s in range(int(sims)):
        # initial state by init distribution
        st = int(rng.choice(K, p=init / np.sum(init)))
        for t in range(int(steps)):
            out[s, t] = st
            st = int(rng.choice(K, p=A[st]))
    return out


def simulate_hmm_mc(
    prices: np.ndarray,
    horizon: int,
    n_states: int = 2,
    n_sims: int = 500,
    seed: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """Fit a simple Gaussian HMM to log-returns and simulate future price paths.

    Returns a dict with keys:
    - price_paths: (sims, horizon)
    - return_paths: (sims, horizon)
    - state_paths: (sims, horizon)
    - mu: (K,)
    - sigma: (K,)
    - trans: (K,K)
    - init: (K,)
    """
    rng = np.random.RandomState(seed)
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 5:
        raise ValueError("Not enough prices for HMM calibration")
    rets = np.diff(_safe_log(prices))
    rets = rets[np.isfinite(rets)]
    if rets.size < 4:
        raise ValueError("Not enough returns for HMM calibration")

    # Fit mixture and derive transition matrix
    w, mu, sigma, gamma, _ = fit_gaussian_mixture_1d(rets, n_states=n_states, max_iter=80, tol=1e-6, seed=seed)
    A = estimate_transition_matrix_from_gamma(gamma)
    init = gamma[-1] / float(np.sum(gamma[-1]))
    K = mu.shape[0]

    # Simulate state paths
    states = simulate_markov_chain(A, init, steps=int(horizon), sims=int(n_sims), rng=rng)

    # Sample returns conditional on states
    ret_paths = np.zeros_like(states, dtype=float)
    for k in range(K):
        idx = (states == k)
        n_k = int(np.sum(idx))
        if n_k > 0:
            ret_paths[idx] = rng.normal(loc=mu[k], scale=max(sigma[k], 1e-12), size=n_k)

    # Build price paths from last observed price
    last_price = float(prices[-1])
    price_paths = np.zeros_like(ret_paths, dtype=float)
    cur = np.full(int(n_sims), last_price, dtype=float)
    for t in range(int(horizon)):
        cur = cur * np.exp(ret_paths[:, t])
        price_paths[:, t] = cur

    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'state_paths': states,
        'mu': mu,
        'sigma': sigma,
        'trans': A,
        'init': init,
    }


def simulate_gbm_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """Calibrate GBM from historical log-returns and simulate forward paths.

    Returns dict with keys 'price_paths' and 'return_paths'.
    """
    rng = np.random.RandomState(seed)
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 5:
        raise ValueError("Not enough prices for GBM calibration")
    rets = np.diff(_safe_log(prices))
    rets = rets[np.isfinite(rets)]
    if rets.size < 2:
        raise ValueError("Not enough returns for GBM calibration")
    mu = float(np.mean(rets))
    sigma = float(np.std(rets) + 1e-12)
    last_price = float(prices[-1])

    ret_paths = rng.normal(loc=mu, scale=sigma, size=(int(n_sims), int(horizon)))
    price_paths = np.zeros_like(ret_paths)
    cur = np.full(int(n_sims), last_price, dtype=float)
    for t in range(int(horizon)):
        cur = cur * np.exp(ret_paths[:, t])
        price_paths[:, t] = cur
    return {'price_paths': price_paths, 'return_paths': ret_paths}


def summarize_paths(
    price_paths: np.ndarray,
    return_paths: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Summarize simulated paths into per-step statistics: mean and CI bounds.

    Returns keys:
    - price_mean, price_lower, price_upper
    - return_mean, return_lower, return_upper (when return_paths provided)
    """
    q_lo = float(alpha / 2.0)
    q_hi = float(1.0 - alpha / 2.0)
    price_mean = np.nanmean(price_paths, axis=0)
    price_lower = np.nanquantile(price_paths, q_lo, axis=0)
    price_upper = np.nanquantile(price_paths, q_hi, axis=0)
    out: Dict[str, np.ndarray] = {
        'price_mean': price_mean,
        'price_lower': price_lower,
        'price_upper': price_upper,
    }
    if return_paths is not None:
        ret_mean = np.nanmean(return_paths, axis=0)
        ret_lower = np.nanquantile(return_paths, q_lo, axis=0)
        ret_upper = np.nanquantile(return_paths, q_hi, axis=0)
        out.update({
            'return_mean': ret_mean,
            'return_lower': ret_lower,
            'return_upper': ret_upper,
        })
    return out


def gbm_single_barrier_upcross_prob(
    s0: float,
    barrier: float,
    mu: float,
    sigma: float,
    T: float,
) -> float:
    """Closed-form probability that GBM S hits upper barrier before time T.

    For GBM: S_t = S0 * exp((mu - 0.5*sigma^2) t + sigma W_t)
    Let X_t = ln S_t evolve as arithmetic Brownian motion with drift m = mu - 0.5*sigma^2
    and volatility sigma. The probability that sup_{t<=T} X_t >= a (a=ln B) is:

    P = Phi((x0 - a + m T)/(sigma * sqrt(T)))
        + exp(2 m (a - x0) / sigma^2) * Phi((x0 - a - m T)/(sigma * sqrt(T)))

    where x0 = ln S0, a = ln B, Phi is the standard normal CDF.
    """
    import math
    from math import log, sqrt
    from scipy.stats import norm
    if T <= 0 or sigma <= 0 or barrier <= 0 or s0 <= 0:
        return 0.0
    x0 = math.log(float(s0))
    a = math.log(float(barrier))
    m = float(mu) - 0.5 * float(sigma) * float(sigma)
    srt = float(sigma) * math.sqrt(float(T))
    z1 = (x0 - a + m * T) / srt
    z2 = (x0 - a - m * T) / srt
    term = math.exp(2.0 * m * (a - x0) / (sigma * sigma))
    return float(norm.cdf(z1) + term * norm.cdf(z2))
