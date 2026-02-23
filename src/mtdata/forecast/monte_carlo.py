from __future__ import annotations

"""Monte Carlo simulation utilities, including a simple HMM-based regime model.

This module provides two primary helpers:
- simulate_gbm_mc: Calibrate GBM from recent log-returns and simulate paths
- simulate_hmm_mc: Fit a light-weight Gaussian HMM (2–3 states) to log-returns
  using an EM-like procedure and simulate regime-switching returns forward.
- simulate_garch_mc: Fit a GARCH(1,1) model (requires 'arch' package) and simulate.
- simulate_bootstrap_mc: Circular block bootstrap from historical returns.
- simulate_heston_mc: Heston stochastic volatility simulation (Euler).
- simulate_jump_diffusion_mc: Merton-style jump diffusion simulation.

This module relies on numpy/pandas plus sklearn (GaussianMixture) and arch
(bootstrap/GARCH).
"""

from typing import Dict, Tuple, Optional
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from arch.bootstrap import CircularBlockBootstrap


def _normalize_probability_vector(
    probs: np.ndarray,
    *,
    fallback_index: Optional[int] = None,
) -> np.ndarray:
    """Normalize a probability vector with robust fallbacks."""
    vec = np.asarray(probs, dtype=float).copy()
    vec = np.where(np.isfinite(vec) & (vec >= 0.0), vec, 0.0)
    total = float(np.sum(vec))
    if total > 0.0:
        return vec / total
    n = int(vec.size)
    if n <= 0:
        return vec
    out = np.zeros(n, dtype=float)
    if fallback_index is not None and 0 <= int(fallback_index) < n:
        out[int(fallback_index)] = 1.0
        return out
    out[:] = 1.0 / float(n)
    return out


def _normalize_transition_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize transition matrix rows; fallback to self-loops on degenerate rows."""
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return np.eye(max(1, int(A.shape[0] if A.ndim > 0 else 1)), dtype=float)
    K = int(A.shape[0])
    out = np.zeros_like(A, dtype=float)
    for i in range(K):
        out[i] = _normalize_probability_vector(A[i], fallback_index=i)
    return out


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def fit_gaussian_mixture_1d(
    x: np.ndarray, n_states: int = 2, max_iter: int = 50, tol: float = 1e-6, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a 1D Gaussian mixture and return (weights, means, sigmas, gamma, ll).

    - x: 1D array of observations (e.g., log-returns)
    - n_states: number of Gaussian components
    - returns:
        w: shape (K,)
        mu: shape (K,)
        sigma: shape (K,)
        gamma: responsibilities, shape (N, K)
        ll: final log-likelihood
    """
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
        ll = float(-0.5 * N * (np.log(2.0 * np.pi) + 2.0 * np.log(max(sigma[0], 1e-12))))
        return w, mu, sigma, gamma, ll

    model = GaussianMixture(
        n_components=K,
        covariance_type='diag',
        reg_covar=1e-12,
        max_iter=int(max_iter),
        tol=float(tol),
        n_init=1,
        random_state=seed,
    )
    x2 = x.reshape(-1, 1)
    model.fit(x2)
    gamma = np.asarray(model.predict_proba(x2), dtype=float)
    w = np.asarray(model.weights_, dtype=float).reshape(-1)
    mu = np.asarray(model.means_, dtype=float).reshape(-1)
    sigma = np.sqrt(np.clip(np.asarray(model.covariances_, dtype=float).reshape(-1), 1e-12, None))
    ll = float(model.score(x2) * N)

    order = np.argsort(mu)
    return w[order], mu[order], sigma[order], gamma[:, order], ll


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
    return _normalize_transition_matrix(counts)


def simulate_markov_chain(A: np.ndarray, init: np.ndarray, steps: int, sims: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Simulate discrete Markov states.

    Returns array of shape (sims, steps) with state indices [0..K-1].
    """
    if rng is None:
        rng = np.random.RandomState(123)
    A = _normalize_transition_matrix(np.asarray(A, dtype=float))
    init = _normalize_probability_vector(np.asarray(init, dtype=float))
    K = A.shape[0]
    out = np.zeros((int(sims), int(steps)), dtype=int)
    for s in range(int(sims)):
        # initial state by init distribution
        st = int(rng.choice(K, p=init))
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
    init = _normalize_probability_vector(gamma[-1])
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

    Returns dict with keys 'price_paths', 'return_paths', 'mu', and 'sigma'.
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
    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'mu': float(mu),
        'sigma': float(sigma),
    }


def simulate_heston_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    kappa: Optional[float] = None,
    theta: Optional[float] = None,
    xi: Optional[float] = None,
    rho: Optional[float] = None,
    v0: Optional[float] = None,
    dt: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Simulate a Heston stochastic volatility model using Euler discretization."""
    rng = np.random.RandomState(seed)
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 10:
        raise ValueError("Not enough prices for Heston calibration")

    rets = np.diff(_safe_log(prices))
    rets = rets[np.isfinite(rets)]
    if rets.size < 5:
        raise ValueError("Not enough returns for Heston calibration")

    mu = float(np.mean(rets))
    ret_var = float(np.var(rets, ddof=1))
    ret_var = max(ret_var, 1e-10)

    theta_val = float(theta) if theta is not None else ret_var
    v0_val = float(v0) if v0 is not None else float(np.var(rets[-50:], ddof=1) if rets.size >= 50 else ret_var)
    kappa_val = float(kappa) if kappa is not None else 2.0
    xi_val = float(xi) if xi is not None else max(1e-6, 0.5 * np.sqrt(theta_val))
    rho_val = float(rho) if rho is not None else -0.3
    rho_val = float(np.clip(rho_val, -0.99, 0.99))

    last_price = float(prices[-1])
    price_paths = np.zeros((int(n_sims), int(horizon)), dtype=float)
    ret_paths = np.zeros_like(price_paths, dtype=float)
    vol_paths = np.zeros_like(price_paths, dtype=float)

    cur = np.full(int(n_sims), last_price, dtype=float)
    v = np.full(int(n_sims), max(v0_val, 1e-10), dtype=float)
    sqrt_dt = float(np.sqrt(dt))

    for t in range(int(horizon)):
        z1 = rng.normal(size=int(n_sims))
        z2 = rng.normal(size=int(n_sims))
        z2 = rho_val * z1 + np.sqrt(max(1.0 - rho_val * rho_val, 0.0)) * z2

        v_pos = np.clip(v, 0.0, None)
        dv = kappa_val * (theta_val - v_pos) * dt + xi_val * np.sqrt(v_pos) * sqrt_dt * z2
        v = np.clip(v_pos + dv, 1e-10, None)

        ret = (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1
        cur = cur * np.exp(ret)

        ret_paths[:, t] = ret
        price_paths[:, t] = cur
        vol_paths[:, t] = v

    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'vol_paths': vol_paths,
        'params': {
            'mu': mu,
            'kappa': kappa_val,
            'theta': theta_val,
            'xi': xi_val,
            'rho': rho_val,
            'v0': v0_val,
        },
    }


def simulate_jump_diffusion_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    jump_lambda: Optional[float] = None,
    jump_mu: Optional[float] = None,
    jump_sigma: Optional[float] = None,
    jump_threshold: float = 3.0,
    dt: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Simulate a Merton jump-diffusion model."""
    rng = np.random.RandomState(seed)
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 10:
        raise ValueError("Not enough prices for jump diffusion calibration")

    rets = np.diff(_safe_log(prices))
    rets = rets[np.isfinite(rets)]
    if rets.size < 5:
        raise ValueError("Not enough returns for jump diffusion calibration")

    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1)) + 1e-12

    jump_mask = np.abs(rets - mu) > (float(jump_threshold) * sigma)
    jump_rets = rets[jump_mask]
    jump_freq = float(jump_rets.size) / float(max(1, rets.size))
    lambda_val = float(jump_lambda) if jump_lambda is not None else float(np.clip(jump_freq, 0.0, 1.0))
    if jump_rets.size >= 3:
        mu_j = float(jump_mu) if jump_mu is not None else float(np.mean(jump_rets))
        sigma_j = float(jump_sigma) if jump_sigma is not None else float(np.std(jump_rets, ddof=1))
    else:
        mu_j = float(jump_mu) if jump_mu is not None else 0.0
        sigma_j = float(jump_sigma) if jump_sigma is not None else float(max(0.5 * sigma, 1e-6))

    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    drift = mu - lambda_val * k

    last_price = float(prices[-1])
    price_paths = np.zeros((int(n_sims), int(horizon)), dtype=float)
    ret_paths = np.zeros_like(price_paths, dtype=float)

    cur = np.full(int(n_sims), last_price, dtype=float)
    sqrt_dt = float(np.sqrt(dt))

    for t in range(int(horizon)):
        z = rng.normal(size=int(n_sims))
        n_j = rng.poisson(lam=lambda_val * dt, size=int(n_sims))
        jump = mu_j * n_j + sigma_j * np.sqrt(n_j.astype(float)) * rng.normal(size=int(n_sims))
        ret = drift * dt + sigma * sqrt_dt * z + jump
        cur = cur * np.exp(ret)
        ret_paths[:, t] = ret
        price_paths[:, t] = cur

    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'params': {
            'mu': mu,
            'sigma': sigma,
            'jump_lambda': lambda_val,
            'jump_mu': mu_j,
            'jump_sigma': sigma_j,
            'jump_threshold': float(jump_threshold),
        },
    }


def simulate_garch_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    p_order: int = 1,
    q_order: int = 1,
) -> Dict[str, np.ndarray]:
    """Fit GARCH(p,q) model and simulate forward paths.
    
    Requires 'arch' package.
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("The 'arch' library is required for GARCH simulations.")

    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 50:  # GARCH needs decent history
        raise ValueError("Not enough prices for GARCH calibration (need > 50)")
    
    rets = np.diff(_safe_log(prices))
    rets = rets[np.isfinite(rets)]
    
    # Scale returns for numerical stability (common practice with GARCH)
    scale = 100.0
    rets_scaled = rets * scale
    
    # Fit GARCH(p,q)
    # Use Mean='Zero' assuming daily drift is negligible compared to vol for short horizons,
    # or 'Constant' to capture drift. Let's use Constant.
    am = arch_model(rets_scaled, vol='Garch', p=p_order, q=q_order, dist='Normal', mean='Constant')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # disp='off' prevents printing convergence info
        res = am.fit(disp='off', show_warning=False)
    
    # Forecast via simulation
    # 'simulations' arg is number of paths
    sim_rng = np.random.RandomState(int(seed) if seed is not None else 42)
    forecasts = res.forecast(
        horizon=horizon,
        method='simulation',
        simulations=n_sims,
        rng=sim_rng.standard_normal,
        random_state=sim_rng,
        reindex=False,
    )
    
    # Get simulated paths (1, n_sims, horizon) -> (n_sims, horizon)
    # forecasts.simulations.values contains the simulated returns
    sim_rets_scaled = forecasts.simulations.values[-1]
    
    # Unscale
    sim_rets = sim_rets_scaled / scale
    
    # Reconstruct prices
    last_price = float(prices[-1])
    # cumsum of log returns
    cum_rets = np.cumsum(sim_rets, axis=1)
    price_paths = last_price * np.exp(cum_rets)
    
    return {
        'price_paths': price_paths,
        'return_paths': sim_rets,
        'model_summary': str(res.summary()),
    }


def simulate_bootstrap_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    block_size: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Circular Block Bootstrap simulation."""
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    rets = np.diff(_safe_log(prices))
    rets = rets[np.isfinite(rets)]
    n = len(rets)
    
    if n < 10:
        raise ValueError("Not enough returns for bootstrapping")
    
    if block_size is None:
        # Politis & White rule of thumb approx n^(1/3)
        block_size = int(max(1, n ** (1.0/3.0)))
    
    block_size = max(1, int(block_size))
    bs = CircularBlockBootstrap(block_size, rets, seed=seed)
    sim_rets = np.zeros((int(n_sims), int(horizon)), dtype=float)
    for s_idx, boot in enumerate(bs.bootstrap(int(n_sims))):
        sample = np.asarray(boot[0][0], dtype=float).reshape(-1)
        if sample.size < int(horizon):
            reps = int(np.ceil(int(horizon) / max(1, sample.size)))
            sample = np.tile(sample, reps)
        sim_rets[s_idx] = sample[: int(horizon)]
    
    # Reconstruct prices
    last_price = float(prices[-1])
    cum_rets = np.cumsum(sim_rets, axis=1)
    price_paths = last_price * np.exp(cum_rets)
    
    return {
        'price_paths': price_paths,
        'return_paths': sim_rets,
        'block_size': np.array(block_size) # store as array for consistency
    }


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

    try:
        s0_f = float(s0)
        barrier_f = float(barrier)
        mu_f = float(mu)
        sigma_f = float(sigma)
        T_f = float(T)
    except Exception:
        return 0.0

    if not all(map(math.isfinite, (s0_f, barrier_f, mu_f, sigma_f, T_f))):
        return 0.0

    if s0_f <= 0.0 or barrier_f <= 0.0:
        return 0.0

    # Already at/above the upper barrier at t=0.
    if barrier_f <= s0_f:
        return 1.0

    if T_f <= 0.0:
        return 0.0

    # Deterministic limiting case.
    if sigma_f <= 0.0:
        if mu_f <= 0.0:
            return 0.0
        return 1.0 if (s0_f * math.exp(mu_f * T_f) >= barrier_f) else 0.0

    from scipy.stats import norm

    x0 = math.log(s0_f)
    a = math.log(barrier_f)
    m = mu_f - 0.5 * sigma_f * sigma_f
    srt = sigma_f * math.sqrt(T_f)
    z1 = (x0 - a + m * T_f) / srt
    z2 = (x0 - a - m * T_f) / srt
    term = math.exp(2.0 * m * (a - x0) / (sigma_f * sigma_f))
    p = float(norm.cdf(z1) + term * norm.cdf(z2))
    if not math.isfinite(p):
        return 0.0
    return float(min(1.0, max(0.0, p)))
