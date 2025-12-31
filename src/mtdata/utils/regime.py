from __future__ import annotations

"""Regime and change-point utilities.

Includes a lightweight Bayesian Online Change-Point Detection (BOCPD)
implementation for Gaussian data with unknown mean/variance using a
conjugate Normal-Inverse-Gamma model, following Adams & MacKay (2007).

This file is self-contained (NumPy-only) and suitable for streaming or
offline runs over a few thousand observations.
"""

from typing import Dict
import numpy as np


def _student_t_logpdf(x: np.ndarray, mu: np.ndarray, lam: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Log of Student-t predictive pdf for BOCPD with N-IG posterior.

    Parameters are arrays aligned over run lengths:
      - x: scalar or array broadcastable to params
      - mu: posterior mean
      - lam: kappa (precision scaling for the mean)
      - alpha: shape (nu/2)
      - beta: scale

    Predictive distribution:
      nu = 2*alpha
      scale^2 = beta*(lam+1)/(alpha*lam)
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    lam = np.asarray(lam, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    nu = 2.0 * alpha
    # scale^2
    s2 = beta * (lam + 1.0) / (alpha * lam)
    z = (x - mu)
    # Student-t logpdf up to constants
    # log t_nu(x | mu, s2) = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(nu*pi*s2) - (nu+1)/2*log(1 + (z^2)/(nu*s2))
    from scipy.special import gammaln  # optional, only a few calls
    term1 = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
    term2 = -0.5 * (np.log(nu * np.pi) + np.log(s2))
    term3 = -((nu + 1.0) / 2.0) * np.log(1.0 + (z * z) / (nu * s2))
    return term1 + term2 + term3


def bocpd_gaussian(
    x: np.ndarray,
    hazard_lambda: int = 250,
    max_run_length: int = 1000,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Bayesian Online Change-Point Detection for Gaussian data with unknown mean/variance.

    Returns a dict with:
      - cp_prob: P(run_length=0 | x_1:t) per t (change-point probability)
      - run_length_map: MAP run length per t
      - log_joint: log joint probabilities matrix truncated by max_run_length

    Notes:
      - Complexity is O(T * R) where R = max_run_length.
      - Uses a constant hazard H = 1 / hazard_lambda.
      - Requires scipy for gammaln (used only inside logpdf).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if T == 0:
        return {"cp_prob": np.array([]), "run_length_map": np.array([]), "log_joint": np.zeros((0, 0))}

    H = 1.0 / float(max(1, hazard_lambda))
    R = int(max(10, min(max_run_length, T)))

    # Posterior parameters per run-length
    mu = np.full(R, float(mu0), dtype=float)
    kappa = np.full(R, float(kappa0), dtype=float)
    alpha = np.full(R, float(alpha0), dtype=float)
    beta = np.full(R, float(beta0), dtype=float)

    # log r_t(r): joint prob of data and run-length r at time t
    log_r = -np.inf * np.ones(R, dtype=float)
    log_r[0] = 0.0  # start with r=0

    cp_prob = np.zeros(T, dtype=float)
    rl_map = np.zeros(T, dtype=float)

    for t in range(T):
        xt = x[t]
        # Predictive probabilities for all run-lengths (based on previous posteriors)
        log_pred = _student_t_logpdf(xt, mu, kappa, alpha, beta)

        # Growth probabilities (r -> r+1)
        log_growth = log_pred + log_r + np.log(1.0 - H)

        # Changepoint probability (r -> 0) sum over r
        log_cp = np.logaddexp.reduce(log_pred + log_r + np.log(H))

        # New run-length distribution
        new_log_r = -np.inf * np.ones(R, dtype=float)
        new_log_r[0] = log_cp
        new_log_r[1:] = log_growth[:-1]

        # Normalize to avoid under/overflow
        m = np.max(new_log_r)
        new_log_r = new_log_r - m
        log_r = new_log_r + np.log(np.sum(np.exp(new_log_r))) - np.log(np.sum(np.exp(new_log_r)))  # keep normalized
        # cp probability is P(r=0 | x_1:t)
        probs = np.exp(new_log_r - np.log(np.sum(np.exp(new_log_r))))
        cp_prob[t] = float(probs[0])
        rl_map[t] = float(np.argmax(probs))

        # Update posterior params for all run-lengths with xt (conjugate update)
        # Shift parameters for growth paths; r=0 uses the priors
        # We maintain params aligned to run-length indexes (0..R-1)
        mu_new = np.empty_like(mu)
        kappa_new = np.empty_like(kappa)
        alpha_new = np.empty_like(alpha)
        beta_new = np.empty_like(beta)

        # r=0 (changepoint): reset to prior updated with xt as first point
        kappa_new[0] = kappa0 + 1.0
        mu_new[0] = (kappa0 * mu0 + xt) / kappa_new[0]
        alpha_new[0] = alpha0 + 0.5
        beta_new[0] = beta0 + 0.5 * (kappa0 * (xt - mu0) ** 2) / kappa_new[0]

        # r>0: grow previous segments
        mu_prev = mu[:-1]
        kappa_prev = kappa[:-1]
        alpha_prev = alpha[:-1]
        beta_prev = beta[:-1]
        kappa_new[1:] = kappa_prev + 1.0
        mu_new[1:] = (kappa_prev * mu_prev + xt) / kappa_new[1:]
        alpha_new[1:] = alpha_prev + 0.5
        beta_new[1:] = beta_prev + 0.5 * (kappa_prev * (xt - mu_prev) ** 2) / kappa_new[1:]

        mu, kappa, alpha, beta = mu_new, kappa_new, alpha_new, beta_new

    # log_joint matrix is not returned to save memory (can be added if needed)
    return {
        "cp_prob": cp_prob,
        "run_length_map": rl_map,
    }

