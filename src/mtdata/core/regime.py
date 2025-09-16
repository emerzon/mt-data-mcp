from typing import Any, Dict, Optional, List, Literal
import numpy as np
import pandas as pd

from .server import mcp, _auto_connect_wrapper
from .schema import TimeframeLiteral, DenoiseSpec
from ..forecast.common import fetch_history as _fetch_history
from ..utils.utils import _format_time_minimal as _format_time_minimal_util
from ..utils.denoise import _apply_denoise as _apply_denoise_util


@mcp.tool()
@_auto_connect_wrapper
def detect_regimes(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 800,
    method: Literal['bocpd','hmm','ms_ar'] = 'bocpd',  # type: ignore
    target: Literal['return','price'] = 'return',  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    threshold: float = 0.5,
    output: Literal['full','summary','compact'] = 'full',  # type: ignore
    lookback: int = 300,
) -> Dict[str, Any]:
    """Detect regimes and/or change-points over the last `limit` bars.

    - method: 'bocpd' (Bayesian online change-point; Gaussian), 'hmm' (Gaussian mixture/HMM-lite), or 'ms_ar' (Markov-switching AR via statsmodels if available).
    - target: 'return' (default; log returns) or 'price'.
    - params: method-specific kwargs, e.g., bocpd: hazard_lambda, max_run_length; hmm: n_states; ms_ar: k_regimes, order.
    - denoise: optional denoising on 'close' prior to target transform.
    - threshold: decision threshold for change-point marking (bocpd cp_prob >= threshold).
    """
    try:
        p = dict(params or {})
        df = _fetch_history(symbol, timeframe, int(max(limit, 50)), as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history"}
        base_col = 'close'
        if denoise:
            try:
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        y = df[base_col].astype(float).to_numpy()
        times = df['time'].astype(float).to_numpy()
        if target == 'return':
            with np.errstate(divide='ignore', invalid='ignore'):
                x = np.diff(np.log(np.maximum(y, 1e-12)))
            x = x[np.isfinite(x)]
            t = times[1: 1 + x.size]
        else:
            x = y[np.isfinite(y)]
            t = times[: x.size]

        # format times
        t_fmt = [_format_time_minimal_util(tt) for tt in t]

        if method == 'bocpd':
            from ..utils.regime import bocpd_gaussian
            hazard_lambda = int(p.get('hazard_lambda', 250))
            max_rl = int(p.get('max_run_length', min(1000, x.size)))
            res = bocpd_gaussian(x, hazard_lambda=hazard_lambda, max_run_length=max_rl)
            cp_prob = res.get('cp_prob', np.zeros_like(x, dtype=float))
            cp_idx = [int(i) for i, v in enumerate(cp_prob) if float(v) >= float(threshold)]
            cps = [{"idx": i, "time": t_fmt[i], "prob": float(cp_prob[i])} for i in cp_idx]
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "cp_prob": [float(v) for v in np.asarray(cp_prob, dtype=float).tolist()],
                "change_points": cps,
                "threshold": float(threshold),
                "params_used": {"hazard_lambda": hazard_lambda, "max_run_length": max_rl},
            }
            if output in ('summary','compact'):
                n = min(int(lookback), len(cp_prob))
                tail = np.asarray(cp_prob[-n:], dtype=float) if n > 0 else np.asarray(cp_prob, dtype=float)
                recent_cps = [c for c in cps if c.get('idx', 0) >= (len(cp_prob) - n)]
                summary = {
                    "lookback": int(n),
                    "last_cp_prob": float(cp_prob[-1]) if len(cp_prob) else float('nan'),
                    "max_cp_prob": float(np.nanmax(tail)) if tail.size else float('nan'),
                    "mean_cp_prob": float(np.nanmean(tail)) if tail.size else float('nan'),
                    "change_points_count": int(len(recent_cps)),
                    "recent_change_points": recent_cps[-5:],
                }
                if output == 'summary':
                    payload = {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method, "summary": summary}
                else:  # compact: keep cp_prob only as tail
                    payload["summary"] = summary
                    if n > 0:
                        payload["cp_prob"] = [float(v) for v in tail.tolist()]
                        payload["times"] = t_fmt[-n:]
                        payload["change_points"] = recent_cps
            return payload

        elif method == 'ms_ar':
            try:
                from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression  # type: ignore
            except Exception:
                return {"error": "statsmodels MarkovRegression not available. Install statsmodels."}
            k_regimes = int(p.get('k_regimes', 2))
            order = int(p.get('order', 0))
            try:
                mod = MarkovRegression(endog=x, k_regimes=max(2, k_regimes), trend='c', order=max(0, order), switching_variance=True)
                res = mod.fit(disp=False, maxiter=int(p.get('maxiter', 100)))
                smoothed = res.smoothed_marginal_probabilities
                # choose most probable regime per time
                state = np.argmax(smoothed, axis=0)
                probs = smoothed.T  # shape (T, K)
            except Exception as ex:
                return {"error": f"MS-AR fitting error: {ex}"}
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in state.tolist()],
                "state_probabilities": [[float(v) for v in row] for row in probs.tolist()],
                "params_used": {"k_regimes": k_regimes, "order": order},
            }
            if output in ('summary','compact'):
                n = min(int(lookback), len(state))
                st_tail = state[-n:] if n > 0 else state
                last_s = int(state[-1]) if len(state) else None
                # Shares
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {int(k): float(c) / float(len(st_tail) or 1) for k, c in zip(unique, counts)}
                summary = {"lookback": int(n), "last_state": last_s, "state_shares": shares}
                if output == 'summary':
                    payload = {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method, "summary": summary}
                else:
                    payload["summary"] = summary
                    if n > 0:
                        payload["state"] = [int(s) for s in st_tail.tolist()]
                        payload["times"] = t_fmt[-n:]
                        payload["state_probabilities"] = [[float(v) for v in row] for row in probs.tolist()][-n:]
            return payload

        else:  # 'hmm' (mixture/HMM-lite)
            try:
                from ..forecast.monte_carlo import fit_gaussian_mixture_1d
            except Exception as ex:
                return {"error": f"HMM-lite import error: {ex}"}
            n_states = int(p.get('n_states', 2))
            w, mu, sigma, gamma, _ = fit_gaussian_mixture_1d(x, n_states=max(2, n_states))
            state = np.argmax(gamma, axis=1) if gamma.ndim == 2 and gamma.shape[0] == x.size else np.zeros(x.size, dtype=int)
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in state.tolist()],
                "state_probabilities": [[float(v) for v in row] for row in (gamma.tolist() if gamma.ndim == 2 else np.zeros((x.size, n_states)).tolist())],
                "regime_params": {"weights": [float(v) for v in w.tolist()], "mu": [float(v) for v in mu.tolist()], "sigma": [float(v) for v in sigma.tolist()]},
            }
            if output in ('summary','compact'):
                n = min(int(lookback), len(state))
                st_tail = state[-n:] if n > 0 else state
                last_s = int(state[-1]) if len(state) else None
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {int(k): float(c) / float(len(st_tail) or 1) for k, c in zip(unique, counts)}
                # Order states by sigma ascending for interpretability
                order = np.argsort(sigma)
                ranks = {int(s): int(r) for r, s in enumerate(order)}
                summary = {
                    "lookback": int(n),
                    "last_state": last_s,
                    "state_shares": shares,
                    "state_sigma": {int(i): float(sigma[i]) for i in range(len(sigma))},
                    "state_order_by_sigma": ranks,
                }
                if output == 'summary':
                    payload = {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method, "summary": summary}
                else:
                    payload["summary"] = summary
                    if n > 0:
                        payload["state"] = [int(s) for s in st_tail.tolist()]
                        payload["times"] = t_fmt[-n:]
                        # If gamma present, truncate probabilities too
                        if isinstance(gamma, np.ndarray) and gamma.ndim == 2 and gamma.shape[0] >= n:
                            payload["state_probabilities"] = [[float(v) for v in row] for row in gamma[-n:].tolist()]
            return payload
    except Exception as e:
        return {"error": f"Error detecting regimes: {str(e)}"}
