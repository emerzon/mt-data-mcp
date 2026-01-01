from typing import Any, Dict, Optional, List, Literal
import numpy as np
import pandas as pd

from .server import mcp, _auto_connect_wrapper
from .schema import TimeframeLiteral, DenoiseSpec
from ..forecast.common import fetch_history as _fetch_history
from ..utils.utils import _format_time_minimal
from ..utils.denoise import _resolve_denoise_base_col


def _consolidate_payload(payload: Dict[str, Any], method: str, output_mode: str, include_series: bool = False) -> Dict[str, Any]:
    """Consolidate time series into regime segments and restructure payload."""
    try:
        times = payload.get("times")
        if not times or not isinstance(times, list):
            return payload

        # Prepare consolidation
        segments: List[Dict[str, Any]] = []
        
        # Extract states/regimes
        states = []
        probs = []
        is_change_point = []
        
        if method == "bocpd":
            # For BOCPD, we define regimes by change points
            # We can create a 'regime_id' that increments at each CP
            # We also look at 'change_points' list in payload
            cps_idx = set()
            if "change_points" in payload and isinstance(payload["change_points"], list):
                for cp in payload["change_points"]:
                    if isinstance(cp, dict) and "idx" in cp:
                        cps_idx.add(cp["idx"])
            
            curr_regime = 0
            # Reconstruct per-step state
            for i in range(len(times)):
                if i in cps_idx:
                    curr_regime += 1
                states.append(curr_regime)
                
            # Probs
            raw_probs = payload.get("cp_prob")
            if isinstance(raw_probs, list):
                probs = raw_probs
            else:
                probs = [0.0] * len(times)
                
        elif method in ("ms_ar", "hmm", "clustering"):
            raw_state = payload.get("state")
            if isinstance(raw_state, list):
                states = raw_state
            
            # Probs
            # structure is usually list of lists [ [p0, p1...], ... ]
            raw_probs = payload.get("state_probabilities")
            # We might just store the max prob or the prob of the current state?
            if isinstance(raw_probs, list) and raw_probs:
                if isinstance(raw_probs[0], list):
                    # Pick prob of selected state
                    for s, p_vec in zip(states, raw_probs):
                        if isinstance(p_vec, list) and 0 <= s < len(p_vec):
                            probs.append(p_vec[s])
                        else:
                            probs.append(None)
                else:
                    probs = raw_probs # Should not happen based on current logic but safe fallback

        if not states or len(states) != len(times):
            # Fallback if creation failed
            return payload

        # Consolidate
        # Loop through
        curr_start = times[0]
        curr_state = states[0]
        curr_prob_sum = 0.0
        curr_count = 0
        
        i = 0
        while i < len(times):
             t = times[i]
             s = states[i]
             p = probs[i] if i < len(probs) and probs[i] is not None else 0.0
             
             # Check change (state change)
             # For BOCPD, 's' changes exactly at CP.
             if s != curr_state and curr_count > 0:
                 # close segment
                 avg_prob = curr_prob_sum / max(1, curr_count)
                 segments.append({
                     "start": curr_start,
                     "end": times[i-1] if i > 0 else curr_start,
                     "duration": curr_count,
                     "regime": curr_state, # state ID or regime ID
                     "confidence": avg_prob # average prob of being in this state/regime (for HMM) or CP prob (BOCPD - meaningless inside segment usually)
                 })
                 # New segment
                 curr_start = t
                 curr_state = s
                 curr_prob_sum = 0.0
                 curr_count = 0
             
             curr_prob_sum += p
             curr_count += 1
             i += 1
             
        # Final segment
        if curr_count > 0:
            avg_prob = curr_prob_sum / max(1, curr_count)
            segments.append({
                "start": curr_start,
                "end": times[-1],
                "duration": curr_count,
                "regime": curr_state,
                "confidence": avg_prob
            })

        # Post-process segments for readability
        # For BOCPD, 'confidence' is avg cp_prob which is usually low except at edges.
        # Maybe we want the PEAK prob? or just drop it.
        # For HMM, 'confidence' is avg prob of that state.
        
        final_segments = []
        for i, seg in enumerate(segments):
            row = {
                "start": seg["start"],
                "end": seg["end"],
                "bars": seg["duration"],
                "regime": seg["regime"]
            }
            if method != 'bocpd':
                row["avg_conf"] = round(seg["confidence"], 4)
            final_segments.append(row)

        # Restructure Payload
        # We want 'regimes' to be the MAIN table.
        # We want to hide raw series under 'series' if output='full'.
        
        new_payload = {
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "method": payload.get("method"),
            "success": True
        }
        
        # Copy summary if exists
        if "summary" in payload:
            new_payload["summary"] = payload["summary"]
            
        # Add consolidated table
        new_payload["regimes"] = final_segments
        
        # Handle raw series
        if output_mode == 'full' and include_series:
            series_data = {}
            for k in ["times", "cp_prob", "state", "state_probabilities", "change_points"]:
                if k in payload:
                    series_data[k] = payload[k]
            new_payload["series"] = series_data
        elif output_mode == 'compact' and include_series:
            # Maybe keep tail of series in 'series'?
            series_data = {}
            for k in ["times", "cp_prob", "state"]:
                 if k in payload:
                     series_data[k] = payload[k] # Already truncated by caller?
            new_payload["series"] = series_data

        # Add params
        if "params_used" in payload:
            new_payload["params_used"] = payload["params_used"]
            
        return new_payload

    except Exception as e:
        # Fallback to original payload on error
        payload["consolidation_error"] = str(e)
        return payload


def _summary_only_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal payload for `output='summary'` (no regimes/series)."""
    out: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "timeframe": payload.get("timeframe"),
        "method": payload.get("method"),
        "target": payload.get("target"),
        "success": bool(payload.get("success", True)),
    }
    if "summary" in payload:
        out["summary"] = payload["summary"]
    if "params_used" in payload:
        out["params_used"] = payload["params_used"]
    if "threshold" in payload:
        out["threshold"] = payload["threshold"]
    return out


@mcp.tool()
@_auto_connect_wrapper
def regime_detect(
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
    include_series: bool = False,
) -> Dict[str, Any]:
    """Detect regimes and/or change-points over the last `limit` bars.

    - method: 'bocpd' (Bayesian online change-point; Gaussian), 'hmm' (Gaussian mixture/HMM-lite), or 'ms_ar' (Markov-switching AR).
    - include_series: If True, include raw time series data (probs, states) in output even if output='full'. Default False.
    - output:
        - 'full': Returns consolidated 'regimes' table. Raw 'series' included only if include_series=True.
        - 'summary': Returns stats only.
        - 'compact': Returns 'regimes' and tail of 'series'.
    """
    try:
        p = dict(params or {})
        df = _fetch_history(symbol, timeframe, int(max(limit, 50)), as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history"}
        base_col = _resolve_denoise_base_col(df, denoise, base_col='close', default_when='pre_ti')
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
        t_fmt = [_format_time_minimal(tt) for tt in t]

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
                payload["summary"] = summary
                if output == "summary":
                    return _summary_only_payload(payload)
                if output == 'compact' and n > 0:
                    # Compact mode uses the tail of the series; remap CP indices so they
                    # remain consistent with the truncated `times` array used by consolidation.
                    tail_offset = len(t_fmt) - n
                    payload["times"] = t_fmt[-n:]
                    payload["cp_prob"] = payload["cp_prob"][-n:]
                    tail_cps: List[Dict[str, Any]] = []
                    for cp in payload.get("change_points", []):
                        if not isinstance(cp, dict):
                            continue
                        idx = cp.get("idx")
                        if isinstance(idx, int) and idx >= tail_offset:
                            cp_tail = dict(cp)
                            cp_tail["idx"] = idx - tail_offset
                            tail_cps.append(cp_tail)
                    payload["change_points"] = tail_cps

            return _consolidate_payload(payload, method, output, include_series=include_series)

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
                if hasattr(smoothed, "values"):
                    smoothed = smoothed.values
                state = np.argmax(smoothed, axis=1)
                probs = smoothed
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
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {int(k): float(c) / float(len(st_tail) or 1) for k, c in zip(unique, counts)}
                summary = {"lookback": int(n), "last_state": last_s, "state_shares": shares}
                payload["summary"] = summary
                if output == "summary":
                    return _summary_only_payload(payload)
                if output == 'compact' and n > 0:
                    payload["times"] = t_fmt[-n:]
                    payload["state"] = payload["state"][-n:]
                    payload["state_probabilities"] = payload["state_probabilities"][-n:]
            
            return _consolidate_payload(payload, method, output, include_series=include_series)

        elif method == 'hmm':  # 'hmm' (mixture/HMM-lite)
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
                order = np.argsort(sigma)
                ranks = {int(s): int(r) for r, s in enumerate(order)}
                summary = {
                    "lookback": int(n),
                    "last_state": last_s,
                    "state_shares": shares,
                    "state_sigma": {int(i): float(sigma[i]) for i in range(len(sigma))},
                    "state_order_by_sigma": ranks,
                }
                payload["summary"] = summary
                if output == "summary":
                    return _summary_only_payload(payload)
                if output == 'compact' and n > 0:
                    payload["times"] = t_fmt[-n:]
                    payload["state"] = payload["state"][-n:]
                    if isinstance(gamma, np.ndarray) and len(gamma) >= n:
                         payload["state_probabilities"] = payload["state_probabilities"][-n:]

            return _consolidate_payload(payload, method, output, include_series=include_series)

        elif method == 'clustering':
            try:
                from .features import extract_rolling_features
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
            except ImportError as ex:
                return {"error": f"Clustering dependencies missing: {ex}"}

            window_size = int(p.get('window_size', 20))
            k_regimes = int(p.get('k_regimes', 3))
            use_pca = bool(p.get('use_pca', True))
            n_components = int(p.get('n_components', 3))

            # Extract features (use 'return' or 'price'? 'return' is stationary, usually better)
            # x is already computed based on target input
            features_df = extract_rolling_features(x, window_size=window_size)
            
            # Align features with time
            # valid_indices are where features are not NaN
            valid_mask = ~features_df.isna().any(axis=1)
            X_valid = features_df.loc[valid_mask]
            
            if X_valid.empty:
                 return {"error": "Not enough data for feature extraction (check window_size)"}

            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)

            # PCA
            if use_pca and X_scaled.shape[1] > n_components:
                pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
                X_final = pca.fit_transform(X_scaled)
            else:
                X_final = X_scaled

            # Cluster
            kmeans = KMeans(n_clusters=k_regimes, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X_final)
            
            # Map back to full length
            # features_df has same length as x (n), but with NaNs at start
            # We want to fill the result states.
            # 0-fill or forward fill or -1? 
            # Let's say -1 for undefined.
            full_states = np.full(len(x), -1, dtype=int)
            full_states[valid_mask] = labels
            
            # Probabilities? KMeans doesn't give probs easily (distance based).
            # We can use 1.0 for the assigned cluster or distance-to-center logic.
            # Simplified: 1.0 for assigned.
            full_probs = np.zeros((len(x), k_regimes))
            full_probs[valid_mask, labels] = 1.0
            
            # Reconstruct payload
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in full_states.tolist()],
                "state_probabilities": [[float(v) for v in row] for row in full_probs.tolist()],
                "params_used": {
                    "k_regimes": k_regimes, 
                    "window_size": window_size,
                    "use_pca": use_pca,
                    "n_components": n_components
                },
            }
            
            # Summary stats
            if output in ('summary','compact'):
                n_summary = min(int(lookback), len(full_states))
                st_tail = full_states[-n_summary:] if n_summary > 0 else full_states
                # Filter out -1
                st_tail_valid = st_tail[st_tail != -1]
                
                unique, counts = np.unique(st_tail_valid, return_counts=True)
                shares = {int(k): float(c) / float(len(st_tail_valid) or 1) for k, c in zip(unique, counts)}
                
                summary = {
                    "lookback": int(n_summary),
                    "last_state": int(full_states[-1]) if len(full_states) else None,
                    "state_shares": shares
                }
                payload["summary"] = summary

                if output == "summary":
                     return _summary_only_payload(payload)
                if output == 'compact' and n_summary > 0:
                     payload["times"] = t_fmt[-n_summary:]
                     payload["state"] = payload["state"][-n_summary:]
                     payload["state_probabilities"] = payload["state_probabilities"][-n_summary:]

            return _consolidate_payload(payload, method, output, include_series=include_series)

    except Exception as e:
        return {"error": f"Error detecting regimes: {str(e)}"}
