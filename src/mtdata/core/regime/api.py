"""Regime detection implementation."""
import logging
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ...forecast.common import fetch_history as _fetch_history
from ...forecast.common import log_returns_from_prices as _log_returns_from_prices
from ...utils.denoise import _resolve_denoise_base_col
from ...utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise
from ...utils.utils import _format_time_minimal
from .. import features as _features_module
from .._mcp_instance import mcp
from ..execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from ..features import extract_rolling_features
from ..mt5_gateway import get_mt5_gateway, mt5_connection_error
from ..schema import DenoiseSpec, TimeframeLiteral
from .crypto import (
    _CRYPTO_SYMBOL_HINTS,
    _is_probably_crypto_symbol,
)
from .methods.bocpd import (
    _auto_calibrate_bocpd_params,
    _bocpd_reliability_score,
    _default_bocpd_cp_threshold,
    _default_bocpd_hazard_lambda,
    _filter_bocpd_change_points,
    _walkforward_quantile_threshold_calibration,
)
from .methods.hmm import _hmm_reliability_from_gamma
from .methods.ms_ar import _ms_ar_reliability_from_smoothed
from .payload import (
    _consolidate_payload,
    _summary_only_payload,
)

# Import from package submodules directly to avoid circular imports
from .smoothing import (
    _canonicalize_regime_labels,
    _count_state_transitions,
    _normalize_state_probability_matrix,
    _smooth_short_state_runs,
    _state_runs,
)

logger = logging.getLogger(__name__)


def _regime_connection_error() -> Optional[Dict[str, Any]]:
    return mt5_connection_error(
        get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
    )


def _coerce_param(
    params: Dict[str, Any],
    key: str,
    *,
    default: Any,
    cast: Any,
    error: Optional[str] = None,
) -> tuple[Any, Optional[str]]:
    raw = params.get(key, default)
    if raw is None:
        return default, None
    try:
        return cast(raw), None
    except Exception:
        if error is not None:
            return None, error
        return default, None


def _summary_window_size(lookback: int, size: int) -> int:
    try:
        lookback_i = int(lookback)
    except Exception:
        lookback_i = int(size)
    return min(max(lookback_i, 0), int(size))


def _resolve_bocpd_priors(
    params: Dict[str, Any],
    series: np.ndarray,
) -> Dict[str, float]:
    """Extract BOCPD prior hyper-parameters from *params* dict.

    If a prior param (mu0, kappa0, alpha0, beta0) is explicitly provided,
    use it.  Otherwise fall back to data-driven defaults derived from the
    series statistics, which are more appropriate than the hard-coded
    ``bocpd_gaussian`` defaults (mu0=0, kappa0=1, alpha0=1, beta0=1)
    when the data mean / variance is far from those assumptions.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    # Data-driven defaults
    if x.size >= 10:
        mu_data = float(np.mean(x))
        var_data = float(np.var(x, ddof=0))
        var_safe = max(var_data, 1e-16)
        dd_mu0 = mu_data
        dd_kappa0 = 1.0
        dd_alpha0 = max(1.0, x.size / 20.0)
        dd_beta0 = max(1e-8, var_safe * dd_alpha0)
    else:
        dd_mu0, dd_kappa0, dd_alpha0, dd_beta0 = 0.0, 1.0, 1.0, 1.0

    mode = str(params.get("prior_mode", "data_driven") or "data_driven").strip().lower()
    if mode == "fixed":
        dd_mu0, dd_kappa0, dd_alpha0, dd_beta0 = 0.0, 1.0, 1.0, 1.0

    mu0, _ = _coerce_param(params, "mu0", default=dd_mu0, cast=float)
    kappa0, _ = _coerce_param(params, "kappa0", default=dd_kappa0, cast=float)
    alpha0, _ = _coerce_param(params, "alpha0", default=dd_alpha0, cast=float)
    beta0, _ = _coerce_param(params, "beta0", default=dd_beta0, cast=float)

    return {
        "mu0": float(mu0),
        "kappa0": max(1e-8, float(kappa0)),
        "alpha0": max(0.5, float(alpha0)),
        "beta0": max(1e-12, float(beta0)),
    }


def _apply_bocpd_output_mode(
    payload: Dict[str, Any],
    *,
    output: str,
    lookback: int,
    cp_prob: np.ndarray,
    change_points: List[Dict[str, Any]],
    raw_cp_idx: List[int],
    reliability: Dict[str, Any],
    expected_fa_rate: float,
    calibration_age_bars: int,
    tuning_hint: Optional[str],
) -> Dict[str, Any]:
    n = _summary_window_size(lookback, len(cp_prob))
    tail = np.asarray(cp_prob[-n:], dtype=float) if n > 0 else np.asarray(cp_prob, dtype=float)
    recent_floor = len(cp_prob) - n
    recent_cps = [cp for cp in change_points if cp.get('idx', 0) >= recent_floor]
    summary = {
        "lookback": int(n),
        "last_cp_prob": float(cp_prob[-1]) if len(cp_prob) else float('nan'),
        "max_cp_prob": float(np.nanmax(tail)) if tail.size else float('nan'),
        "mean_cp_prob": float(np.nanmean(tail)) if tail.size else float('nan'),
        "change_points_count": int(len(recent_cps)),
        "raw_change_points_count": int(sum(1 for idx in raw_cp_idx if int(idx) >= recent_floor)),
        "filtered_change_points_count": int(
            max(0, sum(1 for idx in raw_cp_idx if int(idx) >= recent_floor) - int(len(recent_cps)))
        ),
        "recent_change_points": recent_cps[-5:],
        "confidence": float(reliability.get("confidence", 0.0)),
        "expected_false_alarm_rate": float(reliability.get("expected_false_alarm_rate", expected_fa_rate)),
        "calibration_age_bars": int(reliability.get("calibration_age_bars", calibration_age_bars)),
    }
    if tuning_hint is not None:
        summary["tuning_hint"] = tuning_hint
    payload["summary"] = summary
    if output == "summary":
        return _summary_only_payload(payload)
    if output == 'compact' and n > 0:
        tail_offset = len(payload.get("times", [])) - n
        payload["times"] = payload["times"][-n:]
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
    return payload


def _apply_state_output_mode(
    payload: Dict[str, Any],
    *,
    output: str,
    lookback: int,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    payload["summary"] = summary
    if output == "summary":
        return _summary_only_payload(payload)
    n = _summary_window_size(lookback, len(payload.get("state", [])))
    if output == 'compact' and n > 0:
        payload["times"] = payload["times"][-n:]
        payload["state"] = payload["state"][-n:]
        if isinstance(payload.get("state_probabilities"), list):
            payload["state_probabilities"] = payload["state_probabilities"][-n:]
    return payload


@mcp.tool()
def regime_detect(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 800,
    method: Literal['bocpd','hmm','ms_ar','clustering'] = 'bocpd',  # type: ignore
    target: Literal['return','price'] = 'return',  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    threshold: float = 0.5,
    output: Literal['full','summary','compact'] = 'compact',  # type: ignore
    lookback: int = 300,
    include_series: bool = False,
    min_regime_bars: int = 5,
) -> Dict[str, Any]:
    """Detect regimes and/or change-points over the last `limit` bars.

    - method: 'bocpd' (Bayesian online change-point; Gaussian), 'hmm' (Gaussian mixture/HMM-lite),
      'ms_ar' (Markov-switching AR), or 'clustering' (rolling-feature clustering via tsfresh + KMeans).
    - params (bocpd): optional `hazard_mode` = auto_default|auto_calibrated (defaults to auto_calibrated).
      Explicit `hazard_lambda` / `cp_threshold` always take precedence over auto selection.
      Optional robustness params:
        `cp_threshold_calibration_mode` (default `walkforward_quantile`),
        `threshold_target_false_alarm_rate`,
        `cp_confirm_bars` (default `1`, live-oriented),
        `min_cp_distance_bars`, `cp_edge_multiplier`.
    - include_series: If True, include raw time series data (probs, states) in output even if output='full'. Default False.
    - min_regime_bars: Merge short state runs (< this many bars) for state-based methods to reduce flicker.
    - output:
        - 'compact' (default): Returns recent consolidated 'regimes' and method summary.
        - 'full': Returns full consolidated 'regimes'. Raw 'series' included only if include_series=True.
        - 'summary': Returns stats only.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="regime_detect",
        symbol=symbol,
        timeframe=timeframe,
        method=method,
        target=target,
        output=output,
        limit=limit,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="regime_detect",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            target=target,
            output=output,
            limit=limit,
        )
        return result

    connection_error = _regime_connection_error()
    if connection_error is not None:
        return _finish(connection_error)
    try:
        p = dict(params or {})
        min_regime_bars_val, min_regime_bars_error = _coerce_param(
            p,
            "min_regime_bars",
            default=min_regime_bars,
            cast=int,
            error="min_regime_bars must be an integer >= 1.",
        )
        if min_regime_bars_error is not None:
            return _finish({"error": min_regime_bars_error})
        if min_regime_bars_val < 1:
            return _finish({"error": "min_regime_bars must be >= 1."})
        df = _fetch_history(symbol, timeframe, int(max(limit, 50)), as_of=None)
        if len(df) < 10:
            return _finish({"error": "Insufficient history"})
        base_col = _resolve_denoise_base_col(df, denoise, base_col='close', default_when='pre_ti')
        y = df[base_col].astype(float).to_numpy()
        times = df['time'].astype(float).to_numpy()
        try:
            return_series = _log_returns_from_prices(y)
        except ValueError as exc:
            return _finish({"error": str(exc)})
        calibration_returns = return_series
        calibration_returns = calibration_returns[np.isfinite(calibration_returns)]
        if target == 'return':
            x_raw = return_series
            return_mask = np.isfinite(x_raw)
            x = x_raw[return_mask]
            t = times[1:][return_mask]
        else:
            price_mask = np.isfinite(y)
            x = y[price_mask]
            t = times[price_mask]

        if x.size < 2:
            return _finish({"error": "Insufficient finite observations after filter"})

        # format times
        t_fmt = [_format_time_minimal(tt) for tt in t]

        if method == 'bocpd':
            from ...utils.regime import bocpd_gaussian
            hazard_mode = str(p.get("hazard_mode", "auto_calibrated") or "auto_calibrated").strip().lower()
            if hazard_mode in {"auto", "calibrated"}:
                hazard_mode = "auto_calibrated"
            if hazard_mode not in {"auto_default", "auto_calibrated"}:
                hazard_mode = "auto_calibrated"

            hazard_src = "params"
            threshold_src = "arg"
            calibration_info: Optional[Dict[str, Any]] = None
            threshold_calibration_info: Optional[Dict[str, Any]] = None

            auto_hazard = _default_bocpd_hazard_lambda(symbol, timeframe)
            auto_threshold = _default_bocpd_cp_threshold(symbol, timeframe)
            if hazard_mode == "auto_calibrated":
                auto_hazard, auto_threshold, calibration_info = _auto_calibrate_bocpd_params(
                    returns=calibration_returns, symbol=symbol, timeframe=timeframe
                )

            if "hazard_lambda" in p and p.get("hazard_lambda") is not None:
                hazard_lambda = int(p.get("hazard_lambda"))
            else:
                hazard_lambda = int(auto_hazard)
                hazard_src = "auto_calibrated" if hazard_mode == "auto_calibrated" else "auto_default"
            if "cp_threshold" in p and p.get("cp_threshold") is not None:
                threshold_used = float(p.get("cp_threshold"))
                threshold_src = "params.cp_threshold"
            elif "threshold" in p and p.get("threshold") is not None:
                threshold_used = float(p.get("threshold"))
                threshold_src = "params.threshold"
            else:
                if abs(float(threshold) - 0.5) <= 1e-12:
                    threshold_used = float(auto_threshold)
                    threshold_src = "auto_calibrated" if hazard_mode == "auto_calibrated" else "auto_default"
                else:
                    threshold_used = float(threshold)
                    threshold_src = "arg"
            max_rl, _ = _coerce_param(
                p,
                'max_run_length',
                default=min(1000, x.size),
                cast=int,
            )
            threshold_cal_mode = str(
                p.get("cp_threshold_calibration_mode", "walkforward_quantile")
                or "walkforward_quantile"
            ).strip().lower()
            if threshold_cal_mode in {"auto", "walkforward", "quantile"}:
                threshold_cal_mode = "walkforward_quantile"
            if threshold_src in {"auto_calibrated", "auto_default"} and threshold_cal_mode == "walkforward_quantile":
                target_fa, _ = _coerce_param(
                    p,
                    "threshold_target_false_alarm_rate",
                    default=0.02,
                    cast=float,
                )
                cal_window, _ = _coerce_param(
                    p,
                    "threshold_calibration_window",
                    default=None,
                    cast=int,
                )
                cal_step, _ = _coerce_param(
                    p,
                    "threshold_calibration_step",
                    default=None,
                    cast=int,
                )
                cal_max_windows, _ = _coerce_param(
                    p,
                    "threshold_calibration_max_windows",
                    default=6,
                    cast=int,
                )
                cal_boot, _ = _coerce_param(
                    p,
                    "threshold_calibration_bootstraps",
                    default=2,
                    cast=int,
                )
                threshold_used, threshold_calibration_info = _walkforward_quantile_threshold_calibration(
                    series=x,
                    hazard_lambda=hazard_lambda,
                    base_threshold=threshold_used,
                    target_false_alarm_rate=target_fa,
                    window=cal_window,
                    step=cal_step,
                    max_windows=cal_max_windows,
                    bootstrap_runs=cal_boot,
                )
            bocpd_priors = _resolve_bocpd_priors(p, x)
            res = bocpd_gaussian(
                x,
                hazard_lambda=hazard_lambda,
                max_run_length=max_rl,
                mu0=bocpd_priors["mu0"],
                kappa0=bocpd_priors["kappa0"],
                alpha0=bocpd_priors["alpha0"],
                beta0=bocpd_priors["beta0"],
            )
            cp_prob = np.asarray(res.get('cp_prob', np.zeros_like(x, dtype=float)), dtype=float)
            raw_cp_idx = [int(i) for i, v in enumerate(cp_prob.tolist()) if np.isfinite(v) and float(v) >= float(threshold_used)]
            cp_confirm_bars, _ = _coerce_param(
                p,
                "cp_confirm_bars",
                default=1,
                cast=int,
            )
            cp_confirm_relaxed_mult, _ = _coerce_param(
                p,
                "cp_confirm_relaxed_mult",
                default=0.90,
                cast=float,
            )
            if "cp_edge_multiplier" in p and p.get("cp_edge_multiplier") is not None:
                cp_edge_multiplier, _ = _coerce_param(
                    p,
                    "cp_edge_multiplier",
                    default=1.08,
                    cast=float,
                )
            else:
                # When threshold is already calibrated via walk-forward null quantiles,
                # avoid double-tightening the edge gate.
                if (
                    threshold_src in {"auto_calibrated", "auto_default"}
                    and isinstance(threshold_calibration_info, dict)
                    and bool(threshold_calibration_info.get("calibrated", False))
                ):
                    cp_edge_multiplier = 1.0
                else:
                    cp_edge_multiplier = 1.08
            min_cp_distance_bars, _ = _coerce_param(
                p,
                "min_cp_distance_bars",
                default=max(2, min_regime_bars_val),
                cast=int,
            )
            cp_idx, cp_filter_meta = _filter_bocpd_change_points(
                cp_prob=cp_prob,
                threshold=float(threshold_used),
                min_distance_bars=int(max(1, min_cp_distance_bars)),
                min_regime_bars=int(max(1, min_regime_bars_val)),
                confirm_bars=int(max(1, cp_confirm_bars)),
                confirm_relaxed_mult=float(cp_confirm_relaxed_mult),
                edge_multiplier=float(cp_edge_multiplier),
            )
            cps = [{"idx": i, "time": t_fmt[i], "prob": float(cp_prob[i])} for i in cp_idx]
            tuning_hint: Optional[str] = None
            if len(cps) == 0:
                if len(raw_cp_idx) > 0 and int(cp_filter_meta.get("filtered_count", 0)) > 0:
                    tuning_hint = (
                        "Change-point candidates were filtered by robustness guards "
                        "(confirmation/cooldown/edge checks). Tune cp_confirm_bars, "
                        "min_cp_distance_bars, or cp_edge_multiplier if needed."
                    )
                else:
                    tuning_hint = (
                        "No change points detected. Try lowering threshold or reducing "
                        f"hazard_lambda (currently {hazard_lambda}); active threshold={threshold_used:.2f}."
                    )
            if isinstance(threshold_calibration_info, dict):
                expected_fa_rate = float(
                    threshold_calibration_info.get("target_false_alarm_rate", 0.02)
                )
                calibration_age_bars = int(
                    threshold_calibration_info.get(
                        "points",
                        calibration_info.get("points", 0) if isinstance(calibration_info, dict) else 0,
                    )
                )
                threshold_calibrated = bool(threshold_calibration_info.get("calibrated", False))
            else:
                expected_fa_rate = 0.02
                calibration_age_bars = int(
                    calibration_info.get("points", 0) if isinstance(calibration_info, dict) else 0
                )
                threshold_calibrated = False
            reliability = _bocpd_reliability_score(
                cp_prob=cp_prob,
                cp_indices=cp_idx,
                threshold=float(threshold_used),
                lookback=int(lookback),
                min_regime_bars=int(max(1, min_regime_bars_val)),
                expected_false_alarm_rate=float(expected_fa_rate),
                calibration_age_bars=int(calibration_age_bars),
                threshold_calibrated=bool(threshold_calibrated),
            )
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "cp_prob": [float(v) for v in np.asarray(cp_prob, dtype=float).tolist()],
                "change_points": cps,
                "threshold": float(threshold_used),
                "reliability": reliability,
                "params_used": {
                    "hazard_lambda": hazard_lambda,
                    "hazard_lambda_source": hazard_src,
                    "cp_threshold": float(threshold_used),
                    "cp_threshold_source": threshold_src,
                    "hazard_mode": hazard_mode,
                    "max_run_length": max_rl,
                    "cp_filter": cp_filter_meta,
                    "priors": bocpd_priors,
                },
            }
            if isinstance(calibration_info, dict):
                payload["params_used"]["auto_calibration"] = calibration_info
            if isinstance(threshold_calibration_info, dict):
                payload["params_used"]["cp_threshold_calibration"] = threshold_calibration_info
            if tuning_hint is not None:
                payload["tuning_hint"] = tuning_hint
            if output in ('summary','compact'):
                payload = _apply_bocpd_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    cp_prob=cp_prob,
                    change_points=cps,
                    raw_cp_idx=raw_cp_idx,
                    reliability=reliability,
                    expected_fa_rate=expected_fa_rate,
                    calibration_age_bars=calibration_age_bars,
                    tuning_hint=tuning_hint,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(_consolidate_payload(payload, method, output, include_series=include_series))

        elif method == 'ms_ar':
            try:
                from statsmodels.tsa.regime_switching.markov_regression import (
                    MarkovRegression,  # type: ignore
                )
            except Exception:
                return _finish({"error": "statsmodels MarkovRegression not available. Install statsmodels."})
            k_regimes, _ = _coerce_param(p, 'k_regimes', default=2, cast=int)
            order, _ = _coerce_param(p, 'order', default=0, cast=int)
            try:
                mod = MarkovRegression(endog=x, k_regimes=max(2, k_regimes), trend='c', order=max(0, order), switching_variance=True)
                maxiter, _ = _coerce_param(p, 'maxiter', default=100, cast=int)
                res = mod.fit(disp=False, maxiter=maxiter)
                smoothed = res.smoothed_marginal_probabilities
                if hasattr(smoothed, "values"):
                    smoothed = smoothed.values
                probs = np.asarray(smoothed, dtype=float)
                state = np.argmax(probs, axis=1)
                state, probs, smoothing_meta = _smooth_short_state_runs(
                    state=np.asarray(state, dtype=int),
                    probs=probs,
                    min_regime_bars=min_regime_bars_val,
                )
                state, probs, canon_meta = _canonicalize_regime_labels(
                    state, probs, x,
                )
                smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)
                mle_retvals = getattr(res, "mle_retvals", None)
                converged = None
                if isinstance(mle_retvals, dict):
                    converged = mle_retvals.get("converged")
                elif mle_retvals is not None and hasattr(mle_retvals, "get"):
                    try:
                        converged = mle_retvals.get("converged")
                    except Exception:
                        converged = getattr(mle_retvals, "converged", None)
            except Exception as ex:
                return _finish({"error": f"MS-AR fitting error: {ex}"})
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in state.tolist()],
                "state_probabilities": [[float(v) for v in row] for row in probs.tolist()],
                "params_used": {
                    "k_regimes": k_regimes,
                    "order": order,
                    "min_regime_bars": int(min_regime_bars_val),
                    "smoothing_applied": bool(smoothing_meta.get("smoothing_applied", False)),
                    "transitions_before": int(smoothing_meta.get("transitions_before", 0)),
                    "transitions_after": int(smoothing_meta.get("transitions_after", 0)),
                },
            }
            if converged is not None:
                payload["params_used"]["converged"] = bool(converged)
                if converged is False:
                    payload["warnings"] = ["MS-AR model did not converge; regime probabilities may be unreliable."]
            # Add reliability info
            reliability = _ms_ar_reliability_from_smoothed(
                smoothed_probs=probs,
                params_used=payload["params_used"],
            )
            payload["reliability"] = reliability

            if output in ('summary','compact'):
                n = _summary_window_size(lookback, len(state))
                st_tail = state[-n:] if n > 0 else state
                last_s = int(state[-1]) if len(state) else None
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {int(k): float(c) / float(len(st_tail) or 1) for k, c in zip(unique, counts)}
                summary = {
                    "lookback": int(n),
                    "last_state": last_s,
                    "state_shares": shares,
                    "transitions_before": int(smoothing_meta.get("transitions_before", 0)),
                    "transitions_after": int(smoothing_meta.get("transitions_after", 0)),
                    "smoothing_applied": bool(smoothing_meta.get("smoothing_applied", False)),
                }
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(_consolidate_payload(payload, method, output, include_series=include_series))

        elif method == 'hmm':  # 'hmm' (mixture/HMM-lite)
            n_states, n_states_error = _coerce_param(
                p,
                'n_states',
                default=2,
                cast=int,
                error="n_states must be an integer >= 2 for hmm.",
            )
            if n_states_error is not None:
                return _finish({"error": n_states_error})
            if n_states < 2:
                return _finish({"error": "n_states must be >= 2 for hmm."})
            try:
                from ...forecast.monte_carlo import fit_gaussian_mixture_1d
            except Exception as ex:
                return _finish({"error": f"HMM-lite import error: {ex}"})
            fit_gaussian_mixture_1d = globals().get("fit_gaussian_mixture_1d", fit_gaussian_mixture_1d)
            w, mu, sigma, gamma, _ = fit_gaussian_mixture_1d(x, n_states=n_states)
            gamma_matrix = _normalize_state_probability_matrix(
                gamma,
                rows=x.size,
                requested_states=n_states,
            )
            state = np.argmax(gamma_matrix, axis=1) if gamma_matrix.size else np.zeros(x.size, dtype=int)
            gamma_smoothed: Optional[np.ndarray] = gamma_matrix
            state, gamma_smoothed, smoothing_meta = _smooth_short_state_runs(
                state=np.asarray(state, dtype=int),
                probs=gamma_smoothed,
                min_regime_bars=min_regime_bars_val,
            )
            state, gamma_smoothed, canon_meta = _canonicalize_regime_labels(
                state, gamma_smoothed, x,
            )
            smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)
            gamma_for_payload = (
                gamma_smoothed
                if isinstance(gamma_smoothed, np.ndarray)
                else gamma_matrix
            )
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in state.tolist()],
                "state_probabilities": [
                    [float(v) for v in row]
                    for row in gamma_for_payload.tolist()
                ],
                "regime_params": {"weights": [float(v) for v in w.tolist()], "mu": [float(v) for v in mu.tolist()], "sigma": [float(v) for v in sigma.tolist()]},
                "params_used": {
                    "n_states": int(n_states),
                    "fitted_n_states": int(len(mu)),
                    "min_regime_bars": int(min_regime_bars_val),
                    "smoothing_applied": bool(smoothing_meta.get("smoothing_applied", False)),
                    "transitions_before": int(smoothing_meta.get("transitions_before", 0)),
                    "transitions_after": int(smoothing_meta.get("transitions_after", 0)),
                },
            }
            # Add reliability info
            reliability = _hmm_reliability_from_gamma(gamma_for_payload)
            payload["reliability"] = reliability

            if output in ('summary','compact'):
                n = _summary_window_size(lookback, len(state))
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
                    "transitions_before": int(smoothing_meta.get("transitions_before", 0)),
                    "transitions_after": int(smoothing_meta.get("transitions_after", 0)),
                    "smoothing_applied": bool(smoothing_meta.get("smoothing_applied", False)),
                }
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(_consolidate_payload(payload, method, output, include_series=include_series))

        elif method == 'clustering':
            try:
                standard_scaler_cls = globals().get("StandardScaler")
                kmeans_cls = globals().get("KMeans")
                pca_cls = globals().get("PCA")
                if standard_scaler_cls is None:
                    from sklearn.preprocessing import (
                        StandardScaler as standard_scaler_cls,
                    )
                if kmeans_cls is None:
                    from sklearn.cluster import KMeans as kmeans_cls
                if pca_cls is None:
                    from sklearn.decomposition import PCA as pca_cls
            except ImportError as ex:
                return _finish({"error": f"Clustering dependencies missing: {ex}"})
            window_size, _ = _coerce_param(p, 'window_size', default=20, cast=int)
            k_regimes, _ = _coerce_param(p, 'k_regimes', default=3, cast=int)
            use_pca = bool(p.get('use_pca', True))
            n_components, _ = _coerce_param(p, 'n_components', default=3, cast=int)
            warnings: List[str] = []
            if target == 'price':
                warnings.append(
                    "Clustering on price features may produce level-dependent regimes. Consider target='return'."
                )

            # Extract features (use 'return' or 'price'? 'return' is stationary, usually better)
            # x is already computed based on target input
            extract_rolling_features_impl = globals().get("extract_rolling_features", extract_rolling_features)
            if extract_rolling_features_impl is extract_rolling_features:
                extract_rolling_features_impl = _features_module.extract_rolling_features
            features_df = extract_rolling_features_impl(x, window_size=window_size)

            # Align features with time
            # valid_indices are where features are not NaN
            valid_mask = ~features_df.isna().any(axis=1)
            X_valid = features_df.loc[valid_mask]

            if X_valid.empty:
                 return _finish({"error": "Not enough data for feature extraction (check window_size)"})

            # Normalize
            scaler = standard_scaler_cls()
            X_scaled = scaler.fit_transform(X_valid)

            # PCA
            if use_pca and X_scaled.shape[1] > n_components:
                pca = pca_cls(n_components=min(n_components, X_scaled.shape[1]))
                X_final = pca.fit_transform(X_scaled)
            else:
                X_final = X_scaled

            # Cluster — seed centroids from evenly-spaced rows so KMeans++
            # init is skipped.  KMeans++ triggers joblib CPU-topology probing
            # which blocks indefinitely in asyncio.to_thread workers on Windows.
            n_samples = X_final.shape[0]
            if n_samples < k_regimes:
                return _finish({"error": f"Not enough samples ({n_samples}) for {k_regimes} clusters"})
            idx = np.round(np.linspace(0, n_samples - 1, k_regimes)).astype(int)
            kmeans = kmeans_cls(
                n_clusters=k_regimes, random_state=42, n_init=1,
                init=X_final[idx],
            )
            labels = kmeans.fit_predict(X_final)

            # Smooth short runs and canonicalize on valid slice only
            valid_probs = np.zeros((int(valid_mask.sum()), k_regimes))
            valid_probs[np.arange(len(labels)), labels] = 1.0
            labels, valid_probs, smoothing_meta = _smooth_short_state_runs(
                state=np.asarray(labels, dtype=int),
                probs=valid_probs,
                min_regime_bars=min_regime_bars_val,
            )
            labels, valid_probs, canon_meta = _canonicalize_regime_labels(
                labels, valid_probs, x[valid_mask],
            )
            smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)

            # Map back to full length (-1 for undefined leading window)
            full_states = np.full(len(x), -1, dtype=int)
            full_states[valid_mask] = labels

            full_probs = np.zeros((len(x), k_regimes))
            full_probs[valid_mask] = valid_probs

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
                    "n_components": n_components,
                    "min_regime_bars": int(min_regime_bars_val),
                    "smoothing_applied": smoothing_meta.get("smoothing_applied", False),
                },
            }
            if warnings:
                payload["warnings"] = warnings

            # Summary stats
            if output in ('summary','compact'):
                n_summary = _summary_window_size(lookback, len(full_states))
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
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(_consolidate_payload(payload, method, output, include_series=include_series))

    except Exception as e:
        return _finish({"error": f"Error detecting regimes: {str(e)}"})
