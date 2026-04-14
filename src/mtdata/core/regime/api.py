"""Regime detection implementation."""

import logging
import time
import warnings
from collections import Counter
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


_DIRECTION_SIGNALS = frozenset({"bullish", "bearish", "neutral"})
_VOLATILITY_SIGNALS = frozenset(
    {"very_low_vol", "low_vol", "moderate_vol", "high_vol", "very_high_vol"}
)


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _lookup_regime_info_entry(regime_info: Any, regime_id: Any) -> Dict[str, Any]:
    if not isinstance(regime_info, dict):
        return {}

    candidates: List[Any] = []
    if regime_id is not None:
        candidates.append(regime_id)
        try:
            candidates.append(int(regime_id))
        except (TypeError, ValueError):
            pass
        candidates.append(str(regime_id))

    for candidate in candidates:
        details = regime_info.get(candidate)
        if isinstance(details, dict):
            return details
    return {}


def _normalize_direction_signal(
    label: Any,
    *,
    mean_return: Any = None,
) -> Optional[str]:
    text = str(label or "").strip().lower()
    if "bullish" in text or "positive" in text:
        return "bullish"
    if "bearish" in text or "negative" in text:
        return "bearish"
    if "neutral" in text:
        return "neutral"

    mean_value = _coerce_optional_float(mean_return)
    if mean_value is None:
        return None
    if abs(mean_value) < 1e-4:
        return "neutral"
    return "bullish" if mean_value > 0 else "bearish"


def _normalize_volatility_signal(
    label: Any,
    *,
    volatility: Any = None,
) -> Optional[str]:
    text = str(label or "").strip().lower()
    if "very_high_vol" in text or "extreme_vol" in text:
        return "very_high_vol"
    if "moderate_vol" in text or "mod_vol" in text:
        return "moderate_vol"
    if "very_low_vol" in text:
        return "very_low_vol"
    if "high_vol" in text or "volatile" in text:
        return "high_vol"
    if "low_vol" in text or "stable" in text:
        return "low_vol"

    sigma = _coerce_optional_float(volatility)
    if sigma is None:
        return None
    if sigma < 0.0005:
        return "very_low_vol"
    if sigma < 0.001:
        return "low_vol"
    if sigma < 0.003:
        return "moderate_vol"
    if sigma < 0.006:
        return "high_vol"
    return "very_high_vol"


def _summarize_current_regime_for_comparison(
    method: str,
    result: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None

    if method == "bocpd":
        current_segment = result.get("current_segment")
        if not isinstance(current_segment, dict) or not current_segment:
            legacy_current = result.get("current_regime")
            if isinstance(legacy_current, dict) and legacy_current:
                current_segment = {
                    "started_at": legacy_current.get("since"),
                    "bars_since_change": legacy_current.get("bars"),
                }
            else:
                segments = result.get("segments")
                if not isinstance(segments, list) or not segments:
                    segments = result.get("regimes")
                if isinstance(segments, list) and segments and isinstance(segments[-1], dict):
                    last = segments[-1]
                    current_segment = {
                        "started_at": last.get("started_at", last.get("start")),
                        "bars_since_change": last.get("bars"),
                    }
                else:
                    return None

        entry = {
            key: current_segment.get(key)
            for key in (
                "status",
                "started_at",
                "bars_since_change",
                "transition_risk",
                "latest_transition_probability",
            )
            if current_segment.get(key) is not None
        }

        transition_summary = result.get("transition_summary")
        if not isinstance(transition_summary, dict):
            transition_summary = result.get("summary")
        if isinstance(transition_summary, dict):
            recent_change_points_count = transition_summary.get(
                "recent_change_points_count",
                transition_summary.get("change_points_count"),
            )
            if recent_change_points_count is not None:
                entry["recent_change_points_count"] = recent_change_points_count
            for key in ("recent_transition_activity", "calibration_status"):
                value = transition_summary.get(key)
                if value is not None:
                    entry[key] = value

        segment_context = result.get("segment_context")
        if isinstance(segment_context, dict):
            for key in ("bias", "return_pct", "volatility_pct"):
                value = segment_context.get(key)
                if value is not None:
                    entry[key] = value

        return entry or None

    if method == "rule_based":
        regime = result.get("regime")
        if not isinstance(regime, dict):
            return None
        entry = {
            key: regime.get(key)
            for key in (
                "state",
                "direction",
                "trend_strength",
                "efficiency_ratio",
                "window_bars",
                "window_move_pct",
                "signal_source",
            )
            if regime.get(key) is not None
        }
        return entry or None

    current = result.get("current_regime")
    if not isinstance(current, dict) or not current:
        regimes = result.get("regimes")
        if isinstance(regimes, list) and regimes and isinstance(regimes[-1], dict):
            last = regimes[-1]
            current = {
                "regime_id": last.get("regime"),
                "label": last.get("label"),
                "confidence": last.get("avg_conf"),
                "since": last.get("start"),
                "bars": last.get("bars"),
            }
        else:
            return None

    regime_id = current.get("regime_id")
    regime_stats = _lookup_regime_info_entry(result.get("regime_info"), regime_id)
    label = current.get("label")
    if label is None and regime_stats:
        label = regime_stats.get("label")

    entry: Dict[str, Any] = {
        key: current.get(key)
        for key in ("regime_id", "since", "bars")
        if current.get(key) is not None
    }
    if label is not None:
        entry["label"] = label

    regime_confidence = current.get("regime_confidence")
    if regime_confidence is None:
        regime_confidence = current.get("confidence")
    if regime_confidence is not None:
        entry["regime_confidence"] = regime_confidence

    direction = None
    if method in {"hmm", "ms_ar", "ensemble"}:
        direction = _normalize_direction_signal(
            label,
            mean_return=regime_stats.get("mean_return"),
        )
    elif method == "clustering":
        direction = _normalize_direction_signal(label)
    if direction is not None:
        entry["direction"] = direction

    volatility = None
    if method in {"hmm", "ms_ar", "garch", "ensemble", "wavelet"}:
        volatility = _normalize_volatility_signal(
            label,
            volatility=regime_stats.get("volatility"),
        )
    if volatility is not None:
        entry["volatility"] = volatility

    for key in ("mean_return_pct", "volatility_pct"):
        value = regime_stats.get(key)
        if value is not None:
            entry[key] = value

    if method == "bocpd":
        summary = result.get("summary")
        if isinstance(summary, dict):
            for key in ("last_cp_prob", "change_points_count"):
                value = summary.get(key)
                if value is not None:
                    entry[key] = value

    return entry or None


def _build_semantic_agreement(current_regimes: Dict[str, Any]) -> Dict[str, Any]:
    agreement: Dict[str, Any] = {"basis": "semantic_signals"}

    direction_votes = {
        method: entry["direction"]
        for method, entry in current_regimes.items()
        if isinstance(entry, dict) and entry.get("direction") in _DIRECTION_SIGNALS
    }
    volatility_votes = {
        method: entry["volatility"]
        for method, entry in current_regimes.items()
        if isinstance(entry, dict) and entry.get("volatility") in _VOLATILITY_SIGNALS
    }

    def _consensus(votes: Dict[str, str]) -> Optional[Dict[str, Any]]:
        if len(votes) < 2:
            return None
        counts = Counter(votes.values())
        majority, count = counts.most_common(1)[0]
        return {
            "majority": majority,
            "agreement_pct": round(count / len(votes) * 100.0, 2),
            "methods_considered": list(votes.keys()),
        }

    direction_consensus = _consensus(direction_votes)
    if direction_consensus is not None:
        agreement["direction"] = direction_consensus

    volatility_consensus = _consensus(volatility_votes)
    if volatility_consensus is not None:
        agreement["volatility"] = volatility_consensus

    return agreement


def _build_all_method_comparison(results_by_method: Dict[str, Any]) -> Dict[str, Any]:
    current_regimes: Dict[str, Any] = {}
    for method, result in results_by_method.items():
        current_regimes[method] = _summarize_current_regime_for_comparison(
            method,
            result,
        )

    return {
        "methods_run": list(results_by_method.keys()),
        "current_regimes": current_regimes,
        "agreement": _build_semantic_agreement(current_regimes),
    }


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
    tail = (
        np.asarray(cp_prob[-n:], dtype=float)
        if n > 0
        else np.asarray(cp_prob, dtype=float)
    )
    recent_floor = len(cp_prob) - n
    recent_cps = [cp for cp in change_points if cp.get("idx", 0) >= recent_floor]
    summary = {
        "lookback": int(n),
        "last_cp_prob": float(cp_prob[-1]) if len(cp_prob) else float("nan"),
        "max_cp_prob": float(np.nanmax(tail)) if tail.size else float("nan"),
        "mean_cp_prob": float(np.nanmean(tail)) if tail.size else float("nan"),
        "change_points_count": int(len(recent_cps)),
        "raw_change_points_count": int(
            sum(1 for idx in raw_cp_idx if int(idx) >= recent_floor)
        ),
        "filtered_change_points_count": int(
            max(
                0,
                sum(1 for idx in raw_cp_idx if int(idx) >= recent_floor)
                - int(len(recent_cps)),
            )
        ),
        "recent_change_points": recent_cps[-5:],
        "confidence": float(reliability.get("confidence", 0.0)),
        "expected_false_alarm_rate": float(
            reliability.get("expected_false_alarm_rate", expected_fa_rate)
        ),
        "calibration_age_bars": int(
            reliability.get("calibration_age_bars", calibration_age_bars)
        ),
    }
    if tuning_hint is not None:
        summary["tuning_hint"] = tuning_hint
    payload["summary"] = summary
    if output == "summary":
        return _summary_only_payload(payload)
    if output == "compact" and n > 0:
        tail_offset = len(payload.get("times", [])) - n
        payload["times"] = payload["times"][-n:]
        payload["cp_prob"] = payload["cp_prob"][-n:]
        if isinstance(payload.get("_series_values"), list):
            payload["_series_values"] = payload["_series_values"][-n:]
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
    """Apply output mode filtering.

    - 'summary': Return stats only, no regimes
    - 'compact': Trading-focused (regimes, current_regime, regime_info, reliability)
    - 'full': Research-focused (adds raw series, params, technical details)
    """
    payload["summary"] = summary
    if output == "summary":
        return _summary_only_payload(payload)
    # Note: Raw series (times, state, state_probabilities) are now handled
    # in _consolidate_payload based on output_mode
    return payload


# Timeframe-based default parameters for regime detection
_TIMEFRAME_DEFAULTS: Dict[str, Dict[str, int]] = {
    # Intraday high-frequency
    "M1": {"lookback": 3000, "min_regime_bars": 30},  # ~2 days, 30 min regimes
    "M5": {"lookback": 2000, "min_regime_bars": 12},  # ~7 days, 1 hour regimes
    "M15": {"lookback": 1000, "min_regime_bars": 8},  # ~10 days, 2 hour regimes
    "M30": {"lookback": 800, "min_regime_bars": 6},  # ~16 days, 3 hour regimes
    # Standard intraday/swing
    "H1": {"lookback": 500, "min_regime_bars": 4},  # ~21 days, 4 hour regimes
    "H2": {"lookback": 400, "min_regime_bars": 3},  # ~33 days, 6 hour regimes
    "H4": {"lookback": 300, "min_regime_bars": 3},  # ~50 days, 12 hour regimes
    "H6": {"lookback": 250, "min_regime_bars": 2},  # ~62 days, 12 hour regimes
    "H8": {"lookback": 200, "min_regime_bars": 2},  # ~66 days, 16 hour regimes
    "H12": {"lookback": 150, "min_regime_bars": 2},  # ~75 days, 24 hour regimes
    # Daily and higher
    "D1": {"lookback": 200, "min_regime_bars": 2},  # ~200 days, 2 day regimes
    "W1": {"lookback": 100, "min_regime_bars": 2},  # ~100 weeks, 2 week regimes
    "MN1": {"lookback": 48, "min_regime_bars": 2},  # ~48 months, 2 month regimes
}


def _get_timeframe_defaults(timeframe: str) -> Dict[str, int]:
    """Get sensible defaults for regime detection based on timeframe.

    Higher frequency timeframes need more bars for meaningful analysis
    and higher min_regime_bars to avoid micro-noise.
    """
    tf = str(timeframe).strip().upper()
    return _TIMEFRAME_DEFAULTS.get(tf, {"lookback": 300, "min_regime_bars": 5})


@mcp.tool()
def regime_detect(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 800,
    method: Literal[
        "bocpd",
        "hmm",
        "ms_ar",
        "clustering",
        "garch",
        "rule_based",
        "wavelet",
        "ensemble",
        "all",
    ] = "all",  # type: ignore
    target: Literal["return", "price"] = "return",  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    threshold: float = 0.5,
    detail: Literal["full", "summary", "compact"] = "compact",  # type: ignore
    lookback: int = -1,  # -1 means use timeframe-based default
    include_series: bool = False,
    min_regime_bars: int = -1,  # -1 means use timeframe-based default
    max_regimes: int = 10,  # Maximum regimes to show in compact mode
) -> Dict[str, Any]:
    """Detect regimes and/or change-points over the last `limit` bars.

    - method: Default is 'all' (runs all methods and returns individual results for comparison).
      Also available: 'bocpd' (Bayesian online change-point; Gaussian), 'hmm' (Gaussian mixture/HMM-lite),
      'ms_ar' (Markov-switching AR), 'clustering' (rolling-feature clustering via tsfresh + KMeans/Spectral),
      'garch' (GARCH-based volatility regimes), 'rule_based' (trend/ranging/transition classification),
      'wavelet' (multi-resolution wavelet energy regime detection via PyWavelets),
      'ensemble' (consensus across multiple methods).
    - params (clustering): optional `algorithm` = 'kmeans' (default) | 'spectral' (sklearn SpectralClustering).
      Optional `affinity` for spectral (default 'nearest_neighbors').
    - params (wavelet): optional `wavelet` (default 'db4'), `level` (auto), `n_states` (default 3),
      `energy_window` (default 30 bars).
    - params (ensemble): optional `methods` list (default ['bocpd', 'hmm', 'clustering']),
      `voting` = 'soft' (probability averaging, default) | 'hard' (majority vote).
    - params (bocpd): optional `hazard_mode` = auto_default|auto_calibrated (defaults to auto_calibrated).
      Explicit `hazard_lambda` / `cp_threshold` always take precedence over auto selection.
      Optional robustness params:
        `cp_threshold_calibration_mode` (default `walkforward_quantile`),
        `threshold_target_false_alarm_rate`,
        `cp_confirm_bars` (default `1`, live-oriented),
        `min_cp_distance_bars`, `cp_edge_multiplier`.
    - include_series: If True, include raw time series data (probs, states) in output even if detail='full'. Default False.
    - lookback: Number of recent bars to include in summary/compact detail. Default -1 uses timeframe-based defaults:
        M1: 3000, M5: 2000, M15: 1000, M30: 800, H1: 500, H2: 400, H4: 300, H6-H12: 200-150, D1: 200, W1: 100, MN1: 48
    - min_regime_bars: Merge short state runs (< this many bars) for state-based methods to reduce flicker.
        Default -1 uses timeframe-based defaults: M1: 30, M5: 12, M15-M30: 6-8, H1-H4: 3-4, D1+: 2
    - max_regimes: Maximum number of regime segments to show in compact mode (default 10).
        Most recent segments/regimes are shown. Full mode shows all available windows.
    - detail:
        - 'compact' (default): Returns recent consolidated output. BOCPD uses
          `current_segment` / `segments`; state-based methods return
          `current_regime` / `regimes`.
        - 'full': Returns full consolidated output. Raw 'series' included only if include_series=True.
        - 'summary': Returns stats only.

    Output Structure (state-based methods: hmm, ms_ar, clustering, garch, wavelet, ensemble):
        - success: bool - Whether detection succeeded
        - symbol: str - Symbol analyzed
        - timeframe: str - Timeframe used
        - method: str - Method used
        - target: str - 'return' or 'price'
        - regimes: List[Dict] - Regime segments with start, end, bars, regime ID, label, confidence
        - regime_info: Dict - Descriptive info for each regime (label, mean_return, volatility, etc.)
        - summary: Dict - Quick stats including last_state, state_shares, transitions, smoothing status
        - state_probabilities: List[List[float]] - Probability of each regime at each bar (full output only)
        - reliability: Dict - Confidence score and source (method-dependent)
        - params_used: Dict - Parameters actually used
        - warnings: List[str] - Any warnings (optional)

    Method-Specific Notes:
        - 'bocpd': Returns transition-oriented compact/full output:
          `current_segment`, `transition_summary`, `segment_context`, and `segments`.
          These describe whether a new change point has been confirmed, how long the
          current segment has persisted, and derived bias/volatility context from the
          target series. Raw `cp_prob` and `change_points` remain available in `series`
          when include_series=True. Reliability is based on calibration quality.
          Best for detecting transition timing.
        - 'hmm', 'ms_ar', 'clustering': Return 'state' array and 'state_probabilities'.
          Labels like 'positive_low_vol' describe regime characteristics (return + volatility).
          Reliability based on model fit or cluster separation.
        - 'garch': Volatility regime detection using GARCH(1,1) model.
          n_states is AUTO-DETECTED by default based on volatility coefficient of variation (CV):
            CV > 2.0 → 4 states (very_low/low/high/very_high) for very volatile assets (BTC, meme stocks)
            CV > 1.0 → 3 states (low/moderate/high) for moderately volatile assets
            CV ≤ 1.0 → 2 states (low/high) for stable assets (major forex pairs)
          Explicit n_states parameter overrides auto-detection.
          Uses percentile-based classification with volatility characteristics reported in output.
        - 'rule_based': Returns single 'regime' dict with 'state' (trending/ranging/transition),
          'direction' (bullish/bearish/neutral), trend_strength, efficiency_ratio.
          Trend metrics use the recent price window so direction/window_move_pct stay coherent
          even when target='return'. Best for quick trend classification.
        - 'wavelet': Returns 'regime_params' with 'energy_profiles' showing frequency distribution.
          Best for detecting regimes at different time scales.
        - 'ensemble': Consensus across multiple methods with AUTO-DETECTED n_states.
          n_states determined by return distribution kurtosis:
            kurtosis > 6.0 → 6 states
            kurtosis > 4.5 → 5 states
            kurtosis > 3.5 → 4 states
            kurtosis ≤ 3.5 → 3 states
          Labels are derived from observed return sign and volatility tiers so they stay aligned
          with regime statistics. 'ensemble_info' shows voting method and mean_agreement.
          Explicit n_states overrides auto-detection.
        - 'all': Returns 'comparison' dict with current regime per method and semantic agreement
          metrics on comparable direction/volatility signals, plus 'results' with full output
          from each method. Best for method comparison.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="regime_detect",
        symbol=symbol,
        timeframe=timeframe,
        method=method,
        target=target,
        detail=detail,
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
            detail=detail,
            limit=limit,
        )
        return result

    output = str(detail).strip().lower()
    connection_error = _regime_connection_error()
    if connection_error is not None:
        return _finish(connection_error)
    try:
        p = dict(params or {})

        # Apply timeframe-based defaults if not explicitly provided
        tf_defaults = _get_timeframe_defaults(timeframe)
        effective_lookback = lookback if lookback >= 0 else tf_defaults["lookback"]
        effective_min_regime_bars = (
            min_regime_bars if min_regime_bars >= 0 else tf_defaults["min_regime_bars"]
        )

        min_regime_bars_val, min_regime_bars_error = _coerce_param(
            p,
            "min_regime_bars",
            default=effective_min_regime_bars,
            cast=int,
            error="min_regime_bars must be an integer >= 1.",
        )
        if min_regime_bars_error is not None:
            return _finish({"error": min_regime_bars_error})
        if min_regime_bars_val < 1:
            return _finish({"error": "min_regime_bars must be >= 1."})

        # Override lookback with effective value (will be used throughout function)
        lookback = p.get("lookback", effective_lookback)
        df = _fetch_history(symbol, timeframe, int(max(limit, 50)), as_of=None)
        if len(df) < 10:
            return _finish({"error": "Insufficient history"})
        base_col = _resolve_denoise_base_col(
            df, denoise, base_col="close", default_when="pre_ti"
        )
        y = df[base_col].astype(float).to_numpy()
        times = df["time"].astype(float).to_numpy()
        price_mask = np.isfinite(y)
        price_series = y[price_mask]
        try:
            return_series = _log_returns_from_prices(y)
        except ValueError as exc:
            return _finish({"error": str(exc)})
        calibration_returns = return_series
        calibration_returns = calibration_returns[np.isfinite(calibration_returns)]
        if target == "return":
            x_raw = return_series
            return_mask = np.isfinite(x_raw)
            x = x_raw[return_mask]
            t = times[1:][return_mask]
        else:
            x = price_series
            t = times[price_mask]

        if x.size < 2:
            return _finish({"error": "Insufficient finite observations after filter"})

        # format times
        t_fmt = [_format_time_minimal(tt) for tt in t]

        if method == "bocpd":
            from ...utils.regime import bocpd_gaussian

            hazard_mode = (
                str(p.get("hazard_mode", "auto_calibrated") or "auto_calibrated")
                .strip()
                .lower()
            )
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
                auto_hazard, auto_threshold, calibration_info = (
                    _auto_calibrate_bocpd_params(
                        returns=calibration_returns, symbol=symbol, timeframe=timeframe
                    )
                )

            if "hazard_lambda" in p and p.get("hazard_lambda") is not None:
                hazard_lambda = int(p.get("hazard_lambda"))
            else:
                hazard_lambda = int(auto_hazard)
                hazard_src = (
                    "auto_calibrated"
                    if hazard_mode == "auto_calibrated"
                    else "auto_default"
                )
            if "cp_threshold" in p and p.get("cp_threshold") is not None:
                threshold_used = float(p.get("cp_threshold"))
                threshold_src = "params.cp_threshold"
            elif "threshold" in p and p.get("threshold") is not None:
                threshold_used = float(p.get("threshold"))
                threshold_src = "params.threshold"
            else:
                if abs(float(threshold) - 0.5) <= 1e-12:
                    threshold_used = float(auto_threshold)
                    threshold_src = (
                        "auto_calibrated"
                        if hazard_mode == "auto_calibrated"
                        else "auto_default"
                    )
                else:
                    threshold_used = float(threshold)
                    threshold_src = "arg"
            max_rl, _ = _coerce_param(
                p,
                "max_run_length",
                default=min(1000, x.size),
                cast=int,
            )
            threshold_cal_mode = (
                str(
                    p.get("cp_threshold_calibration_mode", "walkforward_quantile")
                    or "walkforward_quantile"
                )
                .strip()
                .lower()
            )
            if threshold_cal_mode in {"auto", "walkforward", "quantile"}:
                threshold_cal_mode = "walkforward_quantile"
            if (
                threshold_src in {"auto_calibrated", "auto_default"}
                and threshold_cal_mode == "walkforward_quantile"
            ):
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
                threshold_used, threshold_calibration_info = (
                    _walkforward_quantile_threshold_calibration(
                        series=x,
                        hazard_lambda=hazard_lambda,
                        base_threshold=threshold_used,
                        target_false_alarm_rate=target_fa,
                        window=cal_window,
                        step=cal_step,
                        max_windows=cal_max_windows,
                        bootstrap_runs=cal_boot,
                    )
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
            cp_prob = np.asarray(
                res.get("cp_prob", np.zeros_like(x, dtype=float)), dtype=float
            )
            raw_cp_idx = [
                int(i)
                for i, v in enumerate(cp_prob.tolist())
                if np.isfinite(v) and float(v) >= float(threshold_used)
            ]
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
            cps = [
                {"idx": i, "time": t_fmt[i], "prob": float(cp_prob[i])} for i in cp_idx
            ]
            tuning_hint: Optional[str] = None
            if len(cps) == 0:
                if (
                    len(raw_cp_idx) > 0
                    and int(cp_filter_meta.get("filtered_count", 0)) > 0
                ):
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
                        calibration_info.get("points", 0)
                        if isinstance(calibration_info, dict)
                        else 0,
                    )
                )
                threshold_calibrated = bool(
                    threshold_calibration_info.get("calibrated", False)
                )
            else:
                expected_fa_rate = 0.02
                calibration_age_bars = int(
                    calibration_info.get("points", 0)
                    if isinstance(calibration_info, dict)
                    else 0
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
                "cp_prob": [
                    float(v) for v in np.asarray(cp_prob, dtype=float).tolist()
                ],
                "change_points": cps,
                "_series_values": [
                    float(v) for v in np.asarray(x, dtype=float).tolist()
                ],
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
                payload["params_used"]["cp_threshold_calibration"] = (
                    threshold_calibration_info
                )
            if tuning_hint is not None:
                payload["tuning_hint"] = tuning_hint
            if output in ("summary", "compact"):
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

            return _finish(
                _consolidate_payload(
                    payload,
                    method,
                    output,
                    include_series=include_series,
                    max_regimes=max_regimes,
                )
            )

        elif method == "ms_ar":
            try:
                from statsmodels.tsa.regime_switching.markov_regression import (
                    MarkovRegression,  # type: ignore
                )
            except Exception:
                return _finish(
                    {
                        "error": "statsmodels MarkovRegression not available. Install statsmodels."
                    }
                )
            k_regimes, _ = _coerce_param(p, "k_regimes", default=2, cast=int)
            order, _ = _coerce_param(p, "order", default=0, cast=int)
            try:
                mod = MarkovRegression(
                    endog=x,
                    k_regimes=max(2, k_regimes),
                    trend="c",
                    order=max(0, order),
                    switching_variance=True,
                )
                maxiter, _ = _coerce_param(p, "maxiter", default=100, cast=int)
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
                    state,
                    probs,
                    x,
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

            # Build regime parameters (mean/vol per regime)
            msar_regime_params = {"mean_return": [], "volatility": []}
            for s in range(k_regimes):
                mask = state == s
                if mask.any():
                    msar_regime_params["mean_return"].append(float(np.mean(x[mask])))
                    msar_regime_params["volatility"].append(float(np.std(x[mask])))
                else:
                    msar_regime_params["mean_return"].append(0.0)
                    msar_regime_params["volatility"].append(0.0)

            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in state.tolist()],
                "state_probabilities": [
                    [float(v) for v in row] for row in probs.tolist()
                ],
                "regime_params": msar_regime_params,
                "params_used": {
                    "k_regimes": k_regimes,
                    "order": order,
                    "min_regime_bars": int(min_regime_bars_val),
                    "relabeled": bool(canon_meta.get("relabeled", False)),
                    "smoothing_applied": bool(
                        smoothing_meta.get("smoothing_applied", False)
                    ),
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                },
            }
            if canon_meta.get("mapping"):
                payload["params_used"]["label_mapping"] = canon_meta["mapping"]
            if converged is not None:
                payload["params_used"]["converged"] = bool(converged)
                if converged is False:
                    payload["warnings"] = [
                        "MS-AR model did not converge; regime probabilities may be unreliable."
                    ]
            # Add reliability info
            reliability = _ms_ar_reliability_from_smoothed(
                smoothed_probs=probs,
                params_used=payload["params_used"],
            )
            payload["reliability"] = reliability

            if output in ("summary", "compact"):
                n = _summary_window_size(lookback, len(state))
                st_tail = state[-n:] if n > 0 else state
                last_s = int(state[-1]) if len(state) else None
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail) or 1)
                    for k, c in zip(unique, counts)
                }
                summary = {
                    "lookback": int(n),
                    "last_state": last_s,
                    "state_shares": shares,
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                    "smoothing_applied": bool(
                        smoothing_meta.get("smoothing_applied", False)
                    ),
                }
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(
                _consolidate_payload(
                    payload,
                    method,
                    output,
                    include_series=include_series,
                    max_regimes=max_regimes,
                )
            )

        elif method == "hmm":  # 'hmm' (mixture/HMM-lite)
            n_states, n_states_error = _coerce_param(
                p,
                "n_states",
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
            fit_gaussian_mixture_1d = globals().get(
                "fit_gaussian_mixture_1d", fit_gaussian_mixture_1d
            )
            w, mu, sigma, gamma, _ = fit_gaussian_mixture_1d(x, n_states=n_states)
            gamma_matrix = _normalize_state_probability_matrix(
                gamma,
                rows=x.size,
                requested_states=n_states,
            )
            state = (
                np.argmax(gamma_matrix, axis=1)
                if gamma_matrix.size
                else np.zeros(x.size, dtype=int)
            )
            gamma_smoothed: Optional[np.ndarray] = gamma_matrix
            state, gamma_smoothed, smoothing_meta = _smooth_short_state_runs(
                state=np.asarray(state, dtype=int),
                probs=gamma_smoothed,
                min_regime_bars=min_regime_bars_val,
            )
            state, gamma_smoothed, canon_meta = _canonicalize_regime_labels(
                state,
                gamma_smoothed,
                x,
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
                    [float(v) for v in row] for row in gamma_for_payload.tolist()
                ],
                "regime_params": {
                    "weights": [float(v) for v in w.tolist()],
                    "mu": [float(v) for v in mu.tolist()],
                    "sigma": [float(v) for v in sigma.tolist()],
                },
                "params_used": {
                    "n_states": int(n_states),
                    "fitted_n_states": int(len(mu)),
                    "min_regime_bars": int(min_regime_bars_val),
                    "relabeled": bool(canon_meta.get("relabeled", False)),
                    "smoothing_applied": bool(
                        smoothing_meta.get("smoothing_applied", False)
                    ),
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                },
            }
            if canon_meta.get("mapping"):
                payload["params_used"]["label_mapping"] = canon_meta["mapping"]
            # Add reliability info
            reliability = _hmm_reliability_from_gamma(gamma_for_payload)
            payload["reliability"] = reliability

            if output in ("summary", "compact"):
                n = _summary_window_size(lookback, len(state))
                st_tail = state[-n:] if n > 0 else state
                last_s = int(state[-1]) if len(state) else None
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail) or 1)
                    for k, c in zip(unique, counts)
                }
                order = np.argsort(sigma)
                ranks = {int(s): int(r) for r, s in enumerate(order)}
                summary = {
                    "lookback": int(n),
                    "last_state": last_s,
                    "state_shares": shares,
                    "state_sigma": {int(i): float(sigma[i]) for i in range(len(sigma))},
                    "state_order_by_sigma": ranks,
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                    "smoothing_applied": bool(
                        smoothing_meta.get("smoothing_applied", False)
                    ),
                }
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(
                _consolidate_payload(
                    payload,
                    method,
                    output,
                    include_series=include_series,
                    max_regimes=max_regimes,
                )
            )

        elif method == "clustering":
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
                algorithm = str(p.get("algorithm", "kmeans")).strip().lower()
                spectral_cls = None
                if algorithm == "spectral":
                    _sc = globals().get("SpectralClustering")
                    if _sc is None:
                        from sklearn.cluster import (
                            SpectralClustering as _sc,
                        )
                    spectral_cls = _sc
            except ImportError as ex:
                return _finish({"error": f"Clustering dependencies missing: {ex}"})
            window_size, _ = _coerce_param(p, "window_size", default=20, cast=int)
            k_regimes, _ = _coerce_param(p, "k_regimes", default=3, cast=int)
            use_pca = bool(p.get("use_pca", True))
            n_components, _ = _coerce_param(p, "n_components", default=3, cast=int)
            clustering_warnings: List[str] = []
            if target == "price":
                clustering_warnings.append(
                    "Clustering on price features may produce level-dependent regimes. Consider target='return'."
                )

            # Extract features (use 'return' or 'price'? 'return' is stationary, usually better)
            # x is already computed based on target input
            extract_rolling_features_impl = globals().get(
                "extract_rolling_features", extract_rolling_features
            )
            if extract_rolling_features_impl is extract_rolling_features:
                extract_rolling_features_impl = (
                    _features_module.extract_rolling_features
                )
            features_df = extract_rolling_features_impl(x, window_size=window_size)

            # Align features with time
            # valid_indices are where features are not NaN
            valid_mask = ~features_df.isna().any(axis=1)
            X_valid = features_df.loc[valid_mask]

            if X_valid.empty:
                return _finish(
                    {
                        "error": "Not enough data for feature extraction (check window_size)"
                    }
                )

            # Normalize
            scaler = standard_scaler_cls()
            X_scaled = scaler.fit_transform(X_valid)

            # PCA
            if use_pca and X_scaled.shape[1] > n_components:
                pca = pca_cls(n_components=min(n_components, X_scaled.shape[1]))
                X_final = pca.fit_transform(X_scaled)
            else:
                X_final = X_scaled

            # Cluster
            n_samples = X_final.shape[0]
            if n_samples < k_regimes:
                return _finish(
                    {
                        "error": f"Not enough samples ({n_samples}) for {k_regimes} clusters"
                    }
                )

            if algorithm == "spectral" and spectral_cls is not None:
                affinity = str(p.get("affinity", "nearest_neighbors")).strip().lower()
                sc_kwargs: Dict[str, Any] = {
                    "n_clusters": k_regimes,
                    "affinity": affinity,
                    "random_state": 42,
                    "assign_labels": "kmeans",
                    "n_init": 1,
                }
                if affinity == "nearest_neighbors":
                    sc_kwargs["n_neighbors"] = min(
                        max(5, n_samples // 10), n_samples - 1
                    )
                sc = spectral_cls(**sc_kwargs)
                labels = sc.fit_predict(X_final)
            else:
                # KMeans — seed centroids from evenly-spaced rows so KMeans++
                # init is skipped.  KMeans++ triggers joblib CPU-topology probing
                # which blocks indefinitely in asyncio.to_thread workers on Windows.
                idx = np.round(np.linspace(0, n_samples - 1, k_regimes)).astype(int)
                kmeans = kmeans_cls(
                    n_clusters=k_regimes,
                    random_state=42,
                    n_init=1,
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
                labels,
                valid_probs,
                x[valid_mask],
            )
            smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)

            # Map back to full length (-1 for undefined leading window)
            full_states = np.full(len(x), -1, dtype=int)
            full_states[valid_mask] = labels

            full_probs = np.zeros((len(x), k_regimes))
            full_probs[valid_mask] = valid_probs

            # Build regime parameters from data
            clustering_regime_params = {"mean_return": [], "volatility": []}
            for s in range(k_regimes):
                mask = full_states == s
                if mask.any():
                    clustering_regime_params["mean_return"].append(
                        float(np.mean(x[mask]))
                    )
                    clustering_regime_params["volatility"].append(
                        float(np.std(x[mask]))
                    )
                else:
                    clustering_regime_params["mean_return"].append(0.0)
                    clustering_regime_params["volatility"].append(0.0)

            # Reconstruct payload
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in full_states.tolist()],
                "state_probabilities": [
                    [float(v) for v in row] for row in full_probs.tolist()
                ],
                "regime_params": clustering_regime_params,
                "params_used": {
                    "k_regimes": k_regimes,
                    "algorithm": algorithm,
                    "window_size": window_size,
                    "use_pca": use_pca,
                    "n_components": n_components,
                    "min_regime_bars": int(min_regime_bars_val),
                    "smoothing_applied": smoothing_meta.get("smoothing_applied", False),
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                },
            }
            if clustering_warnings:
                payload["warnings"] = clustering_warnings

            # Summary stats
            if output in ("summary", "compact"):
                n_summary = _summary_window_size(lookback, len(full_states))
                st_tail = full_states[-n_summary:] if n_summary > 0 else full_states
                # Filter out -1
                st_tail_valid = st_tail[st_tail != -1]

                unique, counts = np.unique(st_tail_valid, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail_valid) or 1)
                    for k, c in zip(unique, counts)
                }

                summary = {
                    "lookback": int(n_summary),
                    "last_state": int(full_states[-1]) if len(full_states) else None,
                    "state_shares": shares,
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                    "smoothing_applied": bool(
                        smoothing_meta.get("smoothing_applied", False)
                    ),
                }
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            # Add reliability based on cluster separation
            # Compute variance ratio: between-cluster variance / total variance
            total_var = float(np.var(x)) if len(x) > 1 else 0.0
            if total_var > 1e-9:
                between_var = 0.0
                overall_mean = float(np.mean(x))
                for s in range(k_regimes):
                    mask = full_states == s
                    if mask.any():
                        cluster_mean = float(np.mean(x[mask]))
                        cluster_size = int(mask.sum())
                        between_var += cluster_size * (cluster_mean - overall_mean) ** 2
                between_var /= len(x)
                variance_ratio = between_var / total_var
                reliability_score = min(
                    1.0, variance_ratio * 2
                )  # Scale for interpretability
            else:
                variance_ratio = 0.0
                reliability_score = 0.0

            payload["reliability"] = {
                "confidence": round(reliability_score, 4),
                "variance_ratio": round(variance_ratio, 4),
                "source": "cluster_separation",
            }

            return _finish(
                _consolidate_payload(
                    payload,
                    method,
                    output,
                    include_series=include_series,
                    max_regimes=max_regimes,
                )
            )

        elif method == "garch":
            # GARCH-based volatility regime detection
            garch_warnings: List[str] = []
            try:
                from arch import arch_model
            except ImportError:
                return _finish(
                    {
                        "error": "arch package required for GARCH regime detection. Install: pip install arch"
                    }
                )

            # Auto-detect optimal n_states if not explicitly provided
            # Based on volatility distribution characteristics
            n_states_input = p.get("n_states")

            if n_states_input is None:
                # Calculate rolling realized volatility for better characterization
                # Use 20-bar rolling window to capture volatility clustering
                window = min(20, len(x) // 4)
                if window < 5:
                    window = 5

                # Rolling standard deviation of returns
                rolling_vol = np.array(
                    [np.std(x[max(0, i - window) : i + 1]) for i in range(len(x))]
                )
                rolling_vol = rolling_vol[np.isfinite(rolling_vol) & (rolling_vol > 0)]

                if len(rolling_vol) > 10:
                    # Use ratio of 90th to 10th percentile to measure volatility range
                    vol_p90 = np.percentile(rolling_vol, 90)
                    vol_p10 = np.percentile(rolling_vol, 10)
                    vol_ratio = vol_p90 / vol_p10 if vol_p10 > 1e-9 else 1.0

                    # Also calculate kurtosis of returns (fat tails indicator)
                    returns_kurt = (
                        float(np.mean((x - np.mean(x)) ** 4) / (np.std(x) ** 4))
                        if np.std(x) > 1e-9
                        else 3.0
                    )

                    # Infer optimal states based on vol_ratio and kurtosis
                    # High vol_ratio (10+) or high kurtosis (>6) suggests need for more states
                    if vol_ratio > 10.0 or returns_kurt > 6.0:
                        n_states_auto = (
                            4  # Very volatile - need very_low/low/high/very_high
                        )
                    elif vol_ratio > 5.0 or returns_kurt > 4.0:
                        n_states_auto = 3  # Moderately volatile - low/moderate/high
                    else:
                        n_states_auto = 2  # Stable - binary classification sufficient

                    auto_detect_metrics = {
                        "vol_ratio_90_10": round(vol_ratio, 2),
                        "returns_kurtosis": round(returns_kurt, 2),
                    }
                else:
                    # Insufficient data, default to 3 states
                    n_states_auto = 3
                    auto_detect_metrics = {}

                n_states_garch = n_states_auto
                garch_auto_n_states = True
            else:
                n_states_garch, _ = _coerce_param(p, "n_states", default=3, cast=int)
                garch_auto_n_states = False
            garch_p, _ = _coerce_param(p, "p_order", default=1, cast=int)
            garch_q, _ = _coerce_param(p, "q_order", default=1, cast=int)
            vol_threshold, _ = _coerce_param(
                p, "vol_threshold", default=None, cast=float
            )

            if n_states_garch < 2:
                return _finish({"error": "n_states must be >= 2 for garch method."})

            # Fit GARCH model
            try:
                # Scale returns for numerical stability
                scale = 100.0
                x_scaled = x * scale

                am = arch_model(
                    x_scaled,
                    vol="GARCH",
                    p=max(1, garch_p),
                    q=max(1, garch_q),
                    dist="normal",
                    mean="Constant",
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = am.fit(disp="off", show_warning=False)

                # Extract conditional volatility
                conditional_vol = res.conditional_volatility / scale  # Unscale

                # Create regime states based on volatility levels
                # Strategy: sort volatilities and assign regimes based on percentiles
                # State 0 = lowest vol, State N-1 = highest vol
                valid_vol = conditional_vol[np.isfinite(conditional_vol)]

                if len(valid_vol) < n_states_garch * 10:
                    return _finish(
                        {
                            "error": f"Insufficient data for GARCH regime detection (need {n_states_garch * 10}+ bars)"
                        }
                    )

                # Determine volatility thresholds
                if vol_threshold is not None and n_states_garch == 2:
                    # Binary classification with explicit threshold
                    thresholds = [vol_threshold]
                else:
                    # Use percentiles for n_states
                    percentiles = np.linspace(0, 100, n_states_garch + 1)[1:-1]
                    thresholds = [np.percentile(valid_vol, p) for p in percentiles]

                # Assign states based on volatility levels
                state = np.zeros(len(conditional_vol), dtype=int)
                for i, thresh in enumerate(thresholds):
                    state[conditional_vol > thresh] = i + 1

                # Handle non-finite values
                state[~np.isfinite(conditional_vol)] = -1

                # Create probability matrix (hard assignment for GARCH)
                probs = np.zeros((len(state), n_states_garch))
                for i in range(len(state)):
                    if 0 <= state[i] < n_states_garch:
                        probs[i, state[i]] = 1.0

                # Smooth short runs
                state, probs, smoothing_meta = _smooth_short_state_runs(
                    state=np.asarray(state, dtype=int),
                    probs=probs,
                    min_regime_bars=min_regime_bars_val,
                )

                # Build regime parameters
                regime_params = {"volatility": [], "mean_return": []}
                for s in range(n_states_garch):
                    mask = state == s
                    if mask.any():
                        regime_params["volatility"].append(
                            float(np.mean(conditional_vol[mask]))
                        )
                        regime_params["mean_return"].append(
                            float(np.mean(x[mask])) if mask.sum() > 0 else 0.0
                        )
                    else:
                        regime_params["volatility"].append(0.0)
                        regime_params["mean_return"].append(0.0)

                # Build payload
                # Check if n_states seems appropriate for this asset
                garch_warnings = []
                vol_std = float(np.std(valid_vol))
                vol_mean = float(np.mean(valid_vol))
                cv = (
                    vol_std / vol_mean if vol_mean > 1e-9 else 0
                )  # Coefficient of variation

                # Heuristic: High CV (>1.0) suggests volatile asset needing more states
                if cv > 1.0 and n_states_garch < 3:
                    garch_warnings.append(
                        f"High volatility variation detected (CV={cv:.2f}). "
                        f"Consider n_states=3 or 4 for better regime separation."
                    )

                # Build volatility characteristics for transparency
                vol_characteristics = {
                    "cv": round(cv, 4),
                    "mean": round(float(np.mean(valid_vol)), 6),
                    "std": round(float(np.std(valid_vol)), 6),
                    "percentile_33": round(float(np.percentile(valid_vol, 33)), 6),
                    "percentile_66": round(float(np.percentile(valid_vol, 66)), 6),
                }

                # Add auto-detection metrics if applicable
                if garch_auto_n_states and auto_detect_metrics:
                    vol_characteristics["auto_detection"] = auto_detect_metrics

                payload = {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "method": method,
                    "target": target,
                    "times": t_fmt,
                    "state": [int(s) for s in state.tolist()],
                    "state_probabilities": [
                        [float(v) for v in row] for row in probs.tolist()
                    ],
                    "conditional_volatility": [
                        float(v) for v in conditional_vol.tolist()
                    ],
                    "regime_params": regime_params,
                    "params_used": {
                        "n_states": int(n_states_garch),
                        "n_states_auto": bool(garch_auto_n_states),
                        "p_order": int(garch_p),
                        "q_order": int(garch_q),
                        "min_regime_bars": int(min_regime_bars_val),
                        "smoothing_applied": bool(
                            smoothing_meta.get("smoothing_applied", False)
                        ),
                        "transitions_before": int(
                            smoothing_meta.get("transitions_before", 0)
                        ),
                        "transitions_after": int(
                            smoothing_meta.get("transitions_after", 0)
                        ),
                    },
                    "volatility_characteristics": vol_characteristics,
                }
                if garch_warnings:
                    payload["warnings"] = garch_warnings

                # Add reliability estimate based on model fit
                if hasattr(res, "aic") and hasattr(res, "bic"):
                    payload["model_fit"] = {
                        "aic": float(res.aic),
                        "bic": float(res.bic),
                        "loglikelihood": float(res.loglikelihood)
                        if hasattr(res, "loglikelihood")
                        else None,
                    }
                    # Reliability: lower BIC = better model = higher reliability
                    # Normalize to 0-1 scale (approximate)
                    bic = float(res.bic)
                    n_samples = len(x)
                    # Heuristic: BIC per sample < -5 is very good, > 0 is poor
                    bic_per_sample = bic / n_samples if n_samples > 0 else 0
                    reliability_score = max(
                        0.0, min(1.0, 1.0 - (bic_per_sample + 5) / 10)
                    )
                    payload["reliability"] = {
                        "confidence": round(reliability_score, 4),
                        "bic_per_sample": round(bic_per_sample, 4),
                        "source": "bic_normalized",
                    }

                # Add summary for compact/summary output
                if output in ("summary", "compact"):
                    n = _summary_window_size(lookback, len(state))
                    st_tail = state[-n:] if n > 0 else state
                    vol_tail = conditional_vol[-n:] if n > 0 else conditional_vol
                    last_s = int(state[-1]) if len(state) else None

                    unique, counts = np.unique(
                        st_tail[st_tail >= 0], return_counts=True
                    )
                    shares = {
                        int(k): float(c) / float(len(st_tail[st_tail >= 0]) or 1)
                        for k, c in zip(unique, counts)
                    }

                    summary = {
                        "lookback": int(n),
                        "last_state": last_s,
                        "state_shares": shares,
                        "current_conditional_vol": float(conditional_vol[-1])
                        if len(conditional_vol)
                        else None,
                        "avg_conditional_vol": float(np.mean(vol_tail))
                        if len(vol_tail)
                        else None,
                        "transitions_before": int(
                            smoothing_meta.get("transitions_before", 0)
                        ),
                        "transitions_after": int(
                            smoothing_meta.get("transitions_after", 0)
                        ),
                        "smoothing_applied": bool(
                            smoothing_meta.get("smoothing_applied", False)
                        ),
                    }
                    payload = _apply_state_output_mode(
                        payload,
                        output=output,
                        lookback=lookback,
                        summary=summary,
                    )
                    if output == "summary":
                        return _finish(payload)

                return _finish(
                    _consolidate_payload(
                        payload,
                        method,
                        output,
                        include_series=include_series,
                        max_regimes=max_regimes,
                    )
                )

            except Exception as ex:
                return _finish({"error": f"GARCH regime detection failed: {str(ex)}"})

        elif method == "rule_based":
            # Rule-based trend/ranging/transition detection
            # Based on the internal _infer_market_regime from patterns_support.py

            # Get parameters
            efficiency_threshold, _ = _coerce_param(
                p, "efficiency_threshold", default=0.35, cast=float
            )
            trend_strength_threshold, _ = _coerce_param(
                p, "trend_strength_threshold", default=1.25, cast=float
            )
            window_bars, _ = _coerce_param(
                p, "window_bars", default=min(len(price_series), 160), cast=int
            )
            # Ensure window isn't too large
            window_bars = min(window_bars, len(price_series))

            if window_bars < 20:
                return _finish(
                    {
                        "error": f"Insufficient data for rule-based regime (need 20+ bars, got {window_bars})"
                    }
                )

            # Use the recent price window so direction and movement metrics stay
            # meaningful even when the requested target is return-based.
            segment = price_series[-window_bars:]

            # Calculate metrics
            diffs = np.diff(segment)
            finite_diffs = diffs[np.isfinite(diffs)]
            path_length = (
                float(np.sum(np.abs(finite_diffs))) if finite_diffs.size else 0.0
            )
            move = float(segment[-1] - segment[0])
            base_price = float(segment[0]) if abs(float(segment[0])) > 1e-9 else 1e-9

            # Trend strength: move relative to volatility
            trend_strength = float(abs(move) / max(float(np.nanstd(segment)), 1e-9))

            # Efficiency ratio: how direct was the move
            efficiency_ratio = float(abs(move) / max(path_length, 1e-9))

            # Determine regime
            if (
                efficiency_ratio >= efficiency_threshold
                and trend_strength >= trend_strength_threshold
            ):
                regime_state = "trending"
            elif efficiency_ratio <= max(0.1, 0.55 * efficiency_threshold):
                regime_state = "ranging"
            else:
                regime_state = "transition"

            # Determine direction
            if move > 1e-9:
                direction = "bullish"
            elif move < -1e-9:
                direction = "bearish"
            else:
                direction = "neutral"

            # Build payload - single regime for the window
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "regime": {
                    "state": regime_state,
                    "direction": direction,
                    "trend_strength": round(trend_strength, 4),
                    "efficiency_ratio": round(efficiency_ratio, 4),
                    "window_bars": int(window_bars),
                    "window_move_pct": round((move / base_price) * 100.0, 4),
                    "signal_source": "price",
                },
                "params_used": {
                    "efficiency_threshold": float(efficiency_threshold),
                    "trend_strength_threshold": float(trend_strength_threshold),
                    "window_bars": int(window_bars),
                    "signal_source": "price",
                },
            }

            return _finish(payload)

        elif method == "wavelet":
            # Multi-resolution wavelet energy regime detection.
            # Decomposes the series via DWT, computes rolling energy at each
            # decomposition level, then clusters the energy feature vectors
            # to identify regimes that differ in frequency content.
            try:
                import pywt as _pywt
            except ImportError:
                return _finish(
                    {
                        "error": "PyWavelets required for wavelet regime detection. "
                        "Install: pip install PyWavelets"
                    }
                )

            wavelet_name = str(p.get("wavelet", "db4")).strip()
            n_states_wv, _ = _coerce_param(p, "n_states", default=3, cast=int)
            energy_window, _ = _coerce_param(p, "energy_window", default=30, cast=int)

            if n_states_wv < 2:
                return _finish({"error": "n_states must be >= 2 for wavelet method."})
            if len(x) < energy_window + 10:
                return _finish(
                    {
                        "error": f"Insufficient data for wavelet regime detection "
                        f"(need {energy_window + 10}+ bars, got {len(x)})"
                    }
                )

            # Determine decomposition level
            try:
                w = _pywt.Wavelet(wavelet_name)
            except Exception:
                return _finish({"error": f"Unknown wavelet: {wavelet_name}"})
            max_level = _pywt.dwt_max_level(len(x), w.dec_len)
            user_level = p.get("level")
            if user_level is not None:
                level = max(1, min(int(user_level), max_level))
            else:
                level = max(1, min(4, max_level))

            # DWT decomposition
            coeffs = _pywt.wavedec(x, wavelet_name, mode="periodization", level=level)
            # coeffs[0] = approx (low-freq trend), coeffs[1..level] = details (high→low freq)

            # Reconstruct each detail band at full length
            bands: List[np.ndarray] = []
            for i in range(1, len(coeffs)):
                # Zero out all coefficients except band i
                zeroed = [np.zeros_like(c) for c in coeffs]
                zeroed[i] = coeffs[i]
                band = _pywt.waverec(zeroed, wavelet_name, mode="periodization")
                bands.append(np.asarray(band[: len(x)], dtype=float))

            if not bands:
                return _finish(
                    {"error": "Wavelet decomposition produced no detail bands."}
                )

            # Compute rolling energy (variance) for each band
            n_bars = len(x)
            n_bands = len(bands)
            energy_matrix = np.zeros((n_bars, n_bands))
            half_win = energy_window // 2
            for bi, band in enumerate(bands):
                sq = band**2
                # Cumulative sum for fast rolling mean
                cs = np.concatenate([[0.0], np.cumsum(sq)])
                for t in range(n_bars):
                    lo = max(0, t - half_win)
                    hi = min(n_bars, t + half_win + 1)
                    energy_matrix[t, bi] = (cs[hi] - cs[lo]) / max(1, hi - lo)

            # Normalize energy rows to proportions (energy distribution across scales)
            row_sums = energy_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-16, 1.0, row_sums)
            energy_props = energy_matrix / row_sums

            # Cluster energy profiles into regimes using KMeans
            # (sklearn is already available from clustering branch pattern)
            try:
                from sklearn.cluster import KMeans as _WvKMeans
                from sklearn.preprocessing import StandardScaler as _WvScaler
            except ImportError:
                return _finish(
                    {"error": "sklearn required for wavelet regime clustering."}
                )

            # Skip leading bars where energy window isn't fully populated
            valid_start = min(energy_window, n_bars - 1)
            E_valid = energy_props[valid_start:]
            if len(E_valid) < n_states_wv:
                return _finish(
                    {
                        "error": f"Not enough valid bars ({len(E_valid)}) for "
                        f"{n_states_wv} wavelet regimes."
                    }
                )

            scaler = _WvScaler()
            E_scaled = scaler.fit_transform(E_valid)

            n_valid = E_scaled.shape[0]
            idx = np.round(np.linspace(0, n_valid - 1, n_states_wv)).astype(int)
            km = _WvKMeans(
                n_clusters=n_states_wv,
                random_state=42,
                n_init=1,
                init=E_scaled[idx],
            )
            labels = km.fit_predict(E_scaled)

            # Build probability matrix from cluster distances
            distances = km.transform(E_scaled)  # (n_valid, n_states_wv)
            inv_dist = 1.0 / (distances + 1e-8)
            probs_valid = inv_dist / inv_dist.sum(axis=1, keepdims=True)

            # Smooth and canonicalize
            labels, probs_valid, smoothing_meta = _smooth_short_state_runs(
                state=np.asarray(labels, dtype=int),
                probs=probs_valid,
                min_regime_bars=min_regime_bars_val,
            )
            labels, probs_valid, canon_meta = _canonicalize_regime_labels(
                labels,
                probs_valid,
                x[valid_start:],
            )
            smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)

            # Map back to full length
            full_states = np.full(n_bars, -1, dtype=int)
            full_states[valid_start:] = labels
            full_probs = np.zeros((n_bars, n_states_wv))
            full_probs[valid_start:] = probs_valid

            # Compute per-regime energy profiles for interpretability
            regime_energy_profiles: Dict[str, Any] = {}
            wavelet_regime_params: Dict[str, Any] = {
                "mean_return": [],
                "volatility": [],
                "energy_profiles": regime_energy_profiles,
                "n_bands": n_bands,
                "band_labels": [f"D{i}" for i in range(1, n_bands + 1)],
            }
            x_valid = x[valid_start:]
            for s in range(n_states_wv):
                mask = labels == s
                if mask.any():
                    wavelet_regime_params["mean_return"].append(
                        float(np.mean(x_valid[mask]))
                    )
                    wavelet_regime_params["volatility"].append(
                        float(np.std(x_valid[mask]))
                    )
                    profile = energy_props[valid_start:][mask].mean(axis=0)
                    regime_energy_profiles[str(s)] = {
                        f"band_{bi}_energy": round(float(v), 6)
                        for bi, v in enumerate(profile)
                    }
                else:
                    wavelet_regime_params["mean_return"].append(0.0)
                    wavelet_regime_params["volatility"].append(0.0)

            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in full_states.tolist()],
                "state_probabilities": [
                    [float(v) for v in row] for row in full_probs.tolist()
                ],
                "regime_params": wavelet_regime_params,
                "params_used": {
                    "wavelet": wavelet_name,
                    "level": level,
                    "n_states": n_states_wv,
                    "energy_window": energy_window,
                    "min_regime_bars": int(min_regime_bars_val),
                    "smoothing_applied": smoothing_meta.get("smoothing_applied", False),
                },
            }

            if output in ("summary", "compact"):
                n_summary = _summary_window_size(lookback, len(full_states))
                st_tail = full_states[-n_summary:] if n_summary > 0 else full_states
                st_tail_valid = st_tail[st_tail != -1]
                unique, counts = np.unique(st_tail_valid, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail_valid) or 1)
                    for k, c in zip(unique, counts)
                }
                summary = {
                    "lookback": int(n_summary),
                    "last_state": int(full_states[-1]) if len(full_states) else None,
                    "state_shares": shares,
                }
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(
                _consolidate_payload(
                    payload,
                    method,
                    output,
                    include_series=include_series,
                    max_regimes=max_regimes,
                )
            )

        elif method == "ensemble":
            # Consensus regime detection: run multiple fast methods and
            # aggregate their state_probabilities via soft or hard voting.
            _STATE_METHODS = {"hmm", "ms_ar", "clustering", "garch", "wavelet"}
            default_sub = ["bocpd", "hmm", "clustering", "wavelet"]
            sub_methods_raw = p.get("methods", default_sub)
            if isinstance(sub_methods_raw, str):
                sub_methods_raw = [m.strip() for m in sub_methods_raw.split(",")]
            sub_methods = [
                m for m in sub_methods_raw if m not in ("ensemble", "all", "rule_based")
            ]
            if not sub_methods:
                return _finish({"error": "No valid sub-methods for ensemble."})

            voting = str(p.get("voting", "soft")).strip().lower()

            # Auto-detect optimal n_states if not explicitly provided
            n_states_input = p.get("n_states")
            if n_states_input is None:
                # Analyze return distribution characteristics
                returns_kurt = (
                    float(np.mean((x - np.mean(x)) ** 4) / (np.std(x) ** 4))
                    if np.std(x) > 1e-9
                    else 3.0
                )

                # High kurtosis (>5) suggests fat tails needing more granular regimes
                if returns_kurt > 6.0:
                    n_states_auto = 6  # Very granular: strong_bearish to strong_bullish
                elif returns_kurt > 4.5:
                    n_states_auto = 5  # Rich detail with neutral center
                elif returns_kurt > 3.5:
                    n_states_auto = 4  # Standard: bearish_low/high + bullish_low/high
                else:
                    n_states_auto = 3  # Simple: bearish/neutral/bullish

                n_states_ens = n_states_auto
                ens_auto_n_states = True
                ens_auto_metrics = {"returns_kurtosis": round(returns_kurt, 2)}
            else:
                n_states_ens, _ = _coerce_param(p, "n_states", default=4, cast=int)
                ens_auto_n_states = False
                ens_auto_metrics = {}

            if n_states_ens < 2:
                return _finish({"error": "n_states must be >= 2 for ensemble."})

            # Run each sub-method with include_series so we get raw state data
            sub_results: List[Dict[str, Any]] = []
            sub_errors: List[str] = []
            for sm in sub_methods:
                sub_params = dict(p)
                sub_params.pop("methods", None)
                sub_params.pop("voting", None)
                # Only pass n_states to sub-methods if explicitly provided in params
                # This allows methods like GARCH to auto-detect optimal n_states
                if sm in _STATE_METHODS:
                    # Don't override n_states for garch - let it auto-detect
                    if sm != "garch":
                        sub_params.setdefault("n_states", n_states_ens)
                        sub_params.setdefault("k_regimes", n_states_ens)
                    # For garch, if n_states is already in sub_params, keep it
                    # Otherwise leave it out to trigger auto-detection
                try:
                    sr = regime_detect(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        method=sm,  # type: ignore[arg-type]
                        target=target,
                        params=sub_params,
                        denoise=denoise,
                        threshold=threshold,
                        detail="full",
                        lookback=lookback,
                        include_series=True,
                        min_regime_bars=min_regime_bars,
                        __cli_raw=True,
                    )
                except Exception as exc:
                    sub_errors.append(f"{sm}: {exc}")
                    continue
                if isinstance(sr, dict) and sr.get("error"):
                    sub_errors.append(f"{sm}: {sr['error']}")
                    continue
                sub_results.append({"method": sm, "result": sr})

            if not sub_results:
                return _finish(
                    {
                        "error": f"All ensemble sub-methods failed: {'; '.join(sub_errors)}"
                    }
                )

            # Extract state arrays from sub-results
            # For BOCPD (changepoint), convert cp_prob > threshold to binary state
            state_arrays: List[np.ndarray] = []
            prob_arrays: List[np.ndarray] = []  # (n_bars, n_states) per method
            method_names: List[str] = []
            ref_len = len(t_fmt)

            for sr_info in sub_results:
                sm_name = sr_info["method"]
                sr = sr_info["result"]
                series = sr.get("series", {})

                if sm_name == "bocpd":
                    # BOCPD returns cp_prob, not state — convert to binary
                    cp_prob = series.get("cp_prob", sr.get("cp_prob", []))
                    if cp_prob and len(cp_prob) == ref_len:
                        cp_arr = np.asarray(cp_prob, dtype=float)
                        st = np.where(cp_arr > threshold, 1, 0).astype(int)
                        pr = np.zeros((ref_len, n_states_ens))
                        pr[:, 0] = 1.0 - cp_arr
                        if n_states_ens >= 2:
                            pr[:, 1] = cp_arr
                        state_arrays.append(st)
                        prob_arrays.append(pr)
                        method_names.append(sm_name)
                    continue

                # State-based methods
                raw_state = series.get("state", sr.get("state", []))
                raw_probs = series.get(
                    "state_probabilities", sr.get("state_probabilities", [])
                )
                if not raw_state or len(raw_state) != ref_len:
                    continue

                st = np.asarray(raw_state, dtype=int)
                if raw_probs and len(raw_probs) == ref_len:
                    pr = np.asarray(raw_probs, dtype=float)
                    # Pad or trim columns to n_states_ens
                    if pr.shape[1] < n_states_ens:
                        pr = np.pad(pr, ((0, 0), (0, n_states_ens - pr.shape[1])))
                    elif pr.shape[1] > n_states_ens:
                        pr = pr[:, :n_states_ens]
                else:
                    # Hard assignment fallback
                    pr = np.zeros((ref_len, n_states_ens))
                    for i, s in enumerate(st):
                        if 0 <= s < n_states_ens:
                            pr[i, s] = 1.0

                state_arrays.append(st)
                prob_arrays.append(pr)
                method_names.append(sm_name)

            if not prob_arrays:
                return _finish({"error": "No sub-methods produced usable state data."})

            # Aggregate
            if voting == "hard":
                # Majority vote
                stacked = np.stack(state_arrays, axis=0)  # (n_methods, n_bars)
                from scipy import stats as _sp_stats

                mode_result = _sp_stats.mode(stacked, axis=0, keepdims=False)
                ensemble_state = np.asarray(mode_result.mode, dtype=int).ravel()
                # Build probs from vote fractions
                n_methods = len(state_arrays)
                ensemble_probs = np.zeros((ref_len, n_states_ens))
                for mi in range(n_methods):
                    for t_idx in range(ref_len):
                        s = state_arrays[mi][t_idx]
                        if 0 <= s < n_states_ens:
                            ensemble_probs[t_idx, s] += 1.0
                ensemble_probs /= max(n_methods, 1)
            else:
                # Soft voting: average probabilities
                stacked_probs = np.stack(
                    prob_arrays, axis=0
                )  # (n_methods, n_bars, n_states)
                ensemble_probs = np.mean(stacked_probs, axis=0)
                ensemble_state = np.argmax(ensemble_probs, axis=1).astype(int)

            # Smooth and canonicalize
            ensemble_state, ensemble_probs, smoothing_meta = _smooth_short_state_runs(
                state=ensemble_state,
                probs=ensemble_probs,
                min_regime_bars=min_regime_bars_val,
            )
            ensemble_state, ensemble_probs, canon_meta = _canonicalize_regime_labels(
                ensemble_state,
                ensemble_probs,
                x,
            )
            smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)

            # Agreement score: fraction of methods that agree per bar
            agreement = np.zeros(ref_len)
            for t_idx in range(ref_len):
                votes = [
                    sa[t_idx] for sa in state_arrays if 0 <= sa[t_idx] < n_states_ens
                ]
                if votes:
                    most_common = max(set(votes), key=votes.count)
                    agreement[t_idx] = votes.count(most_common) / len(votes)

            # Compute regime parameters (mean, vol) for each ensemble state
            mean_agreement = round(float(np.mean(agreement)), 4)
            ensemble_regime_params = {
                "mean_return": [],
                "volatility": [],
                "mean_agreement": mean_agreement,  # Backward compatibility
            }
            for s in range(n_states_ens):
                mask = ensemble_state == s
                if mask.any():
                    ensemble_regime_params["mean_return"].append(
                        float(np.mean(x[mask]))
                    )
                    ensemble_regime_params["volatility"].append(float(np.std(x[mask])))
                else:
                    ensemble_regime_params["mean_return"].append(0.0)
                    ensemble_regime_params["volatility"].append(0.0)

            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in ensemble_state.tolist()],
                "state_probabilities": [
                    [float(v) for v in row] for row in ensemble_probs.tolist()
                ],
                "regime_params": ensemble_regime_params,
                "ensemble_info": {
                    "sub_methods": method_names,
                    "voting": voting,
                    "mean_agreement": mean_agreement,
                },
                "params_used": {
                    "methods": method_names,
                    "voting": voting,
                    "n_states": n_states_ens,
                    "n_states_auto": bool(ens_auto_n_states),
                    "n_methods_succeeded": len(method_names),
                    "min_regime_bars": int(min_regime_bars_val),
                    "smoothing_applied": smoothing_meta.get("smoothing_applied", False),
                },
            }
            if ens_auto_metrics:
                payload["auto_detection"] = ens_auto_metrics
            if sub_errors:
                payload["warnings"] = [f"Sub-method errors: {'; '.join(sub_errors)}"]

            if output in ("summary", "compact"):
                n_summary = _summary_window_size(lookback, len(ensemble_state))
                st_tail = (
                    ensemble_state[-n_summary:] if n_summary > 0 else ensemble_state
                )
                st_tail_valid = st_tail[st_tail >= 0]
                unique, counts = np.unique(st_tail_valid, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail_valid) or 1)
                    for k, c in zip(unique, counts)
                }
                summary = {
                    "lookback": int(n_summary),
                    "last_state": int(ensemble_state[-1])
                    if len(ensemble_state)
                    else None,
                    "state_shares": shares,
                    "mean_agreement": round(float(np.mean(agreement)), 4),
                }
                payload = _apply_state_output_mode(
                    payload,
                    output=output,
                    lookback=lookback,
                    summary=summary,
                )
                if output == "summary":
                    return _finish(payload)

            return _finish(
                _consolidate_payload(
                    payload,
                    method,
                    output,
                    include_series=include_series,
                    max_regimes=max_regimes,
                )
            )

        elif method == "all":
            # Run all methods and return individual results for comparison
            all_methods = [
                "bocpd",
                "hmm",
                "ms_ar",
                "clustering",
                "garch",
                "wavelet",
                "rule_based",
            ]
            results_by_method: Dict[str, Any] = {}
            all_errors: List[str] = []

            for m in all_methods:
                try:
                    sub_params = dict(p)
                    # Only set default n_states for methods that don't auto-detect
                    # GARCH auto-detects optimal n_states, don't force a default
                    if m in ("hmm", "ms_ar", "clustering"):
                        sub_params.setdefault("n_states", 2)
                        sub_params.setdefault("k_regimes", 2)
                    # GARCH: if n_states not explicitly set, leave it out for auto-detection
                    sr = regime_detect(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        method=m,  # type: ignore[arg-type]
                        target=target,
                        params=sub_params,
                        denoise=denoise,
                        threshold=threshold,
                        detail="compact",
                        lookback=lookback,
                        include_series=False,
                        min_regime_bars=min_regime_bars,
                        __cli_raw=True,
                    )
                    if isinstance(sr, dict) and not sr.get("error"):
                        # Strip redundant fields that are already at top level
                        # (symbol, timeframe, method, target, success)
                        cleaned_result = {
                            k: v
                            for k, v in sr.items()
                            if k
                            not in (
                                "symbol",
                                "timeframe",
                                "method",
                                "target",
                                "success",
                            )
                        }
                        results_by_method[m] = cleaned_result
                    else:
                        all_errors.append(f"{m}: {sr.get('error', 'unknown error')}")
                except Exception as exc:
                    all_errors.append(f"{m}: {exc}")

            if not results_by_method:
                return _finish(
                    {"error": f"All methods failed: {'; '.join(all_errors)}"}
                )

            # Also run ensemble to provide consensus view
            try:
                ens_params = dict(p)
                ens_params["methods"] = list(
                    results_by_method.keys()
                )  # Use methods that succeeded
                ensemble_result = regime_detect(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    method="ensemble",
                    target=target,
                    params=ens_params,
                    denoise=denoise,
                    threshold=threshold,
                    detail="compact",
                    lookback=lookback,
                    include_series=False,
                    min_regime_bars=min_regime_bars,
                    __cli_raw=True,
                )
                if isinstance(ensemble_result, dict) and not ensemble_result.get(
                    "error"
                ):
                    # Strip redundant fields
                    results_by_method["ensemble"] = {
                        k: v
                        for k, v in ensemble_result.items()
                        if k
                        not in ("symbol", "timeframe", "method", "target", "success")
                    }
            except Exception:
                # Ensemble is optional, don't fail if it errors
                pass

            comparison = _build_all_method_comparison(results_by_method)
            comparison["methods_failed"] = [e.split(":")[0] for e in all_errors]

            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "comparison": comparison,
                "results": results_by_method,
                "params_used": {
                    "methods_attempted": all_methods,
                    "methods_succeeded": list(results_by_method.keys()),
                    "methods_failed": [e.split(":")[0] for e in all_errors],
                },
            }
            if all_errors:
                payload["warnings"] = [f"Method errors: {'; '.join(all_errors)}"]

            return _finish(payload)

    except Exception as e:
        return _finish({"error": f"Error detecting regimes: {str(e)}"})
