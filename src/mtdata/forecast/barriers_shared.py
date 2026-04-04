from typing import Any, Dict, Optional, List, Literal, Tuple, Set
import math
import warnings
import numpy as np
from ..shared.schema import TimeframeLiteral, DenoiseSpec
from ..shared.constants import TIMEFRAME_SECONDS
from .common import fetch_history as _fetch_history, log_returns_from_prices as _log_returns_from_prices
from ..utils.utils import _safe_float, parse_kv_or_json as _parse_kv_or_json
from ..utils.barriers import (
    get_pip_size as _get_pip_size,
    resolve_barrier_prices as _resolve_barrier_prices,
    normalize_trade_direction,
    barrier_prices_are_valid as _barrier_prices_are_valid,
)
from .monte_carlo import (
    simulate_gbm_mc as _simulate_gbm_mc, 
    simulate_hmm_mc as _simulate_hmm_mc, 
    simulate_garch_mc as _simulate_garch_mc,
    simulate_bootstrap_mc as _simulate_bootstrap_mc,
    simulate_heston_mc as _simulate_heston_mc,
    simulate_jump_diffusion_mc as _simulate_jump_diffusion_mc,
    gbm_single_barrier_upcross_prob as _gbm_upcross_prob
)
from .barrier_stats import _confidence_interval_wilson_proportion

BARRIER_GRID_PRESETS = {
    'scalp': {
        'tp_min': 0.08, 'tp_max': 0.60, 'tp_steps': 7,
        'sl_min': 0.20, 'sl_max': 1.20, 'sl_steps': 7,
    },
    'intraday': {
        'tp_min': 0.25, 'tp_max': 1.50, 'tp_steps': 7,
        'sl_min': 0.25, 'sl_max': 2.50, 'sl_steps': 9,
    },
    'swing': {
        'tp_min': 0.60, 'tp_max': 3.50, 'tp_steps': 7,
        'sl_min': 0.50, 'sl_max': 4.50, 'sl_steps': 8,
    },
    'position': {
        'tp_min': 1.00, 'tp_max': 8.00, 'tp_steps': 8,
        'sl_min': 0.75, 'sl_max': 6.00, 'sl_steps': 8,
    },
}

PHANTOM_PROFIT_NO_HIT_THRESHOLD = 0.50
PHANTOM_PROFIT_LOSS_THRESHOLD = 0.01
DEGENERATE_OBJECTIVE_MIN_RESOLVE = 0.20
LOW_CONFIDENCE_CI_THRESHOLD = 0.10


def _binomial_wilson_95(p_hat: float, n: int) -> Tuple[float, float]:
    """Wilson 95% interval for a Bernoulli proportion."""
    return _confidence_interval_wilson_proportion(float(p_hat), int(n), confidence=0.95)


def _binomial_se(p_hat: float, n: int) -> float:
    n_i = int(n)
    if n_i <= 0:
        return float("nan")
    p = float(np.clip(float(p_hat), 0.0, 1.0))
    return float(math.sqrt(max(0.0, p * (1.0 - p) / n_i)))


def _least_negative_ref(best_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Compact pointer payload for non-viable outputs to avoid duplicating `best`."""
    if not isinstance(best_row, dict):
        return None
    return {
        "ref": "best",
        "ev": best_row.get("ev"),
        "edge": best_row.get("edge"),
        "edge_vs_breakeven": best_row.get("edge_vs_breakeven"),
        "kelly": best_row.get("kelly"),
        "phantom_profit_risk": best_row.get("phantom_profit_risk"),
        "reason": "No viable candidate was found; see `best` for full details.",
    }


def _scale_price_paths_to_reference(
    price_paths: np.ndarray,
    *,
    simulated_anchor_price: Any,
    reference_price: Any,
) -> np.ndarray:
    """Rescale simulated paths so barrier scoring uses the same entry anchor."""
    paths = np.asarray(price_paths, dtype=float)
    anchor = _safe_float(simulated_anchor_price)
    ref = _safe_float(reference_price)
    if anchor is None or anchor <= 0.0 or ref is None or ref <= 0.0:
        return paths
    scale = float(ref / anchor)
    if not np.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) <= 1e-12:
        return paths
    return paths * scale


def _sort_candidate_results(res_list: List[Dict[str, Any]], objective_val: str) -> None:
    def _metric(key: str, *, descending: bool) -> Any:
        default = float("-inf") if descending else float("inf")

        def _resolve(row: Dict[str, Any]) -> float:
            value = _safe_float(row.get(key))
            return default if value is None else float(value)

        return _resolve

    if objective_val == 'edge':
        res_list.sort(key=_metric('edge', descending=True), reverse=True)
    elif objective_val == 'ev':
        res_list.sort(key=_metric('ev', descending=True), reverse=True)
    elif objective_val == 'ev_cond':
        res_list.sort(key=_metric('ev_cond', descending=True), reverse=True)
    elif objective_val == 'ev_per_bar':
        res_list.sort(key=_metric('ev_per_bar', descending=True), reverse=True)
    elif objective_val == 'kelly':
        res_list.sort(key=_metric('kelly', descending=True), reverse=True)
    elif objective_val == 'kelly_cond':
        res_list.sort(key=_metric('kelly_cond', descending=True), reverse=True)
    elif objective_val == 'prob_tp_first':
        res_list.sort(key=_metric('prob_tp_first', descending=True), reverse=True)
    elif objective_val == 'prob_resolve':
        res_list.sort(key=_metric('prob_resolve', descending=True), reverse=True)
    elif objective_val == 'profit_factor':
        res_list.sort(key=_metric('profit_factor', descending=True), reverse=True)
    elif objective_val == 'min_loss_prob':
        res_list.sort(key=_metric('prob_loss', descending=False))
    elif objective_val == 'utility':
        res_list.sort(key=_metric('utility', descending=True), reverse=True)
    else:
        res_list.sort(key=_metric('ev', descending=True), reverse=True)


def _annotate_candidate_metrics(row: Optional[Dict[str, Any]], cost_per_trade: float = 0.0) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return row

    rr = _safe_float(row.get("rr"))
    prob_win = _safe_float(row.get("prob_win"))
    prob_loss = _safe_float(row.get("prob_loss"))
    prob_tp_first = _safe_float(row.get("prob_tp_first"))
    prob_sl_first = _safe_float(row.get("prob_sl_first"))
    prob_no_hit = _safe_float(row.get("prob_no_hit"))
    prob_resolve = _safe_float(row.get("prob_resolve"))
    tp = _safe_float(row.get("tp"))
    sl = _safe_float(row.get("sl"))
    cost = max(0.0, float(cost_per_trade))
    effective_prob_win = prob_tp_first if prob_tp_first is not None else prob_win
    effective_prob_loss = prob_sl_first if prob_sl_first is not None else prob_loss

    if prob_resolve is None and prob_no_hit is not None:
        prob_resolve = float(max(0.0, min(1.0, 1.0 - prob_no_hit)))
        row["prob_resolve"] = prob_resolve
    if prob_no_hit is None and prob_resolve is not None:
        prob_no_hit = float(max(0.0, min(1.0, 1.0 - prob_resolve)))
        row["prob_no_hit"] = prob_no_hit

    breakeven_win_rate: Optional[float] = None
    if rr is not None and rr > 0.0:
        breakeven_win_rate = float(1.0 / (1.0 + rr))
        row["breakeven_win_rate"] = breakeven_win_rate

    breakeven_win_rate_net: Optional[float] = None
    if cost > 0.0 and tp is not None and sl is not None:
        net_reward = tp - cost
        net_risk = sl + cost
        if net_reward > 0.0 and net_risk > 0.0:
            breakeven_win_rate_net = float(net_risk / (net_reward + net_risk))
            row["breakeven_win_rate_net"] = breakeven_win_rate_net
        elif net_reward <= 0.0:
            breakeven_win_rate_net = 1.0
            row["breakeven_win_rate_net"] = breakeven_win_rate_net

    edge_vs_breakeven: Optional[float] = None
    effective_breakeven = breakeven_win_rate_net if breakeven_win_rate_net is not None else breakeven_win_rate
    if effective_prob_win is not None and effective_breakeven is not None:
        edge_vs_breakeven = float(effective_prob_win - effective_breakeven)
        row["edge_vs_breakeven"] = edge_vs_breakeven

    phantom_profit_risk = bool(
        effective_prob_loss is not None
        and prob_no_hit is not None
        and edge_vs_breakeven is not None
        and effective_prob_loss < PHANTOM_PROFIT_LOSS_THRESHOLD
        and prob_no_hit > PHANTOM_PROFIT_NO_HIT_THRESHOLD
        and edge_vs_breakeven < 0.0
    )
    row["phantom_profit_risk"] = phantom_profit_risk

    prob_win_ci = row.get("prob_win_ci95")
    low_confidence = False
    if isinstance(prob_win_ci, dict):
        ci_lo = _safe_float(prob_win_ci.get("low"))
        ci_hi = _safe_float(prob_win_ci.get("high"))
        if ci_lo is not None and ci_hi is not None:
            ci_width = ci_hi - ci_lo
            low_confidence = ci_width > LOW_CONFIDENCE_CI_THRESHOLD
            row["prob_win_ci_width"] = float(ci_width)
    row["low_confidence"] = low_confidence

    return row


def _candidate_is_viable(row: Optional[Dict[str, Any]], cost_per_trade: float = 0.0) -> bool:
    if not isinstance(row, dict):
        return False
    _annotate_candidate_metrics(row, cost_per_trade=cost_per_trade)
    ev_value = _safe_float(row.get("ev"))
    if ev_value is None or ev_value < 0.0:
        return False
    if bool(row.get("phantom_profit_risk")):
        return False
    return True


def _candidate_status_reason(row: Optional[Dict[str, Any]], cost_per_trade: float = 0.0) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    _annotate_candidate_metrics(row, cost_per_trade=cost_per_trade)
    ev_value = _safe_float(row.get("ev"))
    if ev_value is None:
        return "Selected candidate is missing a finite EV estimate."
    if ev_value < 0.0:
        return "Selected candidate has negative EV."
    if bool(row.get("phantom_profit_risk")):
        return (
            "Selected candidate relies on unresolved paths and a near-zero loss rate "
            "to appear profitable."
        )
    return None


def _build_selection_diagnostics(row: Optional[Dict[str, Any]], cost_per_trade: float = 0.0) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    _annotate_candidate_metrics(row, cost_per_trade=cost_per_trade)

    warnings_out: List[str] = []
    ev_edge_conflict = False
    ev_edge_conflict_reason: Optional[str] = None
    caution: Optional[str] = None

    best_ev = _safe_float(row.get("ev"))
    if best_ev is not None and best_ev < 0.0:
        warnings_out.append(
            "Best candidate has negative EV; objective preference may not align with profitability."
        )

    best_kelly = _safe_float(row.get("kelly"))
    if best_kelly is not None and best_kelly < 0.0:
        warnings_out.append(
            "Best candidate has negative Kelly fraction; sizing should be zero or very conservative."
        )

    best_edge = _safe_float(row.get("edge"))
    win_rate = _safe_float(row.get("prob_tp_first"))
    if win_rate is None:
        win_rate = _safe_float(row.get("prob_win"))
    if best_edge is not None and best_edge < 0.0:
        if win_rate is not None:
            warnings_out.append(
                "Best candidate has negative edge "
                f"({best_edge:.3f}) with win rate {win_rate:.1%}; "
                "positive EV depends on reward/risk skew."
            )
        else:
            warnings_out.append(
                f"Best candidate has negative edge ({best_edge:.3f}); "
                "positive EV may depend on reward/risk skew."
            )

    breakeven_win_rate = _safe_float(row.get("breakeven_win_rate_net"))
    if breakeven_win_rate is None:
        breakeven_win_rate = _safe_float(row.get("breakeven_win_rate"))
    edge_vs_breakeven = _safe_float(row.get("edge_vs_breakeven"))
    if edge_vs_breakeven is not None and edge_vs_breakeven < 0.0 and breakeven_win_rate is not None:
        if win_rate is not None:
            warnings_out.append(
                f"Win rate {win_rate:.1%} is below the break-even threshold "
                f"{breakeven_win_rate:.1%}; unresolved paths or payoff skew are carrying EV."
            )
        else:
            warnings_out.append(
                f"Break-even win rate is {breakeven_win_rate:.1%}; "
                "selected candidate is below that threshold."
            )

    prob_no_hit = _safe_float(row.get("prob_no_hit"))
    if prob_no_hit is not None and prob_no_hit > PHANTOM_PROFIT_NO_HIT_THRESHOLD:
        warnings_out.append(
            f"More than {PHANTOM_PROFIT_NO_HIT_THRESHOLD:.0%} of simulations did not reach either barrier "
            f"before the horizon ({prob_no_hit:.1%} no-hit)."
        )

    if best_ev is not None and edge_vs_breakeven is not None:
        if (best_ev > 0.0 and edge_vs_breakeven < 0.0) or (best_ev < 0.0 and edge_vs_breakeven > 0.0):
            ev_edge_conflict = True
            ev_edge_conflict_reason = "ev and edge_vs_breakeven have opposite signs"
            warnings_out.append(
                "CAUTION: EV and break-even-adjusted edge disagree; inspect win probability, "
                "reward/risk, and unresolved-path share before trading."
            )

    if bool(row.get("phantom_profit_risk")):
        ev_edge_conflict = True
        ev_edge_conflict_reason = (
            "positive EV is dominated by unresolved paths with near-zero loss probability"
        )
        warnings_out.append(
            "CAUTION: SL barrier was not reached in most simulations while the observed loss rate "
            "was near zero. Positive EV may reflect horizon boundary effects, not trading edge."
        )
        caution = (
            "Selected candidate is dominated by unresolved paths; inspect `prob_no_hit`, "
            "`prob_resolve`, and `edge_vs_breakeven` before trading."
        )
    elif ev_edge_conflict:
        caution = (
            "EV and break-even-adjusted edge conflict for the selected candidate; inspect win "
            "probability and break-even threshold before trading."
        )

    deduped_warnings: List[str] = []
    seen: Set[str] = set()
    for message in warnings_out:
        msg = str(message).strip()
        if not msg or msg in seen:
            continue
        seen.add(msg)
        deduped_warnings.append(msg)

    out: Dict[str, Any] = {}
    if deduped_warnings:
        out["selection_warnings"] = deduped_warnings
    if ev_edge_conflict:
        out["ev_edge_conflict"] = True
        if ev_edge_conflict_reason:
            out["ev_edge_conflict_reason"] = ev_edge_conflict_reason
    if caution:
        out["caution"] = caution
    if bool(row.get("low_confidence")):
        ci_w = _safe_float(row.get("prob_win_ci_width"))
        ci_msg = f" (CI width: {ci_w:.1%})" if ci_w is not None else ""
        out["confidence_warning"] = (
            f"Win probability estimate has wide confidence interval{ci_msg}. "
            "Increase n_sims or use search_profile='long' for tighter estimates."
        )
        out["low_confidence"] = True
        prob_win_val = _safe_float(row.get("prob_win"))
        if prob_win_val is not None:
            # n needed for CI width ≤ 0.05: Wilson CI ≈ 2*z*sqrt(p*(1-p)/n) ≤ 0.05
            target_width = 0.05
            z = 1.96
            pq = prob_win_val * (1.0 - prob_win_val)
            min_sims = int(math.ceil((2.0 * z) ** 2 * pq / (target_width ** 2)))
            out["min_sims_recommended"] = max(min_sims, 2000)
    return out


def _build_actionability_payload(
    *,
    status: str,
    status_reason: Optional[str] = None,
    row: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    warning: Optional[str] = None,
    ensemble_degraded: bool = False,
) -> Dict[str, Any]:
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    flags: List[str] = []

    if status == "no_candidates":
        flags.append("status_no_candidates")
    elif status == "non_viable":
        flags.append("status_non_viable")

    if isinstance(row, dict) and bool(row.get("phantom_profit_risk")):
        flags.append("phantom_profit_risk")
    if bool(diag.get("ev_edge_conflict")):
        flags.append("ev_edge_conflict")
    if bool(diag.get("low_confidence")):
        flags.append("low_confidence")
    if diag.get("selection_warnings"):
        flags.append("selection_warnings")
    if warning:
        flags.append("warning")
    if ensemble_degraded:
        flags.append("ensemble_degraded")

    deduped_flags: List[str] = []
    seen: Set[str] = set()
    for flag in flags:
        key = str(flag).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_flags.append(key)

    actionability = "actionable"
    trade_gate_passed = True
    actionability_reason = "No blocking diagnostics detected."

    if status != "ok" or "phantom_profit_risk" in seen or "ev_edge_conflict" in seen:
        actionability = "blocked"
        trade_gate_passed = False
        if status != "ok":
            actionability_reason = (
                str(status_reason).strip()
                if status_reason else f"Optimizer status is {status!r}; do not trade this setup."
            )
        elif "phantom_profit_risk" in seen:
            actionability_reason = (
                "Positive EV is dominated by unresolved paths with near-zero observed loss; "
                "skip this setup."
            )
        else:
            actionability_reason = (
                "EV and break-even-adjusted edge conflict; manual review is required before trading."
            )
    elif {
        "selection_warnings",
        "low_confidence",
        "warning",
        "ensemble_degraded",
    }.intersection(seen):
        actionability = "review"
        trade_gate_passed = False
        if warning:
            actionability_reason = str(warning).strip()
        elif diag.get("confidence_warning"):
            actionability_reason = str(diag["confidence_warning"]).strip()
        elif diag.get("selection_warnings"):
            warnings_list = diag.get("selection_warnings")
            if isinstance(warnings_list, list) and warnings_list:
                actionability_reason = str(warnings_list[0]).strip()
            else:
                actionability_reason = "Selection diagnostics require manual review before trading."
        else:
            actionability_reason = "Selection diagnostics require manual review before trading."

    return {
        "actionability": actionability,
        "trade_gate_passed": trade_gate_passed,
        "actionability_reason": actionability_reason,
        "actionability_flags": deduped_flags,
    }


def _is_crypto_symbol(symbol: str) -> bool:
    sym = str(symbol or "").upper()
    crypto_tokens = {
        "BTC", "ETH", "XRP", "LTC", "SOL", "ADA", "DOGE", "BNB", "DOT",
        "AVAX", "LINK", "TRX", "MATIC", "NEAR", "ATOM", "FIL", "UNI",
    }
    return any(tok in sym for tok in crypto_tokens)


def _auto_barrier_method(
    symbol: str,
    timeframe: str,
    prices: np.ndarray,
    horizon: Optional[int] = None,
) -> Tuple[str, str]:
    """Heuristically choose a barrier simulation method.

    The thresholds in this selector are pragmatic heuristics (history length,
    volatility clustering, tails/jumps) intended for robust defaults, not a
    universal optimum.
    """
    tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0) or 0
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    horizon_val: Optional[int] = None
    try:
        horizon_val = int(horizon) if horizon is not None else None
    except Exception:
        horizon_val = None
    prefer_bridge = horizon_val is not None and horizon_val <= 12
    if prices.size < 10:
        if prefer_bridge:
            return "mc_gbm_bb", "auto: insufficient history; gbm_bb (short horizon)"
        return "mc_gbm", "auto: insufficient history; gbm baseline"

    rets = _log_returns_from_prices(prices)
    rets = rets[np.isfinite(rets)]
    if rets.size < 5:
        if prefer_bridge:
            return "mc_gbm_bb", "auto: insufficient returns; gbm_bb (short horizon)"
        return "mc_gbm", "auto: insufficient returns; gbm baseline"
    if rets.size < 30:
        if prefer_bridge:
            return "mc_gbm_bb", "auto: limited history; gbm_bb (short horizon)"
        return "mc_gbm", "auto: limited history; gbm baseline"

    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1)) + 1e-12
    z = (rets - mu) / sigma
    kurt = float(np.mean(z ** 4) - 3.0) if z.size > 0 else 0.0
    jump_ratio = float(np.max(np.abs(rets - mu)) / sigma) if rets.size > 0 else 0.0
    skew = float(np.mean(z ** 3)) if z.size > 0 else 0.0

    r2 = (rets - mu) ** 2
    if r2.size >= 6:
        r2a = r2[1:]
        r2b = r2[:-1]
        denom = float(np.std(r2a) * np.std(r2b)) + 1e-12
        if denom > 0:
            try:
                vol_corr = float(np.corrcoef(r2a, r2b)[0, 1])
                if not np.isfinite(vol_corr):
                    vol_corr = 0.0
            except Exception:
                vol_corr = 0.0
        else:
            vol_corr = 0.0
    else:
        vol_corr = 0.0

    vol_ratio = 1.0
    if rets.size >= 60:
        short_n = min(50, rets.size)
        long_n = min(200, rets.size)
        short_std = float(np.std(rets[-short_n:], ddof=1)) + 1e-12
        long_std = float(np.std(rets[-long_n:], ddof=1)) + 1e-12
        vol_ratio = short_std / long_std

    if _is_crypto_symbol(symbol) or (tf_secs and tf_secs <= 900):
        if kurt > 2.0 or jump_ratio > 4.0:
            return "jump_diffusion", "auto: crypto/short-tf with jumpy tails"

    if kurt > 3.5 or jump_ratio > 5.0:
        return "jump_diffusion", "auto: heavy tails/jump risk"

    if vol_ratio >= 1.6 or vol_ratio <= 0.65:
        if rets.size >= 220:
            return "hmm_mc", "auto: regime shift (volatility change)"

    if vol_corr > 0.3 and r2.size >= 150:
        try:
            import arch  # noqa: F401
            return "garch", "auto: strong volatility clustering (garch)"
        except Exception:
            return "heston", "auto: strong volatility clustering (heston fallback)"

    if vol_corr > 0.15:
        return "heston", "auto: volatility clustering"

    if rets.size >= 400 and abs(skew) >= 0.7 and jump_ratio < 4.5:
        return "bootstrap", "auto: non-normal returns; bootstrap"

    if prefer_bridge:
        return "mc_gbm_bb", "auto: mild tails; gbm with bridge correction"
    return "mc_gbm", "auto: mild tails; gbm baseline"


def _brownian_bridge_hits(
    log_paths: np.ndarray,
    barrier_log: float,
    sigma: float,
    *,
    direction: Literal["up", "down"],
    uniform: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros((log_paths.shape[0], log_paths.shape[1] - 1), dtype=bool)
    x0 = log_paths[:, :-1]
    x1 = log_paths[:, 1:]
    s2 = float(sigma * sigma)
    if direction == "up":
        valid = (x0 < barrier_log) & (x1 < barrier_log)
        dist0 = barrier_log - x0
        dist1 = barrier_log - x1
    else:
        valid = (x0 > barrier_log) & (x1 > barrier_log)
        dist0 = x0 - barrier_log
        dist1 = x1 - barrier_log
    expo = -2.0 * dist0 * dist1 / s2
    p = np.exp(np.clip(expo, -200.0, 50.0))
    hits = (uniform < p) & valid
    return hits


def _get_live_reference_price(symbol: str, direction: str) -> Tuple[Optional[float], Optional[str]]:
    """Best-effort live tick reference price; returns (price, source) or (None, None)."""
    try:
        import MetaTrader5 as _mt5  # type: ignore
    except Exception:
        return None, None

    try:
        tick = _mt5.symbol_info_tick(symbol)
    except Exception:
        tick = None
    if tick is None:
        return None, None

    def _valid_price(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if not np.isfinite(out) or out <= 0.0:
            return None
        return out

    bid = _valid_price(getattr(tick, "bid", None))
    ask = _valid_price(getattr(tick, "ask", None))
    last = _valid_price(getattr(tick, "last", None))

    direction_norm, _ = normalize_trade_direction(direction)
    if direction_norm == "long":
        if ask is not None:
            return ask, "live_tick_ask"
        if bid is not None:
            return bid, "live_tick_bid_fallback"
    else:
        if bid is not None:
            return bid, "live_tick_bid"
        if ask is not None:
            return ask, "live_tick_ask_fallback"

    if bid is not None and ask is not None:
        return 0.5 * (bid + ask), "live_tick_mid"
    if last is not None:
        return last, "live_tick_last"
    if bid is not None:
        return bid, "live_tick_bid_only"
    if ask is not None:
        return ask, "live_tick_ask_only"
    return None, None


def _resolve_reference_prices(
    close_values: Any,
    *,
    symbol: str,
    direction: str,
    use_live_price: bool = True,
    live_price_getter: Any = None,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]]:
    closes = np.asarray(close_values, dtype=float)
    if closes.size == 0:
        return None, None, None, None, "Latest close is non-finite; refresh history or enable a live reference price."

    trailing_close = _safe_float(closes[-1])
    finite_closes = closes[np.isfinite(closes)]
    history_anchor = _safe_float(finite_closes[-1]) if finite_closes.size else None
    if history_anchor is None or history_anchor <= 0.0:
        return None, None, None, None, "Latest close is non-finite; refresh history or enable a live reference price."

    if use_live_price:
        getter = live_price_getter if callable(live_price_getter) else _get_live_reference_price
        live_price, live_source = getter(symbol, direction)
        if live_price is not None and np.isfinite(live_price) and float(live_price) > 0.0:
            warning = None
            if trailing_close is None or trailing_close <= 0.0:
                warning = "Latest close is non-finite; using the last finite historical close as the simulation anchor."
            return float(history_anchor), float(live_price), str(live_source or "live_tick"), warning, None

    if trailing_close is None or trailing_close <= 0.0:
        return None, None, None, None, "Latest close is non-finite; refresh history or enable a live reference price."
    return float(trailing_close), float(trailing_close), "close", None, None


