from typing import Any, Dict, Optional, List, Literal, Tuple, Set
import math
import numpy as np
from ..core.schema import TimeframeLiteral, DenoiseSpec
from ..core.constants import TIMEFRAME_SECONDS
from .common import fetch_history as _fetch_history
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from ..utils.barriers import get_pip_size as _get_pip_size, resolve_barrier_prices as _resolve_barrier_prices
from .monte_carlo import (
    simulate_gbm_mc as _simulate_gbm_mc, 
    simulate_hmm_mc as _simulate_hmm_mc, 
    simulate_garch_mc as _simulate_garch_mc,
    simulate_bootstrap_mc as _simulate_bootstrap_mc,
    simulate_heston_mc as _simulate_heston_mc,
    simulate_jump_diffusion_mc as _simulate_jump_diffusion_mc,
    gbm_single_barrier_upcross_prob as _gbm_upcross_prob
)

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

    rets = np.diff(np.log(np.clip(prices, 1e-12, None)))
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

def forecast_barrier_hit_probabilities(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','mc_gbm_bb','hmm_mc','garch','bootstrap','heston','jump_diffusion','auto'] = 'hmm_mc',
    direction: Literal['long','short'] = 'long',
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Monte Carlo barrier analysis: TP/SL hit probabilities within `horizon` bars.

    Notes:
    - Barriers are provided via absolute prices (tp_abs/sl_abs), percentages
      (tp_pct/sl_pct), or ticks (tp_pips/sl_pips; uses `trade_tick_size`).
    - In discrete time, TP and SL can be hit in the same bar. Those ties are
      split 50/50 into `prob_tp_first` and `prob_sl_first`.
    """
    try:
        if timeframe not in TIMEFRAME_SECONDS:
            return {"error": f"Invalid timeframe: {timeframe}"}
        p = _parse_kv_or_json(params)
        # Fetch enough history for calibration
        need = int(max(300, horizon + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        # Current price baseline
        last_price = float(df['close'].astype(float).iloc[-1])
        pip_size = _get_pip_size(symbol)

        # Compute absolute TP/SL prices with explicit trade direction
        dir_long = str(direction).lower() == 'long'
        tp_price, sl_price = _resolve_barrier_prices(
            price=last_price,
            direction=direction,
            tp_abs=tp_abs,
            sl_abs=sl_abs,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            pip_size=pip_size,
            adjust_inverted=True,
        )

        if tp_price is None or sl_price is None:
            return {"error": "Provide barriers via tp_abs/sl_abs or tp_pct/sl_pct or tp_pips/sl_pips"}

        # Build input series (denoise optional)
        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import _apply_denoise as _apply_denoise_util
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        prices = df[base_col].astype(float).to_numpy()

        # Simulate paths
        sims = int(p.get('n_sims', p.get('sims', 2000)) or 2000)
        seed = int(p.get('seed', 42) or 42)
        method_key = str(method).lower().strip()
        method_requested = method_key
        auto_reason = None
        if method_key == 'auto':
            method_key, auto_reason = _auto_barrier_method(
                symbol, timeframe, prices, horizon=int(horizon)
            )
        bb_enabled = method_key == 'mc_gbm_bb'
        
        if method_key in ('mc_gbm', 'mc_gbm_bb'):
            sim = _simulate_gbm_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed))
        elif method_key == 'hmm_mc':
            n_states = int(p.get('n_states', 2) or 2)
            sim = _simulate_hmm_mc(prices, horizon=int(horizon), n_states=int(n_states), n_sims=int(sims), seed=int(seed))
        elif method_key == 'garch':
            p_order = int(p.get('p', 1))
            q_order = int(p.get('q', 1))
            sim = _simulate_garch_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed), p_order=p_order, q_order=q_order)
        elif method_key == 'bootstrap':
            bs = p.get('block_size')
            if bs: bs = int(bs)
            sim = _simulate_bootstrap_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed), block_size=bs)
        elif method_key == 'heston':
            sim = _simulate_heston_mc(
                prices,
                horizon=int(horizon),
                n_sims=int(sims),
                seed=int(seed),
                kappa=p.get('kappa'),
                theta=p.get('theta'),
                xi=p.get('xi'),
                rho=p.get('rho'),
                v0=p.get('v0'),
            )
        elif method_key == 'jump_diffusion':
            sim = _simulate_jump_diffusion_mc(
                prices,
                horizon=int(horizon),
                n_sims=int(sims),
                seed=int(seed),
                jump_lambda=p.get('jump_lambda', p.get('lambda')),
                jump_mu=p.get('jump_mu'),
                jump_sigma=p.get('jump_sigma'),
                jump_threshold=float(p.get('jump_threshold', 3.0)),
            )
        else:
            return {"error": f"Unsupported method: {method}. Use 'mc_gbm', 'mc_gbm_bb', 'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', or 'auto'"}

        price_paths = np.asarray(sim['price_paths'], dtype=float)
        S, H = price_paths.shape

        bb_sigma = 0.0
        bb_uniform_tp = None
        bb_uniform_sl = None
        bb_log_paths = None
        if bb_enabled:
            rets = np.diff(np.log(np.clip(prices, 1e-12, None)))
            rets = rets[np.isfinite(rets)]
            bb_sigma = float(np.std(rets, ddof=1)) if rets.size else 0.0
            if not np.isfinite(bb_sigma) or bb_sigma <= 0:
                bb_enabled = False
            else:
                log_paths = np.log(np.clip(price_paths, 1e-12, None))
                log_s0 = float(np.log(max(last_price, 1e-12)))
                bb_log_paths = np.concatenate([np.full((S, 1), log_s0), log_paths], axis=1)
                rng_bb = np.random.RandomState(int(seed) + 7)
                bb_uniform_tp = rng_bb.rand(S, H)
                bb_uniform_sl = rng_bb.rand(S, H)
        
        # Vectorized hit detection
        # hits_tp/sl: boolean (S, H)
        if dir_long:
            hits_tp = (price_paths >= tp_price)
            hits_sl = (price_paths <= sl_price)
        else:
            hits_tp = (price_paths <= tp_price)
            hits_sl = (price_paths >= sl_price)
        if bb_enabled and bb_log_paths is not None and bb_uniform_tp is not None and bb_uniform_sl is not None:
            tp_dir = "up" if dir_long else "down"
            sl_dir = "down" if dir_long else "up"
            tp_bridge = _brownian_bridge_hits(bb_log_paths, float(np.log(tp_price)), bb_sigma, direction=tp_dir, uniform=bb_uniform_tp)
            sl_bridge = _brownian_bridge_hits(bb_log_paths, float(np.log(sl_price)), bb_sigma, direction=sl_dir, uniform=bb_uniform_sl)
            hits_tp = hits_tp | tp_bridge
            hits_sl = hits_sl | sl_bridge

        # Find first hit index (argmax returns 0 if none found, check any)
        idx_tp = np.argmax(hits_tp, axis=1)
        idx_sl = np.argmax(hits_sl, axis=1)
        
        any_tp = np.any(hits_tp, axis=1)
        any_sl = np.any(hits_sl, axis=1)
        
        # Set index to H (beyond horizon) if no hit
        idx_tp_val = np.where(any_tp, idx_tp, H)
        idx_sl_val = np.where(any_sl, idx_sl, H)
        
        # Determine outcomes
        # TP Win: TP hit detected AND (SL not hit OR TP hit before SL)
        tp_wins = (idx_tp_val < idx_sl_val)
        # SL Win: SL hit detected AND (TP not hit OR SL hit before TP)
        sl_wins = (idx_sl_val < idx_tp_val)
        # Tie: Both hit at same index (rare but possible in discrete time)
        ties = (idx_tp_val == idx_sl_val) & (idx_tp_val < H)
        # No hit: Both H
        no_hits = (idx_tp_val == H) & (idx_sl_val == H)

        tp_first = np.sum(tp_wins)
        sl_first = np.sum(sl_wins)
        both_tie = np.sum(ties)
        no_hit = np.sum(no_hits)

        # Collect hit times (1-based) for stats
        # TP stats include strict wins and ties
        t_hit_tp = (idx_tp_val[tp_wins | ties] + 1).tolist()
        t_hit_sl = (idx_sl_val[sl_wins | ties] + 1).tolist()

        # Cumulative hit curves (hit at or before t)
        def _compute_cum_curve(indices, valid_mask, length):
            valid_indices = indices[valid_mask]
            if valid_indices.size == 0:
                return np.zeros(length, dtype=float)
            # bincount counts occurrences of each index
            counts = np.bincount(valid_indices, minlength=length)
            if counts.size > length:
                counts = counts[:length]
            return np.cumsum(counts).astype(float)

        tp_any_by_t = _compute_cum_curve(idx_tp, any_tp, H)
        sl_any_by_t = _compute_cum_curve(idx_sl, any_sl, H)

        S_f = float(S)
        prob_tp_first = (tp_first + 0.5 * both_tie) / S_f
        prob_sl_first = (sl_first + 0.5 * both_tie) / S_f
        prob_no_hit = no_hit / S_f
        tp_any_curve = (tp_any_by_t / S_f).tolist()
        sl_any_curve = (sl_any_by_t / S_f).tolist()

        def _stats(arr: list[int]) -> Dict[str, float]:
            if not arr:
                return {"mean": float('nan'), "median": float('nan')}
            a = np.asarray(arr, dtype=float)
            return {"mean": float(a.mean()), "median": float(np.median(a))}

        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0)
        tp_stats = _stats(t_hit_tp)
        sl_stats = _stats(t_hit_sl)
        def _finite_or_none(x: float) -> Optional[float]:
            try:
                return float(x) if np.isfinite(x) else None
            except Exception:
                return None
        # Directional interpretation:
        # - For long: TP is above last_price, SL is below; prob_tp_first is long win probability.
        # - For short: TP is below last_price, SL is above; prob_tp_first is short win probability.
        edge = float(prob_tp_first - prob_sl_first)
        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_key,
            "horizon": int(horizon),
            "direction": direction,
            "last_price": last_price,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "prob_tp_first": float(prob_tp_first),
            "prob_sl_first": float(prob_sl_first),
            "prob_no_hit": float(prob_no_hit),
            "edge": float(edge),
            "tp_hit_prob_by_t": [float(v) for v in tp_any_curve],
            "sl_hit_prob_by_t": [float(v) for v in sl_any_curve],
            "time_to_tp_bars": tp_stats,
            "time_to_sl_bars": sl_stats,
            "time_to_tp_seconds": {k: _finite_or_none(v * tf_secs) for k, v in tp_stats.items()},
            "time_to_sl_seconds": {k: _finite_or_none(v * tf_secs) for k, v in sl_stats.items()},
            "params_used": {k: p[k] for k in p if k in {"n_sims", "seed", "n_states", "p", "q", "block_size", "kappa", "theta", "xi", "rho", "v0", "jump_lambda", "jump_mu", "jump_sigma", "jump_threshold", "lambda"}},
        }
        if method_requested != method_key:
            out["method_requested"] = method_requested
            out["method_used"] = method_key
            if auto_reason:
                out["auto_reason"] = auto_reason
        if bb_enabled:
            out["bridge_correction"] = True
        if 'model_summary' in sim:
            out['model_summary'] = str(sim['model_summary'])
            
        return out
    except Exception as e:
        return {"error": f"Error computing barrier probabilities: {str(e)}"}

def forecast_barrier_closed_form(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    direction: Literal['long','short'] = 'long',
    barrier: float = 0.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Closed-form single-barrier hit probability for GBM within horizon.

    Direction semantics:
    - "long": probability of reaching an upper barrier (price >= barrier).
    - "short": probability of reaching a lower barrier (price <= barrier).
    """
    try:
        need = int(max(400, horizon + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history"}
        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import _apply_denoise as _apply_denoise_util
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        prices = np.asarray(df[base_col].astype(float).to_numpy(), dtype=float)
        prices = prices[np.isfinite(prices)]
        if prices.size < 5:
            return {"error": "Insufficient prices"}
        s0 = float(prices[-1])
        if barrier <= 0:
            return {"error": "Provide a positive barrier price"}
        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}
        T = float(tf_secs * int(horizon)) / (365.0 * 24.0 * 3600.0)
        if mu is None or sigma is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.diff(np.log(np.maximum(prices, 1e-12)))
            r = r[np.isfinite(r)]
            if r.size < 5:
                return {"error": "Insufficient returns for calibration"}
            mu_hat = float(np.mean(r)) * (365.0 * 24.0 * 3600.0 / tf_secs)
            sigma_hat = float(np.std(r, ddof=1)) * (365.0 * 24.0 * 3600.0 / tf_secs) ** 0.5
            if mu is None:
                mu = mu_hat
            if sigma is None:
                sigma = sigma_hat
        log_drift = float(mu)
        sigma_val = float(sigma)
        if sigma_val <= 0:
            return {"error": "Sigma must be positive"}
        sigma_sq = sigma_val * sigma_val
        gbm_drift = log_drift + 0.5 * sigma_sq
        dir_lower = str(direction).lower()
        direction_norm = 'short' if dir_lower in {'short', 'down'} else 'long'
        if direction_norm == 'short':
            s0_inv = 1.0 / s0
            b_inv = 1.0 / float(barrier)
            inv_drift = sigma_sq - gbm_drift
            prob = _gbm_upcross_prob(s0_inv, b_inv, float(inv_drift), sigma_val, float(T))
        else:
            prob = _gbm_upcross_prob(s0, float(barrier), float(gbm_drift), sigma_val, float(T))
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "direction": direction_norm,
            "last_price": s0,
            "barrier": float(barrier),
            "mu_annual": float(gbm_drift),
            "log_drift_annual": float(log_drift),
            "sigma_annual": sigma_val,
            "prob_hit": float(prob),
        }
    except Exception as e:
        return {"error": f"Error computing closed-form barrier probability: {str(e)}"}

def forecast_barrier_optimize(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','mc_gbm_bb','hmm_mc','garch','bootstrap','heston','jump_diffusion','auto'] = 'hmm_mc',
    direction: Literal['long','short'] = 'long',
    mode: Literal['pct','pips'] = 'pct',
    tp_min: float = 0.25,
    tp_max: float = 1.5,
    tp_steps: int = 7,
    sl_min: float = 0.25,
    sl_max: float = 2.5,
    sl_steps: int = 9,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    objective: Literal[
        'edge',
        'prob_tp_first',
        'prob_resolve',
        'kelly',
        'kelly_cond',
        'ev',
        'ev_cond',
        'ev_per_bar',
        'profit_factor',
        'min_loss_prob',
        'utility',
    ] = 'edge',
    return_grid: bool = True,
    top_k: Optional[int] = None,
    output: Literal['full','summary'] = 'full',
    grid_style: Literal['fixed','volatility','ratio','preset'] = 'fixed',
    preset: Optional[str] = None,
    vol_window: int = 250,
    vol_min_mult: float = 0.5,
    vol_max_mult: float = 4.0,
    vol_steps: int = 7,
    vol_sl_extra: float = 1.8,
    vol_floor_pct: float = 0.15,
    vol_floor_pips: float = 8.0,
    ratio_min: float = 0.5,
    ratio_max: float = 4.0,
    ratio_steps: int = 8,
    refine: bool = False,
    refine_radius: float = 0.3,
    refine_steps: int = 5,
    min_prob_win: Optional[float] = None,
    max_prob_no_hit: Optional[float] = None,
    max_median_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers over a grid of candidate levels.

    Unit conventions:
    - mode="pct": tp/sl are percentage *points* (e.g., tp=0.5 means +0.5%).
    - mode="pips": tp/sl are ticks (trade_tick_size units).

    Grid styles:
    - fixed/preset/volatility generate tp/sl directly in the selected `mode`.
    - ratio treats `ratio_min/max` as reward/risk = tp/sl (TP distance divided
      by SL distance), with SL sampled from `sl_min/max`.

    Metrics:
    - ev/ev_cond/ev_per_bar are reported in the same units as tp/sl (pct points
      or ticks). `ev_per_bar` divides by mean resolution time (bars).
    """
    try:
        if timeframe not in TIMEFRAME_SECONDS:
            return {"error": f"Invalid timeframe: {timeframe}"}

        params_dict = _parse_kv_or_json(params)
        mode_val = str(mode).lower()
        objective_val = str(objective).lower()
        valid_objectives = {
            'edge',
            'prob_tp_first',
            'prob_resolve',
            'kelly',
            'kelly_cond',
            'ev',
            'ev_cond',
            'ev_per_bar',
            'profit_factor',
            'min_loss_prob',
            'utility',
        }
        if objective_val not in valid_objectives:
            objective_val = 'edge'

        grid_style_val = str(params_dict.get('grid_style', grid_style)).lower()
        if grid_style_val not in {'fixed', 'volatility', 'ratio', 'preset'}:
            grid_style_val = 'fixed'
        preset_candidate = params_dict.get('grid_preset', params_dict.get('preset', preset))
        preset_val = str(preset_candidate).lower() if isinstance(preset_candidate, str) and preset_candidate else None

        refine_flag = bool(params_dict.get('refine', refine))
        refine_radius_val = max(0.0, float(params_dict.get('refine_radius', refine_radius)))
        refine_steps_val = max(2, int(params_dict.get('refine_steps', refine_steps)))

        ratio_min_val = float(params_dict.get('ratio_min', ratio_min))
        ratio_max_val = float(params_dict.get('ratio_max', ratio_max))
        ratio_steps_val = max(2, int(params_dict.get('ratio_steps', ratio_steps)))
        if ratio_min_val <= 0:
            ratio_min_val = ratio_min
        if ratio_max_val < ratio_min_val:
            ratio_max_val = ratio_min_val

        vol_window_val = int(params_dict.get('vol_window', vol_window))
        vol_min_mult_val = float(params_dict.get('vol_min_mult', vol_min_mult))
        vol_max_mult_val = float(params_dict.get('vol_max_mult', vol_max_mult))
        vol_steps_val = max(2, int(params_dict.get('vol_steps', vol_steps)))
        vol_sl_extra_val = float(params_dict.get('vol_sl_extra', vol_sl_extra))
        vol_sl_multiplier_val = float(params_dict.get('vol_sl_multiplier', vol_sl_extra_val))
        vol_sl_steps_val = max(vol_steps_val, int(params_dict.get('vol_sl_steps', vol_steps_val + 2)))
        vol_floor_pct_val = float(params_dict.get('vol_floor_pct', vol_floor_pct))
        vol_floor_pips_val = float(params_dict.get('vol_floor_pips', vol_floor_pips))

        # Optional risk/reward filter applied across all grid styles
        rr_min_val = params_dict.get('rr_min')
        rr_max_val = params_dict.get('rr_max')
        try:
            rr_min_val = float(rr_min_val) if rr_min_val is not None else None
        except Exception:
            rr_min_val = None
        try:
            rr_max_val = float(rr_max_val) if rr_max_val is not None else None
        except Exception:
            rr_max_val = None
        if rr_min_val is not None and rr_min_val <= 0:
            rr_min_val = None
        if rr_max_val is not None and rr_max_val <= 0:
            rr_max_val = None

        min_prob_win_val = params_dict.get('min_prob_win', min_prob_win)
        max_prob_no_hit_val = params_dict.get('max_prob_no_hit', max_prob_no_hit)
        max_median_time_val = params_dict.get('max_median_time', max_median_time)
        try:
            min_prob_win_val = float(min_prob_win_val) if min_prob_win_val is not None else None
        except Exception:
            min_prob_win_val = None
        try:
            max_prob_no_hit_val = float(max_prob_no_hit_val) if max_prob_no_hit_val is not None else None
        except Exception:
            max_prob_no_hit_val = None
        try:
            max_median_time_val = float(max_median_time_val) if max_median_time_val is not None else None
        except Exception:
            max_median_time_val = None
        if min_prob_win_val is not None:
            if not np.isfinite(min_prob_win_val):
                min_prob_win_val = None
            else:
                min_prob_win_val = max(0.0, min(1.0, min_prob_win_val))
        if max_prob_no_hit_val is not None:
            if not np.isfinite(max_prob_no_hit_val):
                max_prob_no_hit_val = None
            else:
                max_prob_no_hit_val = max(0.0, min(1.0, max_prob_no_hit_val))
        if max_median_time_val is not None:
            if not np.isfinite(max_median_time_val) or max_median_time_val <= 0:
                max_median_time_val = None

        tp_min_val = float(params_dict.get('tp_min', tp_min))
        tp_max_val = float(params_dict.get('tp_max', tp_max))
        tp_steps_val = max(1, int(params_dict.get('tp_steps', tp_steps)))
        sl_min_val = float(params_dict.get('sl_min', sl_min))
        sl_max_val = float(params_dict.get('sl_max', sl_max))
        sl_steps_val = max(1, int(params_dict.get('sl_steps', sl_steps)))

        need = int(max(300, horizon + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        last_price = float(df['close'].astype(float).iloc[-1])

        pip_size = _get_pip_size(symbol)
        if mode_val == 'pips' and (pip_size is None or pip_size <= 0):
            return {"error": "Tick size unavailable for this symbol; use mode='pct' or provide absolute barriers."}

        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import _apply_denoise as _apply_denoise_util
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        prices = df[base_col].astype(float).to_numpy()

        sims = int(params_dict.get('n_sims', params_dict.get('sims', 4000)) or 4000)
        seed = int(params_dict.get('seed', 42) or 42)
        n_seeds = int(params_dict.get('n_seeds', 1) or 1)
        paths_list: List[np.ndarray] = []
        method_name = str(method).lower().strip()
        method_requested = method_name
        auto_reason = None
        if method_name == 'auto':
            method_name, auto_reason = _auto_barrier_method(
                symbol, timeframe, prices, horizon=int(horizon)
            )
        bb_enabled = method_name == 'mc_gbm_bb'
        
        if method_name in ('mc_gbm', 'mc_gbm_bb'):
            for offset in range(max(1, n_seeds)):
                sim = _simulate_gbm_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed + offset))
                paths_list.append(np.asarray(sim['price_paths'], dtype=float))
        elif method_name == 'hmm_mc':
            n_states = int(params_dict.get('n_states', 2) or 2)
            for offset in range(max(1, n_seeds)):
                sim = _simulate_hmm_mc(prices, horizon=int(horizon), n_states=int(n_states), n_sims=int(sims), seed=int(seed + offset))
                paths_list.append(np.asarray(sim['price_paths'], dtype=float))
        elif method_name == 'garch':
            p_order = int(params_dict.get('p', 1))
            q_order = int(params_dict.get('q', 1))
            for offset in range(max(1, n_seeds)):
                sim = _simulate_garch_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed + offset), p_order=p_order, q_order=q_order)
                paths_list.append(np.asarray(sim['price_paths'], dtype=float))
        elif method_name == 'bootstrap':
            bs = params_dict.get('block_size')
            if bs: bs = int(bs)
            for offset in range(max(1, n_seeds)):
                sim = _simulate_bootstrap_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed + offset), block_size=bs)
                paths_list.append(np.asarray(sim['price_paths'], dtype=float))
        elif method_name == 'heston':
            for offset in range(max(1, n_seeds)):
                sim = _simulate_heston_mc(
                    prices,
                    horizon=int(horizon),
                    n_sims=int(sims),
                    seed=int(seed + offset),
                    kappa=params_dict.get('kappa'),
                    theta=params_dict.get('theta'),
                    xi=params_dict.get('xi'),
                    rho=params_dict.get('rho'),
                    v0=params_dict.get('v0'),
                )
                paths_list.append(np.asarray(sim['price_paths'], dtype=float))
        elif method_name == 'jump_diffusion':
            for offset in range(max(1, n_seeds)):
                sim = _simulate_jump_diffusion_mc(
                    prices,
                    horizon=int(horizon),
                    n_sims=int(sims),
                    seed=int(seed + offset),
                    jump_lambda=params_dict.get('jump_lambda', params_dict.get('lambda')),
                    jump_mu=params_dict.get('jump_mu'),
                    jump_sigma=params_dict.get('jump_sigma'),
                    jump_threshold=float(params_dict.get('jump_threshold', 3.0)),
                )
                paths_list.append(np.asarray(sim['price_paths'], dtype=float))
        else:
            return {"error": f"Unsupported method: {method}. Use 'mc_gbm', 'mc_gbm_bb', 'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', or 'auto'"}

        paths = np.vstack(paths_list) if len(paths_list) > 1 else paths_list[0]
        S, H = paths.shape
        bb_sigma = 0.0
        bb_uniform_tp = None
        bb_uniform_sl = None
        bb_log_paths = None
        if bb_enabled:
            rets = np.diff(np.log(np.clip(prices, 1e-12, None)))
            rets = rets[np.isfinite(rets)]
            bb_sigma = float(np.std(rets, ddof=1)) if rets.size else 0.0
            if not np.isfinite(bb_sigma) or bb_sigma <= 0:
                bb_enabled = False
            else:
                log_paths = np.log(np.clip(paths, 1e-12, None))
                log_s0 = float(np.log(max(last_price, 1e-12)))
                bb_log_paths = np.concatenate([np.full((S, 1), log_s0), log_paths], axis=1)
                rng_bb = np.random.RandomState(int(seed) + 7)
                bb_uniform_tp = rng_bb.rand(S, H)
                bb_uniform_sl = rng_bb.rand(S, H)
        last_idx = H - 1

        def _linspace(a: float, b: float, n: int) -> np.ndarray:
            try:
                return np.linspace(float(a), float(b), int(max(1, n)))
            except Exception:
                return np.array([float(a)])

        seen: Set[Tuple[int, int]] = set()
        base_candidates: List[Tuple[float, float]] = []

        def _push(tp_unit: float, sl_unit: float, bucket: List[Tuple[float, float]]) -> None:
            try:
                tp_val = float(tp_unit)
                sl_val = float(sl_unit)
            except (TypeError, ValueError):
                return
            if not np.isfinite(tp_val) or not np.isfinite(sl_val):
                return
            if tp_val <= 0 or sl_val <= 0:
                return
            key = (int(round(tp_val * 1e6)), int(round(sl_val * 1e6)))
            if key in seen:
                return
            seen.add(key)
            bucket.append((tp_val, sl_val))

        def _add_fixed(bucket: List[Tuple[float, float]], tp_a: float, tp_b: float, tp_n: int, sl_a: float, sl_b: float, sl_n: int) -> None:
            for tp_val in _linspace(tp_a, tp_b, tp_n):
                for sl_val in _linspace(sl_a, sl_b, sl_n):
                    _push(tp_val, sl_val, bucket)

        vol_context: Optional[Dict[str, Any]] = None

        if grid_style_val == 'preset':
            preset_key = preset_val or 'intraday'
            cfg = BARRIER_GRID_PRESETS.get(preset_key, BARRIER_GRID_PRESETS['intraday'])
            if mode_val == 'pct':
                _add_fixed(base_candidates, cfg['tp_min'], cfg['tp_max'], int(cfg['tp_steps']), cfg['sl_min'], cfg['sl_max'], int(cfg['sl_steps']))
            else:
                scale = (float(last_price) / float(pip_size)) / 100.0
                _add_fixed(base_candidates, cfg['tp_min'] * scale, cfg['tp_max'] * scale, int(cfg['tp_steps']), cfg['sl_min'] * scale, cfg['sl_max'] * scale, int(cfg['sl_steps']))
        
        elif grid_style_val == 'volatility':
            # Calculate simple volatility over window
            rets = np.diff(np.log(np.clip(prices, 1e-12, None)))
            rets = rets[np.isfinite(rets)]
            if rets.size > vol_window_val:
                rets = rets[-vol_window_val:]
            vol_per_bar = float(np.std(rets)) if rets.size else 0.0
            vol_horizon = vol_per_bar * np.sqrt(horizon)

            # Convert to percentage space for baseline
            vol_pct = vol_horizon * 100.0

            if mode_val == 'pct':
                tp_start = max(vol_floor_pct_val, vol_pct * vol_min_mult_val)
                tp_end = max(tp_start * 1.1, vol_pct * vol_max_mult_val)
                sl_start = max(vol_floor_pct_val, vol_pct * vol_min_mult_val * 0.8)
                _add_fixed(base_candidates, tp_start, tp_end, vol_steps_val, sl_start, sl_start * vol_sl_multiplier_val, vol_sl_steps_val)
            else:
                # Convert volatility to ticks and apply tick floor when in pips mode
                vol_pips = (vol_pct / 100.0) * (last_price / float(pip_size))
                tp_start = max(vol_floor_pips_val, vol_pips * vol_min_mult_val)
                tp_end = max(tp_start * 1.1, vol_pips * vol_max_mult_val)
                sl_start = max(vol_floor_pips_val, vol_pips * vol_min_mult_val * 0.8)
                _add_fixed(base_candidates, tp_start, tp_end, vol_steps_val, sl_start, sl_start * vol_sl_multiplier_val, vol_sl_steps_val)
            
        elif grid_style_val == 'ratio':
            # Fixed SL grid, TP derived from ratios
            sl_start = sl_min_val
            sl_end = sl_max_val
            for sl_val in _linspace(sl_start, sl_end, sl_steps_val):
                for r in _linspace(ratio_min_val, ratio_max_val, ratio_steps_val):
                    _push(sl_val * r, sl_val, base_candidates)
        
        else: # fixed
            _add_fixed(base_candidates, tp_min_val, tp_max_val, tp_steps_val, sl_min_val, sl_max_val, sl_steps_val)

        # Evaluate candidates
        results: List[Dict[str, Any]] = []
        dir_long = str(direction).lower() == 'long'

        def _evaluate(bucket: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for tp_unit, sl_unit in bucket:
                # Convert to price levels
                if mode_val == 'pct':
                    if dir_long:
                        tp_p = last_price * (1.0 + tp_unit/100.0)
                        sl_p = last_price * (1.0 - sl_unit/100.0)
                    else:
                        tp_p = last_price * (1.0 - tp_unit/100.0)
                        sl_p = last_price * (1.0 + sl_unit/100.0)
                else: # pips
                    if dir_long:
                        tp_p = last_price + tp_unit * pip_size
                        sl_p = last_price - sl_unit * pip_size
                    else:
                        tp_p = last_price - tp_unit * pip_size
                        sl_p = last_price + sl_unit * pip_size

                # Vectorized hit detection
                if dir_long:
                    hit_tp = (paths >= tp_p)
                    hit_sl = (paths <= sl_p)
                else:
                    hit_tp = (paths <= tp_p)
                    hit_sl = (paths >= sl_p)
                if bb_enabled and bb_log_paths is not None and bb_uniform_tp is not None and bb_uniform_sl is not None:
                    tp_dir = "up" if dir_long else "down"
                    sl_dir = "down" if dir_long else "up"
                    tp_bridge = _brownian_bridge_hits(bb_log_paths, float(np.log(tp_p)), bb_sigma, direction=tp_dir, uniform=bb_uniform_tp)
                    sl_bridge = _brownian_bridge_hits(bb_log_paths, float(np.log(sl_p)), bb_sigma, direction=sl_dir, uniform=bb_uniform_sl)
                    hit_tp = hit_tp | tp_bridge
                    hit_sl = hit_sl | sl_bridge

                any_tp = hit_tp.any(axis=1)
                any_sl = hit_sl.any(axis=1)
                
                first_tp = hit_tp.argmax(axis=1)
                first_sl = hit_sl.argmax(axis=1)

                first_tp[~any_tp] = H
                first_sl[~any_sl] = H

                wins = (first_tp < first_sl)
                losses = (first_sl < first_tp)
                ties = (first_tp == first_sl) & (first_tp < H)

                n_wins = wins.sum()
                n_losses = losses.sum()

                prob_win = n_wins / S
                prob_loss = n_losses / S
                prob_tie = ties.sum() / S
                prob_neutral = max(0.0, 1.0 - prob_win - prob_loss - prob_tie)
                prob_resolve = 1.0 - prob_neutral

                risk = sl_unit
                reward = tp_unit
                rr = reward / risk if risk > 0 else 0

                if rr_min_val and rr < rr_min_val:
                    continue
                if rr_max_val and rr > rr_max_val:
                    continue

                ev_val = prob_win * reward - prob_loss * risk
                edge = prob_win - prob_loss

                kelly_val = 0.0
                if rr > 0:
                    kelly_val = prob_win - (prob_loss / rr)

                # Conditional metrics (ignore neutral paths)
                active = prob_win + prob_loss
                if active > 0:
                    prob_win_c = prob_win / active
                    prob_loss_c = prob_loss / active
                    ev_cond = prob_win_c * reward - prob_loss_c * risk
                    kelly_cond = prob_win_c - (prob_loss_c / rr if rr > 0 else 0.0)
                else:
                    ev_cond = 0.0
                    kelly_cond = 0.0

                resolve_mask = (first_tp < H) | (first_sl < H)
                if np.any(resolve_mask):
                    resolve_times = np.minimum(first_tp, first_sl)[resolve_mask] + 1
                    t_res_mean = float(np.mean(resolve_times)) if resolve_times.size else None
                    t_res_med = float(np.median(resolve_times)) if resolve_times.size else None
                else:
                    t_res_mean = None
                    t_res_med = None

                ev_per_bar = 0.0
                if t_res_mean and t_res_mean > 0:
                    ev_per_bar = ev_val / t_res_mean

                profit_factor = 0.0
                denom = prob_loss * risk
                if denom > 0:
                    profit_factor = (prob_win * reward) / denom
                elif prob_win > 0:
                    profit_factor = 1e9

                reward_frac = 0.0
                risk_frac = 0.0
                if last_price > 0:
                    reward_frac = abs(tp_p - last_price) / last_price
                    risk_frac = abs(sl_p - last_price) / last_price
                if risk_frac >= 1.0:
                    risk_frac = 0.999
                utility_val = (prob_win * math.log1p(reward_frac)) + (prob_loss * math.log1p(-risk_frac))

                if min_prob_win_val is not None and prob_win < min_prob_win_val:
                    continue
                if max_prob_no_hit_val is not None and prob_neutral > max_prob_no_hit_val:
                    continue
                if max_median_time_val is not None:
                    if t_res_med is None or t_res_med > max_median_time_val:
                        continue

                # Hit time medians (bars, 1-based) for transparency
                t_hit_tp = (first_tp[wins | ties] + 1)
                t_hit_sl = (first_sl[losses | ties] + 1)
                t_tp_med = float(np.median(t_hit_tp)) if t_hit_tp.size else None
                t_sl_med = float(np.median(t_hit_sl)) if t_hit_sl.size else None

                res = {
                    'tp': tp_unit,
                    'sl': sl_unit,
                    'rr': rr,
                    'tp_price': float(tp_p),
                    'sl_price': float(sl_p),
                    'prob_win': prob_win,
                    'prob_loss': prob_loss,
                    'prob_tp_first': prob_win,
                    'prob_sl_first': prob_loss,
                    'prob_no_hit': prob_neutral,
                    'prob_tie': prob_tie,
                    'prob_resolve': prob_resolve,
                    'ev': ev_val,
                    'ev_cond': ev_cond,
                    'edge': edge,
                    'kelly': kelly_val,
                    'kelly_cond': kelly_cond,
                    'ev_per_bar': ev_per_bar,
                    'profit_factor': profit_factor,
                    'utility': utility_val,
                    't_hit_tp_median': t_tp_med,
                    't_hit_sl_median': t_sl_med,
                    't_hit_resolve_mean': t_res_mean,
                    't_hit_resolve_median': t_res_med,
                }
                out.append(res)
            return out

        results.extend(_evaluate(base_candidates))

        def _sort(res_list: List[Dict[str, Any]]) -> None:
            if objective_val == 'edge':
                res_list.sort(key=lambda x: x['edge'], reverse=True)
            elif objective_val == 'ev':
                res_list.sort(key=lambda x: x['ev'], reverse=True)
            elif objective_val == 'ev_cond':
                res_list.sort(key=lambda x: x['ev_cond'], reverse=True)
            elif objective_val == 'ev_per_bar':
                res_list.sort(key=lambda x: x['ev_per_bar'], reverse=True)
            elif objective_val == 'kelly':
                res_list.sort(key=lambda x: x['kelly'], reverse=True)
            elif objective_val == 'kelly_cond':
                res_list.sort(key=lambda x: x['kelly_cond'], reverse=True)
            elif objective_val == 'prob_tp_first':
                res_list.sort(key=lambda x: x['prob_win'], reverse=True)
            elif objective_val == 'prob_resolve':
                res_list.sort(key=lambda x: x['prob_resolve'], reverse=True)
            elif objective_val == 'profit_factor':
                res_list.sort(key=lambda x: x['profit_factor'], reverse=True)
            elif objective_val == 'min_loss_prob':
                res_list.sort(key=lambda x: x['prob_loss'])
            elif objective_val == 'utility':
                res_list.sort(key=lambda x: x['utility'], reverse=True)

        _sort(results)

        if refine_flag and results:
            best_seed = results[0]
            tp_c = best_seed['tp']
            sl_c = best_seed['sl']
            tp_a = max(1e-9, tp_c * (1.0 - refine_radius_val))
            tp_b = tp_c * (1.0 + refine_radius_val)
            sl_a = max(1e-9, sl_c * (1.0 - refine_radius_val))
            sl_b = sl_c * (1.0 + refine_radius_val)
            refine_candidates: List[Tuple[float, float]] = []
            _add_fixed(refine_candidates, tp_a, tp_b, refine_steps_val, sl_a, sl_b, refine_steps_val)
            results.extend(_evaluate(refine_candidates))
            _sort(results)

        if top_k:
            results = results[:top_k]

        grid_out = results if return_grid else None
        if output == 'summary' and grid_out is not None:
            limit = top_k or min(10, len(grid_out))
            grid_out = grid_out[:limit]
            
        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_name,
            "horizon": horizon,
            "direction": direction,
            "mode": mode,
            "objective": objective,
            "results": results,
            "best": results[0] if results else None,
            "grid": grid_out
        }
        if method_requested != method_name:
            out["method_requested"] = method_requested
            out["method_used"] = method_name
            if auto_reason:
                out["auto_reason"] = auto_reason
        if bb_enabled:
            out["bridge_correction"] = True
        return out

    except Exception as e:
        return {"error": f"Error optimizing barriers: {str(e)}"}
