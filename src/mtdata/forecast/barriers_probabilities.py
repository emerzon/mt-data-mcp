from typing import Any, Dict, Optional, List, Literal
import traceback
import numpy as np
from ..shared.schema import TimeframeLiteral, DenoiseSpec
from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.validators import unsupported_timeframe_seconds_error
from .common import fetch_history as _fetch_history, log_returns_from_prices as _log_returns_from_prices
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
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

from .barriers_shared import (
    _auto_barrier_method,
    _binomial_se,
    _binomial_wilson_95,
    _brownian_bridge_hits,
    _get_live_reference_price,
    _resolve_reference_prices,
    _scale_price_paths_to_reference,
)


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
        try:
            horizon_val = int(horizon)
        except Exception:
            return {"error": f"Invalid horizon: {horizon}. Must be a positive integer."}
        if horizon_val <= 0:
            return {"error": f"Invalid horizon: {horizon_val}. Must be >= 1."}
        direction_norm, direction_error = normalize_trade_direction(direction)
        if direction_error:
            return {"error": direction_error}
        p = _parse_kv_or_json(params)
        warnings_out: List[str] = []
        # Fetch enough history for calibration
        need = int(max(300, horizon_val + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        # Current price baseline
        last_price_close, last_price, last_price_source, price_warning, price_error = _resolve_reference_prices(
            df['close'].astype(float).to_numpy(),
            symbol=symbol,
            direction=direction_norm,
            use_live_price=True,
            live_price_getter=_get_live_reference_price,
        )
        if price_error:
            return {"error": price_error}
        if price_warning:
            warnings_out.append(price_warning)
        pip_size = _get_pip_size(symbol)

        # Compute absolute TP/SL prices with explicit trade direction
        dir_long = direction_norm == 'long'
        tp_price, sl_price = _resolve_barrier_prices(
            price=last_price,
            direction=direction_norm,
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
        if not _barrier_prices_are_valid(
            price=last_price,
            direction=direction_norm,
            tp_price=tp_price,
            sl_price=sl_price,
        ):
            return {"error": "Resolved TP/SL barriers are invalid for the current price."}

        # Build input series (denoise optional)
        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import _apply_denoise as _apply_denoise_util
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception as ex:
                warnings_out.append(f"Denoise request failed; using raw close prices instead: {ex}")
        prices = df[base_col].astype(float).to_numpy()

        # Simulate paths
        sims = int(p.get('n_sims', p.get('sims', 2000)) or 2000)
        if sims <= 0:
            return {"error": f"Invalid n_sims: {sims}. Must be >= 1."}
        seed_raw = p.get('seed')
        seed = int(seed_raw) if seed_raw is not None else None
        request_seed_base = (
            int(seed)
            if seed is not None
            else int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        )
        method_key = str(method).lower().strip()
        method_requested = method_key
        auto_reason = None
        if method_key == 'auto':
            method_key, auto_reason = _auto_barrier_method(
                symbol, timeframe, prices, horizon=horizon_val
            )
        bb_enabled = method_key == 'mc_gbm_bb'
        
        try:
            if method_key in ('mc_gbm', 'mc_gbm_bb'):
                sim = _simulate_gbm_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=int(request_seed_base),
                )
            elif method_key == 'hmm_mc':
                n_states = int(p.get('n_states', 2) or 2)
                sim = _simulate_hmm_mc(
                    prices,
                    horizon=horizon_val,
                    n_states=int(n_states),
                    n_sims=int(sims),
                    seed=int(request_seed_base),
                )
            elif method_key == 'garch':
                p_order = int(p.get('p', 1))
                q_order = int(p.get('q', 1))
                sim = _simulate_garch_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=int(request_seed_base),
                    p_order=p_order,
                    q_order=q_order,
                )
            elif method_key == 'bootstrap':
                bs = p.get('block_size')
                if bs: bs = int(bs)
                sim = _simulate_bootstrap_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=int(request_seed_base),
                    block_size=bs,
                )
            elif method_key == 'heston':
                sim = _simulate_heston_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=int(request_seed_base),
                    kappa=p.get('kappa'),
                    theta=p.get('theta'),
                    xi=p.get('xi'),
                    rho=p.get('rho'),
                    v0=p.get('v0'),
                )
            elif method_key == 'jump_diffusion':
                sim = _simulate_jump_diffusion_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=int(request_seed_base),
                    jump_lambda=p.get('jump_lambda', p.get('lambda')),
                    jump_mu=p.get('jump_mu'),
                    jump_sigma=p.get('jump_sigma'),
                    jump_threshold=float(p.get('jump_threshold', 3.0)),
                )
            else:
                return {"error": f"Unsupported method: {method}. Use 'mc_gbm', 'mc_gbm_bb', 'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', or 'auto'"}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            return {
                "error": f"Simulation failed ({method_key}): {e}",
                "error_type": "simulation_failure",
                "traceback_summary": traceback.format_exc()[-500:],
            }

        price_paths = np.asarray(sim['price_paths'], dtype=float)
        S, H = price_paths.shape
        try:
            sim_anchor_price = float(prices[-1])
        except Exception:
            sim_anchor_price = float(last_price_close)
        price_paths = _scale_price_paths_to_reference(
            price_paths,
            simulated_anchor_price=sim_anchor_price,
            reference_price=last_price,
        )

        bb_sigma = 0.0
        bb_uniform_tp = None
        bb_uniform_sl = None
        bb_log_paths = None
        if bb_enabled:
            rets = _log_returns_from_prices(prices)
            rets = rets[np.isfinite(rets)]
            bb_sigma = float(np.std(rets, ddof=1)) if rets.size else 0.0
            if not np.isfinite(bb_sigma) or bb_sigma <= 0:
                bb_enabled = False
            else:
                log_paths = np.log(np.clip(price_paths, 1e-12, None))
                log_s0 = float(np.log(max(last_price, 1e-12)))
                bb_log_paths = np.concatenate([np.full((S, 1), log_s0), log_paths], axis=1)
                rng_bb = np.random.RandomState(int(request_seed_base) + 7)
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
        prob_tie = both_tie / S_f
        prob_no_hit = no_hit / S_f
        tp_any_curve = (tp_any_by_t / S_f).tolist()
        sl_any_curve = (sl_any_by_t / S_f).tolist()
        tp_lo, tp_hi = _binomial_wilson_95(prob_tp_first, int(S))
        sl_lo, sl_hi = _binomial_wilson_95(prob_sl_first, int(S))
        tie_lo, tie_hi = _binomial_wilson_95(prob_tie, int(S))
        no_hit_lo, no_hit_hi = _binomial_wilson_95(prob_no_hit, int(S))

        def _stats(arr: list[int]) -> Dict[str, float]:
            if not arr:
                return {"mean": float('nan'), "median": float('nan')}
            a = np.asarray(arr, dtype=float)
            return {"mean": float(a.mean()), "median": float(np.median(a))}

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
            "horizon": horizon_val,
            "direction": direction_norm,
            "last_price": last_price,
            "last_price_close": float(last_price_close),
            "last_price_source": last_price_source,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "prob_tp_first": float(prob_tp_first),
            "prob_sl_first": float(prob_sl_first),
            "prob_tie": float(prob_tie),
            "prob_no_hit": float(prob_no_hit),
            "prob_tp_first_se": _binomial_se(prob_tp_first, int(S)),
            "prob_sl_first_se": _binomial_se(prob_sl_first, int(S)),
            "prob_tie_se": _binomial_se(prob_tie, int(S)),
            "prob_no_hit_se": _binomial_se(prob_no_hit, int(S)),
            "prob_tp_first_ci95": {"low": float(tp_lo), "high": float(tp_hi)},
            "prob_sl_first_ci95": {"low": float(sl_lo), "high": float(sl_hi)},
            "prob_tie_ci95": {"low": float(tie_lo), "high": float(tie_hi)},
            "prob_no_hit_ci95": {"low": float(no_hit_lo), "high": float(no_hit_hi)},
            "edge": float(edge),
            "tp_hit_prob_by_t": [float(v) for v in tp_any_curve],
            "sl_hit_prob_by_t": [float(v) for v in sl_any_curve],
            "time_to_tp_bars": tp_stats,
            "time_to_sl_bars": sl_stats,
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
        # Expose simulation model metadata (e.g. HMM fitted vs requested states)
        _meta_keys = ('fitted_n_states', 'requested_n_states', 'model_type')
        sim_meta = {k: sim[k] for k in _meta_keys if k in sim}
        if 'mu' in sim:
            import numpy as _np
            sim_meta['mu'] = [float(v) for v in _np.asarray(sim['mu']).ravel()]
        if 'sigma' in sim:
            import numpy as _np
            sim_meta['sigma'] = [float(v) for v in _np.asarray(sim['sigma']).ravel()]
        if sim_meta:
            out['sim_meta'] = sim_meta
        if warnings_out:
            out["warnings"] = warnings_out
             
        return out
    except (KeyError, AttributeError, IndexError):
        raise
    except Exception as e:
        return {
            "error": f"Error computing barrier probabilities: {str(e)}",
            "error_type": type(e).__name__,
            "traceback_summary": traceback.format_exc()[-500:],
        }

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
        direction_norm, direction_error = normalize_trade_direction(direction)
        if direction_error:
            return {"error": direction_error}
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
            return {"error": unsupported_timeframe_seconds_error(timeframe)}
        T = float(tf_secs * int(horizon)) / (365.0 * 24.0 * 3600.0)
        if mu is None or sigma is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = _log_returns_from_prices(prices)
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
        if direction_norm == 'short':
            s0_inv = 1.0 / s0
            b_inv = 1.0 / float(barrier)
            inv_drift = sigma_sq - gbm_drift
            prob = _gbm_upcross_prob(s0_inv, b_inv, float(inv_drift), sigma_val, float(T))
        else:
            prob = _gbm_upcross_prob(s0, float(barrier), float(gbm_drift), sigma_val, float(T))
        already_hit = (
            (direction_norm == 'long' and barrier <= s0)
            or (direction_norm == 'short' and barrier >= s0)
        )
        result = {
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
        if already_hit:
            result["already_hit"] = True
        return result
    except (KeyError, AttributeError, IndexError):
        raise
    except Exception as e:
        return {
            "error": f"Error computing closed-form barrier probability: {str(e)}",
            "error_type": type(e).__name__,
            "traceback_summary": traceback.format_exc()[-500:],
        }
