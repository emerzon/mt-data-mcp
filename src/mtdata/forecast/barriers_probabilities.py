import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, TimeframeLiteral
from ..shared.validators import unsupported_timeframe_seconds_error
from ..utils.barriers import (
    barrier_prices_are_valid as _barrier_prices_are_valid,
)
from ..utils.barriers import (
    get_pip_size as _get_pip_size,
)
from ..utils.barriers import (
    normalize_same_bar_policy,
    normalize_trade_direction,
    resolve_same_bar_probabilities,
    validate_barrier_unit_family_exclusivity,
)
from ..utils.barriers import (
    resolve_barrier_prices as _resolve_barrier_prices,
)
from ..utils.freshness import (
    closed_session_context,
    format_age_seconds,
    format_freshness_label,
)
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.time import (
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
)
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from .barriers_shared import (
    _auto_barrier_method,
    _binomial_se,
    _binomial_wilson_95,
    _brownian_bridge_hits,
    _get_live_reference_price,
    _resolve_reference_prices,
    _scale_price_paths_to_reference,
    _stable_barrier_seed,
    _symbol_price_precision,
    barrier_method_error,
    normalize_barrier_method,
    normalize_barrier_seed,
    offset_barrier_seed,
)
from .common import fetch_history as _fetch_history
from .common import log_returns_from_prices as _log_returns_from_prices
from .monte_carlo import gbm_single_barrier_upcross_prob as _gbm_upcross_prob
from .monte_carlo import simulate_bootstrap_mc as _simulate_bootstrap_mc
from .monte_carlo import simulate_garch_mc as _simulate_garch_mc
from .monte_carlo import simulate_gbm_mc as _simulate_gbm_mc
from .monte_carlo import simulate_heston_mc as _simulate_heston_mc
from .monte_carlo import simulate_hmm_mc as _simulate_hmm_mc
from .monte_carlo import simulate_jump_diffusion_mc as _simulate_jump_diffusion_mc


def _format_barrier_epoch(epoch: float) -> str:
    formatter = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
    return formatter(float(epoch))


def _coerce_epoch(value: Any) -> Optional[float]:
    try:
        epoch = float(value)
    except Exception:
        return None
    if not np.isfinite(epoch) or epoch <= 0.0:
        return None
    if epoch > 10_000_000_000:
        epoch /= 1000.0
    return epoch


def _history_freshness_context(
    df: Any,
    timeframe: str,
    *,
    symbol: Optional[str] = None,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"history_bars_used": int(len(df))}
    try:
        last_epoch = _coerce_epoch(df["time"].iloc[-1])
    except Exception:
        last_epoch = None
    if last_epoch is None:
        return out

    if now_epoch is None:
        now_epoch = datetime.now(timezone.utc).timestamp()
    timeframe_seconds = max(1, int(TIMEFRAME_SECONDS.get(timeframe, 0) or 0))
    completed_bar_end = last_epoch + timeframe_seconds
    age_seconds = max(0, int(round(now_epoch - completed_bar_end)))
    stale_after = timeframe_seconds
    data_stale = age_seconds > stale_after if stale_after > 0 else None
    age_text = format_age_seconds(age_seconds)
    out.update(
        {
            "history_last_bar_open": _format_barrier_epoch(last_epoch),
            "history_last_bar_open_epoch": float(last_epoch),
            "data_as_of": _format_barrier_epoch(completed_bar_end),
            "data_as_of_epoch": float(completed_bar_end),
            "data_freshness_seconds": age_seconds,
            "data_stale": data_stale,
            "stale_after_seconds": stale_after,
            "freshness_basis": "last_completed_bar_end",
            "input_bar_policy": "closed_bars_only",
        }
    )
    closed_session = closed_session_context(
        symbol,
        now_epoch=now_epoch,
        item="data",
        data_age_seconds=age_seconds,
    )
    if closed_session:
        out.update(closed_session)
    history_policy_ok = not bool(out.get("data_stale")) and not bool(closed_session)
    out["history_policy_ok"] = history_policy_ok
    freshness = format_freshness_label(
        data_stale=out.get("data_stale"),
        market_status=(
            out.get("market_status")
            if out.get("freshness_policy_relaxed") is not False
            else None
        ),
        market_status_reason=(
            out.get("market_status_reason")
            if out.get("freshness_policy_relaxed") is not False
            else None
        ),
        age_seconds=age_seconds,
        age_text=age_text,
        item="data",
    )
    if freshness:
        out["freshness"] = freshness
    return out


def _live_reference_time_context(
    symbol: str,
    timeframe: str,
    *,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        from ..utils.mt5 import mt5 as _mt5
    except Exception:
        return {}
    try:
        tick = _mt5.symbol_info_tick(symbol)
    except Exception:
        tick = None
    if tick is None:
        return {}

    epoch = _coerce_epoch(getattr(tick, "time_msc", None))
    if epoch is None:
        epoch = _coerce_epoch(getattr(tick, "time", None))
    if epoch is None:
        return {}

    if now_epoch is None:
        now_epoch = datetime.now(timezone.utc).timestamp()
    freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=epoch,
        now_epoch=now_epoch,
        item="reference price",
        age_rounder=lambda value: max(0, int(round(value))),
    )
    age_seconds = freshness.get("data_age_seconds")
    out = {
        "reference_price_time": _format_barrier_epoch(epoch),
        "reference_price_time_epoch": float(epoch),
        "reference_price_age_seconds": age_seconds,
        "reference_price_age": format_age_seconds(age_seconds),
        "reference_price_stale": freshness.get("data_stale"),
        "reference_freshness_state": freshness.get("freshness_state"),
        "reference_live_max_age_seconds": freshness.get("live_max_age_seconds"),
        "reference_usable_for_live": freshness.get("usable_for_live_trading"),
    }
    for key in (
        "market_status",
        "market_status_reason",
        "market_status_source",
        "freshness_policy_relaxed",
    ):
        if freshness.get(key) is not None:
            out[key] = freshness[key]
    return out


def _abs_barrier_side_error(
    *,
    price: float,
    direction: str,
    tp_abs: Optional[float],
    sl_abs: Optional[float],
) -> Optional[str]:
    """Reject user-supplied absolute TP/SL levels on the wrong side of price.

    ``tp_abs``/``sl_abs`` are absolute price levels (not offsets). A mis-sided
    level is almost always a unit mistake (e.g. passing an intended offset like
    ``sl_abs=500``), which the silent inversion nudge would otherwise mask with
    meaningless probabilities. Return an actionable error instead.
    """
    try:
        ref = float(price)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ref):
        return None
    is_long = direction == "long"
    problems: List[str] = []
    for name, value, must_be_above in (
        ("tp_abs", tp_abs, is_long),
        ("sl_abs", sl_abs, not is_long),
    ):
        if value is None:
            continue
        try:
            level = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(level):
            continue
        if must_be_above and level <= ref:
            problems.append(f"{name} ({level:g}) must be above the reference price ({ref:g})")
        elif not must_be_above and level >= ref:
            problems.append(f"{name} ({level:g}) must be below the reference price ({ref:g})")
    if not problems:
        return None
    side = "long" if is_long else "short"
    return (
        f"For a {side} position, " + " and ".join(problems)
        + ". tp_abs/sl_abs are absolute price levels, not offsets; use tp_pct/sl_pct "
        "or tp_ticks/sl_ticks to specify distances from the reference price."
    )


def forecast_barrier_hit_probabilities(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','mc_gbm_bb','hmm_mc','garch','bootstrap','heston','jump_diffusion','auto'] = 'mc_gbm_bb',
    direction: Literal['long','short'] = 'long',
    same_bar_policy: Literal['sl_first','tp_first','neutral'] = 'sl_first',
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_ticks: Optional[float] = None,
    sl_ticks: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Monte Carlo barrier analysis: TP/SL hit probabilities within `horizon` bars.

    Notes:
    - Barriers are provided via absolute prices (tp_abs/sl_abs), percentages
      (tp_pct/sl_pct), or ticks (tp_ticks/sl_ticks; uses `trade_tick_size`).
      Use exactly one unit family per request; mixed units are rejected.
    - In discrete time, TP and SL can be hit in the same bar. Resolution is
      controlled explicitly by `same_bar_policy`.
    - The default GBM Brownian-bridge method accounts for barrier touches
      between simulated bar closes. Other path methods disclose close-only
      hit detection in their output.
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
        try:
            same_bar_policy_value = normalize_same_bar_policy(same_bar_policy)
        except ValueError as exc:
            return {"error": str(exc)}
        p = _parse_kv_or_json(params)
        warnings_out: List[str] = []
        try:
            barrier_values = validate_barrier_unit_family_exclusivity(
                {
                    "tp_abs": tp_abs,
                    "sl_abs": sl_abs,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_ticks": tp_ticks,
                    "sl_ticks": sl_ticks,
                }
            )
        except ValueError as exc:
            return {"error": str(exc)}
        # Fetch enough history for calibration
        need = int(max(2000, horizon_val + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        freshness_context = _history_freshness_context(
            df,
            timeframe,
            symbol=symbol,
        )
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

        abs_side_error = _abs_barrier_side_error(
            price=last_price, direction=direction_norm, tp_abs=tp_abs, sl_abs=sl_abs
        )
        if abs_side_error:
            return {"error": abs_side_error}

        # Compute absolute TP/SL prices with explicit trade direction
        dir_long = direction_norm == 'long'
        tp_price, sl_price = _resolve_barrier_prices(
            price=last_price,
            direction=direction_norm,
            tp_abs=tp_abs,
            sl_abs=sl_abs,
            tp_pct=barrier_values.get("tp_pct"),
            sl_pct=barrier_values.get("sl_pct"),
            tp_ticks=barrier_values.get("tp_ticks"),
            sl_ticks=barrier_values.get("sl_ticks"),
            pip_size=pip_size,
        )

        if tp_price is None or sl_price is None:
            return {
                "error": (
                    "Missing barriers. Provide either tp_pct and sl_pct, "
                    "tp_abs and sl_abs, or tp_ticks and sl_ticks."
                )
            }
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
                from ..utils.denoise import apply_denoise as apply_denoise_util
                added = apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception as ex:
                warnings_out.append(f"Denoise request failed; using raw close prices instead: {ex}")
        prices = df[base_col].astype(float).to_numpy()

        # Simulate paths
        sims = int(p.get('n_sims', p.get('sims', 2000)) or 2000)
        if sims <= 0:
            return {"error": f"Invalid n_sims: {sims}. Must be >= 1."}
        method_key = normalize_barrier_method(method)
        if method_key is None:
            return {"error": barrier_method_error(method)}
        method_requested = method_key
        auto_reason = None
        if method_key == 'auto':
            method_key, auto_reason = _auto_barrier_method(
                symbol, timeframe, prices, horizon=horizon_val
            )
        bb_enabled = method_key == 'mc_gbm_bb'
        seed_raw = p.get('seed')
        seed_provided = seed_raw is not None
        request_seed_base = (
            normalize_barrier_seed(seed_raw)
            if seed_provided
            else _stable_barrier_seed(
                "forecast_barrier_prob",
                symbol,
                timeframe,
                horizon_val,
                method_key,
                direction_norm,
                float(last_price),
                float(tp_price),
                float(sl_price),
                int(sims),
                int(len(prices)),
                float(prices[-1]),
                {k: v for k, v in p.items() if k != "seed"},
            )
        )
        
        try:
            if method_key in ('mc_gbm', 'mc_gbm_bb'):
                sim = _simulate_gbm_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                )
            elif method_key == 'hmm_mc':
                n_states = int(p.get('n_states', 2) or 2)
                sim = _simulate_hmm_mc(
                    prices,
                    horizon=horizon_val,
                    n_states=int(n_states),
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                )
            elif method_key == 'garch':
                p_order = int(p.get('p', 1))
                q_order = int(p.get('q', 1))
                sim = _simulate_garch_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
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
                    seed=normalize_barrier_seed(request_seed_base),
                    block_size=bs,
                )
            elif method_key == 'heston':
                sim = _simulate_heston_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
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
                    seed=normalize_barrier_seed(request_seed_base),
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
                rng_bb = np.random.RandomState(offset_barrier_seed(request_seed_base, 7))
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
        resolved_probabilities = resolve_same_bar_probabilities(
            tp_strict=tp_first / S_f,
            sl_strict=sl_first / S_f,
            same_bar=both_tie / S_f,
            no_hit=no_hit / S_f,
            policy=same_bar_policy_value,
        )
        prob_tp_first = resolved_probabilities["prob_tp_first"]
        prob_sl_first = resolved_probabilities["prob_sl_first"]
        prob_same_bar = resolved_probabilities["prob_same_bar"]
        prob_no_hit = resolved_probabilities["prob_no_hit"]
        tp_any_curve = (tp_any_by_t / S_f).tolist()
        sl_any_curve = (sl_any_by_t / S_f).tolist()
        tp_lo, tp_hi = _binomial_wilson_95(prob_tp_first, int(S))
        sl_lo, sl_hi = _binomial_wilson_95(prob_sl_first, int(S))
        tie_lo, tie_hi = _binomial_wilson_95(prob_same_bar, int(S))
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
        probability_edge = float(prob_tp_first - prob_sl_first)
        price_precision = _symbol_price_precision(symbol)
        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_key,
            "intra_bar_hit_detection": (
                "brownian_bridge" if bb_enabled else "simulated_bar_close"
            ),
            "horizon": horizon_val,
            "direction": direction_norm,
            "same_bar_policy": same_bar_policy_value,
            "last_price": last_price,
            "last_price_close": float(last_price_close),
            "last_price_source": last_price_source,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "n_sims": int(S),
            "seed": int(request_seed_base),
            "seed_source": "params" if seed_provided else "request",
            **resolved_probabilities,
            "prob_tp_first_se": _binomial_se(prob_tp_first, int(S)),
            "prob_sl_first_se": _binomial_se(prob_sl_first, int(S)),
            "prob_same_bar_se": _binomial_se(prob_same_bar, int(S)),
            "prob_no_hit_se": _binomial_se(prob_no_hit, int(S)),
            "prob_tp_first_ci95": {"low": float(tp_lo), "high": float(tp_hi)},
            "prob_sl_first_ci95": {"low": float(sl_lo), "high": float(sl_hi)},
            "prob_same_bar_ci95": {"low": float(tie_lo), "high": float(tie_hi)},
            "prob_no_hit_ci95": {"low": float(no_hit_lo), "high": float(no_hit_hi)},
            "probability_edge": probability_edge,
            "tp_hit_prob_by_t": [float(v) for v in tp_any_curve],
            "sl_hit_prob_by_t": [float(v) for v in sl_any_curve],
            "time_to_tp_bars": tp_stats,
            "time_to_sl_bars": sl_stats,
        }
        out.update(freshness_context)
        out["model_data_usable_for_live"] = bool(
            freshness_context.get("history_policy_ok")
        )
        if str(last_price_source or "").startswith("live_tick"):
            reference_context = _live_reference_time_context(symbol, timeframe)
            out.update(reference_context)
            out["usable_for_live_trading"] = bool(
                out.get("model_data_usable_for_live")
                and reference_context.get("reference_usable_for_live")
            )
            out["usable_for_live_trading_basis"] = (
                "model_history_and_reference_quote"
            )
            blockers = []
            if not out.get("model_data_usable_for_live"):
                blockers.append("model_history_outside_policy")
            if not reference_context.get("reference_usable_for_live"):
                blockers.append("reference_quote_not_live")
            out["execution_blockers"] = blockers
            if (
                reference_context.get("reference_price_stale") is True
                or reference_context.get("market_status") == "closed"
            ):
                out["last_price_source"] = str(last_price_source).replace(
                    "live_tick", "last_tick", 1
                )
        elif freshness_context.get("data_as_of"):
            out["reference_price_time"] = freshness_context.get("data_as_of")
            out["reference_price_time_epoch"] = freshness_context.get("data_as_of_epoch")
        out["conditioning_note"] = (
            "Probabilities use closed bars through "
            f"{out.get('data_as_of')}; barriers are measured from "
            f"{out.get('last_price_source')}."
        )
        if not out.get("usable_for_live_trading"):
            warnings_out.append(
                "Barrier output is not execution-ready because the model history "
                "or reference quote is outside its live freshness policy."
            )
        if price_precision is not None:
            out["price_precision"] = int(price_precision)
        if method_requested != method_key:
            out["method_requested"] = method_requested
            out["method_used"] = method_key
            if auto_reason:
                out["auto_reason"] = auto_reason
        if bb_enabled:
            out["bridge_correction"] = True
        else:
            warnings_out.append(
                "Barrier hits are evaluated at simulated bar closes; transient "
                "intra-bar touches may be undercounted."
            )
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
            requested_states = sim_meta.get("requested_n_states")
            fitted_states = sim_meta.get("fitted_n_states")
            if (
                isinstance(requested_states, (int, float))
                and isinstance(fitted_states, (int, float))
                and int(fitted_states) < int(requested_states)
            ):
                warnings_out.append(
                    "HMM state collapse: requested "
                    f"{int(requested_states)} states but fitted {int(fitted_states)}; "
                    "probabilities use the reduced-state model."
                )
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
        need = int(max(2000, horizon + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history"}
        freshness_context = _history_freshness_context(
            df,
            timeframe,
            symbol=symbol,
        )
        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import apply_denoise as apply_denoise_util
                added = apply_denoise_util(df, denoise, default_when='pre_ti')
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
        price_precision = _symbol_price_precision(symbol)
        result = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "direction": direction_norm,
            "last_price": s0,
            "last_price_source": "candle_close",
            "barrier": float(barrier),
            "mu_annual": float(gbm_drift),
            "log_drift_annual": float(log_drift),
            "sigma_annual": sigma_val,
            "prob_hit": float(prob),
        }
        result.update(freshness_context)
        if freshness_context.get("data_as_of"):
            result["reference_price_time"] = freshness_context.get("data_as_of")
            result["reference_price_time_epoch"] = freshness_context.get("data_as_of_epoch")
        if price_precision is not None:
            result["price_precision"] = int(price_precision)
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

