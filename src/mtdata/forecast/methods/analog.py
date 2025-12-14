
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ...utils.patterns import build_index, PatternIndex
from ...core.schema import TimeframeLiteral
from ...utils.mt5 import _mt5_epoch_to_utc
from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry

@ForecastRegistry.register("analog")
class AnalogMethod(ForecastMethod):
    """
    Analog forecasting (Nearest Neighbor) using historical pattern matching.
    
    Finds the top-k most similar historical windows to the current market state
    and projects their future trajectories. Supports multi-timeframe consensus.
    """
    
    @property
    def name(self) -> str:
        return "analog"

    @property
    def category(self) -> str:
        return "analog"

    @property
    def required_packages(self) -> List[str]:
        return ["scipy", "numpy", "pandas"]
        
    @property
    def supports_features(self) -> Dict[str, bool]:
        # Index is built on price levels; return inputs would need separate index logic.
        return {"price": True, "return": False, "volatility": False, "ci": True}

    def _run_single_timeframe(
        self,
        symbol: str,
        timeframe: str,
        horizon: int,
        params: Dict[str, Any],
        query_vector: Optional[np.ndarray] = None,
        *,
        as_of: Optional[Any] = None,
        drop_last_live: bool = True,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Run analog logic for a single timeframe.
        Returns (List[ProjectedPaths], AnalogsMetadataList).
        Paths are arrays of length 'horizon'.
        """
        # Config
        window_size = int(params.get('window_size', 64))
        search_depth = int(params.get('search_depth', 5000))
        top_k = int(params.get('top_k', 20))
        metric = str(params.get('metric', 'euclidean'))
        scale = str(params.get('scale', 'zscore'))
        refine_metric = str(params.get('refine_metric', 'dtw'))
        search_engine = str(params.get('search_engine', 'ckdtree'))
        drop_last_live = bool(params.get('drop_last_live', drop_last_live))
        # Profile-based engines expect z-normalized inputs
        idx_scale = 'zscore' if search_engine in ('matrix_profile', 'mass') else scale
        if search_engine in ('matrix_profile', 'mass') and metric.lower() not in ('euclidean', 'l2'):
            metric = 'euclidean'
        
        # Build Index (for HISTORY lookup)
        try:
            # We fetch enough history to find matches, but we do NOT rely on this fetch for the query vector if provided.
            idx = build_index(
                symbols=[str(symbol)],
                timeframe=str(timeframe),
                window_size=window_size,
                future_size=int(horizon),
                max_bars_per_symbol=search_depth + window_size + int(horizon) + 100,
                scale=idx_scale,
                metric=metric,
                engine=search_engine,
                as_of=as_of,
                drop_last_live=drop_last_live,
            )
        except Exception:
            return [], []
            
        if not idx.X.shape[0]:
            return [], []

        # Determine Query Vector
        if query_vector is not None:
            # Validate length
            if len(query_vector) < window_size:
                if len(query_vector) == 0:
                     return [], []
                # Pad with leading values
                pad_width = window_size - len(query_vector)
                query_window = np.pad(query_vector, (pad_width, 0), mode='edge')
            else:
                query_window = query_vector[-window_size:]
        else:
            # Fallback to internal latest
            full_series = idx.get_symbol_series(str(symbol))
            if full_series is None or len(full_series) < window_size:
                return [], []
            query_window = full_series[-window_size:]
        
        # Search
        try:
            raw_idxs, raw_dists = idx.search(query_window, top_k=top_k*2 + 5)
        except Exception:
            return [], []
            
        # Filter
        n_windows = idx.X.shape[0]
        valid_candidates = []
        for i, dist in zip(raw_idxs, raw_dists):
            if i >= n_windows - 5:
                continue
            valid_candidates.append((i, dist))
            
        if not valid_candidates:
            return [], []
            
        # Refine
        valid_idxs = np.array([x[0] for x in valid_candidates], dtype=int)
        valid_dists = np.array([x[1] for x in valid_candidates], dtype=float)
        k = min(top_k, len(valid_candidates))
        
        try:
            final_idxs, final_scores = idx.refine_matches(
                query_window, valid_idxs, valid_dists, top_k=k,
                shape_metric=refine_metric, allow_lag=int(window_size * 0.1)
            )
        except Exception:
             return [], []
        
        # Project
        futures = []
        meta_list = []
        current_price = query_window[-1]
        
        for i, score in zip(final_idxs, final_scores):
            try:
                full_seq = idx.get_match_values(i, include_future=True)
                if len(full_seq) <= window_size:
                    continue
                future_part = full_seq[window_size:]
                
                # Pad/Truncate
                f_vals = np.full(horizon, np.nan)
                take = min(len(future_part), horizon)
                f_vals[:take] = future_part[:take]
                
                # Scale
                analog_end_price = full_seq[window_size - 1]
                scale_factor = (current_price / analog_end_price) if analog_end_price > 1e-12 else 1.0
                
                proj_future = f_vals * scale_factor
                
                if take < horizon:
                    proj_future[take:] = proj_future[take-1] if take > 0 else current_price

                # Meta - Get time BEFORE appending to futures to ensure sync
                t_arr = idx.get_match_times(i, include_future=False)
                t_start = t_arr[0] if len(t_arr) > 0 else 0
                meta_obj = {
                    "score": float(score),
                    "date": _mt5_epoch_to_utc(float(t_start)),
                    "index": int(i),
                    "scale_factor": float(scale_factor)
                }

                # Only append if both succeeded
                futures.append(proj_future)
                meta_list.append(meta_obj)

            except Exception:
                continue

        return futures, meta_list

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        # Enforce price-only input to avoid mixing return/volatility series with price-based indices
        base_name = series.name if series is not None else ""
        if series is None or series.empty or (base_name and (base_name.startswith("__") or "return" in base_name or "vol" in base_name)):
            raise ValueError("Analog method supports price series only; provided series is missing or appears to be a derived/return/volatility series")

        symbol = params.get('symbol')
        primary_tf = params.get('timeframe')
        as_of = params.get('as_of')
        ci_alpha = params.get('ci_alpha', 0.05)
        
        if not symbol or not primary_tf:
            raise ValueError("Analog method requires 'symbol' and 'timeframe'")
        try:
            ci_alpha = float(ci_alpha) if ci_alpha is not None else 0.05
        except Exception:
            ci_alpha = 0.05
        if not (0.0 < ci_alpha < 1.0):
            ci_alpha = 0.05

        # Capture actual parameter values used (with defaults)
        window_size = int(params.get('window_size', 64))
        search_depth = int(params.get('search_depth', 5000))
        top_k = int(params.get('top_k', 20))
        metric = str(params.get('metric', 'euclidean'))
        scale = str(params.get('scale', 'zscore'))
        refine_metric = str(params.get('refine_metric', 'dtw'))
        search_engine = str(params.get('search_engine', 'ckdtree'))

        # Use input series for primary query vector
        # Handle case where series is empty or missing
        primary_query = series.values if series is not None and not series.empty else None
        
        # Parse secondary timeframes
        sec_tf_param = params.get('secondary_timeframes')
        secondary_tfs = []
        if sec_tf_param:
            if isinstance(sec_tf_param, str):
                secondary_tfs = [s.strip() for s in sec_tf_param.split(',') if s.strip()]
            elif isinstance(sec_tf_param, list):
                secondary_tfs = [str(s) for s in sec_tf_param]
        
        # 1. Run Primary
        p_futures, p_analogs = self._run_single_timeframe(
            symbol, primary_tf, horizon, params, query_vector=primary_query, as_of=as_of
        )
        
        if not p_futures:
             # Fallback if primary fails?
             # For now, raise.
             raise RuntimeError(f"Primary analog search failed for {symbol} {primary_tf}")

        # Pool of all projected paths (from all timeframes)
        # We keep them as arrays of length 'horizon' (primary horizon)
        # For secondaries, we must resample their paths to match this length/steps.
        
        pool_futures = [f for f in p_futures]
        
        from ...core.constants import TIMEFRAME_SECONDS
        p_sec = TIMEFRAME_SECONDS.get(primary_tf, 3600)

        # 2. Run Secondaries
        for tf in secondary_tfs:
            if tf == primary_tf: continue
            
            s_sec = TIMEFRAME_SECONDS.get(tf, 3600)
            required_duration = horizon * p_sec
            s_horizon = int(np.ceil(required_duration / s_sec))
            s_horizon = max(s_horizon, 3)
            
            # Note: We pass None for query_vector here, so it uses the fetched history's latest
            # (Best effort since we don't have secondary series inputs)
            s_futures, _ = self._run_single_timeframe(
                symbol, tf, s_horizon, params, query_vector=None, as_of=as_of
            )
            
            if s_futures:
                # Resample each path to match primary steps
                # Time steps:
                # Primary: 0, 1*dt_p, 2*dt_p ...
                # Secondary: 0, 1*dt_s, ...
                # We interpolate.
                
                t_p = np.arange(horizon) * p_sec
                t_s = np.arange(s_horizon) * s_sec

                # Require at least one full secondary step covering the primary horizon; otherwise skip to avoid flat/aliased paths
                if t_p[-1] < t_s[1]:
                    continue
                
                for s_path in s_futures:
                    # Interp
                    # s_path is y, t_s is x. we want y at t_p
                    # Ensure s_path is clean
                    valid = np.isfinite(s_path)
                    if not np.any(valid): continue
                    
                    # Stepwise hold (piecewise constant) to avoid over-smoothing when upsampling coarse TFs
                    idxs = np.searchsorted(t_s, t_p, side='right') - 1
                    idxs[idxs < 0] = 0
                    idxs[idxs >= len(s_path)] = len(s_path) - 1
                    step_y = s_path[idxs]
                    pool_futures.append(step_y)

        # 3. Aggregate
        if not pool_futures:
            raise RuntimeError("No valid paths generated")
            
        futures_matrix = np.vstack(pool_futures) # Shape: (TotalPaths, Horizon)
        
        # Percentiles for CI (Distribution of Analogs)
        p50 = np.nanmedian(futures_matrix, axis=0)
        lower_q = max(0.0, min(100.0, (ci_alpha / 2.0) * 100.0))
        upper_q = max(0.0, min(100.0, (1.0 - ci_alpha / 2.0) * 100.0))
        p_lower = np.nanpercentile(futures_matrix, lower_q, axis=0)
        p_upper = np.nanpercentile(futures_matrix, upper_q, axis=0)
        
        # Compute ensemble metrics
        spread = np.nanmean(np.nanstd(futures_matrix, axis=0))
        stdev = float(np.nanmean(np.nanstd(futures_matrix, axis=0)))
        
        # Build params_used with actual values (including defaults)
        params_used = {
            "window_size": window_size,
            "search_depth": search_depth,
            "top_k": top_k,
            "metric": metric,
            "scale": scale,
            "refine_metric": refine_metric,
            "search_engine": search_engine,
            "ci_alpha": ci_alpha,
            "n_paths": int(futures_matrix.shape[0]),
            "stdev": stdev,
        }
        if secondary_tfs:
            params_used["secondary_timeframes"] = secondary_tfs
        
        # Metrics
        metadata = {
            "method": "analog",
            "components": [primary_tf] + secondary_tfs,
            "params_used": params_used,
            # Just first few analogs from primary for display
            "analogs": [
                {"values": p_futures[i].tolist(), "meta": p_analogs[i]}
                for i in range(min(5, len(p_futures)))
            ],
            "ensemble_metrics": {
                "spread": float(spread),
                "n_paths": int(futures_matrix.shape[0])
            }
        }
        
        return ForecastResult(
            forecast=p50,
            ci_values=(p_lower, p_upper),
            params_used=params_used,
            metadata=metadata
        )
