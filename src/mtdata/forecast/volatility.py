from typing import Any, Dict, Optional, List, Literal
from datetime import datetime
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import json
import math

from ..core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..core.schema import TimeframeLiteral, DenoiseSpec
from ..utils.mt5 import _mt5_epoch_to_utc, _mt5_copy_rates_from, _ensure_symbol_ready
from ..utils.utils import _parse_start_datetime as _parse_start_datetime_util
from .common import default_seasonality as _default_seasonality_period, pd_freq_from_timeframe as _pd_freq_from_timeframe

# Optional availability flags (match server discovery)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False
try:
    import importlib.util as _importlib_util
    _NF_AVAILABLE = _importlib_util.find_spec("neuralforecast") is not None
    _MLF_AVAILABLE = _importlib_util.find_spec("mlforecast") is not None
except Exception:
    _NF_AVAILABLE = False
    _MLF_AVAILABLE = False
try:
    from arch import arch_model as _arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except Exception:
    _ARCH_AVAILABLE = False


# Use shared helpers from common.py for seasonality and pandas freq mapping

def _bars_per_year(timeframe: str) -> int:
    """Approximate number of bars per year for a given timeframe.

    Uses 365 days; for intraday frames computes 365*24*3600 / seconds_per_bar.
    """
    try:
        secs = TIMEFRAME_SECONDS.get(timeframe)
        if not secs or secs <= 0:
            return 0
        return int(round((365.0 * 24.0 * 3600.0) / float(secs)))
    except Exception:
        return 0


def forecast_volatility(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 1,
    method: Literal['ewma','parkinson','gk','rs','yang_zhang','rolling_std','har_rv','garch','egarch','gjr_garch','arima','sarima','ets','theta'] = 'ewma',  # type: ignore
    proxy: Optional[Literal['squared_return','abs_return','log_r2']] = None,  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    as_of: Optional[str] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Forecast volatility over `horizon` bars with direct estimators/GARCH or general forecasters on a proxy.

    Direct: ewma, parkinson, gk, rs, yang_zhang, rolling_std, garch, egarch, gjr_garch.
    General: arima, sarima, ets, theta (require `proxy`: squared_return|abs_return|log_r2).
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}
        method_l = str(method).lower().strip()
        valid_direct = {'ewma','parkinson','gk','rs','yang_zhang','rolling_std','har_rv','garch','egarch','gjr_garch'}
        valid_general = {'arima','sarima','ets','theta'}
        if method_l not in valid_direct.union(valid_general):
            return {"error": f"Invalid method: {method}"}
        if method_l in {'garch','egarch','gjr_garch'} and not _ARCH_AVAILABLE:
            return {"error": f"{method_l} requires 'arch' package."}

        # Parse method params: accept dict, JSON string, or k=v pairs
        __stage = 'parse_params'
        if isinstance(params, dict):
            p = dict(params)
        elif isinstance(params, str):
            s = params.strip()
            if (s.startswith('{') and s.endswith('}')):
                try:
                    p = json.loads(s)
                except Exception:
                    # Fallback to colon or equals pairs within braces
                    p = {}
                    toks = [tok for tok in s.strip().strip('{}').split() if tok]
                    i = 0
                    while i < len(toks):
                        tok = toks[i].strip().strip(',')
                        if not tok:
                            i += 1; continue
                        if '=' in tok:
                            k, v = tok.split('=', 1)
                            p[k.strip()] = v.strip().strip(',')
                            i += 1; continue
                        if tok.endswith(':'):
                            key = tok[:-1].strip()
                            val = ''
                            if i + 1 < len(toks):
                                val = toks[i+1].strip().strip(',')
                                i += 2
                            else:
                                i += 1
                            p[key] = val
                            continue
                        i += 1
            else:
                # Parse simple k=v pairs separated by comma/space
                p = {}
                for tok in s.split():
                    t = tok.strip().strip(',')
                    if '=' in t:
                        k, v = t.split('=', 1)
                        p[k.strip()] = v.strip()
                    # ignore stray tokens without '='
        else:
            p = {}

        # If using general forecasters on proxy, compute proxy series and return using internal logic
        if method_l in valid_general:
            # Fetch recent closes and build returns
            # Reuse unified forecast branch for fetching by delegating to data_fetch_candles/forecast_generate where possible is heavy; implement lightweight here
            # Determine lookback bars
            need = max(300, int(horizon) + 50)
            _info_before = mt5.symbol_info(symbol)
            _was_visible = bool(_info_before.visible) if _info_before is not None else None
            err = _ensure_symbol_ready(symbol)
            if err:
                return {"error": err}
            try:
                if as_of:
                    to_dt = _parse_start_datetime_util(as_of)
                    if not to_dt:
                        return {"error": "Invalid as_of time."}
                    rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, need)
                else:
                    _tick = mt5.symbol_info_tick(symbol)
                    if _tick is not None and getattr(_tick, 'time', None):
                        t_utc = _mt5_epoch_to_utc(float(_tick.time))
                        server_now_dt = datetime.utcfromtimestamp(t_utc)
                    else:
                        server_now_dt = datetime.utcnow()
                    rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, need)
            finally:
                if _was_visible is False:
                    try:
                        mt5.symbol_select(symbol, False)
                    except Exception:
                        pass
            if rates is None or len(rates) < 5:
                return {"error": f"Failed to get sufficient rates for {symbol}: {mt5.last_error()}"}
            df = pd.DataFrame(rates)
            if as_of is None and len(df) >= 2:
                df = df.iloc[:-1]
            if len(df) < 5:
                return {"error": "Not enough closed bars"}
            if denoise:
                _apply_denoise(df, denoise, default_when='pre_ti')
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.diff(np.log(np.maximum(df['close'].astype(float).to_numpy(), 1e-12)))
            r = r[np.isfinite(r)]
            if r.size < 10:
                return {"error": "Insufficient returns to estimate volatility proxy"}
            # Build proxy
            if not proxy:
                return {"error": "General methods require 'proxy' (squared_return|abs_return|log_r2)"}
            proxy_l = str(proxy).lower().strip()
            eps = 1e-12
            if proxy_l == 'squared_return':
                y = r * r; back = 'sqrt'
            elif proxy_l == 'abs_return':
                y = np.abs(r); back = 'abs'
            elif proxy_l == 'log_r2':
                y = np.log(r * r + eps); back = 'exp_sqrt'
            else:
                return {"error": f"Unsupported proxy: {proxy}"}
            y = y[np.isfinite(y)]
            fh = int(horizon)
            # Fit general model
            if method_l in {'arima','sarima'}:
                if not _SM_SARIMAX_AVAILABLE:
                    return {"error": "ARIMA/SARIMA require statsmodels"}
                ord_p = int(p.get('p',1)); ord_d = int(p.get('d',0)); ord_q = int(p.get('q',1))
                if method_l == 'sarima':
                    m = _default_seasonality_period(timeframe)
                    seas = (int(p.get('P',0)), int(p.get('D',0)), int(p.get('Q',0)), int(m) if m>=2 else 0)
                else:
                    seas = (0,0,0,0)
                try:
                    endog = pd.Series(y.astype(float))
                    model = _SARIMAX(endog, order=(ord_p,ord_d,ord_q), seasonal_order=seas, enforce_stationarity=True, enforce_invertibility=True)
                    res = model.fit(method='lbfgs', disp=False, maxiter=100)
                    yhat = res.get_forecast(steps=fh).predicted_mean.to_numpy()
                except Exception as ex:
                    return {"error": f"SARIMAX error: {ex}"}
            elif method_l == 'ets':
                if not _SM_ETS_AVAILABLE:
                    return {"error": "ETS requires statsmodels"}
                try:
                    res = _ETS(y.astype(float), trend=None, seasonal=None, initialization_method='heuristic').fit(optimized=True)
                    yhat = np.asarray(res.forecast(fh), dtype=float)
                except Exception as ex:
                    return {"error": f"ETS error: {ex}"}
            elif method_l == 'theta':  # theta on proxy
                yy = y.astype(float); n=yy.size; tt=np.arange(1,n+1,dtype=float)
                A=np.vstack([np.ones(n),tt]).T; coef,_a,_b,_c = np.linalg.lstsq(A, yy, rcond=None); a=float(coef[0]); b=float(coef[1])
                trend_future = a + b * (tt[-1] + np.arange(1, fh+1, dtype=float))
                alpha = float(p.get('alpha', 0.2)); level=float(yy[0])
                for v in yy[1:]: level = alpha*float(v) + (1.0-alpha)*level
                yhat = 0.5*(trend_future + np.full(fh, level, dtype=float))
            elif method_l == 'mlf_rf':
                if not _MLF_AVAILABLE:
                    return {"error": "mlf_rf requires 'mlforecast' and 'scikit-learn'"}
                try:
                    from mlforecast import MLForecast as _MLForecast  # type: ignore
                    from sklearn.ensemble import RandomForestRegressor as _RF  # type: ignore
                    import pandas as _pd
                except Exception as ex:
                    return {"error": f"Failed to import mlforecast/sklearn: {ex}"}
                try:
                    ts = _pd.to_datetime(df['time'].iloc[1:].astype(float), unit='s', utc=True) if r.size == (len(df)-1) else _pd.date_range(periods=len(y), freq=_pd_freq_from_timeframe(timeframe))
                except Exception:
                    import pandas as _pd
                    ts = _pd.date_range(periods=len(y), freq=_pd_freq_from_timeframe(timeframe))
                Y_df = _pd.DataFrame({'unique_id': ['ts']*int(len(y)), 'ds': _pd.Index(ts).to_pydatetime(), 'y': y.astype(float)})
                lags = p.get('lags') or [1,2,3,4,5]
                try:
                    lags = [int(v) for v in lags]
                except Exception:
                    lags = [1,2,3,4,5]
                rf = _RF(n_estimators=int(p.get('n_estimators', 200)), random_state=42)
                try:
                    mlf = _MLForecast(models=[rf], freq=_pd_freq_from_timeframe(timeframe)).add_lags(lags)
                    mlf.fit(Y_df)
                    Yf = mlf.predict(h=int(fh))
                    try:
                        Yf = Yf[Yf['unique_id']=='ts']
                    except Exception:
                        pass
                    yhat = np.asarray((Yf['y'] if 'y' in Yf.columns else Yf.iloc[:, -1]).to_numpy(), dtype=float)
                except Exception as ex:
                    return {"error": f"mlf_rf error: {ex}"}
            elif method_l == 'nhits':
                if not _NF_AVAILABLE:
                    return {"error": "nhits requires 'neuralforecast[torch]'"}
                try:
                    from neuralforecast import NeuralForecast as _NeuralForecast  # type: ignore
                    from neuralforecast.models import NHITS as _NF_NHITS  # type: ignore
                    import pandas as _pd
                except Exception as ex:
                    return {"error": f"Failed to import neuralforecast: {ex}"}
                max_epochs = int(p.get('max_epochs', 30))
                batch_size = int(p.get('batch_size', 32))
                if p.get('input_size') is not None:
                    input_size = int(p['input_size'])
                else:
                    base = max(64, 96)
                    input_size = int(min(len(y), base))
                try:
                    ts = _pd.to_datetime(df['time'].iloc[1:].astype(float), unit='s', utc=True) if r.size == (len(df)-1) else _pd.date_range(periods=len(y), freq=_pd_freq_from_timeframe(timeframe))
                except Exception:
                    import pandas as _pd
                    ts = _pd.date_range(periods=len(y), freq=_pd_freq_from_timeframe(timeframe))
                Y_df = _pd.DataFrame({'unique_id': ['ts']*int(len(y)), 'ds': _pd.Index(ts).to_pydatetime(), 'y': y.astype(float)})
                model = _NF_NHITS(h=int(fh), input_size=int(input_size), max_epochs=int(max_epochs), batch_size=int(batch_size))
                try:
                    nf = _NeuralForecast(models=[model], freq=_pd_freq_from_timeframe(timeframe))
                    nf.fit(df=Y_df, verbose=False)
                    Yf = nf.predict()
                    try:
                        Yf = Yf[Yf['unique_id']=='ts']
                    except Exception:
                        pass
                    pred_col = None
                    for c in list(Yf.columns):
                        if c not in ('unique_id','ds','y'):
                            pred_col = c
                            if c == 'y_hat':
                                break
                    if pred_col is None:
                        return {"error": "nhits prediction columns not found"}
                    yhat = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
                except Exception as ex:
                    return {"error": f"nhits error: {ex}"}
            else:
                return {"error": f"Unsupported general method for volatility proxy: {method_l}"}
            # Back-transform to per-step sigma and aggregate horizon
            if back == 'sqrt':
                sig = np.sqrt(np.clip(yhat, 0.0, None))
            elif back == 'abs':
                sig = np.maximum(0.0, yhat) * math.sqrt(math.pi/2.0)
            else:
                sig = np.sqrt(np.exp(yhat))
            hsig = float(math.sqrt(np.sum(sig[:fh]**2)))
            # Current sigma (baseline)
            sbar = float(np.std(r[-100:], ddof=0) if r.size>=5 else np.std(r, ddof=0))
            bpy = float(365.0*24.0*3600.0/float(tf_secs))
            return {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "proxy": proxy_l,
                    "horizon": int(horizon), "sigma_bar_return": sbar, "sigma_annual_return": float(sbar*math.sqrt(bpy)),
                    "horizon_sigma_return": hsig, "horizon_sigma_annual": float(hsig*math.sqrt(bpy/max(1,int(horizon)))),
                    "params_used": p}

        # Direct volatility methods
        # Fetch history sized by method
        def _need_bars_direct() -> int:
            if method_l == 'ewma':
                lb = int(p.get('lookback', 1500)); return max(lb + 5, int(horizon) + 5)
            if method_l in {'parkinson','gk','rs','yang_zhang','rolling_std'}:
                w = int(p.get('window', 20)); return max(w + int(horizon) + 10, 60)
            if method_l in {'garch','egarch','gjr_garch'}:
                fb = int(p.get('fit_bars', 2000)); return max(fb + 10, int(horizon) + 10)
            return max(300, int(horizon) + 50)

        need = _need_bars_direct()
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        try:
            if as_of:
                to_dt = _parse_start_datetime_util(as_of)
                if not to_dt:
                    return {"error": "Invalid as_of time."}
                rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, need)
            else:
                _tick = mt5.symbol_info_tick(symbol)
                if _tick is not None and getattr(_tick, 'time', None):
                    t_utc = _mt5_epoch_to_utc(float(_tick.time))
                    server_now_dt = datetime.utcfromtimestamp(t_utc)
                else:
                    server_now_dt = datetime.utcnow()
                rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, need)
        finally:
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass
        if rates is None or len(rates) < 3:
            return {"error": f"Failed to get sufficient rates for {symbol}: {mt5.last_error()}"}

        df = pd.DataFrame(rates)
        if as_of is None and len(df) >= 2:
            df = df.iloc[:-1]
        if len(df) < 3:
            return {"error": "Not enough closed bars"}
        # Denoise defaults for range methods
        if denoise and isinstance(denoise, dict) and method_l in {'parkinson','gk','rs','yang_zhang'}:
            dn2 = dict(denoise); dn2.setdefault('columns', ['open','high','low','close'])
            _apply_denoise(df, dn2, default_when='pre_ti')
        elif denoise:
            _apply_denoise(df, denoise, default_when='pre_ti')

        # Compute returns and helpers
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.diff(np.log(np.maximum(df['close'].astype(float).to_numpy(), 1e-12)))
        r = r[np.isfinite(r)]
        if r.size < 5:
            return {"error": "Insufficient returns to estimate volatility"}
        bpy = float(365.0*24.0*3600.0/float(tf_secs))

        if method_l == 'ewma':
            lb = int(p.get('lookback', 1500)); halflife = p.get('halflife'); lam = p.get('lambda_', 0.94)
            tail = r[-lb:] if r.size >= lb else r
            if halflife is not None:
                try: lam = 1.0 - math.log(2.0) / float(halflife)
                except Exception: lam = 0.94
            lam = float(lam)
            w = np.power(lam, np.arange(len(tail)-1, -1, -1, dtype=float)); w /= float(np.sum(w))
            sigma2 = float(np.sum(w * (tail * tail)))
            sbar = math.sqrt(max(0.0, sigma2))
            hsig = float(sbar * math.sqrt(max(1, int(horizon))))
            return {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                    "sigma_bar_return": sbar, "sigma_annual_return": float(sbar*math.sqrt(bpy)),
                    "horizon_sigma_return": hsig, "horizon_sigma_annual": float(hsig*math.sqrt(bpy/max(1,int(horizon)))),
                    "params_used": {"lookback": lb, "lambda_": lam}}

        if method_l in {'parkinson','gk','rs','yang_zhang','rolling_std'}:
            window = int(p.get('window', 20))
            o = df['open'].astype(float).to_numpy(); h = df['high'].astype(float).to_numpy(); l = df['low'].astype(float).to_numpy(); c = df['close'].astype(float).to_numpy()
            if method_l == 'parkinson':
                v = _parkinson_sigma_sq(h, l)
            elif method_l == 'gk':
                v = _garman_klass_sigma_sq(o, h, l, c)
            elif method_l == 'rs':
                v = _rogers_satchell_sigma_sq(o, h, l, c)
            elif method_l == 'yang_zhang':
                with np.errstate(divide='ignore', invalid='ignore'):
                    oc = np.log(np.maximum(o[1:],1e-12)) - np.log(np.maximum(c[:-1],1e-12))
                    co = np.log(np.maximum(c,1e-12)) - np.log(np.maximum(o,1e-12))
                    rs = (np.log(np.maximum(h,1e-12)) - np.log(np.maximum(c,1e-12))) * (np.log(np.maximum(h,1e-12)) - np.log(np.maximum(o,1e-12))) + \
                         (np.log(np.maximum(l,1e-12)) - np.log(np.maximum(c,1e-12))) * (np.log(np.maximum(l,1e-12)) - np.log(np.maximum(o,1e-12)))
                k = 0.34/(1.34 + (window+1)/(window-1)) if window>1 else 0.34
                co_var = pd.Series(co[1:]).rolling(window=window, min_periods=window).var(ddof=0).to_numpy()
                oc_var = pd.Series(oc).rolling(window=window, min_periods=window).var(ddof=0).to_numpy()
                rs_mean = pd.Series(rs).rolling(window=window, min_periods=window).mean().to_numpy()
                v = (co_var + k*oc_var + (1-k)*rs_mean)
            else:
                v = pd.Series(r*r).rolling(window=window, min_periods=window).mean().to_numpy()
            sigma2 = float(v[-1]) if np.isfinite(v[-1]) else float(np.nanmean(v[-window:]))
            sbar = math.sqrt(max(0.0, sigma2))
            hsig = float(sbar * math.sqrt(max(1, int(horizon))))
            return {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                    "sigma_bar_return": sbar, "sigma_annual_return": float(sbar*math.sqrt(bpy)),
                    "horizon_sigma_return": hsig, "horizon_sigma_annual": float(hsig*math.sqrt(bpy/max(1,int(horizon)))),
                    "params_used": {"window": int(window)}}

        if method_l in {'garch','egarch','gjr_garch'}:
            fit_bars = int(p.get('fit_bars', 2000)); mean_model = str(p.get('mean','Zero')).lower(); dist = str(p.get('dist','normal'))
            r_pct = 100.0 * r
            r_fit = r_pct[-fit_bars:] if r_pct.size > fit_bars else r_pct
            try:
                if method_l == 'egarch':
                    am = _arch_model(r_fit, mean=mean_model if mean_model in ('zero','constant') else 'zero', vol='EGARCH', p=int(p.get('p',1)), q=int(p.get('q',1)), dist=dist)
                elif method_l == 'gjr_garch':
                    am = _arch_model(r_fit, mean=mean_model if mean_model in ('zero','constant') else 'zero', vol='GARCH', p=int(p.get('p',1)), o=int(p.get('o',1)), q=int(p.get('q',1)), dist=dist)
                else:
                    am = _arch_model(r_fit, mean=mean_model if mean_model in ('zero','constant') else 'zero', vol='GARCH', p=1, q=1, dist=dist)
                res = am.fit(disp='off')
                fc = res.forecast(horizon=max(1, int(horizon)), reindex=False)
                variances = fc.variance.values[-1]
                sbar = float(math.sqrt(max(0.0, float(variances[0])))) / 100.0
                hsig = float(math.sqrt(max(0.0, float(np.sum(variances))))) / 100.0
                return {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                        "sigma_bar_return": sbar, "sigma_annual_return": float(sbar*math.sqrt(bpy)),
                        "horizon_sigma_return": hsig, "horizon_sigma_annual": float(hsig*math.sqrt(bpy/max(1,int(horizon)))),
                        "params_used": {k: p[k] for k in p}}
            except Exception as ex:
                return {"error": f"{method_l} error: {ex}"}

        # Ensemble meta-method: aggregate multiple base forecasts
        if method_l == 'ensemble':
            try:
                # Determine default base methods based on availability
                default_methods = ['theta', 'fourier_ols']
                if _SM_ETS_AVAILABLE:
                    default_methods.append('holt')
                # ARIMA/SARIMA can be added by user explicitly to avoid latency by default

                base_methods_in = p.get('methods')
                if isinstance(base_methods_in, (list, tuple)) and base_methods_in:
                    base_methods = [str(m).lower().strip() for m in base_methods_in]
                else:
                    base_methods = list(default_methods)
                # Remove invalid or recursive entries
                base_methods = [m for m in base_methods if m in _FORECAST_METHODS and m != 'ensemble']
                # Deduplicate while preserving order
                seen = set()
                base_methods = [m for m in base_methods if not (m in seen or seen.add(m))]
                if not base_methods:
                    return {"error": "Ensemble requires at least one valid base method"}

                aggregator = str(p.get('aggregator', 'mean')).lower()
                weights = p.get('weights')
                expose_components = bool(p.get('expose_components', True))

                # Normalize weights if provided
                w = None
                if isinstance(weights, (list, tuple)) and len(weights) == len(base_methods):
                    try:
                        w_arr = np.array([float(x) for x in weights], dtype=float)
                        if np.all(np.isfinite(w_arr)) and np.any(w_arr > 0):
                            w = w_arr.clip(min=0)
                            s = float(np.sum(w))
                            if s > 0:
                                w = w / s
                            else:
                                w = None
                        else:
                            w = None
                    except Exception:
                        w = None

                comp_results = []
                for bm in base_methods:
                    try:
                        # Pass through common args; avoid per-method params for MVP simplicity
                        r = forecast(
                            symbol=symbol,
                            timeframe=timeframe,
                            method=bm,  # type: ignore
                            horizon=horizon,
                            lookback=lookback,
                            as_of=as_of,
                            params=None,
                            ci_alpha=ci_alpha,
                            target=target,  # type: ignore
                            denoise=denoise,
                            timezone=timezone,
                        )
                        if isinstance(r, dict) and r.get('success') and r.get('forecast_price'):
                            comp_results.append((bm, r))
                    except Exception:
                        continue

                if not comp_results:
                    return {"error": "Ensemble failed: no successful base forecasts"}

                # Establish reference horizon and timestamps from first successful component
                first_method, first_res = comp_results[0]
                ref_prices = np.array(first_res.get('forecast_price', []), dtype=float)
                fh = int(len(ref_prices))
                if fh <= 0:
                    return {"error": "Ensemble failed: empty forecast from base methods"}

                # Collect aligned component arrays; drop any mismatched lengths
                comps_prices = []
                comps_returns = []
                lower_list = []
                upper_list = []
                used_methods = []
                for bm, r in comp_results:
                    fp = r.get('forecast_price')
                    if not isinstance(fp, (list, tuple)) or len(fp) != fh:
                        continue
                    used_methods.append(bm)
                    comps_prices.append(np.array(fp, dtype=float))
                    fr = r.get('forecast_return')
                    if isinstance(fr, (list, tuple)) and len(fr) == fh:
                        comps_returns.append(np.array(fr, dtype=float))
                    lp = r.get('lower_price'); up = r.get('upper_price')
                    if isinstance(lp, (list, tuple)) and isinstance(up, (list, tuple)) and len(lp) == fh and len(up) == fh:
                        lower_list.append(np.array(lp, dtype=float))
                        upper_list.append(np.array(up, dtype=float))

                if len(comps_prices) == 0:
                    return {"error": "Ensemble failed: no aligned component forecasts"}

                M = len(comps_prices)
                # Choose weights
                if aggregator == 'weighted' and w is not None and len(w) == M:
                    w_use = np.array(w, dtype=float)
                else:
                    w_use = np.full(M, 1.0 / M, dtype=float)
                    aggregator = 'mean' if aggregator == 'weighted' else aggregator

                X = np.vstack(comps_prices)  # shape (M, fh)
                if aggregator == 'median':
                    agg_price = np.median(X, axis=0)
                else:  # mean or default
                    agg_price = np.average(X, axis=0, weights=w_use)

                # Aggregate returns if all components provided them; otherwise skip
                if len(comps_returns) == M:
                    XR = np.vstack(comps_returns)
                    if aggregator == 'median':
                        agg_return = np.median(XR, axis=0)
                    else:
                        agg_return = np.average(XR, axis=0, weights=w_use)
                else:
                    agg_return = None

                # Aggregate confidence intervals only if all components have them
                if len(lower_list) == M and len(upper_list) == M:
                    L = np.vstack(lower_list)
                    U = np.vstack(upper_list)
                    if aggregator == 'median':
                        agg_lower = np.median(L, axis=0)
                        agg_upper = np.median(U, axis=0)
                    else:
                        agg_lower = np.average(L, axis=0, weights=w_use)
                        agg_upper = np.average(U, axis=0, weights=w_use)
                else:
                    agg_lower = None
                    agg_upper = None

                # Build payload using the first component as template for metadata/times
                payload: Dict[str, Any] = {
                    "success": True,
                    "symbol": first_res.get('symbol', symbol),
                    "timeframe": first_res.get('timeframe', timeframe),
                    "method": "ensemble",
                    "target": first_res.get('target', str(target)),
                    "params_used": {
                        "base_methods": used_methods,
                        "aggregator": aggregator,
                        "weights": [float(x) for x in (w_use.tolist() if isinstance(w_use, np.ndarray) else [])],
                    },
                    "lookback_used": int(first_res.get('lookback_used', 0)),
                    "horizon": int(first_res.get('horizon', horizon)),
                    "seasonality_period": int(first_res.get('seasonality_period', 0)),
                    "as_of": first_res.get('as_of', as_of or None),
                    "train_start": first_res.get('train_start'),
                    "train_end": first_res.get('train_end'),
                    "times": first_res.get('times'),
                    "forecast_price": [float(v) for v in agg_price.tolist()],
                }
                # Timezone flag passthrough if present
                if 'timezone' in first_res:
                    payload['timezone'] = first_res.get('timezone')
                # Trend: reuse first component's for simplicity
                if 'forecast_trend' in first_res:
                    payload['forecast_trend'] = first_res.get('forecast_trend')
                if agg_return is not None:
                    payload['forecast_return'] = [float(v) for v in agg_return.tolist()]
                if agg_lower is not None and agg_upper is not None and ci_alpha is not None:
                    payload['lower_price'] = [float(v) for v in agg_lower.tolist()]
                    payload['upper_price'] = [float(v) for v in agg_upper.tolist()]
                    payload['ci_alpha'] = float(ci_alpha)

                if expose_components:
                    comps_out = []
                    for i, (bm, r) in enumerate(comp_results):
                        try:
                            comps_out.append({
                                "method": bm,
                                "weight": float(w_use[i]) if i < len(w_use) else float(1.0 / M),
                                "forecast_price": r.get('forecast_price'),
                            })
                        except Exception:
                            continue
                    payload['components'] = comps_out

                return payload
            except Exception as ex:
                return {"error": f"Error computing ensemble forecast: {ex}"}
        # Backward compatibility for 'lambda' -> 'lambda_'
        if 'lambda' in p and 'lambda_' not in p:
            p['lambda_'] = p['lambda']

        # Determine bars required
        if method_l == 'ewma':
            lookback = int(p.get('lookback', 1500))
            need = max(lookback + 2, 100)
        elif method_l in ('parkinson','gk','rs'):
            window = int(p.get('window', 20))
            need = max(window + 2, 50)
        else:  # garch
            fit_bars = int(p.get('fit_bars', 2000))
            need = max(fit_bars + 2, 500)

        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}

        try:
            # Use explicit as-of time if provided, else server time for alignment
            if as_of:
                to_dt = _parse_start_datetime_util(as_of)
                if not to_dt:
                    return {"error": "Invalid as_of_datetime. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, need)
            else:
                _tick = mt5.symbol_info_tick(symbol)
                if _tick is not None and getattr(_tick, 'time', None):
                    t_utc = _mt5_epoch_to_utc(float(_tick.time))
                    server_now_dt = datetime.utcfromtimestamp(t_utc)
                else:
                    server_now_dt = datetime.utcnow()
                rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, need)
        finally:
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass

        if rates is None or len(rates) < 3:
            return {"error": f"Failed to get sufficient rates for {symbol}: {mt5.last_error()}"}

        df = pd.DataFrame(rates)
        # Drop forming last bar only when using current 'now' as anchor; keep all for historical as_of
        if as_of is None and len(df) >= 2:
            df = df.iloc[:-1]
        if len(df) < 3:
            return {"error": "Not enough closed bars to compute volatility"}

        # Optionally denoise relevant columns prior to volatility estimation
        __stage = 'denoise'
        if denoise:
            try:
                # Default columns by method if user didn't provide
                _spec = dict(denoise)
                if 'columns' not in _spec or not _spec.get('columns'):
                    if method_l in ('ewma', 'garch'):
                        _spec['columns'] = ['close']
                    else:  # range-based estimators rely on OHLC
                        _spec['columns'] = ['open', 'high', 'low', 'close']
                _apply_denoise(df, _spec, default_when='pre_ti')
            except Exception:
                # Fail-safe: ignore denoise errors and proceed with raw data
                pass

        # Prefer denoised columns if user asked to keep originals
        def _col(name: str) -> str:
            dn = f"{name}_dn"
            return dn if dn in df.columns else name

        closes = df[_col('close')].to_numpy(dtype=float)
        highs = df[_col('high')].to_numpy(dtype=float) if 'high' in df.columns else None
        lows = df[_col('low')].to_numpy(dtype=float) if 'low' in df.columns else None
        opens = df[_col('open')].to_numpy(dtype=float) if 'open' in df.columns else None
        last_close = float(closes[-1])

        bars_per_year = _bars_per_year(timeframe)
        ann_factor = math.sqrt(bars_per_year) if bars_per_year > 0 else float('nan')

        sigma_bar = float('nan')
        sigma_ann = float('nan')
        sigma_h_bar = float('nan')  # horizon sigma of sum of returns over k bars
        params_used: Dict[str, Any] = {}

        if method_l == 'ewma':
            r = _log_returns_from_closes(closes)
            if r.size < 5:
                return {"error": "Not enough return observations for EWMA"}
            lam = p.get('lambda_')
            hl = p.get('halflife')
            if hl is not None:
                try:
                    hl = float(hl)
                except Exception:
                    hl = None
            if lam is not None:
                try:
                    lam = float(lam)
                except Exception:
                    lam = None
            if lam is not None and (hl is None):
                alpha = 1.0 - float(lam)
                var_series = pd.Series(r).ewm(alpha=alpha, adjust=False).var(bias=False)
                params_used['lambda'] = float(lam)
            else:
                # default halflife if not provided
                if hl is None:
                    # heuristic per timeframe
                    default_hl = 60 if timeframe.startswith('H') else (11 if timeframe == 'D1' else 180)
                    hl = float(p.get('halflife', default_hl))
                var_series = pd.Series(r).ewm(halflife=float(hl), adjust=False).var(bias=False)
                params_used['halflife'] = float(hl)
            v = float(var_series.iloc[-1])
            v = v if math.isfinite(v) and v >= 0 else float('nan')
            sigma_bar = math.sqrt(v) if math.isfinite(v) and v >= 0 else float('nan')
            sigma_h_bar = math.sqrt(max(1, int(horizon)) * v) if math.isfinite(v) and v >= 0 else float('nan')
        elif method_l in ('parkinson','gk','rs'):
            if highs is None or lows is None:
                return {"error": "High/Low data required for range-based estimators"}
            window = int(p.get('window', 20))
            params_used['window'] = window
            if method_l == 'parkinson':
                var_bars = _parkinson_sigma_sq(highs, lows)
            elif method_l == 'gk':
                if opens is None:
                    return {"error": "Open data required for Garman–Klass"}
                var_bars = _garman_klass_sigma_sq(opens, highs, lows, closes)
            else:  # rs
                if opens is None:
                    return {"error": "Open data required for Rogers–Satchell"}
                var_bars = _rogers_satchell_sigma_sq(opens, highs, lows, closes)
            s = pd.Series(var_bars)
            v = float(s.tail(window).mean(skipna=True))
            v = v if math.isfinite(v) and v >= 0 else float('nan')
            sigma_bar = math.sqrt(v) if math.isfinite(v) and v >= 0 else float('nan')
            sigma_h_bar = math.sqrt(max(1, int(horizon)) * v) if math.isfinite(v) and v >= 0 else float('nan')
        elif method_l == 'har_rv':
            # HAR-RV on daily realized variance computed from intraday returns
            try:
                rv_tf = str(p.get('rv_timeframe', 'M5')).upper()
                rv_mt5_tf = TIMEFRAME_MAP.get(rv_tf)
                if rv_mt5_tf is None:
                    return {"error": f"Invalid rv_timeframe: {rv_tf}"}
                days = int(p.get('days', 120))
                w = int(p.get('window_w', 5))
                m = int(p.get('window_m', 22))
                rv_tf_secs = TIMEFRAME_SECONDS.get(rv_tf, 300)
                bars_needed = int(days * max(1, (86400 // max(1, rv_tf_secs))) + 50)
                _info_before = mt5.symbol_info(symbol)
                _was_visible = bool(_info_before.visible) if _info_before is not None else None
                err = _ensure_symbol_ready(symbol)
                if err:
                    return {"error": err}
                try:
                    if as_of:
                        to_dt = _parse_start_datetime_util(as_of)
                        if not to_dt:
                            return {"error": "Invalid as_of time."}
                        rates_rv = _mt5_copy_rates_from(symbol, rv_mt5_tf, to_dt, bars_needed)
                    else:
                        _tick = mt5.symbol_info_tick(symbol)
                        if _tick is not None and getattr(_tick, 'time', None):
                            t_utc = _mt5_epoch_to_utc(float(_tick.time))
                            server_now_dt = datetime.utcfromtimestamp(t_utc)
                        else:
                            server_now_dt = datetime.utcnow()
                        rates_rv = _mt5_copy_rates_from(symbol, rv_mt5_tf, server_now_dt, bars_needed)
                finally:
                    if _was_visible is False:
                        try:
                            mt5.symbol_select(symbol, False)
                        except Exception:
                            pass
                if rates_rv is None or len(rates_rv) < 50:
                    return {"error": f"Failed to get intraday rates for RV: {mt5.last_error()}"}
                dfrv = pd.DataFrame(rates_rv)
                if as_of is None and len(dfrv) >= 2:
                    dfrv = dfrv.iloc[:-1]
                c = dfrv['close'].astype(float).to_numpy()
                if c.size < 10:
                    return {"error": "Insufficient intraday bars for RV"}
                with np.errstate(divide='ignore', invalid='ignore'):
                    rr = np.diff(np.log(np.maximum(c, 1e-12)))
                rr = rr[np.isfinite(rr)]
                dt = pd.to_datetime(dfrv['time'].iloc[1:].astype(float), unit='s', utc=True)
                days_idx = pd.DatetimeIndex(dt).floor('D')
                df_r = pd.DataFrame({'day': days_idx, 'r2': rr * rr})
                daily_rv = df_r.groupby('day')['r2'].sum().astype(float)
                if len(daily_rv) < max(30, m + 5):
                    return {"error": "Not enough daily RV observations for HAR-RV"}
                RV = daily_rv.to_numpy(dtype=float)
                N = RV.size
                # Lagged features
                Dlag = RV[:-1]
                def rmean(arr, k):
                    s = pd.Series(arr)
                    return s.rolling(window=k, min_periods=k).mean().to_numpy()
                Wlag_full = rmean(RV, w)  # aligned to current index
                Mlag_full = rmean(RV, m)
                # Build design for y=RV[1:]
                y = RV[1:]
                Wlag = Wlag_full[:-1]
                Mlag = Mlag_full[:-1]
                Xd = Dlag
                mask = np.isfinite(Xd) & np.isfinite(Wlag) & np.isfinite(Mlag) & np.isfinite(y)
                X = np.vstack([np.ones_like(Xd[mask]), Xd[mask], Wlag[mask], Mlag[mask]]).T
                yv = y[mask]
                if X.shape[0] < 20:
                    return {"error": "Insufficient samples after alignment for HAR-RV"}
                beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
                # Next-day forecast from last lags
                D_last = RV[-1]
                W_last = float(pd.Series(RV).tail(w).mean())
                M_last = float(pd.Series(RV).tail(m).mean())
                rv_next = float(beta[0] + beta[1]*D_last + beta[2]*W_last + beta[3]*M_last)
                rv_next = max(0.0, rv_next)
                # Map to per-bar and horizon sigma for requested timeframe
                tf_secs = TIMEFRAME_SECONDS.get(timeframe)
                if not tf_secs:
                    return {"error": f"Unsupported timeframe seconds for {timeframe}"}
                bars_per_day = float(86400.0 / float(tf_secs))
                sbar = float(math.sqrt(rv_next / bars_per_day))
                h_days = float(int(horizon)) / bars_per_day
                hsig = float(math.sqrt(rv_next * max(h_days, 0.0)))
                bpy = float(365.0 * 24.0 * 3600.0 / float(tf_secs))
                return {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                        "sigma_bar_return": sbar, "sigma_annual_return": float(sbar*math.sqrt(bpy)),
                        "horizon_sigma_return": hsig, "horizon_sigma_annual": float(hsig*math.sqrt(bpy/max(1,int(horizon)))),
                        "params_used": {"rv_timeframe": rv_tf, "window_w": w, "window_m": m,
                                         "beta": [float(b) for b in beta.tolist()],
                                         "days": days}}
            except Exception as ex:
                return {"error": f"HAR-RV error: {ex}"}
        else:  # garch
            r = _log_returns_from_closes(closes)
            if r.size < 100:
                return {"error": "Not enough return observations for GARCH (need >=100)"}
            fit_bars = int(p.get('fit_bars', min(2000, r.size)))
            mean_model = str(p.get('mean', 'Zero'))
            dist = str(p.get('dist', 'normal'))
            params_used.update({'fit_bars': fit_bars, 'mean': mean_model, 'dist': dist})
            r_fit = pd.Series(r[-fit_bars:]) * 100.0  # scale to percent
            try:
                am = _arch_model(r_fit, mean=mean_model.lower(), vol='GARCH', p=1, q=1, dist=dist)
                res = am.fit(disp='off')
                # Current conditional variance (percent^2)
                cond_vol = float(res.conditional_volatility.iloc[-1])  # percent
                sigma_bar = cond_vol / 100.0
                # k-step ahead variance forecasts (percent^2)
                fc = res.forecast(horizon=max(1, int(horizon)), reindex=False)
                var_path = np.array(fc.variance.iloc[-1].values, dtype=float)  # shape (horizon,)
                var_sum = float(np.nansum(var_path))  # percent^2
                sigma_h_bar = math.sqrt(var_sum) / 100.0
            except Exception as ex:
                return {"error": f"GARCH fitting error: {ex}"}

        sigma_ann = sigma_bar * ann_factor if math.isfinite(sigma_bar) and math.isfinite(ann_factor) else float('nan')
        sigma_h_ann = sigma_h_bar * ann_factor if math.isfinite(sigma_h_bar) and math.isfinite(ann_factor) else float('nan')

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_l,
            "params_used": params_used,
            "bars_used": int(len(df)),
            "horizon": int(horizon),
            "last_close": last_close,
            "sigma_bar_return": sigma_bar,
            "sigma_annual_return": sigma_ann,
            "horizon_sigma_return": sigma_h_bar,
            "horizon_sigma_annual": sigma_h_ann,
            "as_of": as_of or None,
        }
    except Exception as e:
        return {"error": f"Error computing volatility forecast: {str(e)}"}

