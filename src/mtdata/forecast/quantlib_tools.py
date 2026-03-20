"""QuantLib-based pricing and calibration helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import datetime as _dt
import math

import numpy as np

from ..services.options_service import get_options_chain


def price_barrier_option_quantlib(
    *,
    spot: float,
    strike: float,
    barrier: float,
    maturity_days: int,
    option_type: str = "call",
    barrier_type: str = "up_out",
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    volatility: float = 0.2,
    rebate: float = 0.0,
) -> Dict[str, Any]:
    """Price a European barrier option with QuantLib."""
    try:
        import QuantLib as ql
    except Exception as ex:
        return {"error": f"QuantLib is required: {ex}"}

    try:
        spot_val = float(spot)
        strike_val = float(strike)
        barrier_val = float(barrier)
        maturity_val = int(maturity_days)
        rf = float(risk_free_rate)
        div = float(dividend_yield)
        vol = float(volatility)
        rebate_val = float(rebate)
    except Exception:
        return {"error": "Invalid numeric input for barrier pricing"}

    if not (
        spot_val > 0
        and strike_val > 0
        and barrier_val > 0
        and maturity_val > 0
        and vol > 0
    ):
        return {
            "error": "spot/strike/barrier/maturity_days/volatility must be positive"
        }

    option_type_norm = str(option_type).strip().lower()
    barrier_type_norm = str(barrier_type).strip().lower()
    opt_map = {"call": ql.Option.Call, "put": ql.Option.Put}
    barrier_map = {
        "up_in": ql.Barrier.UpIn,
        "up_out": ql.Barrier.UpOut,
        "down_in": ql.Barrier.DownIn,
        "down_out": ql.Barrier.DownOut,
    }
    if option_type_norm not in opt_map:
        return {"error": f"Invalid option_type: {option_type}. Use call|put."}
    if barrier_type_norm not in barrier_map:
        return {
            "error": f"Invalid barrier_type: {barrier_type}. Use up_in|up_out|down_in|down_out."
        }

    ql_today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = ql_today
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    maturity = ql_today + int(maturity_val)

    payoff = ql.PlainVanillaPayoff(opt_map[option_type_norm], float(strike_val))
    exercise = ql.EuropeanExercise(maturity)
    barrier_opt = ql.BarrierOption(
        barrier_map[barrier_type_norm],
        float(barrier_val),
        float(rebate_val),
        payoff,
        exercise,
    )

    def _price_with(spot_local: float, vol_local: float) -> float:
        spot_h = ql.QuoteHandle(ql.SimpleQuote(float(spot_local)))
        rf_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(ql_today, float(rf), day_count)
        )
        div_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(ql_today, float(div), day_count)
        )
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(ql_today, calendar, float(vol_local), day_count)
        )
        process = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)
        barrier_opt.setPricingEngine(ql.AnalyticBarrierEngine(process))
        return float(barrier_opt.NPV())

    try:
        npv = _price_with(spot_val, vol)
    except Exception as ex:
        return {"error": f"QuantLib pricing failed: {ex}"}

    delta = float("nan")
    gamma = float("nan")
    vega = float("nan")
    try:
        eps_s = max(1e-4, abs(spot_val) * 1e-2)
        p_up = _price_with(spot_val + eps_s, vol)
        p_dn = _price_with(max(1e-8, spot_val - eps_s), vol)
        delta = (p_up - p_dn) / (2.0 * eps_s)
        gamma = (p_up - 2.0 * npv + p_dn) / (eps_s * eps_s)
        eps_v = max(1e-4, abs(vol) * 5e-2)
        pv_up = _price_with(spot_val, vol + eps_v)
        pv_dn = _price_with(spot_val, max(1e-6, vol - eps_v))
        vega = (pv_up - pv_dn) / (2.0 * eps_v)
    except Exception:
        pass

    return {
        "success": True,
        "price": float(npv),
        "delta": float(delta) if math.isfinite(delta) else None,
        "gamma": float(gamma) if math.isfinite(gamma) else None,
        "vega": float(vega) if math.isfinite(vega) else None,
        "params_used": {
            "spot": float(spot_val),
            "strike": float(strike_val),
            "barrier": float(barrier_val),
            "maturity_days": int(maturity_val),
            "option_type": option_type_norm,
            "barrier_type": barrier_type_norm,
            "risk_free_rate": float(rf),
            "dividend_yield": float(div),
            "volatility": float(vol),
            "rebate": float(rebate_val),
        },
    }


def calibrate_heston_quantlib_from_options(
    *,
    symbol: str,
    expiration: Optional[str] = None,
    option_type: str = "call",
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    min_open_interest: int = 0,
    min_volume: int = 0,
    max_contracts: int = 25,
) -> Dict[str, Any]:
    """Calibrate a Heston model from option-chain implied vols using QuantLib."""
    try:
        import QuantLib as ql
    except Exception as ex:
        return {"error": f"QuantLib is required: {ex}"}

    side = str(option_type or "call").strip().lower()
    if side not in {"call", "put", "both"}:
        return {"error": f"Invalid option_type: {option_type}. Use call|put|both."}

    chain = get_options_chain(
        symbol=symbol,
        expiration=expiration,
        option_type=side,
        min_open_interest=int(min_open_interest),
        min_volume=int(min_volume),
        limit=max(50, int(max_contracts) * 6),
    )
    if isinstance(chain, dict) and chain.get("error"):
        return chain

    contracts = chain.get("options", []) if isinstance(chain, dict) else []
    if not isinstance(contracts, list):
        contracts = []

    spot_val = float(chain.get("underlying_price", float("nan")))
    if not (spot_val == spot_val and spot_val > 0):
        return {"error": "Underlying spot price unavailable from options provider."}

    rows: List[Dict[str, Any]] = []
    for row in contracts:
        if not isinstance(row, dict):
            continue
        strike = float(row.get("strike", float("nan")))
        iv = float(row.get("implied_volatility", float("nan")))
        if not (
            np.isfinite(strike) and strike > 0 and np.isfinite(iv) and 0.01 <= iv <= 5.0
        ):
            continue
        rows.append({"strike": strike, "iv": iv, "side": row.get("side")})

    if len(rows) < 5:
        return {
            "error": "Need at least 5 contracts with valid implied volatility for Heston calibration."
        }

    rows.sort(key=lambda x: abs(float(x["strike"]) - spot_val))
    rows = rows[: max(5, int(max_contracts))]
    expiry_text = str(chain.get("expiration") or "")
    if not expiry_text:
        return {"error": "Options expiration date missing from chain output."}
    try:
        expiry_date = _dt.datetime.strptime(expiry_text, "%Y-%m-%d").date()
    except Exception:
        return {"error": f"Invalid expiration format: {expiry_text}"}
    today = _dt.date.today()
    days_to_expiry = max(1, int((expiry_date - today).days))

    ql_today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = ql_today
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    rf_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(ql_today, float(risk_free_rate), day_count)
    )
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(ql_today, float(dividend_yield), day_count)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot_val)))

    ivs = np.asarray([float(r["iv"]) for r in rows], dtype=float)
    theta0 = float(max(1e-6, np.median(ivs) ** 2))
    v0_0 = float(theta0)
    kappa0 = 1.0
    sigma0 = float(max(0.05, np.std(ivs) * 2.0))
    rho0 = -0.5

    process = ql.HestonProcess(
        rf_ts, div_ts, spot_handle, v0_0, kappa0, theta0, sigma0, rho0
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    helpers: List[Any] = []
    maturity = ql.Period(int(days_to_expiry), ql.Days)
    for row in rows:
        helper = ql.HestonModelHelper(
            maturity,
            calendar,
            float(spot_val),
            float(row["strike"]),
            ql.QuoteHandle(ql.SimpleQuote(float(row["iv"]))),
            rf_ts,
            div_ts,
            ql.BlackCalibrationHelper.ImpliedVolError,
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    try:
        method = ql.LevenbergMarquardt()
        end_criteria = ql.EndCriteria(500, 100, 1e-8, 1e-8, 1e-8)
        model.calibrate(helpers, method, end_criteria)
    except Exception as ex:
        return {"error": f"QuantLib Heston calibration failed: {ex}"}

    errors = [float(h.calibrationError()) for h in helpers]
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")

    return {
        "success": True,
        "symbol": str(symbol).upper().strip(),
        "expiration": expiry_text,
        "days_to_expiry": int(days_to_expiry),
        "contracts_used": int(len(rows)),
        "spot": float(spot_val),
        "calibration_error_rmse": float(rmse) if np.isfinite(rmse) else None,
        "params": {
            "kappa": float(model.kappa()),
            "theta": float(model.theta()),
            "sigma": float(model.sigma()),
            "rho": float(model.rho()),
            "v0": float(model.v0()),
        },
        "risk_free_rate": float(risk_free_rate),
        "dividend_yield": float(dividend_yield),
        "sample_contracts": rows[:10],
    }
