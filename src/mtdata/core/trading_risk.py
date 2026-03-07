"""Trading risk analysis."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ._mcp_instance import mcp
from ..utils.mt5 import _auto_connect_wrapper, mt5_adapter


@mcp.tool()
def trade_risk_analyze(
    symbol: Optional[str] = None,
    desired_risk_pct: Optional[float] = None,
    proposed_entry: Optional[float] = None,
    proposed_sl: Optional[float] = None,
    proposed_tp: Optional[float] = None,
) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
    def _analyze_risk():
        try:
            account = mt5.account_info()
            if account is None:
                return {"error": "Failed to get account info"}

            equity = float(account.equity)
            currency = account.currency
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if positions is None:
                positions = []

            position_risks: List[Dict[str, Any]] = []
            total_risk_currency = 0.0
            positions_without_sl = 0
            total_notional_exposure = 0.0

            for pos in positions:
                try:
                    sym_info = mt5.symbol_info(pos.symbol)
                    if sym_info is None:
                        continue

                    entry_price = float(pos.price_open)
                    sl_price = float(pos.sl) if pos.sl and pos.sl > 0 else None
                    tp_price = float(pos.tp) if pos.tp and pos.tp > 0 else None
                    volume = float(pos.volume)

                    contract_size = float(sym_info.trade_contract_size)
                    point = float(getattr(sym_info, "point", 0.0) or 0.0)
                    tick_value = float(getattr(sym_info, "trade_tick_value", 0.0) or 0.0)
                    tick_size = float(getattr(sym_info, "trade_tick_size", 0.0) or 0.0)
                    if not math.isfinite(tick_size) or tick_size <= 0:
                        tick_size = point if math.isfinite(point) and point > 0 else 0.0
                    tick_value_valid = math.isfinite(tick_value) and tick_value > 0
                    if not math.isfinite(contract_size) or contract_size <= 0:
                        contract_size = 1.0

                    notional_value = abs(volume) * contract_size * entry_price
                    total_notional_exposure += notional_value

                    risk_currency = None
                    risk_pct = None
                    reward_currency = None
                    rr_ratio = None
                    risk_status = "undefined"

                    if sl_price and tick_size > 0 and tick_value_valid:
                        risk_ticks = (entry_price - sl_price) / tick_size if pos.type == 0 else (sl_price - entry_price) / tick_size
                        risk_currency = abs(risk_ticks * tick_value * volume)
                        risk_pct = (risk_currency / equity) * 100.0 if equity > 0 else 0.0
                        total_risk_currency += risk_currency
                        risk_status = "defined"

                        if tp_price:
                            reward_ticks = (tp_price - entry_price) / tick_size if pos.type == 0 else (entry_price - tp_price) / tick_size
                            reward_currency = abs(reward_ticks * tick_value * volume)
                            if risk_currency > 0:
                                rr_ratio = reward_currency / risk_currency
                    elif sl_price:
                        risk_status = "undefined"
                    else:
                        positions_without_sl += 1
                        risk_status = "unlimited"

                    position_risks.append(
                        {
                            "ticket": pos.ticket,
                            "symbol": pos.symbol,
                            "type": "BUY" if pos.type == 0 else "SELL",
                            "volume": volume,
                            "entry": entry_price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "risk_currency": round(risk_currency, 2) if risk_currency else None,
                            "risk_pct": round(risk_pct, 2) if risk_pct else None,
                            "risk_status": risk_status,
                            "notional_value": round(notional_value, 2),
                            "reward_currency": round(reward_currency, 2) if reward_currency else None,
                            "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
                        }
                    )
                except Exception:
                    continue

            total_risk_pct = (total_risk_currency / equity) * 100.0 if equity > 0 else 0.0
            notional_exposure_pct = (total_notional_exposure / equity) * 100.0 if equity > 0 else 0.0

            overall_risk_status = "defined"
            if positions_without_sl > 0:
                overall_risk_status = "unlimited"
            elif total_risk_pct > 10:
                overall_risk_status = "high"
            elif total_risk_pct > 5:
                overall_risk_status = "moderate"
            else:
                overall_risk_status = "low"

            result: Dict[str, Any] = {
                "success": True,
                "account": {"equity": round(equity, 2), "currency": currency},
                "portfolio_risk": {
                    "overall_risk_status": overall_risk_status,
                    "total_risk_currency": round(total_risk_currency, 2),
                    "total_risk_pct": round(total_risk_pct, 2),
                    "positions_count": len(position_risks),
                    "positions_without_sl": positions_without_sl,
                    "notional_exposure": round(total_notional_exposure, 2),
                    "notional_exposure_pct": round(notional_exposure_pct, 2),
                },
                "positions": position_risks,
            }
            if positions_without_sl > 0:
                result["warning"] = f"{positions_without_sl} position(s) without stop loss - UNLIMITED RISK!"

            if desired_risk_pct is not None and proposed_entry is not None and proposed_sl is not None:
                if not symbol:
                    return {"error": "symbol is required for position sizing"}

                sym_info = mt5.symbol_info(symbol)
                if sym_info is None:
                    return {"error": f"Symbol {symbol} not found"}

                contract_size = float(sym_info.trade_contract_size)
                point = float(getattr(sym_info, "point", 0.0) or 0.0)
                tick_value = float(getattr(sym_info, "trade_tick_value", 0.0) or 0.0)
                tick_size = float(getattr(sym_info, "trade_tick_size", 0.0) or 0.0)
                if not math.isfinite(tick_size) or tick_size <= 0:
                    tick_size = point if math.isfinite(point) and point > 0 else 0.0
                min_volume = float(sym_info.volume_min)
                max_volume = float(sym_info.volume_max)
                volume_step = float(sym_info.volume_step)
                if not (math.isfinite(tick_value) and tick_value > 0 and math.isfinite(tick_size) and tick_size > 0):
                    result["position_sizing_error"] = "Symbol tick configuration is invalid for risk sizing"
                    return result
                if not (math.isfinite(volume_step) and volume_step > 0):
                    volume_step = max(min_volume, 0.01)
                if not math.isfinite(contract_size) or contract_size <= 0:
                    contract_size = 1.0

                risk_amount = equity * (desired_risk_pct / 100.0)
                sl_distance_ticks = abs(proposed_entry - proposed_sl) / tick_size
                if sl_distance_ticks > 0:
                    raw_volume = risk_amount / (sl_distance_ticks * tick_value)
                    if not math.isfinite(raw_volume) or raw_volume <= 0:
                        result["position_sizing_error"] = "Calculated volume is invalid"
                        return result

                    volume_steps = math.floor((raw_volume / volume_step) + 1e-12)
                    suggested_volume = volume_steps * volume_step
                    rounding_mode = "rounded_down_to_step"
                    sizing_notes: List[str] = []

                    if suggested_volume < min_volume:
                        suggested_volume = min_volume
                        rounding_mode = "clamped_to_min_volume"
                        sizing_notes.append("Minimum trade volume forces the size up to the broker minimum.")
                    elif suggested_volume > max_volume:
                        suggested_volume = max_volume
                        rounding_mode = "clamped_to_max_volume"
                        sizing_notes.append("Maximum trade volume caps the size below the unconstrained target.")
                    elif suggested_volume < raw_volume:
                        sizing_notes.append("Volume was rounded down to the nearest broker step to avoid exceeding requested risk.")

                    step_txt = f"{volume_step:.10f}".rstrip("0")
                    step_decimals = len(step_txt.split(".")[1]) if "." in step_txt else 0
                    suggested_volume = float(f"{suggested_volume:.{step_decimals}f}") if step_decimals > 0 else float(round(suggested_volume))

                    actual_risk = sl_distance_ticks * tick_value * suggested_volume
                    actual_risk_pct = (actual_risk / equity) * 100.0
                    risk_pct_diff = actual_risk_pct - float(desired_risk_pct)
                    risk_over_target = actual_risk_pct > (float(desired_risk_pct) + 1e-9)
                    overshoot_pct = max(0.0, float(actual_risk_pct) - float(desired_risk_pct))
                    overshoot_currency = max(0.0, float(actual_risk) - float(risk_amount))
                    overshoot_reason = None
                    if risk_over_target:
                        if rounding_mode == "clamped_to_min_volume":
                            overshoot_reason = "min_volume_constraint"
                        elif rounding_mode == "clamped_to_max_volume":
                            overshoot_reason = "max_volume_constraint"
                        elif rounding_mode == "rounded_down_to_step":
                            overshoot_reason = "step_rounding_precision"
                        else:
                            overshoot_reason = "broker_volume_constraints"
                        sizing_notes.append("Actual risk still exceeds the requested level after broker volume constraints.")

                    rr_ratio = None
                    reward_currency = None
                    if proposed_tp is not None:
                        tp_distance_ticks = abs(proposed_tp - proposed_entry) / tick_size
                        reward_currency = tp_distance_ticks * tick_value * suggested_volume
                        if actual_risk > 0:
                            rr_ratio = reward_currency / actual_risk

                    result["position_sizing"] = {
                        "symbol": symbol,
                        "suggested_volume": suggested_volume,
                        "requested_risk_currency": round(risk_amount, 2),
                        "requested_risk_pct": float(desired_risk_pct),
                        "entry": proposed_entry,
                        "sl": proposed_sl,
                        "tp": proposed_tp,
                        "risk_currency": round(actual_risk, 2),
                        "risk_pct": round(actual_risk_pct, 2),
                        "risk_pct_diff": round(risk_pct_diff, 2),
                        "risk_over_target": risk_over_target,
                        "risk_compliance": "exceeds_requested_risk" if risk_over_target else "within_requested_risk",
                        "risk_overshoot_pct": round(overshoot_pct, 2),
                        "risk_overshoot_currency": round(overshoot_currency, 2),
                        "risk_over_target_reason": overshoot_reason,
                        "raw_volume": round(raw_volume, 8),
                        "volume_step": volume_step,
                        "volume_min": min_volume,
                        "volume_max": max_volume,
                        "volume_rounding": rounding_mode,
                        "reward_currency": round(reward_currency, 2) if reward_currency else None,
                        "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
                        "sizing_notes": sizing_notes,
                    }
                    if risk_over_target:
                        result["position_sizing_warning"] = (
                            f"Requested risk {float(desired_risk_pct):.2f}% but actual risk is "
                            f"{float(actual_risk_pct):.2f}% (+{overshoot_pct:.2f}%) after broker volume constraints."
                        )
                        result["risk_alert"] = {
                            "severity": "warning",
                            "code": "risk_overshoot_after_volume_constraints",
                            "reason": overshoot_reason,
                            "requested_risk_pct": float(desired_risk_pct),
                            "actual_risk_pct": round(actual_risk_pct, 2),
                            "overshoot_pct": round(overshoot_pct, 2),
                            "requested_risk_currency": round(risk_amount, 2),
                            "actual_risk_currency": round(actual_risk, 2),
                            "overshoot_currency": round(overshoot_currency, 2),
                        }
                else:
                    result["position_sizing_error"] = "SL distance must be greater than 0"

            return result
        except Exception as exc:
            return {"error": str(exc)}

    return _analyze_risk()
