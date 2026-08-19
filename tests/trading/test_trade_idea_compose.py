from __future__ import annotations

from typing import Any, Dict

import pytest

from mtdata.core.trading.ideas import run_trade_idea_compose
from mtdata.core.trading.ideas_requests import TradeIdeaComposeRequest


def _session(*, usable: bool = True, tradable: bool = True) -> Dict[str, Any]:
    return {
        "success": True,
        "symbol": "EURUSD",
        "is_tradable": tradable,
        "can_open_new_positions": tradable,
        "quote": {
            "symbol": "EURUSD",
            "bid": 1.1000,
            "ask": 1.1002,
            "mid": 1.1001,
            "spread": 0.0002,
            "usable_for_live_trading": usable,
            "data_stale": not usable,
            "execution_blockers": [] if usable else ["quote_not_live_ready"],
        },
        "source": {"provider": "mt5"},
    }


def _forecast(*, trend: str = "up") -> Dict[str, Any]:
    values = [1.1001, 1.1004, 1.1008] if trend == "up" else [1.1001, 1.0998, 1.0994]
    if trend == "flat":
        values = [1.1001, 1.1001, 1.1001]
    direction = "bullish" if trend == "up" else "bearish" if trend == "down" else "neutral"
    return {
        "success": True,
        "method": "theta",
        "library": "native",
        "quantity": "price",
        "horizon": 12,
        "forecast_price": values,
        "trend": trend,
        "forecast_vs_last_price": {
            "direction": direction,
            "direction_basis": "horizon_end",
            "direction_actionable": trend != "flat",
            "direction_status": "interval_confirmed" if trend != "flat" else "neutral",
        },
    }


def _volatility() -> Dict[str, Any]:
    return {
        "success": True,
        "method": "ewma",
        "horizon": 12,
        "volatility_per_bar": 0.0006,
        "volatility_horizon": 0.0021,
    }


def _barriers(*, tp_first: float = 0.42, sl_first: float = 0.31) -> Dict[str, Any]:
    return {
        "success": True,
        "method": "mc_gbm_bb",
        "direction": "long",
        "horizon": 12,
        "reference_price": 1.1002,
        "tp_pct": 0.40,
        "sl_pct": 0.60,
        "tp_price": 1.104602,
        "sl_price": 1.0935988,
        "prob_tp_first": tp_first,
        "prob_sl_first": sl_first,
        "prob_no_hit": round(1.0 - tp_first - sl_first, 4),
        "probability_edge": tp_first - sl_first,
    }


def _confluence() -> Dict[str, Any]:
    return {
        "success": True,
        "levels": [
            {
                "type": "resistance",
                "price": 1.1044,
                "score": 4.2,
                "range": {"low": 1.1042, "high": 1.1046, "width": 0.0004},
            },
            {
                "type": "support",
                "price": 1.0938,
                "score": 3.1,
                "range": {"low": 1.0936, "high": 1.0940, "width": 0.0004},
            },
        ],
    }


def _sizing(*, volume: float = 0.12) -> Dict[str, Any]:
    return {
        "success": True,
        "candidate_valid": True,
        "position_sizing": {
            "suggested_volume": volume,
            "candidate_valid": True,
            "requested_risk_pct": 0.5,
            "status": "ok",
            "entry": 1.1002,
            "sl": 1.0935988,
            "tp": 1.104602,
        },
    }


def _preview(*, preview_ok: bool = True) -> Dict[str, Any]:
    return {
        "success": True,
        "dry_run": True,
        "preview_ok": preview_ok,
        "would_send_order": False,
        "actionability": "preview_only",
        "blockers": [] if preview_ok else ["quote_not_live_ready"],
        "validation": {"live_submission_eligible": preview_ok},
        "guardrails_preview": {"enabled": True, "blocked": False},
    }


def _caller(mapping: Dict[str, Any]):
    def _call(name: str, kwargs: Dict[str, Any]) -> Any:
        payload = mapping.get(name)
        if callable(payload):
            return payload(kwargs)
        if payload is None:
            return {"success": False, "error": f"missing section {name}"}
        return payload

    return _call


def test_trade_idea_compose_quick_preview_path() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["success"] is True
    assert idea["direction"] == "long"
    assert idea["direction_basis"] == "forecast_vs_last_price"
    assert idea["suggested_direction"] == "long"
    assert idea["actionability"] == "preview_only"
    assert idea["preview"]["dry_run"] is True
    assert idea["preview"]["preview_ok"] is True
    assert idea["preview"]["would_send_order"] is False
    assert idea["sizing"]["suggested_volume"] == 0.12
    assert idea["geometry"]["take_profit"] == pytest.approx(1.104602)
    assert idea["geometry"]["stop_loss"] == pytest.approx(1.0935988)
    assert idea["gates"]["preview"]["status"] == "pass"
    assert "source_tool_calls" not in idea
    assert "not an order" in idea["narrative"]


def test_trade_idea_compose_stands_down_on_stale_quote() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(usable=False),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["actionability"] == "research"
    assert idea["sizing"]["suggested_volume"] == 0.0
    assert idea["preview"]["preview_ok"] is False
    assert idea["gates"]["quote_fresh"]["status"] == "fail"
    assert "sizing" not in idea.get("failed_sections", [])


def test_trade_idea_compose_auto_stands_down_when_barriers_disagree() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="auto"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(trend="up"),
                "volatility": _volatility(),
                "barriers": _barriers(tp_first=0.20, sl_first=0.55),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["suggested_direction"] == "long"
    assert idea["gates"]["alignment"]["status"] == "fail"
    assert idea["actionability"] == "research"
    assert idea["sizing"]["suggested_volume"] == 0.0


def test_trade_idea_compose_explicit_direction_keeps_side_when_barriers_are_weak() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(trend="up"),
                "volatility": _volatility(),
                "barriers": _barriers(tp_first=0.20, sl_first=0.55),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["direction"] == "long"
    assert idea["gates"]["barriers"]["status"] == "fail"
    assert idea["actionability"] == "preview_only"
    assert idea["preview"]["preview_ok"] is True


def test_trade_idea_compose_historical_skips_preview() -> None:
    calls: list[str] = []

    def _tracking(name: str, kwargs: Dict[str, Any]) -> Any:
        calls.append(name)
        return {
            "session": _session(),
            "forecast": _forecast(),
            "volatility": _volatility(),
            "barriers": _barriers(),
        }[name]

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", as_of="2026-01-15"),
        call_section=_tracking,
    )

    assert "session" not in calls
    assert "sizing" not in calls
    assert "preview" not in calls
    assert "quote" not in idea
    assert idea["actionability"] == "research"
    assert idea["preview"]["skipped"] is True
    assert idea["gates"]["preview"]["status"] == "skip"


def test_trade_idea_compose_stands_down_on_unconfirmed_direction() -> None:
    forecast = _forecast(trend="down")
    forecast["forecast_price"] = [1.15787, 1.15780, 1.15775]
    forecast["forecast_vs_last_price"] = {
        "direction": "neutral",
        "direction_basis": "horizon_end",
        "direction_actionable": False,
        "direction_status": "neutral",
    }

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": forecast,
                "volatility": _volatility(),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["direction_basis"] == "forecast_vs_last_price"
    assert "suggested_direction" not in idea
    assert idea["forecast"]["trend"] == "down"
    assert idea["gates"]["alignment"]["status"] == "fail"
    assert idea["sizing"]["suggested_volume"] == 0.0


def test_trade_idea_compose_standard_snaps_to_confluence() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", template="standard", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "confluence": _confluence(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["structure"]["levels"]
    snaps = idea["barriers"]["snapped_to_structure"]
    kinds = {row["kind"] for row in snaps}
    assert kinds == {"take_profit", "stop_loss"}
    assert idea["geometry"]["take_profit"] == pytest.approx(1.1044)
    assert idea["geometry"]["stop_loss"] == pytest.approx(1.0938)


def test_trade_idea_compose_never_accepts_live_preview() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": {
                    "success": True,
                    "dry_run": False,
                    "preview_ok": True,
                    "would_send_order": True,
                },
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["preview"]["dry_run"] is True
    assert idea["preview"]["preview_ok"] is False
    assert idea["gates"]["preview"]["status"] == "fail"
    assert idea["sizing"]["suggested_volume"] == 0.0


def test_trade_idea_compose_symbol_not_found_fails_closed() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="NOPE"),
        call_section=_caller(
            {
                "session": {
                    "success": False,
                    "error": "Symbol 'NOPE' was not found.",
                    "error_code": "symbol_not_found",
                }
            }
        ),
    )

    assert idea["success"] is False
    assert idea["error_code"] == "symbol_not_found"


def test_trade_idea_compose_full_detail_keeps_source_calls() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", detail="full"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    names = [row["name"] for row in idea["source_tool_calls"]]
    assert names == ["session", "forecast", "volatility", "barriers", "sizing", "preview"]


def test_trade_idea_compose_partial_failure_when_volatility_fails() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": {"success": False, "error": "ewma unavailable"},
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["success"] is True
    assert idea["partial_failure"] is True
    assert "volatility" in idea["failed_sections"]
    assert idea["direction"] == "long"
