"""Tests for core/report.py — report_generate tool.

Covers lines 45-245 by mocking template functions and external data fetching.
"""
import logging
import warnings
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mtdata.utils.mt5 import MT5ConnectionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _get_report_generate():
    from mtdata.core.report import report_generate
    raw = _unwrap(report_generate)

    def _call(symbol, **kwargs):
        from mtdata.core import report as report_mod
        from mtdata.core.report.requests import ReportGenerateRequest

        with patch.object(report_mod, "ensure_mt5_connection_or_raise", return_value=None):
            if kwargs.get("format") == "toon":
                kwargs.pop("format")
            kwargs.setdefault("detail", "full")
            return raw(request=ReportGenerateRequest(symbol=symbol, **kwargs))

    return _call


def _make_report(sections=None, error=None):
    """Build a minimal report dict."""
    rep: Dict[str, Any] = {}
    if error:
        rep["error"] = error
    if sections is not None:
        rep["sections"] = sections
    return rep


def test_report_generate_request_rejects_removed_output_field():
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="output was removed; use json"):
        ReportGenerateRequest(symbol="EURUSD", output="json")


def test_report_generate_request_rejects_removed_structured_format_alias():
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="format was removed; use json"):
        ReportGenerateRequest(symbol="EURUSD", format="structured")


def test_report_generate_request_defaults_to_compact_detail():
    from mtdata.core.report.requests import ReportGenerateRequest

    request = ReportGenerateRequest(symbol="EURUSD")

    assert request.detail == "compact"


def test_basic_report_volatility_failure_keeps_method_errors(monkeypatch):
    from mtdata.core.report_templates import basic as template_basic_mod

    def fake_raw_result(func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "forecast_volatility_estimate":
            return {"error": f"{kwargs.get('method')} unavailable"}
        return {"error": "stubbed section"}

    monkeypatch.setattr(template_basic_mod, "_get_raw_result", fake_raw_result)

    out = template_basic_mod.template_basic(
        "EURUSD",
        horizon=3,
        denoise=None,
        params={"timeframe": "H1"},
    )

    volatility = out["sections"]["volatility"]
    assert volatility["error"] == "Volatility estimation failed."
    assert volatility["hint"].startswith("Run forecast_volatility_estimate")
    assert volatility["errors"][0]["method"] == "ewma"
    assert "unavailable" in volatility["errors"][0]["error"]


def test_run_report_generate_logs_finish_event(caplog):
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    basic_template = MagicMock(return_value={"sections": {}, "diagnostics": {}})
    with patch("mtdata.core.report_templates.template_basic", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_minimal", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_advanced", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_scalping", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_intraday", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_swing", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_position", basic_template, create=True), \
         caplog.at_level("DEBUG", logger="mtdata.core.report.use_cases"):
        result = run_report_generate(
            ReportGenerateRequest(symbol="EURUSD"),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert isinstance(result, dict)
    assert any(
        "event=finish operation=report_generate success=True" in record.message
        for record in caplog.records
    )


def test_report_generate_returns_connection_error_payload(monkeypatch):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(report_mod, "ensure_mt5_connection_or_raise", fail_connection)

    out = raw(request=report_mod.ReportGenerateRequest(symbol="EURUSD"))

    assert out["success"] is False
    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)


def test_report_generate_logs_finish_event(monkeypatch, caplog):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)
    monkeypatch.setattr(report_mod, "_report_connection_error", lambda: None)
    monkeypatch.setattr(report_mod, "run_report_generate", lambda *args, **kwargs: {"success": True, "sections": {}})

    with caplog.at_level(logging.DEBUG, logger=report_mod.logger.name):
        out = raw(request=report_mod.ReportGenerateRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=report_generate success=True" in record.message
        for record in caplog.records
    )


def test_report_generate_compact_structured_payload(monkeypatch):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)
    monkeypatch.setattr(report_mod, "_report_connection_error", lambda: None)
    monkeypatch.setattr(
        report_mod,
        "run_report_generate",
        lambda *args, **kwargs: {
            "success": True,
            "detail": "compact",
            "summary": ["close=1.10"],
            "summary_structured": {"market": {"close": 1.10}},
            "sections_status": {"summary": {"ok": 1, "partial": 0, "error": 0}},
        },
    )

    out = raw(request=report_mod.ReportGenerateRequest(symbol="EURUSD"))

    assert out["detail"] == "compact"
    assert "summary" in out
    assert "sections" not in out


def test_compact_report_payload_orders_structured_summary_first():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "summary": ["close=1.10"],
            "summary_structured": {"market": {"close": 1.10}},
            "sections_status": {"summary": {"ok": 1, "partial": 0, "error": 0}},
        },
        symbol="EURUSD",
        template="basic",
    )

    assert list(out).index("summary_structured") < list(out).index("summary")


def _make_full_sections():
    """Create a rich sections dict to exercise summary extraction."""
    return {
        "context": {
            "last_snapshot": {
                "close": 1.1020,
                "EMA_20": 1.1010,
                "EMA_50": 1.1000,
                "RSI_14": 55.5,
            },
        },
        "pivot": {
            "levels": [
                {"level": "R1", "classic": 1.1060},
                {"level": "PP", "classic": 1.1020},
                {"level": "S1", "classic": 1.0980},
            ],
            "methods": [{"method": "classic"}],
        },
        "volatility": {
            "horizon_sigma_price": 0.0045,
        },
        "forecast": {
            "method": "EMA",
        },
        "barriers": {
            "long": {
                "best": {"tp": 1.5, "sl": 0.8, "edge": 0.3},
            },
            "short": {
                "best": {"tp": 1.2, "sl": 0.7, "edge": 0.2},
            },
        },
    }


_TEMPLATES_PATH = "mtdata.core.report"
_FMT_NUM = "mtdata.core.report.format_number"
_GET_IND = "mtdata.core.report._get_indicator_value"


# Template mock shortcuts
def _patch_templates():
    """Return a patch context for the template imports block."""
    return patch(f"{_TEMPLATES_PATH}.template_basic", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_advanced", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_scalping", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_intraday", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_swing", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_position", create=True)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

from mtdata.core.report import _report_error_payload


class TestReportErrorHelpers:

    def test_error_payload_normal(self):
        result = _report_error_payload("oops")
        assert result["success"] is False
        assert result["error"] == "oops"
        assert result["error_code"] == "report_generation_error"
        assert result["operation"] == "report_generate"
        assert isinstance(result.get("request_id"), str)

    def test_error_payload_empty(self):
        result = _report_error_payload("")
        assert result["error"] == "Unknown error."
        assert result["error_code"] == "report_generation_error"


# ---------------------------------------------------------------------------
# report_generate — template dispatch
# ---------------------------------------------------------------------------


class TestReportTemplateDispatch:

    def _run(self, template, format="toon", horizon=None, methods=None, timeframe=None):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())

        mock_templates = {
            "basic": MagicMock(return_value=rep),
            "minimal": MagicMock(return_value=rep),
            "advanced": MagicMock(return_value=rep),
            "scalping": MagicMock(return_value=rep),
            "intraday": MagicMock(return_value=rep),
            "swing": MagicMock(return_value=rep),
            "position": MagicMock(return_value=rep),
        }

        with patch(f"{_TEMPLATES_PATH}._t_basic", mock_templates["basic"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_minimal", mock_templates["minimal"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_advanced", mock_templates["advanced"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_scalping", mock_templates["scalping"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_intraday", mock_templates["intraday"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_swing", mock_templates["swing"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_position", mock_templates["position"], create=True), \
             patch(_FMT_NUM, side_effect=str):
            # Patch the import block inside the function
            mock_mod = MagicMock()
            mock_mod.template_basic = mock_templates["basic"]
            mock_mod.template_minimal = mock_templates["minimal"]
            mock_mod.template_advanced = mock_templates["advanced"]
            mock_mod.template_scalping = mock_templates["scalping"]
            mock_mod.template_intraday = mock_templates["intraday"]
            mock_mod.template_swing = mock_templates["swing"]
            mock_mod.template_position = mock_templates["position"]

            with patch(f"{_TEMPLATES_PATH}.report_generate") as mock_rg:
                # Call the real unwrapped function
                pass

            # Actually run the function with template import patched
            with patch("mtdata.core.report_templates.template_basic", mock_templates["basic"], create=True), \
                 patch("mtdata.core.report_templates.template_minimal", mock_templates["minimal"], create=True), \
                 patch("mtdata.core.report_templates.template_advanced", mock_templates["advanced"], create=True), \
                 patch("mtdata.core.report_templates.template_scalping", mock_templates["scalping"], create=True), \
                 patch("mtdata.core.report_templates.template_intraday", mock_templates["intraday"], create=True), \
                 patch("mtdata.core.report_templates.template_swing", mock_templates["swing"], create=True), \
                 patch("mtdata.core.report_templates.template_position", mock_templates["position"], create=True):
                res = fn("EURUSD", template=template, format=format,
                         horizon=horizon, methods=methods, timeframe=timeframe)
        return res

    def test_basic_template_toon(self):
        res = self._run("basic")
        assert isinstance(res, dict)

    def test_advanced_template(self):
        res = self._run("advanced")
        assert isinstance(res, dict)

    def test_minimal_template(self):
        res = self._run("minimal")
        assert isinstance(res, dict)

    def test_scalping_template(self):
        res = self._run("scalping")
        assert isinstance(res, dict)

    def test_intraday_template(self):
        res = self._run("intraday")
        assert isinstance(res, dict)

    def test_swing_template(self):
        res = self._run("swing")
        assert isinstance(res, dict)

    def test_position_template(self):
        res = self._run("position")
        assert isinstance(res, dict)


class TestReportUnknownTemplate:

    def test_unknown_toon(self):
        fn = _get_report_generate()
        # Patch the import block to succeed
        mock_basic = MagicMock()
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_minimal", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="unknown_xyz", format="toon")
        assert "error" in res


# ---------------------------------------------------------------------------
# Template import failure
# ---------------------------------------------------------------------------


class TestReportTemplateImportError:

    def test_import_error_toon(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            res = fn("EURUSD", template="basic", format="toon")
        assert "error" in res


# ---------------------------------------------------------------------------
# Template returns non-dict / error
# ---------------------------------------------------------------------------


class TestReportBadTemplateReturn:

    def test_non_dict_toon(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value="not a dict")
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_minimal", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", format="toon")
        assert "error" in res

    def test_error_in_report_toon(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value={"error": "data fetch failed"})
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", format="toon")
        assert "error" in res


# ---------------------------------------------------------------------------
# Horizon selection
# ---------------------------------------------------------------------------


class TestReportHorizon:

    def _run_with_horizon(self, horizon=None, params=None, template="basic"):
        fn = _get_report_generate()
        rep = _make_report(sections={})
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_minimal", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            fn("EURUSD", template=template, horizon=horizon, params=params)
        return mock_basic

    def test_default_horizon_basic(self):
        mock = self._run_with_horizon(template="basic")
        args = mock.call_args
        # Second positional arg is horizon
        assert args[0][1] == 12

    def test_explicit_horizon(self):
        mock = self._run_with_horizon(horizon=20, template="basic")
        assert mock.call_args[0][1] == 20

    def test_horizon_from_params(self):
        mock = self._run_with_horizon(params={"horizon": 30}, template="basic")
        assert mock.call_args[0][1] == 30

    def test_minimal_default_horizon(self):
        mock = self._run_with_horizon(template="minimal")
        assert mock.call_args[0][1] == 12

    def test_scalping_default_horizon(self):
        mock = self._run_with_horizon(template="scalping")
        assert mock.call_args[0][1] == 8

    def test_swing_default_horizon(self):
        mock = self._run_with_horizon(template="swing")
        assert mock.call_args[0][1] == 24

    def test_position_default_horizon(self):
        mock = self._run_with_horizon(template="position")
        assert mock.call_args[0][1] == 30


# ---------------------------------------------------------------------------
# Summary extraction — context
# ---------------------------------------------------------------------------


class TestReportSummaryContext:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")
        return res

    def test_close_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("close=" in s for s in res.get("summary", []))

    def test_trend_above_emas(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("above EMAs" in s for s in res.get("summary", []))

    def test_trend_mixed(self):
        sec = _make_full_sections()
        sec["context"]["last_snapshot"]["close"] = 1.0990  # below EMA_50
        res = self._run_report(sec)
        assert any("mixed" in s for s in res.get("summary", []))

    def test_trend_skipped_when_close_missing(self):
        sec = _make_full_sections()
        sec["context"]["last_snapshot"]["close"] = None
        res = self._run_report(sec)
        assert not any("trend:" in s for s in res.get("summary", []))

    def test_rsi_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("RSI=" in s for s in res.get("summary", []))

    def test_summary_key_precedes_sections_in_payload(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        keys = list(res.keys())
        assert keys.index("summary") < keys.index("sections")

    def test_no_context(self):
        res = self._run_report({})
        assert "summary" in res


# ---------------------------------------------------------------------------
# Summary extraction — pivot
# ---------------------------------------------------------------------------


class TestReportSummaryPivot:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            return fn("EURUSD", template="basic", format="toon")

    def test_pivot_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("pivot" in s for s in res.get("summary", []))

    def test_pivot_method_fallback(self):
        """When chosen method not in available methods, fallback to first available."""
        sec = _make_full_sections()
        sec["pivot"]["methods"] = [{"method": "fibonacci"}]
        sec["pivot"]["levels"] = [
            {"level": "R1", "woodie": 1.106},
            {"level": "PP", "woodie": 1.102},
            {"level": "S1", "woodie": 1.098},
        ]
        res = self._run_report(sec)
        # Should fallback to 'woodie' since 'fibonacci' not in level columns
        assert isinstance(res, dict)

    def test_pivot_context_in_summary(self):
        sec = _make_full_sections()
        sec["pivot"]["calculation_basis"] = {
            "session_boundary": "MT5 broker/session calendar",
            "display_timezone": "UTC",
        }
        res = self._run_report(sec)
        assert any("pivot context" in s for s in res.get("summary", []))


# ---------------------------------------------------------------------------
# Summary extraction — volatility & forecast
# ---------------------------------------------------------------------------


class TestReportSummaryVolForecast:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            return fn("EURUSD", template="basic", format="toon")

    def test_vol_sigma(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("sigma=" in s for s in res.get("summary", []))

    def test_vol_return_sigma_fallback(self):
        sec = _make_full_sections()
        sec["volatility"] = {"horizon_sigma_return": 0.003}
        res = self._run_report(sec)
        assert any("sigma=" in s for s in res.get("summary", []))

    def test_forecast_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("forecast=" in s for s in res.get("summary", []))

    def test_forecast_timing_in_summary(self):
        sec = _make_full_sections()
        sec["forecast"].update({
            "last_observation_epoch": 1740948000.0,
            "forecast_start_epoch": 1740951600.0,
            "forecast_anchor": "next_timeframe_bar_after_last_observation",
        })
        res = self._run_report(sec)
        assert any("forecast timing:" in s for s in res.get("summary", []))

    def test_forecast_selection_criteria_in_summary(self):
        sec = _make_full_sections()
        sec["backtest"] = {
            "selection_criteria": {
                "primary_metric": "avg_rmse",
                "rmse_tolerance_pct": 5.0,
                "tie_breaker": "avg_directional_accuracy",
            },
            "best_method": {"method": "naive"},
        }
        res = self._run_report(sec)
        assert any("forecast selection:" in s for s in res.get("summary", []))

    def test_forecast_selection_criteria_includes_min_directional_accuracy(self):
        sec = _make_full_sections()
        sec["backtest"] = {
            "selection_criteria": {
                "primary_metric": "avg_rmse",
                "rmse_tolerance_pct": 5.0,
                "tie_breaker": "avg_directional_accuracy",
                "min_directional_accuracy": 0.55,
            },
            "best_method": {"method": "naive"},
        }
        res = self._run_report(sec)
        assert any("min-dir-acc>=" in s for s in res.get("summary", []))


# ---------------------------------------------------------------------------
# Summary extraction — barriers
# ---------------------------------------------------------------------------


class TestReportSummaryBarriers:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            return fn("EURUSD", template="basic", format="toon")

    def test_long_short_barriers(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        summ = res.get("summary", [])
        assert any("dir=long" in s for s in summ)
        assert any("dir=short" in s for s in summ)

    def test_single_best_barrier(self):
        """Old-style barriers with single best/direction (lines 217-233)."""
        sec = _make_full_sections()
        sec["barriers"] = {
            "best": {"tp": 2.0, "sl": 1.0, "edge": 0.5},
            "direction": "long",
        }
        res = self._run_report(sec)
        assert any("barrier best" in s for s in res.get("summary", []))

    def test_barrier_summary_includes_ev_and_conflict_hint(self):
        sec = _make_full_sections()
        sec["barriers"]["long"]["best"]["ev"] = 0.03
        sec["barriers"]["long"]["best"]["edge"] = -0.1
        res = self._run_report(sec)
        long_line = [s for s in res.get("summary", []) if "barrier best" in s and "dir=long" in s][0]
        assert "ev=" in long_line
        assert "edge=" in long_line
        assert "ev_edge_conflict=true" in long_line
        assert "ev_edge_conflict_reason=" in long_line
        structured = res["summary_structured"]["barriers"]["long"]
        assert structured["ev"] == 0.03
        assert structured["edge"] == -0.1
        assert structured["ev_edge_conflict"] is True
        assert structured["conflict_reason"] == "ev and edge have opposite signs"

    def test_no_barriers_section(self):
        sec = _make_full_sections()
        del sec["barriers"]
        res = self._run_report(sec)
        assert "summary" in res


class TestReportWarnings:

    def test_template_warnings_are_captured_in_diagnostics(self):
        fn = _get_report_generate()

        def _warn_template(*args, **kwargs):
            warnings.warn("model convergence warning", RuntimeWarning)
            return _make_report(sections=_make_full_sections())

        with patch("mtdata.core.report_templates.template_basic", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_advanced", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_scalping", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_intraday", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_swing", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_position", _warn_template, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert isinstance(res, dict)
        assert "diagnostics" in res
        assert "warnings" in res["diagnostics"]
        assert "model convergence warning" in res["diagnostics"]["warnings"][0]

    def test_library_deprecation_warnings_are_not_user_facing(self):
        fn = _get_report_generate()

        def _warn_template(*args, **kwargs):
            warnings.warn("Importing from torchao.dtypes is deprecated", DeprecationWarning)
            warnings.warn("model convergence warning", RuntimeWarning)
            return _make_report(sections=_make_full_sections())

        with patch("mtdata.core.report_templates.template_basic", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_advanced", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_scalping", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_intraday", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_swing", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_position", _warn_template, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        warnings_out = res["diagnostics"]["warnings"]
        assert warnings_out == ["model convergence warning"]

    def test_flat_forecast_is_flagged_in_summary_and_diagnostics(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["forecast"] = {
            "method": "sf_autoarima",
            "forecast_price": [65955.1] * 12,
        }
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert any("forecast=sf_autoarima (flat)" in s for s in res.get("summary", []))
        assert "diagnostics" in res
        assert "warnings" in res["diagnostics"]
        assert any("degenerate" in str(w).lower() for w in res["diagnostics"]["warnings"])

    def test_execution_time_metric_is_added_to_diagnostics(self):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert isinstance(res, dict)
        assert "diagnostics" in res
        assert "execution_time_ms" in res["diagnostics"]
        assert float(res["diagnostics"]["execution_time_ms"]) >= 0.0

    def test_sections_status_summary_is_added(self):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert res["sections_status"]["summary"]["ok"] >= 1
        assert res["sections_status"]["sections"]["forecast"] == "ok"
        assert res["success"] is True

    def test_partial_section_marks_report_partially_complete(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["barriers"] = {
            "long": {"best": {"tp": 1.5, "sl": 0.8, "edge": 0.3}},
            "short": {"error": "optimizer failed"},
        }
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert res["sections_status"]["sections"]["barriers"] == "partial"
        assert res["sections_status"]["summary"]["partial"] >= 1
        assert res["sections_status"]["details"]["barriers"]["errors"][0]["path"] == "short"
        assert "usable data" in res["sections_status"]["definitions"]["partial"]
        assert res["completeness"] == "partial"
        assert res["success"] is True

    def test_sections_status_filters_placeholder_error_noise(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["volatility"] = {
            "summary": {"realized_vol": 0.12},
            "error": "Volatility estimation failed.",
            "estimators": [
                {"error": "no value"},
                {"error": ""},
                {"error": "Volatility estimation failed."},
            ],
        }
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        errors = res["sections_status"]["details"]["volatility"]["errors"]
        assert {"path": "error", "message": "Volatility estimation failed."} in errors
        assert all(item["message"] != "no value" for item in errors)

    def test_error_only_section_marks_report_failed(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["forecast"] = {"error": "forecast failed"}
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert res["sections_status"]["sections"]["forecast"] == "error"
        assert res["completeness"] == "failed"
        assert res["success"] is False


# ---------------------------------------------------------------------------
# Top-level exception
# ---------------------------------------------------------------------------


class TestReportTopLevelException:

    def test_exception_toon(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            # Force a deeper exception
            res = fn("EURUSD", template="basic", format="toon")
        assert isinstance(res, dict)

# ---------------------------------------------------------------------------
# Params passthrough
# ---------------------------------------------------------------------------


class TestReportParams:

    def test_timeframe_in_params(self):
        fn = _get_report_generate()
        rep = _make_report(sections={})
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            fn("EURUSD", template="basic", timeframe="M15")
        # The params dict passed to template should contain 'timeframe'
        call_args = mock_basic.call_args
        p = call_args[0][3]  # 4th positional arg = params
        assert p.get("timeframe") == "M15"

    def test_methods_in_params(self):
        fn = _get_report_generate()
        rep = _make_report(sections={})
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            fn("EURUSD", template="basic", methods=["EMA", "ARIMA"])
        call_args = mock_basic.call_args
        p = call_args[0][3]
        assert p.get("methods") == ["EMA", "ARIMA"]
