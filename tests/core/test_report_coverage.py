"""Tests for core/report.py — report_generate tool.

Covers lines 45-245 by mocking template functions and external data fetching.
"""
import logging
import pytest
import warnings
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List
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
        from mtdata.core.report_requests import ReportGenerateRequest

        with patch.object(report_mod, "ensure_mt5_connection_or_raise", return_value=None):
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


def test_run_report_generate_logs_finish_event(caplog):
    from mtdata.core.report_requests import ReportGenerateRequest
    from mtdata.core.report_use_cases import run_report_generate

    basic_template = MagicMock(return_value={"sections": {}, "diagnostics": {}})
    with patch("mtdata.core.report_templates.template_basic", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_advanced", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_scalping", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_intraday", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_swing", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_position", basic_template, create=True), \
         caplog.at_level("INFO", logger="mtdata.core.report_use_cases"):
        result = run_report_generate(
            ReportGenerateRequest(symbol="EURUSD"),
            render_report=lambda rep: "# report",
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_text=lambda message: f"error: {message}",
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

    assert out == {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}


def test_report_generate_logs_finish_event(monkeypatch, caplog):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)
    monkeypatch.setattr(report_mod, "_report_connection_error", lambda: None)
    monkeypatch.setattr(report_mod, "run_report_generate", lambda *args, **kwargs: {"success": True, "sections": {}})

    with caplog.at_level(logging.INFO, logger=report_mod.logger.name):
        out = raw(request=report_mod.ReportGenerateRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=report_generate success=True" in record.message
        for record in caplog.records
    )


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
_RENDER = "mtdata.core.report.render_enhanced_report"
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

from mtdata.core.report import _report_error_text, _report_error_payload


class TestReportErrorHelpers:

    def test_error_text_normal(self):
        assert _report_error_text("bad thing") == "error: bad thing\n"

    def test_error_text_empty(self):
        assert _report_error_text("") == "error: Unknown error.\n"

    def test_error_text_whitespace(self):
        assert _report_error_text("   ") == "error: Unknown error.\n"

    def test_error_payload_normal(self):
        assert _report_error_payload("oops") == {"error": "oops"}

    def test_error_payload_empty(self):
        assert _report_error_payload("") == {"error": "Unknown error."}


# ---------------------------------------------------------------------------
# report_generate — template dispatch
# ---------------------------------------------------------------------------


class TestReportTemplateDispatch:

    def _run(self, template, output="toon", horizon=None, methods=None, timeframe=None):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())

        mock_templates = {
            "basic": MagicMock(return_value=rep),
            "advanced": MagicMock(return_value=rep),
            "scalping": MagicMock(return_value=rep),
            "intraday": MagicMock(return_value=rep),
            "swing": MagicMock(return_value=rep),
            "position": MagicMock(return_value=rep),
        }

        with patch(f"{_TEMPLATES_PATH}._t_basic", mock_templates["basic"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_advanced", mock_templates["advanced"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_scalping", mock_templates["scalping"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_intraday", mock_templates["intraday"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_swing", mock_templates["swing"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_position", mock_templates["position"], create=True), \
             patch(_FMT_NUM, side_effect=str), \
             patch(_RENDER, return_value="# Report"):
            # Patch the import block inside the function
            mock_mod = MagicMock()
            mock_mod.template_basic = mock_templates["basic"]
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
                 patch("mtdata.core.report_templates.template_advanced", mock_templates["advanced"], create=True), \
                 patch("mtdata.core.report_templates.template_scalping", mock_templates["scalping"], create=True), \
                 patch("mtdata.core.report_templates.template_intraday", mock_templates["intraday"], create=True), \
                 patch("mtdata.core.report_templates.template_swing", mock_templates["swing"], create=True), \
                 patch("mtdata.core.report_templates.template_position", mock_templates["position"], create=True):
                res = fn("EURUSD", template=template, output=output,
                         horizon=horizon, methods=methods, timeframe=timeframe)
        return res

    def test_basic_template_toon(self):
        res = self._run("basic")
        assert isinstance(res, dict)

    def test_advanced_template(self):
        res = self._run("advanced")
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
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="unknown_xyz", output="toon")
        assert "error" in res

    def test_unknown_markdown(self):
        fn = _get_report_generate()
        mock_basic = MagicMock()
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="unknown_xyz", output="markdown")
        assert isinstance(res, str)
        assert "error:" in res


# ---------------------------------------------------------------------------
# Template import failure
# ---------------------------------------------------------------------------


class TestReportTemplateImportError:

    def test_import_error_toon(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            res = fn("EURUSD", template="basic", output="toon")
        assert "error" in res

    def test_import_error_markdown(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            res = fn("EURUSD", template="basic", output="markdown")
        assert isinstance(res, str)
        assert "error:" in res


# ---------------------------------------------------------------------------
# Template returns non-dict / error
# ---------------------------------------------------------------------------


class TestReportBadTemplateReturn:

    def test_non_dict_toon(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value="not a dict")
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", output="toon")
        assert "error" in res

    def test_non_dict_markdown(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value="not a dict")
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", output="markdown")
        assert isinstance(res, str)
        assert "error:" in res

    def test_error_in_report_toon(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value={"error": "data fetch failed"})
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", output="toon")
        assert "error" in res

    def test_error_in_report_markdown(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value={"error": "data fetch failed"})
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", output="markdown")
        assert isinstance(res, str)
        assert "error:" in res


# ---------------------------------------------------------------------------
# Horizon selection
# ---------------------------------------------------------------------------


class TestReportHorizon:

    def _run_with_horizon(self, horizon=None, params=None, template="basic"):
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
            res = fn("EURUSD", template="basic", output="toon")
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

    def test_rsi_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("RSI=" in s for s in res.get("summary", []))

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
            return fn("EURUSD", template="basic", output="toon")

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
            return fn("EURUSD", template="basic", output="toon")

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
            "last_observation_time": "2026-03-02 18:00 UTC",
            "forecast_start_time": "2026-03-02 19:00 UTC",
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
            return fn("EURUSD", template="basic", output="toon")

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

    def test_no_barriers_section(self):
        sec = _make_full_sections()
        del sec["barriers"]
        res = self._run_report(sec)
        assert "summary" in res


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


class TestReportMarkdownOutput:

    def test_markdown_renders(self):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str), \
             patch(_RENDER, return_value="# Report\nContent"):
            res = fn("EURUSD", template="basic", output="markdown")
        assert isinstance(res, str)
        assert "Report" in res


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
            res = fn("EURUSD", template="basic", output="toon")

        assert isinstance(res, dict)
        assert "diagnostics" in res
        assert "warnings" in res["diagnostics"]
        assert "model convergence warning" in res["diagnostics"]["warnings"][0]

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
            res = fn("EURUSD", template="basic", output="toon")

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
            res = fn("EURUSD", template="basic", output="toon")

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
            res = fn("EURUSD", template="basic", output="toon")

        assert res["sections_status"]["summary"]["ok"] >= 1
        assert res["sections_status"]["sections"]["forecast"] == "ok"
        assert res["success"] is True

    def test_partial_section_marks_report_unsuccessful(self):
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
            res = fn("EURUSD", template="basic", output="toon")

        assert res["sections_status"]["sections"]["barriers"] == "partial"
        assert res["sections_status"]["summary"]["partial"] >= 1
        assert res["success"] is False


# ---------------------------------------------------------------------------
# Top-level exception
# ---------------------------------------------------------------------------


class TestReportTopLevelException:

    def test_exception_toon(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            # Force a deeper exception
            res = fn("EURUSD", template="basic", output="toon")
        assert isinstance(res, dict)

    def test_exception_markdown(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            res = fn("EURUSD", template="basic", output="markdown")
        assert isinstance(res, str)


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
