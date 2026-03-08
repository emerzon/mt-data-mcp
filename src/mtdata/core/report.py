from typing import Any, Dict, Union
import logging
import time

from ._mcp_instance import mcp
from .execution_logging import infer_result_success, log_operation_finish, log_operation_start
from .mt5_gateway import create_mt5_gateway
from .report_requests import ReportGenerateRequest
from .report_use_cases import run_report_generate
from .report_utils import render_enhanced_report, format_number, _get_indicator_value
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise

logger = logging.getLogger(__name__)


def _report_error_text(message: Any) -> str:
    text = str(message).strip()
    if not text:
        text = 'Unknown error.'
    return f"error: {text}\n"


def _report_error_payload(message: Any) -> Dict[str, Any]:
    text = str(message).strip()
    if not text:
        text = 'Unknown error.'
    return {"error": text}


def _get_mt5_gateway():
    return create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise)


def _report_connection_error() -> Dict[str, Any] | None:
    mt5 = _get_mt5_gateway()
    try:
        mt5.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return None


def _append_diagnostic_warning(report: Dict[str, Any], message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    warnings_list = diagnostics.get("warnings")
    if not isinstance(warnings_list, list):
        warnings_list = []
    if text not in warnings_list:
        warnings_list.append(text)
    diagnostics["warnings"] = warnings_list
    report["diagnostics"] = diagnostics


@mcp.tool()
def report_generate(
    request: ReportGenerateRequest,
) -> Union[str, Dict[str, Any]]:
    """Generate a consolidated, information-dense analysis report with compact multi-format output.

    - template: 'basic' (context, pivot, EWMA vol, backtest->best forecast, MC barrier grid, patterns)
                'advanced' (adds regimes, HAR-RV, conformal),
                or style-specific ('scalping' | 'intraday' | 'swing' | 'position').
    - params: optional dict to tune steps/spacing, grids, and optionally override timeframe per template via 'timeframe' or methods via 'methods'.
    - denoise: pass-through to candle fetching (e.g., {method:'ema', params:{alpha:0.2}, columns:['close']}).  
    - output: 'toon' (structured TOON) or 'markdown' (rendered report text).
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="report_generate",
        symbol=request.symbol,
        template=request.template,
        output=request.output,
    )

    def _finish(result: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        log_operation_finish(
            logger,
            operation="report_generate",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=request.symbol,
            template=request.template,
            output=request.output,
        )
        return result

    connection_error = _report_connection_error()
    if connection_error is not None:
        return _finish(connection_error)
    return _finish(
        run_report_generate(
            request,
            render_report=render_enhanced_report,
            format_number=format_number,
            get_indicator_value=_get_indicator_value,
            report_error_text=_report_error_text,
            report_error_payload=_report_error_payload,
            append_diagnostic_warning=_append_diagnostic_warning,
        )
    )
