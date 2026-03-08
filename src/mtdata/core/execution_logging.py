"""Shared execution-path logging helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TypeVar

ResultT = TypeVar("ResultT")


def infer_result_success(result: Any) -> bool:
    if isinstance(result, dict):
        error_text = result.get("error")
        if isinstance(error_text, str) and error_text.strip():
            return False
        if error_text not in (None, False):
            return False
        success = result.get("success")
        if isinstance(success, bool):
            return success
        return True
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                error_text = item.get("error")
                if isinstance(error_text, str) and error_text.strip():
                    return False
                if error_text not in (None, False):
                    return False
        return True
    return result is not None


def log_operation_start(logger: logging.Logger, *, operation: str, **fields: Any) -> None:
    logger.debug("event=start operation=%s %s", operation, _format_fields(fields))


def log_operation_finish(
    logger: logging.Logger,
    *,
    operation: str,
    started_at: float,
    success: bool,
    **fields: Any,
) -> None:
    logger.info(
        "event=finish operation=%s success=%s duration_ms=%.3f %s",
        operation,
        bool(success),
        _elapsed_ms(started_at),
        _format_fields(fields),
    )


def log_operation_exception(
    logger: logging.Logger,
    *,
    operation: str,
    started_at: float,
    exc: BaseException,
    **fields: Any,
) -> None:
    logger.exception(
        "event=error operation=%s duration_ms=%.3f %s error=%s",
        operation,
        _elapsed_ms(started_at),
        _format_fields(fields),
        exc,
    )


def run_logged_operation(
    logger: logging.Logger,
    *,
    operation: str,
    func: Callable[[], ResultT],
    success_eval: Optional[Callable[[ResultT], bool]] = None,
    **fields: Any,
) -> ResultT:
    started_at = time.perf_counter()
    log_operation_start(logger, operation=operation, **fields)
    try:
        result = func()
    except Exception as exc:
        log_operation_exception(
            logger,
            operation=operation,
            started_at=started_at,
            exc=exc,
            **fields,
        )
        raise

    success_value = infer_result_success(result) if success_eval is None else bool(success_eval(result))
    log_operation_finish(
        logger,
        operation=operation,
        started_at=started_at,
        success=success_value,
        **fields,
    )
    return result


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - float(started_at)) * 1000.0, 3)


def _format_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            parts.append(f"{key}={text}")
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)
