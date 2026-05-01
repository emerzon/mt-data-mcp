import logging
import time

import pytest

from mtdata.core.execution_logging import (
    log_operation_finish,
    log_operation_start,
    run_logged_operation,
)


def test_run_logged_operation_logs_finish_event(caplog):
    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        result = run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="sample_op",
            item="abc",
            func=lambda: {"success": True},
        )

    assert result["success"] is True
    assert any(
        "event=finish operation=sample_op success=True" in record.message
        for record in caplog.records
    )


def test_run_logged_operation_runs_cleanup_after_top_level_success(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mtdata.core.execution_logging._run_post_operation_cleanup",
        lambda logger, *, operation: calls.append(operation),
    )

    result = run_logged_operation(
        logging.getLogger("mtdata.test.exec"),
        operation="sample_op",
        func=lambda: {"success": True},
    )

    assert result["success"] is True
    assert calls == ["sample_op"]


def test_run_logged_operation_logs_exception_and_reraises(caplog):
    with caplog.at_level(logging.ERROR, logger="mtdata.test.exec"), pytest.raises(RuntimeError, match="boom"):
        run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="sample_fail",
            func=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert any(
        "event=error operation=sample_fail" in record.message
        for record in caplog.records
    )


def test_run_logged_operation_runs_cleanup_after_top_level_exception(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mtdata.core.execution_logging._run_post_operation_cleanup",
        lambda logger, *, operation: calls.append(operation),
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="sample_fail",
            func=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert calls == ["sample_fail"]


def test_nested_same_operation_logs_single_finish_event(caplog):
    logger = logging.getLogger("mtdata.test.exec")

    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        result = run_logged_operation(
            logger,
            operation="sample_op",
            func=lambda: run_logged_operation(
                logger,
                operation="sample_op",
                func=lambda: {"success": True},
            ),
        )

    assert result["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=sample_op success=True" in record.message
    ]
    assert len(finish_records) == 1


def test_nested_operations_run_cleanup_once_for_outer_operation(monkeypatch):
    logger = logging.getLogger("mtdata.test.exec")
    calls = []
    monkeypatch.setattr(
        "mtdata.core.execution_logging._run_post_operation_cleanup",
        lambda logger, *, operation: calls.append(operation),
    )

    result = run_logged_operation(
        logger,
        operation="outer_op",
        func=lambda: run_logged_operation(
            logger,
            operation="inner_op",
            func=lambda: {"success": True},
        ),
    )

    assert result["success"] is True
    assert calls == ["outer_op"]


def test_nested_different_operations_still_log_both_finish_events(caplog):
    logger = logging.getLogger("mtdata.test.exec")

    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        result = run_logged_operation(
            logger,
            operation="outer_op",
            func=lambda: run_logged_operation(
                logger,
                operation="inner_op",
                func=lambda: {"success": True},
            ),
        )

    assert result["success"] is True
    assert any("event=finish operation=outer_op success=True" in record.message for record in caplog.records)
    assert any("event=finish operation=inner_op success=True" in record.message for record in caplog.records)


def test_manual_nested_same_operation_logs_single_finish_event(caplog):
    logger = logging.getLogger("mtdata.test.exec")
    outer_started = time.perf_counter()
    inner_started = time.perf_counter()

    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        log_operation_start(logger, operation="manual_op")
        log_operation_start(logger, operation="manual_op")
        log_operation_finish(logger, operation="manual_op", started_at=inner_started, success=True)
        log_operation_finish(logger, operation="manual_op", started_at=outer_started, success=True)

    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=manual_op success=True" in record.message
    ]
    assert len(finish_records) == 1
