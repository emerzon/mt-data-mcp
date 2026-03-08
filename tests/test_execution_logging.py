import logging

import pytest

from mtdata.core.execution_logging import run_logged_operation


def test_run_logged_operation_logs_finish_event(caplog):
    with caplog.at_level(logging.INFO, logger="mtdata.test.exec"):
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
