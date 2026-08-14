import os
import queue
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtdata.forecast.job_store import JobStore
from mtdata.forecast.model_store import ModelStore
from mtdata.forecast.task_manager import (
    TaskManager,
    _configured_task_ttl_seconds,
    _format_worker_exit,
    _process_training_entry,
    _TrainingSpec,
)


def _make_spec() -> _TrainingSpec:
    return _TrainingSpec(
        task_kind="prepared",
        method_name="heavy",
        data_scope="EURUSD_H1",
        params_hash="hash-1",
        horizon=5,
        seasonality=1,
        params={},
        timeframe="H1",
    )


class _TaskManagerBusinessLogicCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self._job_store = JobStore(path=os.path.join(self._tmpdir, "jobs.sqlite"))
        self.tm = TaskManager(max_workers=1, heavy_limit=1, store=self._store, job_store=self._job_store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestTaskManagerHeavyRuntime(_TaskManagerBusinessLogicCase):
    def test_failed_worker_event_is_logged_with_task_context(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(task.task_id, status="running")

        with self.assertLogs("mtdata.forecast.task_manager", level="ERROR") as logs:
            terminal = self.tm._handle_process_event(
                task.task_id,
                _make_spec(),
                {"type": "failed", "error": "disk full\nwhile saving"},
            )

        self.assertTrue(terminal)
        output = "\n".join(logs.output)
        self.assertIn("event=forecast_training_failed", output)
        self.assertIn(f"task_id={task.task_id}", output)
        self.assertIn("method=heavy", output)
        self.assertIn("data_scope=EURUSD_H1", output)
        self.assertIn("error=disk full while saving", output)

    def test_dead_worker_is_logged_with_process_context(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(task.task_id, status="running")
        process = SimpleNamespace(pid=4321, exitcode=-9)

        with self.assertLogs("mtdata.forecast.task_manager", level="ERROR") as logs:
            self.tm._finalize_dead_process(task.task_id, process)

        output = "\n".join(logs.output)
        self.assertIn("event=forecast_training_worker_died", output)
        self.assertIn("pid=4321", output)
        self.assertIn("exitcode=-9", output)
        expected = "Windows status" if os.name == "nt" else "SIGKILL"
        self.assertIn(expected, output)

    def test_dead_worker_includes_persisted_diagnostic_tail(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(task.task_id, status="running")
        diagnostic_path = os.path.join(self._tmpdir, "worker.log")
        with open(diagnostic_path, "w", encoding="utf-8") as stream:
            stream.write("native fault in model kernel")

        self.tm._finalize_dead_process(
            task.task_id,
            SimpleNamespace(pid=4321, exitcode=-11),
            diagnostic_path=diagnostic_path,
        )

        status = self.tm.get_status(task.task_id)
        self.assertIsNotNone(status)
        self.assertIn("Worker diagnostic tail", status.error)
        self.assertIn("native fault in model kernel", status.error)

    def test_failed_worker_event_persists_bounded_traceback(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(task.task_id, status="running")

        terminal = self.tm._handle_process_event(
            task.task_id,
            _make_spec(),
            {
                "type": "failed",
                "error": "bad scale",
                "exception_type": "ValueError",
                "traceback": "Traceback: model.py line 42",
            },
        )

        status = self.tm.get_status(task.task_id)
        self.assertTrue(terminal)
        self.assertIsNotNone(status)
        self.assertIn("ValueError: bad scale", status.error)
        self.assertIn("model.py line 42", status.error)

    def test_worker_exit_format_explains_posix_signals_and_windows_status(self):
        self.assertIn("SIGKILL", _format_worker_exit(-9, platform_name="posix"))
        self.assertIn("out-of-memory", _format_worker_exit(-9, platform_name="posix"))
        self.assertIn("0xC0000005", _format_worker_exit(-1073741819, platform_name="nt"))
        self.assertIn("access violation", _format_worker_exit(-1073741819, platform_name="nt"))

    def test_task_retention_is_configurable(self):
        with patch.dict(os.environ, {"MTDATA_FORECAST_TASK_TTL_SECONDS": "7200"}):
            self.assertEqual(_configured_task_ttl_seconds(), 7200.0)

    def test_process_entry_emits_python_traceback_and_captures_it(self):
        events = []
        event_queue = SimpleNamespace(put=events.append)
        cancel_event = SimpleNamespace(is_set=lambda: False)
        diagnostic_path = os.path.join(self._tmpdir, "python-failure.log")

        with patch(
            "mtdata.forecast.task_manager._execute_training_spec",
            side_effect=ValueError("bad training scale"),
        ):
            _process_training_entry(
                _make_spec(),
                "task-python-failure",
                self._tmpdir,
                event_queue,
                cancel_event,
                60.0,
                diagnostic_path,
            )

        failed_event = next(event for event in events if event["type"] == "failed")
        self.assertEqual(failed_event["exception_type"], "ValueError")
        self.assertIn("bad training scale", failed_event["traceback"])
        with open(diagnostic_path, encoding="utf-8") as stream:
            self.assertIn("bad training scale", stream.read())

    def test_handle_process_event_marks_malformed_completion_failed(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(
            task.task_id,
            status="running",
            started_at=1.0,
            heartbeat_at=1.0,
        )

        terminal = self.tm._handle_process_event(
            task.task_id,
            _make_spec(),
            {
                "type": "completed",
                "heartbeat_at": 2.0,
                "completed_at": 2.0,
                "result": None,
            },
        )

        status = self.tm.get_status(task.task_id)
        self.assertTrue(terminal)
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "failed")
        self.assertIn("Malformed completion event", status.error)

    def test_handle_process_event_ignores_updates_after_terminal_status(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(
            task.task_id,
            status="failed",
            error="timed out",
            completed_at=1.0,
            heartbeat_at=1.0,
        )

        terminal = self.tm._handle_process_event(
            task.task_id,
            _make_spec(),
            {
                "type": "completed",
                "heartbeat_at": 2.0,
                "completed_at": 2.0,
                "result": {"model_id": "heavy/EURUSD_H1/hash-1"},
            },
        )

        status = self.tm.get_status(task.task_id)
        self.assertTrue(terminal)
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "failed")
        self.assertEqual(status.error, "timed out")
        self.assertIsNone(status.result)

    def test_heavy_queue_transport_error_marks_task_failed(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")

        class _FakeQueue:
            def get(self, timeout=None):
                raise EOFError("pipe closed")

            def get_nowait(self):
                raise queue.Empty

            def close(self):
                return None

            def join_thread(self):
                return None

        class _FakeProcess:
            def __init__(self):
                self.pid = 4321
                self.exitcode = -9

            def start(self):
                return None

            def is_alive(self):
                return False

            def join(self, timeout=None):
                return None

        fake_context = SimpleNamespace(
            Queue=lambda: _FakeQueue(),
            Event=MagicMock(return_value=MagicMock()),
            Process=MagicMock(return_value=_FakeProcess()),
        )

        with patch.object(self.tm, "_mp_context", fake_context):
            self.tm._run_heavy_task(task.task_id, _make_spec(), timeout_seconds=30.0)

        status = self.tm.get_status(task.task_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "failed")
        self.assertIn("communication failed", status.error.lower())

    def test_join_failure_in_finalizer_does_not_crash_runtime(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")

        class _FakeQueue:
            def get(self, timeout=None):
                raise queue.Empty

            def get_nowait(self):
                raise queue.Empty

            def close(self):
                return None

            def join_thread(self):
                return None

        class _FakeProcess:
            def __init__(self):
                self.pid = 4322
                self.exitcode = -9

            def start(self):
                return None

            def is_alive(self):
                return False

            def join(self, timeout=None):
                raise ValueError("cannot join process before it has been started")

        fake_context = SimpleNamespace(
            Queue=lambda: _FakeQueue(),
            Event=MagicMock(return_value=MagicMock()),
            Process=MagicMock(return_value=_FakeProcess()),
        )

        with patch.object(self.tm, "_mp_context", fake_context):
            self.tm._run_heavy_task(task.task_id, _make_spec(), timeout_seconds=30.0)

        status = self.tm.get_status(task.task_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "failed")

    def test_cancelled_dead_worker_without_terminal_event_becomes_cancelled(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(
            task.task_id,
            status="running",
            cancel_requested=True,
            started_at=1.0,
            heartbeat_at=1.0,
        )

        self.tm._finalize_dead_process(
            task.task_id,
            SimpleNamespace(exitcode=-15),
        )

        status = self.tm.get_status(task.task_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "cancelled")
        self.assertTrue(status.cancel_requested)

    def test_cancel_without_active_runtime_does_not_persist_cancel_requested(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(
            task.task_id,
            status="running",
            started_at=time.time(),
            heartbeat_at=time.time(),
        )

        result = self.tm.cancel(task.task_id)
        status = self.tm.get_status(task.task_id)

        self.assertEqual(result["status"], "not_cancelled")
        self.assertFalse(result["cancel_requested"])
        self.assertIsNotNone(status)
        self.assertFalse(status.cancel_requested)

    def test_cancel_finalizes_a_stopped_heavy_worker_before_returning(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(
            task.task_id,
            status="running",
            started_at=time.time(),
            heartbeat_at=time.time(),
        )
        process = MagicMock()
        process.is_alive.side_effect = [True, False]
        control = SimpleNamespace(
            process=process,
            event_queue=MagicMock(),
            cancel_event=MagicMock(),
        )
        self.tm._process_controls[task.task_id] = control

        with patch.object(self.tm, "_stop_process", return_value=True):
            result = self.tm.cancel(task.task_id)

        status = self.tm.get_status(task.task_id)
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["terminated"])
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "cancelled")

    def test_shutdown_marks_running_tasks_terminal(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        self.tm._mutate_task(
            task.task_id,
            status="running",
            started_at=time.time(),
            heartbeat_at=time.time(),
        )

        self.tm.shutdown(wait=False)

        status = self.tm.get_status(task.task_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "failed")
        self.assertIn("shut down", status.error.lower())


if __name__ == "__main__":
    unittest.main()
