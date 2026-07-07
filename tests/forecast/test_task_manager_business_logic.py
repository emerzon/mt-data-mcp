import os
import queue
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtdata.forecast.job_store import JobStore
from mtdata.forecast.model_store import ModelStore
from mtdata.forecast.task_manager import TaskManager, _TrainingSpec


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
