"""Tests for forecast training runtime infrastructure."""

import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mtdata.forecast.interface import ForecastMethod, ForecastResult, TrainedModelHandle, TrainingProgress, TrainResult
from mtdata.forecast.job_store import JobRecord, JobStore
from mtdata.forecast.model_store import ModelStore
from mtdata.forecast.task_manager import TaskManager, TrainingTask, _snapshot


def _make_series(n: int = 100) -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(np.linspace(100, 120, n) + rng.normal(0, 0.5, n))


class _FakeMethod(ForecastMethod):
    def __init__(self, name: str = "fake", delay: float = 0.05, fail: bool = False):
        self._name = name
        self._delay = delay
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def training_category(self) -> str:
        return "fast"

    @property
    def train_supports_cancel(self) -> bool:
        return True

    @property
    def train_supports_progress(self) -> bool:
        return True

    def forecast(self, series, horizon, seasonality, params, **kw) -> ForecastResult:
        return ForecastResult(forecast=np.ones(horizon), params_used=params)

    def train(self, series, horizon, seasonality, params, *, progress_callback=None, cancel_token=None, **kw) -> TrainResult:
        if progress_callback:
            progress_callback(TrainingProgress(step=0, total_steps=2))
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        time.sleep(self._delay)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if self._fail:
            raise RuntimeError("training boom")
        if progress_callback:
            progress_callback(TrainingProgress(step=2, total_steps=2))
        return TrainResult(artifact_bytes=b"model-data", params_used=params or {})


class _FakeHeavyMethod(_FakeMethod):
    @property
    def training_category(self) -> str:
        return "heavy"


class _TaskManagerTestCase(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self._job_store = JobStore(path=os.path.join(self._tmpdir, "jobs.sqlite"))
        self.tm = TaskManager(max_workers=2, heavy_limit=1, store=self._store, job_store=self._job_store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestSnapshot(unittest.TestCase):
    def test_snapshot_returns_copy(self):
        task = TrainingTask(task_id="x", method="m", data_scope="s", params_hash="h")
        snap = _snapshot(task)
        self.assertEqual(snap.task_id, "x")
        snap.status = "failed"
        self.assertEqual(task.status, "pending")


class TestTaskManagerBasic(_TaskManagerTestCase):
    def test_submit_and_complete(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, is_new = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.assertTrue(is_new)

            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)
            self.assertIsNotNone(final)
            self.assertEqual(final.status, "completed")
            self.assertIsInstance(final.result, TrainedModelHandle)
            self.assertIsNotNone(final.started_at)
            self.assertIsNotNone(final.completed_at)

    def test_failed_task_persists_error(self):
        fake = _FakeMethod(delay=0.01, fail=True)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertEqual(final.status, "failed")
        self.assertIn("training boom", final.error)

    def test_canonical_hash_dedups_even_when_legacy_hash_arg_differs(self):
        fake = _FakeMethod(delay=0.2)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            series = _make_series()
            tid1, new1 = self.tm.submit("fake", series, 5, 1, {"lr": 0.01}, "EURUSD_H1", "legacy-a")
            tid2, new2 = self.tm.submit("fake", series, 5, 1, {"lr": 0.01}, "EURUSD_H1", "legacy-b")

        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(tid1, tid2)

    def test_different_params_do_not_dedup(self):
        fake = _FakeMethod(delay=0.2)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            series = _make_series()
            tid1, new1 = self.tm.submit("fake", series, 5, 1, {"lr": 0.01}, "EURUSD_H1")
            tid2, new2 = self.tm.submit("fake", series, 5, 1, {"lr": 0.1}, "EURUSD_H1")

        self.assertTrue(new1)
        self.assertTrue(new2)
        self.assertNotEqual(tid1, tid2)

    def test_progress_and_heartbeat_update_task(self):
        fake = _FakeMethod(delay=0.05)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertIsNotNone(final.progress)
        self.assertEqual(final.progress.total_steps, 2)
        self.assertIsNotNone(final.heartbeat_at)

    def test_wait_for_status_returns_terminal_snapshot(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            waited = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertEqual(waited.status, "completed")


class TestTaskManagerCancelAndList(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self._job_store = JobStore(path=os.path.join(self._tmpdir, "jobs.sqlite"))
        self.tm = TaskManager(max_workers=1, heavy_limit=1, store=self._store, job_store=self._job_store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_list_tasks_empty(self):
        self.assertEqual(self.tm.list_tasks(), [])

    def test_cancel_unknown_task(self):
        result = self.tm.cancel("nonexistent")
        self.assertEqual(result["status"], "not_found")
        self.assertFalse(result["cancel_requested"])

    def test_cancel_pending_task(self):
        fake = _FakeMethod(delay=0.4)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            self.tm.submit("fake", _make_series(), 5, 1, {"lr": 0.01}, "EURUSD_H1")
            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {"lr": 0.02}, "EURUSD_H1")
            result = self.tm.cancel(task_id)
            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertTrue(result["cancel_requested"])
        self.assertEqual(final.status, "cancelled")

    def test_cancel_completed_task(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(task_id, timeout_seconds=5.0)
            result = self.tm.cancel(task_id)

        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["cancel_requested"])

    def test_list_tasks_with_filter(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        completed = self.tm.list_tasks(status="completed")
        pending = self.tm.list_tasks(status="pending")
        self.assertGreaterEqual(len(completed), 1)
        self.assertEqual(len(pending), 0)


class TestTaskManagerCleanup(_TaskManagerTestCase):
    def test_cleanup_removes_old_tasks(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(tid, timeout_seconds=5.0)
            removed = self.tm.cleanup_completed(max_age_seconds=0.0)

        self.assertGreaterEqual(removed, 1)
        self.assertIsNone(self.tm.get_status(tid))

    def test_cleanup_keeps_recent_tasks(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(tid, timeout_seconds=5.0)
            removed = self.tm.cleanup_completed(max_age_seconds=3600.0)

        self.assertEqual(removed, 0)
        self.assertIsNotNone(self.tm.get_status(tid))


class TestHeavyRuntimeScheduling(_TaskManagerTestCase):
    def test_heavy_tasks_use_separate_executor_budget(self):
        fake = _FakeHeavyMethod(delay=0.01)
        active_count = [0]
        max_concurrent = [0]
        lock = threading.Lock()

        def _fake_heavy_runner(task_id, spec, timeout_seconds):
            with lock:
                active_count[0] += 1
                max_concurrent[0] = max(max_concurrent[0], active_count[0])
            time.sleep(0.15)
            with lock:
                active_count[0] -= 1
            self.tm._mutate_task(
                task_id,
                status="completed",
                completed_at=time.time(),
                heartbeat_at=time.time(),
            )

        with (
            patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg,
            patch.object(self.tm, "_run_heavy_task", side_effect=_fake_heavy_runner),
        ):
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "heavy", "supports_training": True}

            t1, _ = self.tm.submit("heavy_a", _make_series(), 5, 1, {"x": 1}, "EURUSD_H1")
            t2, _ = self.tm.submit("heavy_b", _make_series(), 5, 1, {"x": 2}, "GBPUSD_H1")
            self.tm.wait_for_status(t1, timeout_seconds=5.0)
            self.tm.wait_for_status(t2, timeout_seconds=5.0)

        self.assertEqual(max_concurrent[0], 1)


class TestTaskManagerShutdown(unittest.TestCase):
    def test_submit_after_shutdown_raises(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        store = ModelStore(root=tmpdir)
        job_store = JobStore(path=os.path.join(tmpdir, "jobs.sqlite"))
        tm = TaskManager(max_workers=1, store=store, job_store=job_store)
        tm.shutdown(wait=False)

        with self.assertRaises(RuntimeError):
            tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")

        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


class TestSnapshotImmutability(_TaskManagerTestCase):
    def test_get_status_returns_snapshot_not_live_ref(self):
        fake = _FakeMethod(delay=0.2)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            snap1 = self.tm.get_status(tid)
            snap1.status = "cancelled"
            snap2 = self.tm.get_status(tid)

        self.assertNotEqual(snap2.status, "cancelled")


class TestTaskManagerPersistence(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self._job_store = JobStore(path=os.path.join(self._tmpdir, "jobs.sqlite"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_completed_tasks_survive_manager_restart(self):
        fake = _FakeMethod(delay=0.01)
        tm = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}
            tid, _ = tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            tm.wait_for_status(tid, timeout_seconds=5.0)
        tm.shutdown(wait=True)

        tm_restarted = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        try:
            recovered = tm_restarted.get_status(tid)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.status, "completed")
            self.assertIsNotNone(recovered.result)
        finally:
            tm_restarted.shutdown(wait=True)

    def test_incomplete_persisted_tasks_marked_failed_on_recovery(self):
        created_at = time.time() - 5.0
        self._job_store.upsert(
            JobRecord(
                task_id="orphaned-task",
                method="fake",
                data_scope="EURUSD_H1",
                params_hash="abc123",
                status="running",
                created_at=created_at,
                started_at=created_at + 1.0,
                heartbeat_at=created_at + 2.0,
            )
        )

        tm = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        try:
            recovered = tm.get_status("orphaned-task")
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.status, "failed")
            self.assertIn("orphaned", recovered.error)
        finally:
            tm.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
