"""Tests for forecast TaskManager background training infrastructure."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mtdata.forecast.interface import (
    ForecastMethod,
    ForecastResult,
    TrainedModelHandle,
    TrainingProgress,
    TrainResult,
)
from mtdata.forecast.model_store import ModelStore
from mtdata.forecast.task_manager import TaskManager, TrainingTask, _snapshot


def _make_series(n: int = 100) -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(np.linspace(100, 120, n) + rng.normal(0, 0.5, n))


class _FakeMethod(ForecastMethod):
    """A fake method that simulates training with configurable delay."""

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

    def forecast(self, series, horizon, seasonality, params, **kw) -> ForecastResult:
        return ForecastResult(forecast=np.ones(horizon), params_used=params)

    def train(self, series, horizon, seasonality, params, *, progress_callback=None, **kw) -> TrainResult:
        if progress_callback:
            progress_callback(TrainingProgress(step=0, total_steps=2))
        time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("training boom")
        if progress_callback:
            progress_callback(TrainingProgress(step=2, total_steps=2))
        return TrainResult(artifact_bytes=b"model-data", params_used=params or {})


class _FakeHeavyMethod(_FakeMethod):
    @property
    def training_category(self) -> str:
        return "heavy"


# ------------------------------------------------------------------
# Snapshot tests
# ------------------------------------------------------------------

class TestSnapshot(unittest.TestCase):
    def test_snapshot_returns_copy(self):
        task = TrainingTask(task_id="x", method="m", data_scope="s", params_hash="h")
        snap = _snapshot(task)
        self.assertEqual(snap.task_id, "x")
        snap.status = "failed"
        self.assertEqual(task.status, "pending")  # original unchanged


# ------------------------------------------------------------------
# TaskManager basic lifecycle
# ------------------------------------------------------------------

class TestTaskManagerBasic(unittest.TestCase):

    def setUp(self):
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self.tm = TaskManager(max_workers=2, store=self._store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch.object(ForecastMethod, '__init_subclass__', lambda **kw: None)
    def test_submit_and_complete(self):
        fake = _FakeMethod(delay=0.05)
        with patch.object(type(self.tm), 'store', new_callable=lambda: property(lambda s: self._store)):
            pass
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, is_new = self.tm.submit(
                "fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123",
            )
            self.assertTrue(is_new)
            self.assertIsInstance(task_id, str)

            # Wait for completion
            for _ in range(100):
                status = self.tm.get_status(task_id)
                if status and status.status in ("completed", "failed"):
                    break
                time.sleep(0.05)

            final = self.tm.get_status(task_id)
            self.assertIsNotNone(final)
            self.assertEqual(final.status, "completed")
            self.assertIsNotNone(final.result)
            self.assertIsInstance(final.result, TrainedModelHandle)
            self.assertIsNotNone(final.started_at)
            self.assertIsNotNone(final.completed_at)

    def test_submit_failed_task(self):
        fake = _FakeMethod(delay=0.01, fail=True)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123")

            for _ in range(100):
                status = self.tm.get_status(task_id)
                if status and status.status in ("completed", "failed"):
                    break
                time.sleep(0.05)

            final = self.tm.get_status(task_id)
            self.assertEqual(final.status, "failed")
            self.assertIn("training boom", final.error)

    def test_dedup_identical_submissions(self):
        fake = _FakeMethod(delay=0.3)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            series = _make_series()
            tid1, new1 = self.tm.submit("fake", series, 5, 1, {}, "EURUSD_H1", "abc123")
            tid2, new2 = self.tm.submit("fake", series, 5, 1, {}, "EURUSD_H1", "abc123")

            self.assertTrue(new1)
            self.assertFalse(new2)
            self.assertEqual(tid1, tid2)

    def test_different_params_hash_no_dedup(self):
        fake = _FakeMethod(delay=0.3)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            series = _make_series()
            tid1, new1 = self.tm.submit("fake", series, 5, 1, {}, "EURUSD_H1", "hash_a")
            tid2, new2 = self.tm.submit("fake", series, 5, 1, {}, "EURUSD_H1", "hash_b")

            self.assertTrue(new1)
            self.assertTrue(new2)
            self.assertNotEqual(tid1, tid2)

    def test_progress_callback_updates_task(self):
        fake = _FakeMethod(delay=0.1)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123")

            for _ in range(100):
                status = self.tm.get_status(task_id)
                if status and status.status == "completed":
                    break
                time.sleep(0.05)

            final = self.tm.get_status(task_id)
            self.assertIsNotNone(final.progress)
            self.assertEqual(final.progress.total_steps, 2)


# ------------------------------------------------------------------
# Task listing and cancellation
# ------------------------------------------------------------------

class TestTaskManagerListCancel(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self.tm = TaskManager(max_workers=2, store=self._store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_list_tasks_empty(self):
        self.assertEqual(self.tm.list_tasks(), [])

    def test_list_tasks_with_filter(self):
        fake = _FakeMethod(delay=0.01)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123")
            time.sleep(0.3)  # let it complete

            completed = self.tm.list_tasks(status="completed")
            pending = self.tm.list_tasks(status="pending")
            self.assertGreaterEqual(len(completed), 1)
            self.assertEqual(len(pending), 0)

    def test_cancel_unknown_task(self):
        self.assertFalse(self.tm.cancel("nonexistent"))

    def test_cancel_completed_task(self):
        fake = _FakeMethod(delay=0.01)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123")
            time.sleep(0.3)
            self.assertFalse(self.tm.cancel(tid))


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------

class TestTaskManagerCleanup(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self.tm = TaskManager(max_workers=2, store=self._store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cleanup_removes_old_tasks(self):
        fake = _FakeMethod(delay=0.01)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123")
            time.sleep(0.3)

            # With max_age=0, everything completed should be removed
            removed = self.tm.cleanup_completed(max_age_seconds=0.0)
            self.assertGreaterEqual(removed, 1)
            self.assertIsNone(self.tm.get_status(tid))

    def test_cleanup_keeps_recent_tasks(self):
        fake = _FakeMethod(delay=0.01)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123")
            time.sleep(0.3)

            removed = self.tm.cleanup_completed(max_age_seconds=3600.0)
            self.assertEqual(removed, 0)
            self.assertIsNotNone(self.tm.get_status(tid))


# ------------------------------------------------------------------
# Heavy method semaphore
# ------------------------------------------------------------------

class TestHeavySemaphore(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self.tm = TaskManager(max_workers=4, heavy_limit=1, store=self._store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_heavy_methods_serialized(self):
        """Two heavy tasks should not run simultaneously."""
        concurrency_log = []
        active_count = [0]
        max_concurrent = [0]
        lock = threading.Lock()

        class _SlowHeavy(_FakeHeavyMethod):
            def train(self, series, horizon, seasonality, params, *, progress_callback=None, **kw):
                with lock:
                    active_count[0] += 1
                    max_concurrent[0] = max(max_concurrent[0], active_count[0])
                time.sleep(0.15)
                with lock:
                    active_count[0] -= 1
                return TrainResult(artifact_bytes=b"heavy-model", params_used={})

        fake = _SlowHeavy(delay=0)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "heavy", "supports_training": True}

            series = _make_series()
            self.tm.submit("heavy_a", series, 5, 1, {}, "EURUSD_H1", "h1")
            self.tm.submit("heavy_b", series, 5, 1, {}, "GBPUSD_H1", "h2")

            # Wait for both to finish
            time.sleep(1.0)

            self.assertEqual(max_concurrent[0], 1, "Heavy tasks should serialize, max concurrent should be 1")


# ------------------------------------------------------------------
# Shutdown
# ------------------------------------------------------------------

class TestTaskManagerShutdown(unittest.TestCase):

    def test_submit_after_shutdown_raises(self):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        store = ModelStore(root=tmpdir)
        tm = TaskManager(max_workers=1, store=store)
        tm.shutdown(wait=False)

        with self.assertRaises(RuntimeError):
            tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc")

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ------------------------------------------------------------------
# Snapshot immutability
# ------------------------------------------------------------------

class TestSnapshotImmutability(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self.tm = TaskManager(max_workers=1, store=self._store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_get_status_returns_snapshot_not_live_ref(self):
        fake = _FakeMethod(delay=0.3)
        with patch('mtdata.forecast.task_manager.ForecastRegistry') as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1", "abc123")

            snap1 = self.tm.get_status(tid)
            # Mutate the snapshot
            snap1.status = "cancelled"

            # Internal state should be unaffected
            snap2 = self.tm.get_status(tid)
            self.assertNotEqual(snap2.status, "cancelled")


if __name__ == "__main__":
    unittest.main()
