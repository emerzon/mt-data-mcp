import tempfile
import time
import unittest
from pathlib import Path

from mtdata.forecast.job_store import JobRecord, JobStore


class TestJobStore(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = JobStore(path=str(Path(self._tmpdir) / "jobs.sqlite"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_upsert_and_get_roundtrip(self):
        record = JobRecord(
            task_id="task-1",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="hash-1",
            status="running",
            created_at=1000.0,
            progress_payload={"step": 1, "total_steps": 10},
            pid=4321,
            heartbeat_at=1001.0,
        )
        self.store.upsert(record)

        loaded = self.store.get("task-1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.status, "running")
        self.assertEqual(loaded.progress_payload["step"], 1)
        self.assertEqual(loaded.pid, 4321)

    def test_find_active_ignores_completed(self):
        self.store.upsert(
            JobRecord(
                task_id="done-task",
                method="nhits",
                data_scope="EURUSD_H1",
                params_hash="hash-1",
                status="completed",
                created_at=1000.0,
                completed_at=1010.0,
            )
        )
        self.assertIsNone(self.store.find_active("nhits", "EURUSD_H1", "hash-1"))

    def test_cleanup_completed_returns_removed_ids(self):
        finished_at = time.time() - 30.0
        self.store.upsert(
            JobRecord(
                task_id="old-task",
                method="nhits",
                data_scope="EURUSD_H1",
                params_hash="hash-1",
                status="completed",
                created_at=finished_at - 10.0,
                completed_at=finished_at,
            )
        )

        removed = self.store.cleanup_completed(max_age_seconds=0.0)
        self.assertEqual(removed, ["old-task"])
        self.assertIsNone(self.store.get("old-task"))

    def test_mark_active_jobs_failed(self):
        self.store.upsert(
            JobRecord(
                task_id="active-task",
                method="nhits",
                data_scope="EURUSD_H1",
                params_hash="hash-1",
                status="running",
                created_at=time.time(),
            )
        )

        updated = self.store.mark_active_jobs_failed("recovered")
        self.assertEqual(updated, 1)
        loaded = self.store.get("active-task")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.status, "failed")
        self.assertEqual(loaded.error, "recovered")


if __name__ == "__main__":
    unittest.main()
