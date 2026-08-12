import logging
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import pytest

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

    def test_corrupted_payload_json_does_not_break_reads(self):
        self.store.upsert(
            JobRecord(
                task_id="task-corrupt",
                method="nhits",
                data_scope="EURUSD_H1",
                params_hash="hash-1",
                status="running",
                created_at=1000.0,
                progress_payload={"step": 1},
                result_payload={"ok": True},
            )
        )
        with self.store._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress_json = ? WHERE task_id = ?",
                ("{broken json", "task-corrupt"),
            )
            conn.commit()

        loaded = self.store.get("task-corrupt")

        self.assertIsNotNone(loaded)
        self.assertIsNone(loaded.progress_payload)
        self.assertEqual(loaded.result_payload, {"ok": True})
        self.assertEqual(
            [record.task_id for record in self.store.list_jobs()],
            ["task-corrupt"],
        )

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

    def test_mark_stale_active_job_failed_rechecks_heartbeat(self):
        now = time.time()
        self.store.upsert(
            JobRecord(
                task_id="active-task",
                method="nhits",
                data_scope="EURUSD_H1",
                params_hash="hash-1",
                status="running",
                created_at=now - 60.0,
                heartbeat_at=now,
            )
        )

        updated = self.store.mark_stale_active_job_failed(
            "active-task",
            "orphaned",
            stale_before=now - 30.0,
        )

        self.assertFalse(updated)
        self.assertEqual(self.store.get("active-task").status, "running")


def test_corrupt_job_database_is_quarantined_and_recreated(tmp_path, caplog):
    database = tmp_path / "jobs.sqlite"
    corrupt_bytes = b"CORRUPTED-GARBAGE-DATA"
    database.write_bytes(corrupt_bytes)

    with caplog.at_level(logging.WARNING, logger="mtdata.forecast.job_store"):
        store = JobStore(path=str(database))

    quarantined = list(tmp_path.glob("jobs.sqlite.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt_bytes
    assert store.list_jobs() == []
    store.upsert(
        JobRecord(
            task_id="new-task",
            method="theta",
            data_scope="EURUSD_H1",
            params_hash="hash",
            status="pending",
            created_at=1.0,
        )
    )
    assert store.get("new-task") is not None
    assert "has been quarantined" in caplog.text


def test_non_corruption_database_error_is_not_replaced(tmp_path, monkeypatch):
    database = tmp_path / "jobs.sqlite"
    database.write_bytes(b"preserve-me")

    def locked(_self):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(JobStore, "_init_db", locked)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        JobStore(path=str(database))

    assert database.read_bytes() == b"preserve-me"
    assert list(tmp_path.glob("jobs.sqlite.corrupt-*")) == []


if __name__ == "__main__":
    unittest.main()
