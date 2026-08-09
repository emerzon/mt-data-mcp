import logging
import sqlite3

import pytest

from mtdata.forecast.job_store import JobRecord, JobStore


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
