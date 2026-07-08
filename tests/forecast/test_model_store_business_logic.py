import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtdata.forecast.model_store import ModelStore


class TestModelStoreBusinessLogic(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_logic_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=1)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cleanup_expired_skips_locked_model_dir(self):
        self.store.save("m", "d", "p", b"data")
        model_dir = self.store._model_dir("m", "d", "p")
        meta_path = model_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["last_used"] = meta["created_at"] - 10
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with self.store._model_dir_lock(model_dir, blocking=True):
            removed = self.store.cleanup_expired()
            self.assertEqual(removed, 0)
            self.assertTrue(model_dir.exists())

        removed = self.store.cleanup_expired()
        self.assertEqual(removed, 1)
        self.assertFalse(model_dir.exists())

    def test_delete_stages_directory_when_recursive_delete_fails(self):
        handle = self.store.save("m", "d", "p", b"data")
        model_dir = self.store._model_dir("m", "d", "p")
        deleted_root = self.store._deleted_root()

        with patch("mtdata.forecast.model_store.shutil.rmtree", side_effect=OSError("disk full")):
            removed = self.store.delete(handle.model_id)

        self.assertTrue(removed)
        self.assertFalse(model_dir.exists())
        staged_dirs = [path for path in deleted_root.iterdir() if path.is_dir()]
        self.assertEqual(len(staged_dirs), 1)
        self.assertEqual(self.store.list_models(), [])

    def test_read_raw_meta_warns_on_corrupt_metadata(self):
        self.store.save("m", "d", "p", b"data")
        model_dir = self.store._model_dir("m", "d", "p")
        (model_dir / "metadata.json").write_text("{corrupt json", encoding="utf-8")

        with self.assertLogs("mtdata.forecast.model_store", level="WARNING") as captured:
            self.assertIsNone(self.store._read_raw_meta(model_dir))
        self.assertTrue(any("metadata.json" in msg for msg in captured.output))

    def test_read_raw_meta_silent_when_metadata_missing(self):
        model_dir = self.store._model_dir("m", "d", "absent")
        model_dir.mkdir(parents=True, exist_ok=True)
        # No metadata.json present: returns None without emitting a warning.
        self.assertIsNone(self.store._read_raw_meta(model_dir))


if __name__ == "__main__":
    unittest.main()
