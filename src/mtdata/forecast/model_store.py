"""Persistent store for trained forecast model artifacts.

Provides a thread-safe, TTL-aware store that persists trained models to
disk so that subsequent prediction requests can reuse them without
retraining.

Configuration:
    - ``MTDATA_MODEL_STORE``: root directory (default ``~/.mtdata/models``).
    - ``MTDATA_MODEL_TTL_DAYS``: expiry in days (default 7).

Directory layout::

    {root}/{method}/{data_scope}/{params_hash}/
        model.bin          # method-serialized artifact bytes
        metadata.json      # TrainedModelHandle + training metrics

Serialization is **method-owned**: the store persists opaque bytes
produced by ``ForecastMethod.serialize_artifact()`` and delegates
deserialization to ``ForecastMethod.deserialize_artifact()``.

Writes are **atomic**: data is written to a temp file then renamed
via ``os.replace`` to avoid partial artifacts on crash.

Usage::

    from mtdata.forecast.model_store import model_store

    handle = model_store.save(
        method="nhits",
        data_scope="EURUSD_H1",
        params_hash="abc123",
        artifact_bytes=trained_bytes,
        metadata={"epochs": 100, "final_loss": 0.02},
    )

    raw = model_store.load_bytes(handle.model_id)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interface import TrainedModelHandle

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".mtdata", "models")
_DEFAULT_TTL_DAYS = 7.0


def _env_root() -> str:
    return os.environ.get("MTDATA_MODEL_STORE", _DEFAULT_ROOT)


def _env_ttl_seconds() -> float:
    try:
        days = float(os.environ.get("MTDATA_MODEL_TTL_DAYS", _DEFAULT_TTL_DAYS))
    except (TypeError, ValueError):
        days = _DEFAULT_TTL_DAYS
    return max(0.0, days * 86400.0)


class ModelStore:
    """Thread-safe persistent store for trained model artifacts."""

    def __init__(
        self,
        root: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        self._root = Path(root) if root else Path(_env_root())
        self._ttl = ttl_seconds if ttl_seconds is not None else _env_ttl_seconds()
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def _model_dir(self, method: str, data_scope: str, params_hash: str) -> Path:
        safe_scope = data_scope.replace("/", "_").replace("\\", "_")
        return self._root / method / safe_scope / params_hash

    @staticmethod
    def _model_id(method: str, data_scope: str, params_hash: str) -> str:
        safe_scope = data_scope.replace("/", "_").replace("\\", "_")
        return f"{method}/{safe_scope}/{params_hash}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        method: str,
        data_scope: str,
        params_hash: str,
        artifact_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrainedModelHandle:
        """Persist method-serialized artifact bytes and return a handle.

        Writes are atomic (temp file + ``os.replace``).
        """
        model_id = self._model_id(method, data_scope, params_hash)
        model_dir = self._model_dir(method, data_scope, params_hash)

        with self._lock:
            model_dir.mkdir(parents=True, exist_ok=True)

            # Atomic write: artifact
            artifact_path = model_dir / "model.bin"
            self._atomic_write_bytes(artifact_path, artifact_bytes)

            # Build handle
            now = time.time()
            handle = TrainedModelHandle(
                model_id=model_id,
                method=method,
                data_scope=data_scope,
                params_hash=params_hash,
                created_at=now,
                metadata=dict(metadata or {}),
            )

            # Atomic write: metadata (includes last_used)
            meta_dict: Dict[str, Any] = {
                "model_id": handle.model_id,
                "method": handle.method,
                "data_scope": handle.data_scope,
                "params_hash": handle.params_hash,
                "created_at": handle.created_at,
                "last_used": now,
                "metadata": handle.metadata,
            }
            meta_path = model_dir / "metadata.json"
            self._atomic_write_text(
                meta_path, json.dumps(meta_dict, indent=2, default=str),
            )

        logger.debug("Saved model %s to %s", model_id, model_dir)
        return handle

    def load_bytes(self, model_id: str) -> bytes:
        """Load raw artifact bytes by *model_id*.

        Updates ``last_used`` on successful access.
        Raises ``FileNotFoundError`` if the model does not exist.
        """
        parts = model_id.split("/")
        if len(parts) != 3:
            raise FileNotFoundError(f"Invalid model_id format: {model_id}")
        method, data_scope, params_hash = parts
        model_dir = self._model_dir(method, data_scope, params_hash)
        artifact_path = model_dir / "model.bin"

        if not artifact_path.is_file():
            raise FileNotFoundError(f"No model artifact at {artifact_path}")

        data = artifact_path.read_bytes()
        self._touch_last_used(model_dir)
        return data

    def find(
        self,
        method: str,
        data_scope: str,
        params_hash: str,
    ) -> Optional[TrainedModelHandle]:
        """Look up a trained model handle.

        Returns ``None`` if not found or expired (based on ``last_used``).
        """
        model_dir = self._model_dir(method, data_scope, params_hash)
        return self._read_handle(model_dir)

    def list_models(
        self,
        method: Optional[str] = None,
    ) -> List[TrainedModelHandle]:
        """List all stored model handles, optionally filtered by method."""
        handles: List[TrainedModelHandle] = []
        if not self._root.is_dir():
            return handles

        for method_dir in sorted(self._root.iterdir()):
            if not method_dir.is_dir():
                continue
            if method and method_dir.name != method:
                continue
            for scope_dir in sorted(method_dir.iterdir()):
                if not scope_dir.is_dir():
                    continue
                for hash_dir in sorted(scope_dir.iterdir()):
                    if not hash_dir.is_dir():
                        continue
                    handle = self._read_handle(hash_dir, skip_expiry=True)
                    if handle is not None:
                        handles.append(handle)
        return handles

    def delete(self, model_id: str) -> bool:
        """Delete a model by *model_id*. Returns ``True`` if it existed."""
        parts = model_id.split("/")
        if len(parts) != 3:
            return False
        method, data_scope, params_hash = parts
        model_dir = self._model_dir(method, data_scope, params_hash)
        return self._remove_dir(model_dir)

    def cleanup_expired(self) -> int:
        """Remove all models that have exceeded the TTL. Returns count removed."""
        if self._ttl <= 0:
            return 0
        removed = 0
        now = time.time()
        for handle in self.list_models():
            meta = self._read_raw_meta(
                self._model_dir(handle.method, handle.data_scope, handle.params_hash)
            )
            last_used = float(meta.get("last_used", handle.created_at)) if meta else handle.created_at
            if (now - last_used) > self._ttl:
                if self.delete(handle.model_id):
                    removed += 1
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_handle(
        self, model_dir: Path, *, skip_expiry: bool = False,
    ) -> Optional[TrainedModelHandle]:
        meta = self._read_raw_meta(model_dir)
        if meta is None:
            return None

        created_at = float(meta.get("created_at", 0))
        last_used = float(meta.get("last_used", created_at))

        if not skip_expiry and self._ttl > 0 and (time.time() - last_used) > self._ttl:
            self._remove_dir(model_dir)
            return None

        return TrainedModelHandle(
            model_id=meta.get("model_id", ""),
            method=meta.get("method", model_dir.parent.parent.name if model_dir.parent.parent else ""),
            data_scope=meta.get("data_scope", ""),
            params_hash=meta.get("params_hash", model_dir.name),
            created_at=created_at,
            metadata=meta.get("metadata", {}),
        )

    @staticmethod
    def _read_raw_meta(model_dir: Path) -> Optional[Dict[str, Any]]:
        meta_path = model_dir / "metadata.json"
        if not meta_path.is_file():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _touch_last_used(self, model_dir: Path) -> None:
        """Update ``last_used`` in metadata.json without rewriting the artifact."""
        meta = self._read_raw_meta(model_dir)
        if meta is None:
            return
        meta["last_used"] = time.time()
        meta_path = model_dir / "metadata.json"
        try:
            self._atomic_write_text(
                meta_path, json.dumps(meta, indent=2, default=str),
            )
        except Exception as exc:
            logger.debug("Failed to update last_used for %s: %s", model_dir, exc)

    @staticmethod
    def _atomic_write_bytes(target: Path, data: bytes) -> None:
        """Write *data* atomically via temp file + ``os.replace``."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp",
        )
        try:
            os.write(fd, data)
            os.close(fd)
            fd = -1
            os.replace(tmp_path, str(target))
        except BaseException:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_text(target: Path, text: str) -> None:
        """Write *text* atomically via temp file + ``os.replace``."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp",
        )
        try:
            os.write(fd, text.encode("utf-8"))
            os.close(fd)
            fd = -1
            os.replace(tmp_path, str(target))
        except BaseException:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _remove_dir(self, path: Path) -> bool:
        try:
            if path.is_dir():
                shutil.rmtree(path)
                return True
        except Exception as exc:
            logger.debug("Failed to remove %s: %s", path, exc)
        return False


# Module-level singleton
model_store = ModelStore()
