"""Cross-process coverage for temporary MT5 symbol visibility coordination."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Any


def _hold_visibility_lock(
    lock_path: str,
    acquired: Any,
    release: Any,
) -> None:
    os.environ["MTDATA_SYMBOL_VISIBILITY_LOCK"] = lock_path
    from mtdata.utils.mt5 import _symbol_visibility_snapshot_guard

    with _symbol_visibility_snapshot_guard():
        acquired.set()
        release.wait(10.0)


def _enter_visibility_lock(lock_path: str, entered: Any) -> None:
    os.environ["MTDATA_SYMBOL_VISIBILITY_LOCK"] = lock_path
    from mtdata.utils.mt5 import _symbol_visibility_snapshot_guard

    with _symbol_visibility_snapshot_guard():
        entered.set()


def _enter_nested_visibility_lock(lock_path: str, entered: Any) -> None:
    os.environ["MTDATA_SYMBOL_VISIBILITY_LOCK"] = lock_path
    from mtdata.utils.mt5 import _symbol_visibility_snapshot_guard

    with _symbol_visibility_snapshot_guard():
        with _symbol_visibility_snapshot_guard():
            entered.set()


def test_visibility_lock_serializes_independent_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    second_entered = context.Event()
    lock_path = str(tmp_path / "symbol-visibility.lock")
    holder = context.Process(
        target=_hold_visibility_lock,
        args=(lock_path, acquired, release),
    )
    observer = context.Process(
        target=_enter_visibility_lock,
        args=(lock_path, second_entered),
    )

    try:
        holder.start()
        assert acquired.wait(10.0)
        observer.start()
        assert not second_entered.wait(0.3)
        release.set()
        assert second_entered.wait(10.0)
        holder.join(10.0)
        observer.join(10.0)
        assert holder.exitcode == 0
        assert observer.exitcode == 0
    finally:
        release.set()
        for process in (holder, observer):
            if process.is_alive():
                process.terminate()
            process.join(5.0)


def test_visibility_lock_is_reentrant_within_one_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    process = context.Process(
        target=_enter_nested_visibility_lock,
        args=(str(tmp_path / "nested-symbol-visibility.lock"), entered),
    )

    try:
        process.start()
        assert entered.wait(10.0)
        process.join(10.0)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5.0)
