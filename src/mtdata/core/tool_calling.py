from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any


def unwrap_tool_callable(func: Any) -> Any:
    raw = getattr(func, "__wrapped__", None)
    return raw if callable(raw) else func


def resolve_sync_tool_result(result: Any) -> Any:
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    state: dict[str, Any] = {}

    def _runner() -> None:
        try:
            state["result"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover
            state["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in state:
        raise state["error"]
    return state.get("result")


def call_tool_sync_raw(func: Any, *args: Any, cli_raw: bool = False, **kwargs: Any) -> Any:
    target = unwrap_tool_callable(func)
    if not cli_raw:
        return resolve_sync_tool_result(target(*args, **kwargs))

    raw_kwargs = dict(kwargs)
    raw_kwargs["__cli_raw"] = True
    try:
        return resolve_sync_tool_result(target(*args, **raw_kwargs))
    except TypeError:
        return resolve_sync_tool_result(target(*args, **kwargs))
