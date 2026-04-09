"""CLI runtime utilities."""
import io
import logging
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional


def _debug_enabled() -> bool:
    """Check if CLI debug mode is enabled."""
    try:
        v = os.environ.get("MTDATA_CLI_DEBUG", "").strip().lower()
        return v not in ("", "0", "false", "no")
    except Exception:
        return False


def _debug(msg: str) -> None:
    """Print debug message if enabled."""
    if _debug_enabled():
        try:
            print(f"[cli-debug] {msg}", file=sys.stderr)
        except Exception:
            pass


def _argparse_color_enabled() -> bool:
    """Check if argparse color output is enabled."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _configure_cli_logging(*, verbose: bool) -> None:
    """Configure CLI logging."""
    try:
        mtdata_logger = logging.getLogger("mtdata")
        mtdata_logger.setLevel(logging.INFO if verbose else logging.WARNING)
        mtdata_logger.propagate = bool(verbose)
        if not any(isinstance(handler, logging.NullHandler) for handler in mtdata_logger.handlers):
            mtdata_logger.addHandler(logging.NullHandler())
    except Exception:
        pass


@contextmanager
def _temporary_environment(overrides: Dict[str, Optional[str]]):
    """Temporarily override environment variables."""
    previous: Dict[str, Optional[str]] = {}
    missing: set[str] = set()
    for key, value in overrides.items():
        if key in os.environ:
            previous[key] = os.environ.get(key)
        else:
            missing.add(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key in overrides:
            if key in missing:
                os.environ.pop(key, None)
                continue
            restored = previous.get(key)
            if restored is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = restored


@contextmanager
def _suppress_cli_side_output(*, enabled: bool):
    """Suppress stdout/stderr for CLI operations."""
    if not enabled:
        yield
        return
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    previous_disable = logging.root.manager.disable
    env_overrides = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "TQDM_DISABLE": "1",
    }
    try:
        logging.disable(logging.CRITICAL)
        with _temporary_environment(env_overrides):
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                yield
    finally:
        logging.disable(previous_disable)


__all__ = [
    "_debug_enabled",
    "_debug",
    "_argparse_color_enabled",
    "_configure_cli_logging",
    "_temporary_environment",
    "_suppress_cli_side_output",
]
