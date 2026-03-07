from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

TransportLiteral = Literal["stdio", "sse", "streamable-http"]


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _get_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or default


def _normalize_transport(value: Optional[str], *, default: TransportLiteral = "sse") -> TransportLiteral:
    candidate = str(value or default).strip().lower()
    if candidate not in ("stdio", "sse", "streamable-http"):
        return default
    return candidate  # type: ignore[return-value]


@dataclass(frozen=True)
class McpRuntimeSettings:
    transport: TransportLiteral = "sse"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    mount_path: str = "/"
    sse_path: str = "/sse"
    message_path: str = "/message"


@dataclass(frozen=True)
class WebApiRuntimeSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
    webui_directory: str = "webui/dist"


def load_mcp_runtime_settings(
    *,
    transport_override: Optional[str] = None,
    default_transport: TransportLiteral = "sse",
) -> McpRuntimeSettings:
    return McpRuntimeSettings(
        transport=_normalize_transport(transport_override or os.getenv("MCP_TRANSPORT"), default=default_transport),
        host=(os.getenv("FASTMCP_HOST", "0.0.0.0").strip() or "0.0.0.0"),
        port=_get_int_env("FASTMCP_PORT", 8000),
        log_level=(os.getenv("FASTMCP_LOG_LEVEL", "INFO").strip() or "INFO"),
        mount_path=(os.getenv("FASTMCP_MOUNT_PATH", "/").strip() or "/"),
        sse_path=(os.getenv("FASTMCP_SSE_PATH", "/sse").strip() or "/sse"),
        message_path=(os.getenv("FASTMCP_MESSAGE_PATH", "/message").strip() or "/message"),
    )


def load_web_api_runtime_settings() -> WebApiRuntimeSettings:
    return WebApiRuntimeSettings(
        host=(os.getenv("WEBAPI_HOST", "127.0.0.1").strip() or "127.0.0.1"),
        port=_get_int_env("WEBAPI_PORT", 8000),
        cors_origins=_get_csv_env(
            "CORS_ORIGINS",
            ("http://127.0.0.1:5173", "http://localhost:5173"),
        ),
        webui_directory=(os.getenv("WEBUI_DIST_DIR", "webui/dist").strip() or "webui/dist"),
    )


def apply_mcp_runtime_settings(mcp: Any, settings: McpRuntimeSettings) -> None:
    runtime = getattr(mcp, "settings", None)
    if runtime is None:
        return
    runtime.host = settings.host
    runtime.port = settings.port
    runtime.log_level = settings.log_level
    runtime.mount_path = settings.mount_path
    runtime.sse_path = settings.sse_path
    runtime.message_path = settings.message_path
