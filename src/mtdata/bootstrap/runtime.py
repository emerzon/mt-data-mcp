from __future__ import annotations

import hmac
import os
import warnings
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Literal, Optional

from .env import get_bool_env

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


_get_bool_env = get_bool_env


def _normalize_transport(value: Optional[str], *, default: TransportLiteral = "sse") -> TransportLiteral:
    candidate = str(value or default).strip().lower()
    if candidate not in ("stdio", "sse", "streamable-http"):
        return default
    return candidate  # type: ignore[return-value]


def is_loopback_host(value: Optional[str]) -> bool:
    host = str(value or "").strip().lower()
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _require_explicit_remote_bind(host: str, *, allow_remote: bool, host_env: str, allow_remote_env: str) -> str:
    normalized = str(host or "").strip() or "127.0.0.1"
    if is_loopback_host(normalized) or allow_remote:
        return normalized
    warnings.warn(
        f"{host_env}={normalized!r} is a non-loopback bind. "
        f"Set {allow_remote_env}=1 to suppress this warning. "
        f"For local-only startup, set {host_env}=127.0.0.1.",
        RuntimeWarning,
        stacklevel=3,
    )
    return normalized


def _validate_cors_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    if any(str(origin).strip() == "*" for origin in origins):
        raise ValueError(
            "CORS_ORIGINS cannot include '*' while credentialed requests are enabled; specify explicit origins."
        )
    return origins


@dataclass(frozen=True)
class McpRuntimeSettings:
    transport: TransportLiteral = "sse"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    mount_path: str = "/"
    sse_path: str = "/sse"
    message_path: str = "/message"
    allow_remote: bool = False
    auth_token: Optional[str] = None


@dataclass(frozen=True)
class WebApiRuntimeSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
    webui_directory: str = "webui/dist"
    auth_token: Optional[str] = None
    allow_remote: bool = False


def load_mcp_runtime_settings(
    *,
    transport_override: Optional[str] = None,
    default_transport: TransportLiteral = "sse",
) -> McpRuntimeSettings:
    transport = _normalize_transport(transport_override or os.getenv("MCP_TRANSPORT"), default=default_transport)
    allow_remote = _get_bool_env("FASTMCP_ALLOW_REMOTE", False)
    host = (os.getenv("FASTMCP_HOST", "127.0.0.1").strip() or "127.0.0.1")
    if transport != "stdio":
        host = _require_explicit_remote_bind(
            host,
            allow_remote=allow_remote,
            host_env="FASTMCP_HOST",
            allow_remote_env="FASTMCP_ALLOW_REMOTE",
        )
    auth_token_raw = os.getenv("MCP_AUTH_TOKEN")
    auth_token = str(auth_token_raw or "").strip() or None
    if transport != "stdio" and not is_loopback_host(host) and not auth_token:
        raise ValueError(
            "MCP_AUTH_TOKEN is required for non-loopback FASTMCP_HOST values "
            "(SSE/streamable-HTTP expose trading tools)."
        )
    return McpRuntimeSettings(
        transport=transport,
        host=host,
        port=_get_int_env("FASTMCP_PORT", 8000),
        log_level=(os.getenv("FASTMCP_LOG_LEVEL", "INFO").strip() or "INFO"),
        mount_path=(os.getenv("FASTMCP_MOUNT_PATH", "/").strip() or "/"),
        sse_path=(os.getenv("FASTMCP_SSE_PATH", "/sse").strip() or "/sse"),
        message_path=(os.getenv("FASTMCP_MESSAGE_PATH", "/message").strip() or "/message"),
        allow_remote=allow_remote,
        auth_token=auth_token,
    )


def load_web_api_runtime_settings() -> WebApiRuntimeSettings:
    allow_remote = _get_bool_env("WEBAPI_ALLOW_REMOTE", False)
    host = _require_explicit_remote_bind(
        (os.getenv("WEBAPI_HOST", "127.0.0.1").strip() or "127.0.0.1"),
        allow_remote=allow_remote,
        host_env="WEBAPI_HOST",
        allow_remote_env="WEBAPI_ALLOW_REMOTE",
    )
    auth_token_raw = os.getenv("WEBAPI_AUTH_TOKEN")
    auth_token = str(auth_token_raw or "").strip() or None
    if not is_loopback_host(host) and not auth_token:
        raise ValueError("WEBAPI_AUTH_TOKEN is required for non-loopback WEBAPI_HOST values.")
    cors_origins = _validate_cors_origins(
        _get_csv_env(
            "CORS_ORIGINS",
            ("http://127.0.0.1:5173", "http://localhost:5173"),
        )
    )
    return WebApiRuntimeSettings(
        host=host,
        port=_get_int_env("WEBAPI_PORT", 8000),
        cors_origins=cors_origins,
        webui_directory=(os.getenv("WEBUI_DIST_DIR", "webui/dist").strip() or "webui/dist"),
        auth_token=auth_token,
        allow_remote=allow_remote,
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
    if settings.auth_token and settings.transport != "stdio":
        _install_mcp_bearer_auth(mcp, settings.auth_token)


def _install_mcp_bearer_auth(mcp: Any, token: str) -> None:
    """Wrap FastMCP HTTP apps so requests require Bearer / X-API-Key auth."""
    expected = str(token).strip()
    if not expected:
        return
    if getattr(mcp, "_mtdata_auth_installed", False):
        return

    def _authorized(headers: Any) -> bool:
        try:
            auth = str(headers.get("authorization") or headers.get("Authorization") or "")
        except Exception:
            auth = ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
            if provided and hmac.compare_digest(provided, expected):
                return True
        try:
            api_key = str(headers.get("x-api-key") or headers.get("X-API-Key") or "")
        except Exception:
            api_key = ""
        return bool(api_key) and hmac.compare_digest(api_key, expected)

    def _wrap_app_factory(original: Any) -> Any:
        def factory(*args: Any, **kwargs: Any) -> Any:
            app = original(*args, **kwargs)
            try:
                from starlette.middleware.base import BaseHTTPMiddleware
                from starlette.responses import JSONResponse
            except Exception as exc:
                raise RuntimeError(
                    "MCP authentication middleware dependencies are unavailable."
                ) from exc

            class _BearerAuthMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
                    if _authorized(request.headers):
                        return await call_next(request)
                    return JSONResponse(
                        {
                            "error": "Unauthorized",
                            "error_code": "mcp_auth_required",
                            "message": (
                                "MCP HTTP transport requires Authorization: Bearer "
                                "<MCP_AUTH_TOKEN> or X-API-Key."
                            ),
                        },
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )

            try:
                app.add_middleware(_BearerAuthMiddleware)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to attach required MCP authentication middleware."
                ) from exc
            return app

        return factory

    wrapped = 0
    for attr in ("sse_app", "streamable_http_app"):
        original = getattr(mcp, attr, None)
        if callable(original):
            setattr(mcp, attr, _wrap_app_factory(original))
            wrapped += 1
    if wrapped == 0:
        raise RuntimeError("FastMCP exposes no HTTP app factory to protect.")
    setattr(mcp, "_mtdata_auth_installed", True)
