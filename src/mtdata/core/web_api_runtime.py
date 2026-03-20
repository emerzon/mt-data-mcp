"""FastAPI runtime assembly helpers for the Web API."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from ..bootstrap.runtime import WebApiRuntimeSettings, load_web_api_runtime_settings

logger = logging.getLogger(__name__)


def create_web_api_app(settings: WebApiRuntimeSettings | None = None) -> FastAPI:
    """Create the shared FastAPI app with configured CORS middleware."""
    runtime = settings or load_web_api_runtime_settings()
    app = FastAPI(title="mtdata-webui", version="0.1.0")
    origins = list(runtime.cors_origins)
    if not origins:
        origins = ["http://127.0.0.1:5173", "http://localhost:5173"]
    if any(str(origin).strip() == "*" for origin in origins):
        raise ValueError(
            "CORS_ORIGINS cannot include '*' while credentialed requests are enabled; specify explicit origins."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def mount_webui(
    app: FastAPI,
    *,
    directory: str | None = None,
    settings: WebApiRuntimeSettings | None = None,
) -> None:
    """Mount the built SPA when present; ignore missing build artifacts."""
    runtime = settings or load_web_api_runtime_settings()
    try:
        app.mount("/app", StaticFiles(directory=directory or runtime.webui_directory, html=True), name="webui")
    except Exception as exc:
        logger.warning("Skipping Web UI mount for %s: %s", directory or runtime.webui_directory, exc)


def run_webapi(app: FastAPI, settings: WebApiRuntimeSettings | None = None) -> None:
    """Run the FastAPI app with the configured host and port."""
    import uvicorn

    runtime = settings or load_web_api_runtime_settings()
    uvicorn.run(app, host=runtime.host, port=runtime.port)
