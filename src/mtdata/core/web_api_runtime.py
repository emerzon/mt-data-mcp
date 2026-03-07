"""FastAPI runtime assembly helpers for the Web API."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles


def create_web_api_app() -> FastAPI:
    """Create the shared FastAPI app with configured CORS middleware."""
    app = FastAPI(title="mtdata-webui", version="0.1.0")
    origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
        if origin.strip()
    ]
    if not origins:
        origins = ["http://127.0.0.1:5173", "http://localhost:5173"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def mount_webui(app: FastAPI, *, directory: str = "webui/dist") -> None:
    """Mount the built SPA when present; ignore missing build artifacts."""
    try:
        app.mount("/app", StaticFiles(directory=directory, html=True), name="webui")
    except Exception:
        pass


def run_webapi(app: FastAPI) -> None:
    """Run the FastAPI app with the configured host and port."""
    import uvicorn

    host = os.getenv("WEBAPI_HOST", "127.0.0.1")
    port = int(os.getenv("WEBAPI_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
