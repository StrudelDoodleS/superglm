"""FastAPI server for the notebook editor frontend."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from superglm.editor.assets import app_asset_content_type, read_app_asset
from superglm.editor.io import jsonable

_LOGGER = logging.getLogger(__name__)
_CLIENT_ERROR_TYPES = (KeyError, ValueError, TypeError, IndexError, FileNotFoundError)


def create_editor_app(widget: Any) -> FastAPI:
    """Create the local HTTP app that binds browser routes to an editor widget."""
    app = FastAPI(title="SuperGLM Editor", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def require_editor_token(request: Request, call_next):
        if _is_public_path(request.url.path):
            return await call_next(request)
        if _request_token(request) == getattr(widget, "_token", None):
            return await call_next(request)
        return _json_response({"error": "invalid or missing editor token"}, status_code=403)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception(_request: Request, exc: StarletteHTTPException) -> Response:
        message = "not found" if exc.status_code == 404 else str(exc.detail)
        return _json_response({"error": message}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception(_request: Request, exc: RequestValidationError) -> Response:
        return _json_response({"error": str(exc)}, status_code=400)

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        return _asset_response("index.html")

    @app.get("/assets/{asset_path:path}", include_in_schema=False)
    def asset(asset_path: str) -> Response:
        return _asset_response(asset_path)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/state")
    def state() -> Response:
        return _json_response(widget._state())

    @app.get("/health")
    def health() -> Response:
        return _json_response({"ok": True})

    @app.post("/term")
    def set_term(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._set_term(str(payload["term"])))

    @app.post("/select")
    def select(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._select(
                str(payload["term"]),
                [int(v) for v in payload.get("indices", [])],
            )
        )

    @app.post("/op")
    def operate(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._operate(
                str(payload["operation"]),
                None if "term" not in payload else str(payload["term"]),
            )
        )

    @app.post("/drag")
    def drag(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._drag(
                str(payload["term"]),
                [int(v) for v in payload.get("indices", [])],
                float(payload.get("delta", 0.0)),
                None if "values" not in payload else [float(v) for v in payload.get("values", [])],
            )
        )

    @app.post("/control")
    def control(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._control(
                str(payload["term"]),
                int(payload["handle_index"]),
                float(payload["value"]),
                None if "handle_count" not in payload else int(payload["handle_count"]),
            )
        )

    @app.post("/control_count")
    def control_count(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._set_control_count(
                str(payload["term"]),
                int(payload["count"]),
            )
        )

    @app.post("/metrics")
    def metrics(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._metrics(
                str(payload.get("metric", "deviance")),
                None if "source" not in payload else str(payload["source"]),
            )
        )

    @app.post("/summary")
    def summary(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._summary(str(payload.get("source", "original"))))

    @app.post("/report")
    def report(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._report(str(payload.get("report", "validation"))))

    @app.post("/save_model")
    def save_model(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._save_model(
                directory=None if "directory" not in payload else str(payload["directory"]),
                filename=None if "filename" not in payload else str(payload["filename"]),
                path=None if "path" not in payload else str(payload["path"]),
            )
        )

    @app.get("/download_model")
    def download_model(filename: str = "superglm_edited_model.joblib") -> Response:
        try:
            data, safe_name = widget._download_model(filename)
        except _CLIENT_ERROR_TYPES as exc:
            return _json_response({"error": _client_error_message(exc)}, status_code=400)
        except Exception:  # pragma: no cover - surfaced to browser/tests as JSON
            _LOGGER.exception("Unhandled SuperGLM editor download error.")
            return _json_response({"error": "internal editor error"}, status_code=500)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                **_no_store_headers(),
                "Content-Disposition": f'attachment; filename="{safe_name}"',
            },
        )

    @app.post("/native_save_dialog")
    def native_save_dialog(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._native_save_dialog(
                directory=None if "directory" not in payload else str(payload["directory"]),
                filename=None if "filename" not in payload else str(payload["filename"]),
            )
        )

    @app.post("/open_directory")
    def open_directory(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._open_directory(
                None
                if "path" not in payload or payload["path"] in (None, "")
                else str(payload["path"])
            )
        )

    @app.post("/save_directory")
    def save_directory(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._save_directory(
                None
                if "path" not in payload or payload["path"] in (None, "")
                else str(payload["path"])
            )
        )

    @app.post("/refit_offset")
    def refit_offset(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._refit_offset(str(payload.get("method", "auto"))))

    @app.post("/profile_distribution")
    def profile_distribution(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._profile_distribution(
                str(payload.get("parameter", "")),
                **_profile_options(payload),
            )
        )

    @app.post("/profile_distribution/start")
    def start_profile_distribution(
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> Response:
        return _guarded_json(
            lambda: widget._start_profile_distribution_job(
                str(payload.get("parameter", "")),
                **_profile_options(payload),
            )
        )

    @app.get("/profile_distribution/status/{job_id}")
    def profile_distribution_status(job_id: str, wait: bool = False) -> Response:
        return _guarded_json(lambda: widget._profile_distribution_status(job_id, wait=wait))

    @app.post("/collapse_levels")
    def collapse_levels(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._collapse_levels(
                None if "term" not in payload else str(payload["term"]),
                str(payload.get("method", "auto")),
            )
        )

    @app.post("/ungroup_levels")
    def ungroup_levels(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._ungroup_levels(
                None if "term" not in payload else str(payload["term"]),
                str(payload.get("method", "auto")),
            )
        )

    @app.post("/reorder_levels")
    def reorder_levels(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._reorder_levels(
                None if "term" not in payload else str(payload["term"]),
                int(payload.get("target_index", 0)),
            )
        )

    @app.post("/uncollapse_levels")
    def uncollapse_levels() -> Response:
        return _guarded_json(widget._uncollapse_levels)

    return app


class EditorAppServer:
    """Run a FastAPI app for a single editor widget in a background thread."""

    def __init__(self, widget: Any):
        self.app = create_editor_app(widget)
        self._socket = _bound_local_socket()
        self.host, self.port = self._socket.getsockname()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"superglm-editor-{self.port}",
            daemon=True,
        )
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)

    def start(self, timeout: float = 5.0) -> None:
        """Start serving and wait until uvicorn has accepted the bound socket."""
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if self._startup_error is not None:
                raise RuntimeError(
                    "SuperGLM editor server failed to start."
                ) from self._startup_error
            if not self._thread.is_alive():
                raise RuntimeError("SuperGLM editor server stopped before startup completed.")
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out starting SuperGLM editor server.")
            time.sleep(0.01)

    def close(self, timeout: float = 2.0) -> None:
        """Request server shutdown and release the pre-bound socket."""
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(timeout=timeout)
        try:
            self._socket.close()
        except OSError:
            pass

    def _run(self) -> None:
        try:
            self._server.run(sockets=[self._socket])
        except BaseException as exc:  # pragma: no cover - startup path is timing dependent
            self._startup_error = exc
            raise


def _bound_local_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(socket.SOMAXCONN)
    return sock


def _profile_options(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "fit_mode",
        "phi_method",
        "method",
        "xatol",
        "maxiter",
        "p_bounds",
        "n_grid",
        "grid",
        "n_grid_coarse",
        "optimizer",
        "theta_bounds",
        "trace_iterations",
        "verbose",
    }
    options = {key: value for key, value in payload.items() if key in allowed}
    for key in ("p_bounds", "theta_bounds"):
        if key in options and isinstance(options[key], list):
            options[key] = tuple(options[key])
    return options


def _asset_response(path: str) -> Response:
    try:
        data = read_app_asset(path)
    except FileNotFoundError:
        return _json_response({"error": "not found"}, status_code=404)
    return Response(
        content=data,
        media_type=app_asset_content_type(path),
        headers=_no_store_headers(),
    )


def _is_public_path(path: str) -> bool:
    return path == "/" or path == "/favicon.ico" or path.startswith("/assets/")


def _request_token(request: Request) -> str:
    return request.headers.get("X-SuperGLM-Editor-Token", "") or request.query_params.get(
        "token", ""
    )


def _guarded_json(factory: Callable[[], dict[str, Any]]) -> Response:
    try:
        return _json_response(factory())
    except _CLIENT_ERROR_TYPES as exc:
        return _json_response({"error": _client_error_message(exc)}, status_code=400)
    except Exception:  # pragma: no cover - surfaced to browser/tests as JSON
        _LOGGER.exception("Unhandled SuperGLM editor request error.")
        return _json_response({"error": "internal editor error"}, status_code=500)


def _client_error_message(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _json_response(payload: dict[str, Any], *, status_code: int = 200) -> Response:
    import json

    body = json.dumps(jsonable(payload), allow_nan=False).encode("utf-8")
    return Response(
        content=body,
        status_code=status_code,
        media_type="application/json",
        headers=_no_store_headers(),
    )


def _no_store_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store"}


__all__ = ["EditorAppServer", "create_editor_app"]
