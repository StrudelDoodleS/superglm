"""The editor inside a notebook cell, without a local server.

``EditorWidget`` normally serves its page from a local HTTP server, which the
browser can reach only when it runs on the kernel's machine. On a hosted
notebook such as Databricks the browser is elsewhere, so this mode carries
the page's requests over the notebook's own widget messages instead
(anywidget), and runs each one through the same FastAPI app the local server
uses, in process, so every route and its checks are shared.

Two pieces of prior art set the method. jupyter-ws-tunnel (MIT) serves one
ASGI app either over a socket or over a widget comm; here the comm carries
plain request/response pairs. Calling the ASGI app in process follows the
ASGI 3 HTTP specification, as httpx's ``ASGITransport`` does. The page's ES
modules load from blob URLs with their relative import specifiers rewritten
to those URLs, the technique of packagemap-polyfill, which became
es-module-shims; the browser's own module loader still links them, so the
module graph must be acyclic, which :func:`pack_modules` checks.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import posixpath
import re
import threading
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import unquote

from superglm.editor.assets import read_app_asset

_LOGGER = logging.getLogger(__name__)

REQUEST = "superglm.request"
RESPONSE = "superglm.response"
MODULE_PREFIX = "superglm-module:"
ENTRY_MODULE = "main.js"
HOST_MODULE = "api/notebook_host.js"

# Databricks caps one widget message at 5 MB; a reply is sent in parts well
# under that, leaving room for a front end that base64-encodes its buffers.
MESSAGE_PART_BYTES = 1 << 20

_SPECIFIER = re.compile(r"""(\b(?:from|import)\s*)(["'])(\.\.?/[^"']+)\2""")
_STYLESHEET = re.compile(r"""<link\s+rel="stylesheet"\s+href="/assets/([^"]+)">""")
_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script\b[^>]*>\s*", re.S | re.I)


def pack_modules(sources: Mapping[str, str], entry: str = ENTRY_MODULE) -> list[dict[str, str]]:
    """Return the modules ``entry`` reaches, each after the modules it imports.

    Every relative import specifier is rewritten to ``superglm-module:<path>``,
    which the page loader replaces with that module's blob URL. A missing
    module or an import cycle raises ``ValueError``: blob URLs exist only once
    their source is final, so a cycle cannot be linked this way.
    """
    order: list[dict[str, str]] = []
    state: dict[str, str] = {}

    def resolve(importer: str, specifier: str) -> str:
        path = posixpath.normpath(posixpath.join(posixpath.dirname(importer), specifier))
        if path not in sources:
            raise ValueError(f"{importer} imports {specifier!r}, which is not an editor module.")
        return path

    def visit(path: str, chain: tuple[str, ...]) -> None:
        if state.get(path) == "done":
            return
        if state.get(path) == "open":
            cycle = " -> ".join((*chain[chain.index(path) :], path))
            raise ValueError(f"The editor's modules import each other in a cycle: {cycle}.")
        state[path] = "open"
        source = sources[path]
        for match in _SPECIFIER.finditer(source):
            visit(resolve(path, match.group(3)), (*chain, path))
        rewritten = _SPECIFIER.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}{MODULE_PREFIX}"
                f"{resolve(path, match.group(3))}{match.group(2)}"
            ),
            source,
        )
        state[path] = "done"
        order.append({"path": path, "source": rewritten})

    if entry not in sources:
        raise ValueError(f"The editor has no module {entry!r}.")
    visit(entry, ())
    return order


def page_html(index_html: str, read: Callable[[str], str]) -> str:
    """The editor page with its stylesheets inline and its scripts removed.

    The loader writes this into a frame, sets the theme the page's inline
    script would have set, and then runs the modules itself.
    """
    inlined = _STYLESHEET.sub(
        lambda match: f'<style data-asset="{match.group(1)}">\n{read(match.group(1))}</style>',
        index_html,
    )
    return _SCRIPT.sub("", inlined)


def app_bundle() -> dict[str, Any]:
    """The editor page and its modules, as the notebook loader receives them."""

    def read(path: str) -> str:
        return read_app_asset(path).decode("utf-8")

    sources: dict[str, str] = {}
    pending = [ENTRY_MODULE]
    while pending:
        path = pending.pop()
        if path in sources:
            continue
        try:
            sources[path] = read(path)
        except FileNotFoundError:
            continue  # pack_modules names the importer of a missing module
        for match in _SPECIFIER.finditer(sources[path]):
            pending.append(
                posixpath.normpath(posixpath.join(posixpath.dirname(path), match.group(3)))
            )
    return {
        "html": page_html(read("index.html"), read),
        "modules": pack_modules(sources),
        "entry": ENTRY_MODULE,
    }


async def call_asgi(
    app: Any,
    method: str,
    url: str,
    headers: list[tuple[bytes, bytes]],
    body: bytes,
) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    """Run one HTTP request through an ASGI app in this process.

    The request body arrives in one message, and ``http.disconnect`` follows
    only once the response is complete, as ASGI 3 specifies for a client that
    stays connected.
    """
    path, _, query = url.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": unquote(path),
        "raw_path": path.encode("latin-1"),
        "query_string": query.encode("latin-1"),
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
    }
    sent_request = False
    complete = asyncio.Event()
    status = 500
    response_headers: list[tuple[bytes, bytes]] = []
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        await complete.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status, response_headers
        if message["type"] == "http.response.start":
            status = int(message["status"])
            response_headers = list(message.get("headers", []))
        elif message["type"] == "http.response.body":
            chunks.append(bytes(message.get("body", b"")))
            if not message.get("more_body", False):
                complete.set()

    await app(scope, receive, send)
    return status, response_headers, b"".join(chunks)


@functools.cache
def notebook_view_class() -> type:
    try:
        import anywidget
        import traitlets
    except ImportError as exc:
        raise ImportError(
            "The editor inside a notebook cell needs anywidget. "
            "Install it with: pip install 'superglm[notebook]'"
        ) from exc

    class EditorNotebookView(anywidget.AnyWidget):
        """The notebook-side view: a frame that runs the editor page."""

        _esm = read_app_asset(HOST_MODULE).decode("utf-8")
        bundle = traitlets.Dict().tag(sync=True)
        height = traitlets.Int(720).tag(sync=True)

    return EditorNotebookView


class NotebookTransport:
    """Answer an editor page's requests over its widget's messages.

    Requests run on this transport's own event loop thread, as the local
    server's do on uvicorn's, so a long request such as a job-status wait
    never holds the kernel, and FastAPI runs the synchronous route handlers
    in its thread pool as it does behind the server.
    """

    def __init__(self, app: Any, token: str, *, view: Any | None = None):
        self._app = app
        self._token = token.encode("latin-1")
        self.view = notebook_view_class()(bundle=app_bundle()) if view is None else view
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name=f"superglm-editor-notebook-{id(self):x}",
            daemon=True,
        )
        self._thread.start()
        self._closed = False
        self.view.on_msg(self._on_message)

    def _on_message(self, _view: Any, content: Any, _buffers: Any = None) -> None:
        if self._closed or not isinstance(content, dict) or content.get("type") != REQUEST:
            return
        asyncio.run_coroutine_threadsafe(self._answer(content), self._loop)

    async def _answer(self, content: dict[str, Any]) -> None:
        request_id = content.get("id")
        try:
            headers = [
                (str(name).lower().encode("latin-1"), str(value).encode("latin-1"))
                for name, value in content.get("headers") or []
                if str(name).lower() != "x-superglm-editor-token"
            ]
            headers.append((b"x-superglm-editor-token", self._token))
            body = content.get("body")
            status, response_headers, payload = await call_asgi(
                self._app,
                str(content.get("method", "GET")),
                str(content.get("url", "/")),
                headers,
                b"" if body is None else str(body).encode("utf-8"),
            )
            named = [
                [name.decode("latin-1"), value.decode("latin-1")]
                for name, value in response_headers
            ]
        except Exception:
            _LOGGER.exception("Unhandled SuperGLM notebook editor request error.")
            status = 500
            named = [["content-type", "application/json"]]
            payload = json.dumps({"error": "internal editor error"}).encode("utf-8")
        self._reply(request_id, status, named, payload)

    def _reply(
        self, request_id: Any, status: int, headers: list[list[str]], payload: bytes
    ) -> None:
        parts = max(1, -(-len(payload) // MESSAGE_PART_BYTES))
        for part in range(parts):
            chunk = payload[part * MESSAGE_PART_BYTES : (part + 1) * MESSAGE_PART_BYTES]
            message = {
                "type": RESPONSE,
                "id": request_id,
                "status": status,
                "headers": headers,
                "part": part,
                "parts": parts,
            }
            self.view.send(message, buffers=[chunk])

    def mimebundle(self) -> dict[str, Any]:
        """The display data that shows this transport's view."""
        own = getattr(self.view, "_repr_mimebundle_", None)
        if callable(own):
            data = own()
            if isinstance(data, tuple):
                data = data[0]
            if data:
                return dict(data)
        # ipywidgets 7 displays through _ipython_display_; its view is the
        # same widget-view MIME type, which carries only the model id.
        return {
            "text/plain": "SuperGLM editor",
            "application/vnd.jupyter.widget-view+json": {
                "version_major": 2,
                "version_minor": 0,
                "model_id": self.view.model_id,
            },
        }

    def close(self) -> None:
        """Stop answering requests and close the view."""
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._loop.close()
        close = getattr(self.view, "close", None)
        if callable(close):
            close()


__all__ = ["NotebookTransport", "app_bundle", "call_asgi", "page_html", "pack_modules"]
