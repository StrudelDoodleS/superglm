"""Static asset loading for the notebook editor app."""

from __future__ import annotations

from importlib.resources import files

_APP_PACKAGE = "superglm.editor.app"

_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def read_app_asset(path: str) -> bytes:
    """Return a packaged editor-app asset.

    Paths are relative to ``superglm.editor.app``. Absolute paths and parent
    traversal are rejected so the HTTP asset route cannot escape the package.
    """
    if "\\" in path:
        raise FileNotFoundError(path)
    parts = path.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise FileNotFoundError(path)
    resource = files(_APP_PACKAGE).joinpath(*parts)
    if not resource.is_file():
        raise FileNotFoundError(path)
    return resource.read_bytes()


def app_asset_content_type(path: str) -> str:
    """Return the HTTP content type for an editor-app asset."""
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return _CONTENT_TYPES.get(suffix, "application/octet-stream")
