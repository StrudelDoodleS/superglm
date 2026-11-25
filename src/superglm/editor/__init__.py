"""Notebook-friendly editors for fitted SuperGLM main effects."""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.editor.session import EditableTerm, EditorSession, EditRecord, edit

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports none of its modules, so a thread that imports one of them by path while another thread
# runs this body never waits on a sibling that thread holds.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.editor.session": ("EditableTerm", "EditorSession", "EditRecord", "edit"),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = ["EditableTerm", "EditRecord", "EditorSession", "edit"]
