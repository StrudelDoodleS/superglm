"""SuperGLM model package."""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.model.api import SuperGLM as SuperGLM
    from superglm.model.fit_ops import PathResult as PathResult

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports nothing, so a module that imports one of its submodules while another thread first
# loads the package finds no half-run body to wait for.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.model.api": ("SuperGLM",),
    "superglm.model.fit_ops": ("PathResult",),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = ["SuperGLM", "PathResult"]
