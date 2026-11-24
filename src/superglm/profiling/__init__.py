"""Distribution parameter profiling (Tweedie p, NB theta).

# Internal submodules: import siblings directly, not through this __init__.
"""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.profiling.nb import NBProfileResult, NBThetaBoundWarning
    from superglm.profiling.tweedie import TweedieProfileResult

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports none of its modules, so a thread that imports one of them by path while another thread
# runs this body never waits on a sibling that thread holds.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.profiling.nb": ("NBProfileResult", "NBThetaBoundWarning"),
    "superglm.profiling.tweedie": ("TweedieProfileResult",),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = [
    "NBProfileResult",
    "NBThetaBoundWarning",
    "TweedieProfileResult",
]
