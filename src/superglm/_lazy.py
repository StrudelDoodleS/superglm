"""Lazy re-exports for package ``__init__`` modules.

A package that names its public objects in its ``__init__`` imports every module that defines
one, and so everything they import: ``import superglm`` alone took 2 s and 320 MiB. ``attach``
gives the package a module ``__getattr__`` (PEP 562) that imports the defining module, or a
submodule, on first access and keeps the result in the package's namespace, so each name costs
its import once and only for the program that reads it. This is the mechanism of Scientific
Python's SPEC 1 (``lazy_loader.attach``) in the standard library alone, and the one NumPy,
SciPy and scikit-learn document for their own subpackages. Static checkers read the real
imports a package keeps under ``typing.TYPE_CHECKING``; ``tests/test_import_compat.py`` holds
that block and the table passed here equal.

A name is stored in the package's namespace on its first read, as ``lazy_loader.attach`` does
from version 0.6 (0.4 and 0.5 did so only for a function named like its module): a later read is
a dictionary lookup, and a ``mock.patch`` of the module that defines the name, in force at that
first read, leaves its double on the package after the block. Patch the package's own name, or
read it once before patching the module.

Threads. ``import superglm`` loads nothing, so a module first loads on whichever thread first
reads one of its names. Where modules import each other, two such loads at once make CPython's
per-module import locks detect the cycle and break it, which in one thread is a
``_DeadlockError`` and in another a half-initialised module (``ImportError: cannot import name
...``). ``attach`` therefore runs every first load under one reentrant lock, so the loads follow
one another as they would in a single thread. The lock is a leaf: a thread that is inside an
import never waits for it (``_inside_an_import``), because that thread holds import locks that
the holder of this one may need, and neither lock could see the cycle. A thread that waits
holds no import lock, so no cycle runs through it. Two threads that import superglm's modules by
name (``import superglm.model``) are not serialised and rely on CPython's import locks alone.
"""

from __future__ import annotations

import os
import sys
from _thread import RLock
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from typing import Any

#: Held while a name's defining module is imported by a thread that is not already importing.
_FIRST_LOAD = RLock()
if hasattr(os, "register_at_fork"):
    # A child forked while another thread held the lock would wait for it forever.
    os.register_at_fork(after_in_child=_FIRST_LOAD._at_fork_reinit)


def _inside_an_import() -> bool:
    """Whether the calling thread is running the body of a module that is being imported.

    An import holds its module's lock for as long as the module's body runs, and only then, so
    such a body (a ``<module>`` frame of anything but ``__main__``) on the stack is the sign of a
    thread that holds import locks.
    """
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "<module>" and frame.f_globals.get("__name__") != "__main__":
            return True
        frame = frame.f_back
    return False


def attach(
    package: str,
    path: Iterable[str],
    exports: Mapping[str, tuple[str, ...]],
    *,
    missing: Callable[[str], Any] | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """``(__getattr__, __dir__)`` for ``package``, whose names are ``exports[module]``.

    ``exports`` maps each defining module to the names it supplies. A name that is neither one
    of them nor a submodule goes to ``missing`` when given (a package's deprecated aliases), and
    is an ``AttributeError`` otherwise.
    """
    owner = {name: module for module, names in exports.items() for name in names}
    search = list(path)

    def load(name: str) -> Any:
        module = owner.get(name)
        if module is not None:
            value = getattr(import_module(module), name)
        elif name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        else:
            try:
                value = import_module(f"{package}.{name}")
            except ModuleNotFoundError as error:
                if error.name != f"{package}.{name}":
                    raise
                if missing is not None:
                    return missing(name)
                raise AttributeError(f"module {package!r} has no attribute {name!r}") from None
        vars(sys.modules[package])[name] = value
        return value

    def module_getattr(name: str) -> Any:
        if _inside_an_import():
            return load(name)
        with _FIRST_LOAD:
            return load(name)

    def module_dir() -> list[str]:
        from pkgutil import iter_modules

        submodules = (module.name for module in iter_modules(search))
        return sorted({*vars(sys.modules[package]), *owner, *submodules})

    return module_getattr, module_dir
