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
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from typing import Any


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

    def module_getattr(name: str) -> Any:
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

    def module_dir() -> list[str]:
        from pkgutil import iter_modules

        submodules = (module.name for module in iter_modules(search))
        return sorted({*vars(sys.modules[package]), *owner, *submodules})

    return module_getattr, module_dir
