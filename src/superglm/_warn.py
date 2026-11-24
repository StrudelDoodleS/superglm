"""Warning attribution for callers and feature builds."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from threading import local

# Frames under this directory are superglm's own; a warning skips them to land
# on the caller's line however deep the call (Python 3.12, skip_file_prefixes).
_PACKAGE_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep
_feature_warnings = local()


def warn_caller(message: str | Warning, category: type[Warning] = UserWarning) -> None:
    """Warn at the first frame outside superglm (a ``Warning`` instance keeps its own class)."""
    warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_PREFIX,))


@contextmanager
def feature_build_messages(name: str) -> Iterator[None]:
    """Name a feature's errors and report its built-in warnings at the build's caller."""
    pending: list[tuple[str, type[Warning]]] = []
    previous = getattr(_feature_warnings, "build", (None, None))
    _feature_warnings.build = name, pending
    try:
        yield
    except ValueError as err:
        raise ValueError(f"Feature {name!r}: {err}") from err
    finally:
        _feature_warnings.build = previous
    for message, category in pending:
        # This generator, contextlib.__exit__, the build, then its caller.
        warnings.warn(message, category, stacklevel=4)


def warn_feature(
    message: str,
    feature_name: str | None,
    category: type[Warning] = UserWarning,
    *,
    stacklevel: int | None = 1,
) -> None:
    """Use the active build's name for delegated warnings on the emitting thread.

    Outside a build, a named warning or a stacklevel of None uses the first
    caller outside superglm. An unnamed warning with a stacklevel uses that frame.
    """
    name, pending = getattr(_feature_warnings, "build", (None, None))
    if feature_name is None:
        feature_name = name
    if feature_name is None and stacklevel is not None:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
        return
    if feature_name is not None:
        message = f"Feature {feature_name!r}: {message}"
    if pending is None:
        warn_caller(message, category)
    else:
        pending.append((message, category))
