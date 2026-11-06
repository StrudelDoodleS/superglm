"""A private directory for Numba's compile cache, for a process with nowhere else to write one.

Numba keeps the kernels it compiles in a cache directory and needs one it can write to even to
read a cache that is already full: ``NUMBA_CACHE_DIR`` when that is set, otherwise a
``__pycache__`` next to the source file, otherwise the user's cache directory (Numba, *Notes on
Caching* and *Environment variables*). A kernel declared ``cache=True`` stops its module's import
with ``RuntimeError: cannot cache function ...: no locator available`` when none of them can be
written, which is every process on a read-only file system whose only writable place is the
temporary directory (AWS Lambda, a read-only container root).

``ensure_cache_location`` runs when superglm is imported, before anything imports Numba, and in
that one case points ``NUMBA_CACHE_DIR`` at a directory made by ``tempfile.mkdtemp``: new for this
process, readable by its owner alone (``mkdtemp`` creates it with mode 0o700), removed when the
process exits. Numba loads its cache with ``pickle``, so a directory another user could create or
write into is never used: no fixed name under the temporary directory. A ``NUMBA_CACHE_DIR`` or
``NUMBA_CACHE_LOCATOR_CLASSES`` the user set is never changed, and nothing changes while an
ordinary location can be written.
"""

from __future__ import annotations

import os
import sys

_PREFIX = "superglm-numba-cache-"


def _can_write(path: str) -> bool:
    """Whether the process can write the directory Numba would use at ``path``, or create it.

    Numba creates the directory on first use, so the nearest one that exists is the one to test.
    ``os.access`` is a permission query: nothing is created, and nothing beyond ``os`` is imported,
    so ``import superglm`` stays as light as it is.
    """
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent
    return os.access(path, os.W_OK | os.X_OK)


def _user_cache_directory() -> str:
    """The user-wide location in Numba's search order: the platform's per-user cache directory."""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    elif sys.platform == "darwin":
        root = os.path.join(home, "Library", "Caches")
    else:
        root = os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    return os.path.join(root, "numba")


def _ordinary_location_is_writable() -> bool:
    if _can_write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")):
        return True
    user = _user_cache_directory()
    # No home directory: ``expanduser`` leaves ``~`` as it was and the path is relative.
    return os.path.isabs(user) and _can_write(user)


def _remove(directory: str, owner: int) -> None:
    """Remove the directory in the process that made it, not in a forked child that inherited the exit hook."""
    import shutil

    if os.getpid() == owner:
        shutil.rmtree(directory, ignore_errors=True)


def ensure_cache_location() -> str | None:
    """Point Numba at a private temporary directory when it would find none to write; return it.

    Returns ``None`` when nothing changed: the user chose a location, an ordinary one can be
    written, or there is no temporary directory to make one in (Numba then reports the problem as
    it always did).
    """
    if os.environ.get("NUMBA_CACHE_DIR") or os.environ.get("NUMBA_CACHE_LOCATOR_CLASSES"):
        return None
    if _ordinary_location_is_writable():
        return None
    import atexit
    import tempfile

    try:
        directory = tempfile.mkdtemp(prefix=_PREFIX)
    except OSError:
        return None
    os.environ["NUMBA_CACHE_DIR"] = directory
    atexit.register(_remove, directory, os.getpid())
    numba = sys.modules.get("numba")
    if numba is not None:  # imported first: Numba reads its environment once, at import
        numba.config.reload_config()
    return directory
