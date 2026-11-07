"""A private directory for Numba's compile cache, for a process with nowhere else to write one.

Numba keeps the kernels it compiles in a cache directory and needs one it can write to even to
read a cache that is already full: ``NUMBA_CACHE_DIR`` when that is set, otherwise a
``__pycache__`` next to the source file, otherwise the user's cache directory (Numba, *Notes on
Caching* and *Environment variables*). A kernel declared ``cache=True`` stops its module's import
with ``RuntimeError: cannot cache function ...: no locator available`` when none of them can be
written, which is every process on a read-only file system whose only writable place is the
temporary directory (AWS Lambda, a read-only container root).

``ensure_cache_location`` runs when superglm is imported, before anything imports Numba, and in
that one case points Numba at a directory made by ``tempfile.mkdtemp``: new for this process,
readable by its owner alone (mode 0o700), removed when the process exits. Numba loads its cache
with ``pickle``, so a directory another user could create or write into is never used. On Linux and
macOS the directory is made inside a directory of the user's own in the temporary directory
(``superglm-numba-cache-u<uid>``, kept, mode 0o700, checked to be a real directory owned by the
caller before it is used), so a child that outlives the process and makes Numba create the path
again does so where no other user can enter. A child that inherits the setting uses the directory
only while it is still a real directory owned by the caller and closed to others; otherwise it
makes its own. A setting the user chose is never changed: ``NUMBA_CACHE_DIR`` or
``NUMBA_CACHE_LOCATOR_CLASSES`` in the environment, ``numba.config.CACHE_DIR`` set in code before
this import, or ``cache_dir`` in a ``.numba_config.yaml`` in the working directory. Nothing changes
while an ordinary location can be written, and nothing here raises.
"""

from __future__ import annotations

import os
import stat
import sys

_PREFIX = "superglm-numba-cache-"


def _can_write(path: str) -> bool:
    """Whether the process can write the directory Numba would use at ``path``, or create it.

    Numba creates the directory on first use, so the nearest one that exists is the one to test.
    On Linux and macOS ``os.access`` is a permission query: nothing is created, and nothing beyond
    ``os`` is imported, so ``import superglm`` stays as light as it is. CPython's Windows
    ``os.access`` says yes for every directory, so there the directory is asked as Numba asks it,
    by making a file in it.
    """
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent
    if sys.platform != "win32":
        return os.access(path, os.W_OK | os.X_OK)
    import tempfile

    try:
        tempfile.TemporaryFile(dir=path).close()
    except OSError:
        return False
    return True


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


def _private_to_me(path: str) -> bool:
    """Whether ``path`` is a real directory, not a link, that only the caller can enter.

    Mode bits and owners mean nothing on Windows, whose temporary directory is the user's own.
    """
    try:
        status = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(status.st_mode):
        return False
    return sys.platform == "win32" or (status.st_uid == os.getuid() and not status.st_mode & 0o077)


def _private_parent() -> str | None:
    """The caller's own directory in the temporary directory, made if it is not there, or ``None``."""
    if sys.platform == "win32":
        return None
    import tempfile

    parent = os.path.join(tempfile.gettempdir(), f"{_PREFIX}u{os.getuid()}")
    try:
        os.mkdir(parent, 0o700)
    except FileExistsError:
        pass
    except OSError:
        return None
    return parent if _private_to_me(parent) else None


def _yaml_names_a_cache_directory() -> bool:
    """Whether the working directory's ``.numba_config.yaml`` sets ``cache_dir``, which Numba reads."""
    try:
        with open(".numba_config.yaml", encoding="utf-8") as file:
            return any(line.split(":")[0].strip() == "cache_dir" for line in file)
    except (OSError, ValueError):
        return False


def _set_in_code_or_yaml() -> bool:
    """Whether a cache directory is set where the environment does not show it.

    Numba reads its environment and the YAML file once, when it is imported, so a Numba that is
    already loaded holds what was set in ``config.CACHE_DIR``, by the program or by that file; one
    that is not loaded will read the YAML file.
    """
    numba = sys.modules.get("numba")
    if numba is not None:
        return bool(numba.config.CACHE_DIR)
    return _yaml_names_a_cache_directory()


def ensure_cache_location() -> str | None:
    """Point Numba at a private temporary directory when it would find none to write; return it.

    Returns ``None`` when nothing changed: the user chose a location, an ordinary one can be
    written, a private directory a parent process made is still usable, or there is no temporary
    directory to make one in (Numba then reports the problem as it always did). What costs more
    than an environment read is asked only of a process that would otherwise act.
    """
    named = os.environ.get("NUMBA_CACHE_DIR", "")
    inherited = named if os.path.basename(named).startswith(_PREFIX) else None
    if os.environ.get("NUMBA_CACHE_LOCATOR_CLASSES") or (named and inherited is None):
        return None
    if inherited is not None:
        if _private_to_me(inherited):
            return None
    elif _ordinary_location_is_writable() or _set_in_code_or_yaml():
        return None
    import atexit
    import tempfile

    try:
        directory = tempfile.mkdtemp(prefix=_PREFIX, dir=_private_parent())
    except OSError:
        return None
    os.environ["NUMBA_CACHE_DIR"] = directory
    atexit.register(_remove, directory, os.getpid())
    numba = sys.modules.get("numba")
    if numba is not None:  # imported first: it read its environment then
        numba.config.CACHE_DIR = directory
    return directory
