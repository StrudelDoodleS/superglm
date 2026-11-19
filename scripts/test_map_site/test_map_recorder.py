"""Record which repository files each test file reaches, for the test map (scripts/test_map.py).

It runs in every Python process of a map build (``sitecustomize`` starts it) and records, under the
current *context*, the files under the repository root:

- whose functions start running, by qualified name, through ``sys.monitoring``'s ``PY_START``
  event (PEP 669). Each function reports once per context: its event is disabled after the first report and re-enabled
  (``restart_events``) when the context changes. Module and class bodies are not recorded, and
  neither is a function that runs while a ``superglm`` module body executes (an import): which
  test triggers a lazy import is an accident of scheduling, so import-time code is kept apart,
  by function, in ``import_time``;
- that are opened for reading, and the directories listed, through the ``open``, ``os.listdir``
  and ``os.scandir`` audit events (PEP 578), leaving out the import system's own reads.

The context is a test file while its tests run or its module is collected (the pytest plugin in
``test_map_plugin`` switches it), a fixture broader than a function while it is set up, and ``""``
otherwise. A Python subprocess starts under the context it inherits through the environment, and
then follows the context file its pytest process writes at each switch, read every
``POLL_SECONDS``: a pooled worker (joblib's loky, a process pool) outlives the test that started it
and serves the tests after it. The recorder adds its variables to a subprocess given its own
``env``, and notes the context as ``untracked`` when the child runs Python isolated (``-I``, ``-E``
or ``-S``), which skips this module.

``functools`` caches in the repository's modules (other than test files) are cleared whenever the
context moves to another test file, so a cached result computed under one test file does not hide
the code behind it from the next.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

OUT_ENV = "SUPERGLM_TEST_MAP_OUT"
ROOT_ENV = "SUPERGLM_TEST_MAP_ROOT"
CONTEXT_ENV = "SUPERGLM_TEST_MAP_CONTEXT"
CONTEXT_FILE_ENV = "SUPERGLM_TEST_MAP_CONTEXT_FILE"
POLL_SECONDS = 0.05
SITE = str(Path(__file__).resolve().parent)
SESSION = ""
FIXTURE = "fixture:"
# sys.monitoring reserves 0, 1, 2 and 5 for debuggers, coverage, profilers and optimizers.
TOOL_ID = 4
_CO_OPTIMIZED = 0x0001  # set on function code objects, never on a module or class body
_SKIPPED_DIRECTORIES = (".git", ".venv", "__pycache__", ".pytest_cache", "node_modules")
_ISOLATING_FLAGS = frozenset({"-I", "-E", "-S"})
_SOURCE_READERS = (f"{os.sep}linecache.py", f"{os.sep}tokenize.py")

_root = ""
_src = ""
_context = SESSION
_last_test_file = None
_reached: dict[str, dict[str, set[str]]] = {}
_import_time: dict[str, set[str]] = {}
_opened: dict[str, set[str]] = {}
_listed: dict[str, set[str]] = {}
fixtures: dict[str, set[str]] = {}
tests: dict[str, list[float]] = {}
_untracked: set[str] = set()
_paths: dict[str, str | None] = {}
_caches: list = []
_cache_modules = -1
_context_file: Path | None = None  # set in the process that names the contexts
_started = False
_saved = False


def root() -> Path:
    """The repository root the recorder maps files under."""
    return Path(_root)


def _relative(filename: str) -> str | None:
    """``filename`` relative to the root, posix style, or ``None`` outside it or in a skipped tree."""
    if not filename.startswith(_root):
        return None
    relative = filename[len(_root) :].replace(os.sep, "/")
    if relative.startswith("scripts/test_map_site/") or any(
        part in _SKIPPED_DIRECTORIES for part in relative.split("/")
    ):
        return None
    return relative


def _code_path(filename: str) -> str | None:
    try:
        return _paths[filename]
    except KeyError:
        path = _paths[filename] = _relative(filename)
        return path


def _inside_superglm_import() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        code = frame.f_code
        if code.co_name == "<module>" and code.co_filename.startswith(_src):
            return True
        frame = frame.f_back
    return False


def _on_start(code, offset):
    path = _code_path(code.co_filename)
    if path is None or not code.co_flags & _CO_OPTIMIZED:
        return sys.monitoring.DISABLE
    if _inside_superglm_import():
        _import_time.setdefault(path, set()).add(code.co_qualname)
        return None  # it may run again outside an import, in this context
    _reached.setdefault(_context, {}).setdefault(path, set()).add(code.co_qualname)
    return sys.monitoring.DISABLE


def _reading(mode, flags) -> bool:
    if isinstance(mode, str):
        return not any(letter in mode for letter in "wax")
    return isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR) == 0


def _on_audit(event: str, args) -> None:
    if event == "open":
        if not _reading(args[1], args[2]):
            return
        records = _opened
    elif event in ("os.listdir", "os.scandir"):
        records = _listed
    else:
        return
    path = args[0]
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    elif isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str):
        return
    try:
        caller = sys._getframe(1).f_code.co_filename
    except ValueError:  # no Python frame: the interpreter opening its script
        return
    # The import system reading a module or its bytecode is not the test reading the file, and
    # neither is the end-of-run warnings report reading source lines outside every test. Inside a
    # test, ``inspect.getsource`` reads through linecache too, and that read counts.
    if caller.startswith("<frozen") or (_context == SESSION and caller.endswith(_SOURCE_READERS)):
        return
    relative = _relative(os.path.abspath(path))
    if relative is not None:
        records.setdefault(_context, set()).add(relative.rstrip("/") or ".")


def _clear_caches() -> None:
    """Empty every ``functools`` cache defined in a repository module other than a test file."""
    global _caches, _cache_modules
    if len(sys.modules) != _cache_modules:
        _cache_modules = len(sys.modules)
        found = []
        for module in list(sys.modules.values()):
            path = _relative(getattr(module, "__file__", None) or "")
            if path is None or Path(path).name.startswith("test_"):
                continue
            for value in list(vars(module).values()):
                if isinstance(value, functools._lru_cache_wrapper):
                    found.append(value)
        _caches = found
    for cache in _caches:
        cache.cache_clear()


def switch(context: str) -> None:
    """Record what runs from now on under ``context``."""
    global _context, _last_test_file
    if context == _context:
        return
    _context = context
    os.environ[CONTEXT_ENV] = context
    if _context_file is not None:
        partial = _context_file.with_suffix(".partial")
        partial.write_text(context, encoding="utf-8")
        partial.replace(_context_file)
    if context and not context.startswith(FIXTURE) and context != _last_test_file:
        _last_test_file = context
        _clear_caches()
    sys.monitoring.restart_events()


def own_contexts() -> None:
    """Make this process the one that names contexts for itself and the processes it starts."""
    global _context_file
    path = Path(os.environ[OUT_ENV]) / "contexts" / str(os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_context, encoding="utf-8")
    _context_file = path
    os.environ[CONTEXT_FILE_ENV] = str(path)


def _follow(path: Path) -> None:
    """Switch to the context the parent's file names until this process names its own."""
    while _context_file is None:
        time.sleep(POLL_SECONDS)
        try:
            context = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _context_file is None and context != _context:
            switch(context)


def _child_environment(env):
    """``env`` with what a child needs to record; one that names its own map build keeps it."""
    child = dict(env)
    if OUT_ENV in child:
        if child[OUT_ENV] != os.environ.get(OUT_ENV):
            # A build of its own starts at its own session, not at the context copied from ours.
            child.pop(CONTEXT_ENV, None)
            child.pop(CONTEXT_FILE_ENV, None)
        return child
    for name in (OUT_ENV, ROOT_ENV, CONTEXT_FILE_ENV):
        if name in os.environ:
            child[name] = os.environ[name]
    child[CONTEXT_ENV] = _context
    inherited = child.get("PYTHONPATH")
    child["PYTHONPATH"] = SITE if not inherited else SITE + os.pathsep + inherited
    return child


_popen_init = subprocess.Popen.__init__


def _traced_popen_init(self, args, *positional, **keywords):
    if keywords.get("env") is not None:
        keywords["env"] = _child_environment(keywords["env"])
    words = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
    if any(str(word) in _ISOLATING_FLAGS for word in words[1:]):
        _untracked.add(_context)
    _popen_init(self, args, *positional, **keywords)


def start() -> None:
    """Begin recording in this process, under the context its environment names."""
    global _root, _src, _context, _started
    if _started:
        return
    try:
        sys.monitoring.use_tool_id(TOOL_ID, "superglm-test-map")
    except ValueError:
        return
    _root = os.path.join(os.path.realpath(os.environ[ROOT_ENV]), "")
    _src = os.path.join(_root, "src", "")
    _context = os.environ.get(CONTEXT_ENV, SESSION)
    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.PY_START, _on_start)
    sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.PY_START)
    sys.addaudithook(_on_audit)
    subprocess.Popen.__init__ = _traced_popen_init
    atexit.register(save)
    _started = True
    if os.environ.get(CONTEXT_FILE_ENV):
        threading.Thread(
            target=_follow, args=(Path(os.environ[CONTEXT_FILE_ENV]),), daemon=True
        ).start()


def _listing(records: dict[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(values) for key, values in records.items() if values}


def save() -> None:
    """Write this process's records to the output directory, once."""
    global _saved
    if _saved or not _started:
        return
    _saved = True
    sys.monitoring.set_events(TOOL_ID, 0)
    record = {
        "reached": {
            context: {path: sorted(names) for path, names in functions.items()}
            for context, functions in _reached.items()
        },
        "import_time": _listing(_import_time),
        "opened": _listing(_opened),
        "listed": _listing(_listed),
        "fixtures": _listing(fixtures),
        "tests": tests,
        "untracked": sorted(_untracked),
        "process": {"pid": os.getpid(), "parent": os.getppid(), "argv": sys.argv[:6]},
    }
    out = Path(os.environ[OUT_ENV])
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{os.getpid()}-{uuid.uuid4().hex}.json"
    partial = target.with_suffix(".partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    partial.replace(target)
