"""Production code must guard every call that exists on only some supported platforms.

superglm runs on Linux, macOS and Windows (AGENTS.md, "Numerical policy"). A
Unix-only module or function (``fcntl``, ``os.getuid``, ``signal.SIGKILL``) or a
Windows-only one (``msvcrt``) raises on the other platforms, and CI shows it
only on that platform's native runner. Each use must sit under a test of
``sys.platform`` or ``os.name``, after an early return or raise on one, or
inside a ``try`` that catches the failure. The ``fork`` start method is
refused outright: it is unavailable on Windows and unsafe with threads on
macOS. A text file is opened with an explicit encoding: without one, Python
3.12-3.14 on Windows reads and writes the locale's code page, not UTF-8.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src" / "superglm"

MODULES = frozenset(
    {"fcntl", "resource", "termios", "pwd", "grp", "posix", "msvcrt", "winreg", "_winapi"}
)
ATTRIBUTES = {
    "os": frozenset(
        {
            "fork",
            "forkpty",
            "getuid",
            "geteuid",
            "getgid",
            "getegid",
            "getpgid",
            "setsid",
            "killpg",
            "getloadavg",
            "sysconf",
            "sched_getaffinity",
            "nice",
            "wait3",
            "wait4",
            "mkfifo",
            "startfile",
        }
    ),
    "signal": frozenset(
        {"SIGKILL", "SIGALRM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGCHLD", "alarm"}
    ),
}
# Exceptions a try may catch to cover a missing module or function.
CATCHES = frozenset(
    {"ImportError", "ModuleNotFoundError", "AttributeError", "OSError", "Exception"}
)
# Functions whose callers catch the failure, with where they do.
CALLED_GUARDED = {
    "diagnostics/_fit_record.py::_peak_rss_mib": "called only through _safe, which records the error",
}


def _tests_platform(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Attribute)
        and isinstance(sub.value, ast.Name)
        and (sub.value.id, sub.attr) in {("sys", "platform"), ("os", "name")}
        for sub in ast.walk(node)
    )


def _hazard(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import | ast.ImportFrom):
        if isinstance(node, ast.Import):
            hits = [alias.name.split(".")[0] for alias in node.names]
            hits = [name for name in hits if name in MODULES]
        else:
            module = node.module or ""
            hits = [module.split(".")[0]] if module.split(".")[0] in MODULES else []
            hits += [alias.name for alias in node.names if alias.name in ATTRIBUTES.get(module, ())]
        return f"import {', '.join(sorted(hits))}" if hits else None
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.attr in ATTRIBUTES.get(node.value.id, ())
    ):
        return f"{node.value.id}.{node.attr}"
    return None


_ALL_ATTRIBUTES = frozenset().union(*ATTRIBUTES.values())


def _fork_start(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
        and getattr(node.func, "attr", getattr(node.func, "id", None))
        in {"set_start_method", "get_context"}
        and any(isinstance(arg, ast.Constant) and arg.value == "fork" for arg in node.args)
    )


def _unencoded_text(node: ast.AST) -> bool:
    """``read_text``, ``write_text`` or a text-mode ``open``/``io.open`` with no ``encoding``."""
    if not isinstance(node, ast.Call) or any(kw.arg == "encoding" for kw in node.keywords):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in {"read_text", "write_text"}:
        return True
    builtin = isinstance(func, ast.Name) and func.id == "open"
    io_open = (
        isinstance(func, ast.Attribute)
        and func.attr == "open"
        and isinstance(func.value, ast.Name)
        and func.value.id == "io"
    )
    if not (builtin or io_open):
        return False
    mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
    if mode is None and len(node.args) > 1:
        mode = node.args[1]
    if mode is None:
        return True
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" not in mode.value


def _catches(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(name, ast.Name) and name.id in CATCHES for name in names)


def _returns_early_on_platform(function: ast.AST, line: int) -> bool:
    """An earlier ``if <platform test>: return/raise`` in the function's own body."""
    return any(
        isinstance(statement, ast.If)
        and statement.lineno < line
        and _tests_platform(statement.test)
        and any(isinstance(last, ast.Return | ast.Raise) for last in statement.body[-1:])
        for statement in getattr(function, "body", ())
    )


def _guarded(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    child, parent = node, parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.If | ast.IfExp | ast.While) and _tests_platform(parent.test):
            return True
        if isinstance(parent, ast.BoolOp):
            earlier = parent.values[: parent.values.index(child)]
            if any(_tests_platform(value) for value in earlier):
                return True
        if (
            isinstance(parent, ast.Try)
            and child in parent.body
            and any(_catches(handler) for handler in parent.handlers)
        ):
            return True
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef) and (
            _returns_early_on_platform(parent, node.lineno)
        ):
            return True
        child, parent = parent, parents.get(parent)
    return False


def _function_of(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    parent = parents.get(node)
    while parent is not None and not isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
        parent = parents.get(parent)
    return None if parent is None else parent.name


def _offenders(tree: ast.AST, relative: str) -> list[str]:
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found = []
    for node in ast.walk(tree):
        if _fork_start(node):
            found.append(f"{relative}:{node.lineno} the 'fork' start method")
            continue
        if _unencoded_text(node):
            found.append(f"{relative}:{node.lineno} a text file without an encoding")
            continue
        hazard = _hazard(node)
        if hazard is None or _guarded(node, parents):
            continue
        if f"{relative}::{_function_of(node, parents)}" in CALLED_GUARDED:
            continue
        found.append(f"{relative}:{node.lineno} {hazard}")
    return found


def test_platform_specific_calls_are_guarded() -> None:
    paths = sorted(SOURCE.rglob("*.py"))
    assert paths
    offenders = [
        hit
        for path in paths
        for hit in _offenders(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
            path.relative_to(SOURCE).as_posix(),
        )
    ]
    assert not offenders, f"unguarded platform-specific code: {offenders}"


def test_every_listed_guarded_function_exists() -> None:
    for entry in CALLED_GUARDED:
        relative, function = entry.split("::")
        tree = ast.parse((SOURCE / relative).read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == function for node in ast.walk(tree)
        ), entry


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import fcntl", ["m.py:1 import fcntl"]),
        ("import os\nos.getuid()", ["m.py:2 os.getuid"]),
        ("from os import getuid", ["m.py:1 import getuid"]),
        ("import signal\nsignal.SIGKILL", ["m.py:2 signal.SIGKILL"]),
        (
            "import multiprocessing as mp\nmp.set_start_method('fork')",
            ["m.py:2 the 'fork' start method"],
        ),
        ("import sys\nif sys.platform == 'win32':\n    import msvcrt\nelse:\n    import fcntl", []),
        ("import os, sys\nok = sys.platform == 'win32' or os.getuid() == 0", []),
        ("try:\n    import resource\nexcept ImportError:\n    resource = None", []),
        ("try:\n    import resource\nexcept ValueError:\n    pass", ["m.py:2 import resource"]),
        (
            "import os, sys\ndef f():\n    if sys.platform == 'win32':\n        return None\n    return os.getuid()",
            [],
        ),
        (
            "import os, sys\ndef f():\n    x = os.getuid()\n    if sys.platform == 'win32':\n        return None",
            ["m.py:3 os.getuid"],
        ),
        ("import os\nos.getpid()", []),
        ("Path(p).read_text()", ["m.py:1 a text file without an encoding"]),
        ("Path(p).write_text(s, encoding='utf-8')", []),
        ("open(p)", ["m.py:1 a text file without an encoding"]),
        ("open(p, 'w')", ["m.py:1 a text file without an encoding"]),
        ("open(p, 'rb')", []),
        ("open(p, mode='a+b')", []),
        ("open(p, encoding='utf-8')", []),
    ],
)
def test_the_guard_reads_each_form(source, expected) -> None:
    assert _offenders(ast.parse(source), "m.py") == expected
