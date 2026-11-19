"""Pick the quick-suite test files a change reaches, from a map recorded on an earlier commit.

    uv run python scripts/test_map.py build     # record a map of the quick suite at a clean HEAD
    uv run python scripts/test_map.py select    # the selection for the working tree, and why

This is regression test selection by dynamic dependencies, as in Ekstazi (Gligoric, Eloussi and
Marinov, *Practical Regression Test Selection with Dynamic File Dependencies*, ISSTA 2015): record
what each test file runs, then run the test files that ran something the change touches. Python's
imports are eager, so a file-level dependency puts nearly every source file under every test
(Wang, Pradel and Liu, *Names Are All You Need: Effective and Safe Regression Test Selection for
Python*, arXiv:2605.25356, 2026, measure 83% of files per test file), and here most test files fit
a model, so most run every core module. The map therefore records each *function* a test file
runs (pytest-testmon's granularity, from coverage of function blocks), and a change to import-time
code is followed by name, as that paper's name-based dependency propagation does.

A map is recorded by running the quick suite with ``test_map_site`` first on ``PYTHONPATH`` (its
``sitecustomize`` starts ``test_map_recorder`` in every process and ``-p test_map_plugin`` names
the contexts; both describe what they record). It is local, kept beside the git objects
(``<git common dir>/superglm-test-map/<commit>.json``, shared by every worktree, never committed),
and stays usable for any later tree: the files that differ between the map's commit and the
working tree (``git diff <map commit>``) are the change, so commits master made after the map
count as changed too. The map with the fewest such files is used. ``pre_push.py`` records a new
one whenever it runs the whole quick suite on a clean tree, and keeps the newest ``KEEP``.

A test file *ran* a function when one of its tests, its module's collection, or a fixture broader
than a function that it uses started it, in the test process or a Python subprocess. For a changed
Python file, comparing the map's version with the working tree's (``ast``, so comments and
formatting do not count), a test file is selected when:

1. it changed or is new, or it is a contract test (``ALWAYS``);
2. it ran a function whose definition (decorators, signature, docstring, body) changed or was
   removed; or it read the changed file (``inspect.getsource`` included), or listed the directory
   the changed file is in (it may hand the file to a process the recorder cannot see);
3. a method was added to a class whose functions it ran (an override changes dispatch), or a
   function was added with a decorator other than ``PLAIN`` ones (a decorator can register it)
   to a file it ran; or a module-level statement that binds nothing (a registration call)
   changed in a file it ran, or in a file whose functions that statement calls;
4. it reads a name whose value may have changed, followed by name. A name changes when a
   module-level statement that binds it changes (each imported name and each lazy re-export
   entry is a statement of its own; a decorated class statement binds its class, and then every
   test file that ran one of its methods counts), when a statement that reads a changed name or
   calls a changed function binds it (``CAP = limit()`` runs at import), and, in a module that
   compiles with numba (``numba_cache_key.scan``), when it names a changed function or a compiled
   caller of one, since a compiled body never runs as Python and the map cannot see it run. A
   function changes when its definition does, when a function the map saw run during an import
   changed, or when it reads a changed name. Names are followed into each module that imports
   them (``from m import name``, ``import m`` then ``m.name``, a package's lazy re-export table),
   so they pass through package re-exports; a test file that imports a changed name is selected,
   and so is one that ran a function reading it. An attribute assigned in an undecorated class's
   body is read through instances, so a changed one selects the test files that ran any function,
   in any Python file, mentioning its name (``getattr(obj, "name")`` included);
5. a module's or class's docstring changed and it ran a function reading docstrings
   (``DOCSTRING_READERS``), or a module's top-level names changed and it ran a function in
   ``tests/`` that lists names (``ENUMERATORS``);
6. a changed file no recorded run read, or a changed Python file outside ``src/`` and ``tests/``
   whose import-time code changed, is named (by file name) in its source, or it ran a function of
   a file that names it;
7. it started a Python subprocess the recorder could not follow (``-I``, ``-E`` or ``-S``), and a
   Python file outside ``tests/`` changed. A new or deleted module counts as all its names
   changing, and a deleted one also selects every test file that ran any of its functions.

The whole quick suite runs instead when there is no map, when a file that configures every test
changed (``FULL_FILES``, any ``conftest.py``), when a changed function runs before any test (in the
session: conftest hooks, startup), when a ``conftest.py`` reads a changed name, when a function
registered through numba's extension API (``NUMBA_EXTENSIONS``) changed, or when the selection
would take at least ``FULL_SHARE`` of the suite's recorded seconds anyway. A map is recorded only
from a run in which every test passed.

Blind spots: a test whose result depends on state left by another test file (beyond ``functools``
caches, which the recorder clears between test files); a name reached other than by reading it
(``getattr`` with a computed string, ``globals()``); files read by a library's own I/O (a C reader)
rather than Python's ``open``. GitHub's CI runs every test on every platform before a merge, so a
missed test costs a CI round, not a bad merge.
"""

from __future__ import annotations

import argparse
import ast
import copy
import functools
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts" / "test_map_site"
OUT_ENV = "SUPERGLM_TEST_MAP_OUT"
ROOT_ENV = "SUPERGLM_TEST_MAP_ROOT"
VERSION = 2
KEEP = 3
FULL_SHARE = 0.6
QUICK = "not slow and not browser and not docs"
SESSION = ""
# Contract tests: they read the repository's own configuration rather than exercising src/.
ALWAYS = (
    "tests/test_ci_contracts.py",
    "tests/test_extended_dtype_guard.py",
    "tests/test_platform_hazard_guard.py",
    "tests/test_public_api_snapshot.py",
    "tests/test_supply_chain_governance.py",
)
# Files that configure every test run, or the selection itself; a change to one runs everything.
FULL_FILES = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "tests/__init__.py",
        "scripts/run_test_suite.py",
        "scripts/test_map.py",
        "scripts/numba_cache_key.py",
        "scripts/test_map_site/sitecustomize.py",
        "scripts/test_map_site/test_map_plugin.py",
        "scripts/test_map_site/test_map_recorder.py",
    }
)
# Decorators that register nothing: an added function carrying only these is reached by calling it.
PLAIN = frozenset(
    {
        "property",
        "setter",
        "getter",
        "deleter",
        "staticmethod",
        "classmethod",
        "cached_property",
        "abstractmethod",
        "overload",
        "override",
        "wraps",
        "contextmanager",
        "asynccontextmanager",
        "cache",
        "lru_cache",
    }
)
# numba's extension API: what these register changes compiled code in every module.
NUMBA_EXTENSIONS = frozenset(
    {
        "overload",
        "overload_method",
        "overload_attribute",
        "overload_classmethod",
        "intrinsic",
        "lower_builtin",
        "type_callable",
        "register_model",
        "typeof_impl",
    }
)
# What a test reads when it reads a docstring, or lists a module's names.
DOCSTRING_READERS = frozenset({"__doc__", "getdoc", "cleandoc"})
ENUMERATORS = frozenset({"getmembers", "dir", "vars", "__dict__", "__all__", "walk_packages"})
TEST_FILE = re.compile(r"tests/(.*/)?test_[^/]*\.py")
RECORDING_ARGUMENTS = ("-p", "test_map_plugin")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def is_test_file(path: str) -> bool:
    return TEST_FILE.fullmatch(path) is not None


def _is_conftest(path: str) -> bool:
    return PurePosixPath(path).name == "conftest.py"


# -- Recording ---------------------------------------------------------------------------------


def recording_environment(out: Path, root: Path = ROOT) -> dict[str, str]:
    """The variables that make a quick-suite run record a map into ``out``."""
    inherited = os.environ.get("PYTHONPATH")
    site = str(SITE)
    return {
        OUT_ENV: str(out),
        ROOT_ENV: str(root),
        "PYTHONPATH": site if not inherited else site + os.pathsep + inherited,
    }


def merge(out: Path) -> dict:
    """One map from the per-process records a run left in ``out``.

    A test file ran what its own context ran, read or listed, and what each fixture broader than a
    function that its tests use did while it was set up.
    """
    reached: dict[str, dict[str, set[str]]] = {}
    opened: dict[str, set[str]] = {}
    listed: dict[str, set[str]] = {}
    fixtures: dict[str, set[str]] = {}
    import_time: dict[str, set[str]] = {}
    tests: dict[str, list[float]] = {}
    untracked: set[str] = set()
    for path in sorted(out.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for context, functions in record["reached"].items():
            for name, qualnames in functions.items():
                reached.setdefault(context, {}).setdefault(name, set()).update(qualnames)
        for key, target in (
            ("opened", opened),
            ("listed", listed),
            ("fixtures", fixtures),
            ("import_time", import_time),
        ):
            for name, values in record[key].items():
                target.setdefault(name, set()).update(values)
        for name, (count, seconds) in record["tests"].items():
            total = tests.setdefault(name, [0, 0.0])
            total[0] += count
            total[1] += seconds
        untracked.update(record["untracked"])
    files = sorted(tests)

    def contexts(test_file: str) -> list[str]:
        return [test_file, *sorted(fixtures.get(test_file, ()))]

    functions: dict[str, dict[str, set[int]]] = {}
    for index, test_file in enumerate(files):
        for context in contexts(test_file):
            for name, qualnames in reached.get(context, {}).items():
                for qualname in qualnames:
                    functions.setdefault(name, {}).setdefault(qualname, set()).add(index)

    def invert(records: dict[str, set[str]]) -> dict[str, list[str]]:
        index: dict[str, set[str]] = {}
        for test_file in files:
            for context in contexts(test_file):
                for name in records.get(context, ()):
                    index.setdefault(name, set()).add(test_file)
        return {name: sorted(found) for name, found in sorted(index.items())}

    return {
        "version": VERSION,
        "tests": {name: [int(c), round(s, 3)] for name, (c, s) in sorted(tests.items())},
        "files": files,
        "functions": {
            name: {qualname: sorted(ids) for qualname, ids in sorted(by_name.items())}
            for name, by_name in sorted(functions.items())
        },
        "opened": invert(opened),
        "listed": invert(listed),
        # What runs outside every test. Numba and the reports read sources there, so reads
        # outside a test do not count.
        "session": {
            name: sorted(qualnames) for name, qualnames in sorted(reached.get(SESSION, {}).items())
        },
        "import_time": {name: sorted(found) for name, found in sorted(import_time.items())},
        "untracked": sorted(untracked & tests.keys()),
    }


def map_directory(root: Path = ROOT) -> Path:
    common = git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / "superglm-test-map"


def save(data: dict, commit: str, root: Path = ROOT) -> Path:
    """Keep ``data`` as the map of ``commit``, and only the newest ``KEEP`` maps."""
    directory = map_directory(root)
    directory.mkdir(exist_ok=True)
    target = directory / f"{commit}.json"
    partial = target.with_suffix(".partial")
    partial.write_text(json.dumps({**data, "commit": commit}), encoding="utf-8")
    partial.replace(target)
    maps = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in maps[KEEP:]:
        old.unlink()
    return target


# -- Changes -----------------------------------------------------------------------------------


def changes(commit: str, root: Path = ROOT) -> list[tuple[str, str]]:
    """``(status, path)`` for every tracked file the working tree has changed since ``commit``.

    Renames are a deletion and an addition, so both paths count.
    """
    lines = git(root, "diff", "--name-status", "--no-renames", commit).splitlines()
    return [(line[0], line.split("\t", 1)[1]) for line in lines if line]


def best_map(root: Path = ROOT) -> tuple[dict, list[tuple[str, str]]] | None:
    """The stored map with the fewest changed files, and those changes; ``None`` without one."""
    directory = map_directory(root)
    best = None
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != VERSION:
                continue
            changed = changes(data["commit"], root)
        except (json.JSONDecodeError, KeyError, subprocess.CalledProcessError):
            continue  # a map of a commit this clone no longer has, or one half written
        if best is None or len(changed) < len(best[1]):
            best = (data, changed)
    return best


# -- What a module defines and reads -----------------------------------------------------------


def _module_name(path: str) -> str | None:
    parts = PurePosixPath(path).with_suffix("").parts
    if path.startswith("src/superglm/") and path.endswith(".py"):
        parts = parts[1:]
    elif not (path.startswith("tests/") and path.endswith(".py")):
        return None
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _identifiers(node: ast.AST, strings: bool = True) -> frozenset[str]:
    """The names and attribute names ``node`` mentions, and strings that could be one
    (``getattr(obj, "name")``)."""
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif strings and isinstance(child, ast.Constant) and isinstance(child.value, str):
            if child.value.isidentifier():
                found.add(child.value)
    return frozenset(found)


def _called(node: ast.AST) -> frozenset[str]:
    """The names ``node`` calls, or applies as a decorator, by name or attribute."""
    found = set()
    for child in ast.walk(node):
        targets = [child.func] if isinstance(child, ast.Call) else []
        targets += getattr(child, "decorator_list", [])
        for target in targets:
            if isinstance(target, ast.Call):
                target = target.func
            if isinstance(target, ast.Name):
                found.add(target.id)
            elif isinstance(target, ast.Attribute):
                found.add(target.attr)
    return frozenset(found)


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The names of ``node``'s decorators (``a.b(...)`` is ``b``)."""
    found = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        found.add(target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "?"))
    return found


def _bound(node: ast.stmt) -> frozenset[str]:
    """The module-level names a statement (its functions removed) binds."""
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            found.add(child.id)
        elif isinstance(child, (ast.ClassDef, *_FUNCTIONS)):
            found.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            found |= {(alias.asname or alias.name).split(".")[0] for alias in child.names}
    return frozenset(found)


def _without_functions(node: ast.stmt) -> ast.stmt:
    stripped = copy.deepcopy(node)
    for child in ast.walk(stripped):
        for name in ("body", "orelse", "finalbody"):
            body = getattr(child, name, None)
            if isinstance(body, list):
                setattr(child, name, [n for n in body if not isinstance(n, _FUNCTIONS)])
    return stripped


@dataclass
class Shape:
    """A module as a change can reach it: its functions, and its module-level statements."""

    code: str
    functions: dict[str, str]
    reads: dict[str, frozenset[str]]
    decorated: frozenset[str]
    # (names bound, names read, names called, dump) of each module-level statement, its
    # functions removed.
    statements: list[tuple[frozenset[str], frozenset[str], frozenset[str], str]]
    # "Class.attribute" -> dump, for each plain assignment in an undecorated class's body; the
    # rest of the class is its statement above.
    attributes: dict[str, str] = field(default_factory=dict)
    docstrings: str = ""  # the module's and its classes' docstrings
    # Functions a numba extension decorator registers into compiled code everywhere.
    extensions: frozenset[str] = frozenset()

    def bindings(self) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for binds, _, _, dump in self.statements:
            for name in binds or {""}:  # "" collects the statements that bind nothing
                found.setdefault(name, []).append(dump)
        return {name: sorted(dumps) for name, dumps in found.items()}


def shape(text: str | None) -> Shape | None:
    try:
        tree = ast.parse(text or "")
    except SyntaxError:
        return None
    code = ast.dump(tree)
    functions: dict[str, str] = {}
    reads: dict[str, frozenset[str]] = {}
    decorated = set()
    extensions = set()
    docstrings = repr(
        [ast.get_docstring(tree, clean=False)]
        + [
            ast.get_docstring(node, clean=False)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
    )

    def collect(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNCTIONS):
                name = prefix + child.name
                functions[name] = functions.get(name, "") + ast.dump(child)
                reads[name] = reads.get(name, frozenset()) | _identifiers(child)
                if _decorators(child) - PLAIN:
                    decorated.add(name)
                if _decorators(child) & NUMBA_EXTENSIONS:
                    extensions.add(name)
                collect(child, f"{name}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                collect(child, f"{prefix}{child.name}.")
            else:
                collect(child, prefix)

    collect(tree, "")
    statements = []
    attributes: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, _FUNCTIONS):
            continue
        stripped = _without_functions(node)
        _strip_docstrings(stripped)
        if isinstance(stripped, ast.ClassDef) and not stripped.decorator_list:
            # A class attribute is read through instances anywhere, by its name; the class's
            # own statement keeps its bases, keywords and everything else in its body.
            kept = []
            for item in stripped.body:
                targets = [item.target] if isinstance(item, ast.AnnAssign) else []
                targets += item.targets if isinstance(item, ast.Assign) else []
                if targets and all(isinstance(target, ast.Name) for target in targets):
                    for target in targets:
                        key = f"{stripped.name}.{target.id}"
                        attributes[key] = attributes.get(key, "") + ast.dump(item)
                else:
                    kept.append(item)
            stripped.body = kept or [ast.Pass()]
        # Each imported name is its own statement, wherever it sits (``if TYPE_CHECKING:``,
        # ``try``), so adding one name to an import changes no other name's binding; so is
        # each entry of a lazy re-export table.
        for child in list(ast.walk(stripped)):
            for name in ("body", "orelse", "finalbody"):
                body = getattr(child, name, None)
                if isinstance(body, list):
                    setattr(child, name, [n for n in body if not _split_import(n, statements)])
        if _split_import(stripped, statements):
            continue
        _split_lazy_table(stripped, statements)
        if isinstance(stripped, ast.Expr) and isinstance(stripped.value, ast.Constant):
            continue  # the module's docstring, or a bare constant
        statements.append(
            # Strings in a module-level statement are mostly names listed (``__all__``, a lazy
            # re-export table), not read; a function's ``getattr(obj, "name")`` is a read.
            (
                _bound(stripped),
                _identifiers(stripped, strings=False),
                _called(stripped),
                ast.dump(stripped),
            )
        )
    return Shape(
        code,
        functions,
        reads,
        frozenset(decorated),
        statements,
        attributes,
        docstrings,
        frozenset(extensions),
    )


def _split_import(node: ast.stmt, statements: list) -> bool:
    """Add one statement per name ``node`` imports, if it is an import."""
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        return False
    for alias in node.names:
        single = copy.copy(node)
        single.names = [alias]
        bound = frozenset({(alias.asname or alias.name).split(".")[0]})
        statements.append((bound, frozenset(), frozenset(), ast.dump(single)))
    return True


def _split_lazy_table(node: ast.stmt, statements: list) -> None:
    """Add one statement per name a lazy re-export table ``{"a.module": ("Name", ...)}`` binds."""
    value = getattr(node, "value", None)
    if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(value, ast.Dict):
        return
    for key, names in zip(value.keys, value.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(names, (ast.Tuple, ast.List))):
            return
        for item in names.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                bound = frozenset({item.value})
                statements.append(
                    (bound, frozenset(), frozenset(), f"lazy {key.value!r} {item.value}")
                )


class Modules:
    """The Python modules under ``src/superglm`` and ``tests``: their shapes and their imports."""

    def __init__(self, root: Path, read: Callable[[str], str | None]):
        self.root = root
        self.read = read
        tracked = git(root, "ls-files", "--", "src/superglm", "tests").splitlines()
        self.paths = {name: path for path in tracked if (name := _module_name(path)) is not None}
        self._importers: dict[str, set[str]] | None = None

    @functools.cache
    def shape(self, path: str) -> Shape | None:
        return shape(self.read(path))

    @functools.cache
    def bindings(self, path: str) -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
        """What the file at ``path`` takes from repository modules.

        The names it binds from one (``local -> (module, name)``, ``*`` for a star import, and a
        package's lazy re-export table ``{"module": ("name", ...)}``), and the names it binds to a
        module itself (``local -> modules``: ``import a.b`` binds ``a`` to ``a`` and ``a.b``).
        """
        text = self.read(path)
        try:
            tree = ast.parse(text or "")
        except SyntaxError:
            return {}, {}
        name = _module_name(path) or ""
        package = path.endswith("__init__.py")
        names: dict[str, tuple[str, str]] = {}
        modules: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if alias.asname:
                        modules.setdefault(alias.asname, set()).add(alias.name)
                    else:
                        prefixes = {".".join(parts[: i + 1]) for i in range(len(parts))}
                        modules.setdefault(parts[0], set()).update(prefixes)
            elif isinstance(node, ast.ImportFrom):
                base = self._base(name, package, node)
                for alias in node.names:
                    local = alias.asname or alias.name
                    if f"{base}.{alias.name}" in self.paths:
                        modules.setdefault(local, set()).add(f"{base}.{alias.name}")
                    else:
                        names[local if alias.name != "*" else f"*{base}"] = (base, alias.name)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value in self.paths
                        and isinstance(value, (ast.Tuple, ast.List))
                    ):
                        for item in value.elts:
                            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                names[item.value] = (key.value, item.value)
        return names, modules

    def _base(self, name: str, package: bool, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        parent = name if package else name.rpartition(".")[0]
        if node.level > 1:
            parent = parent.rsplit(".", node.level - 1)[0]
        return f"{parent}.{node.module}" if node.module else parent

    def importers(self, path: str) -> set[str]:
        """The modules that take a name from, or bind, the module at ``path``."""
        if self._importers is None:
            index: dict[str, set[str]] = {}
            for source in self.paths.values():
                names, modules = self.bindings(source)
                targets = {module for module, _ in names.values()}
                targets |= {module for found in modules.values() for module in found}
                for module in targets:
                    if module in self.paths:
                        index.setdefault(self.paths[module], set()).add(source)
            self._importers = index
        return self._importers.get(path, set())

    def local_names(self, source: str, path: str, changed: set[str]) -> tuple[set[str], set[str]]:
        """The names ``source`` binds to ``changed`` names of ``path``, and to ``path`` itself."""
        module = _module_name(path)
        names, modules = self.bindings(source)
        local = set()
        for key, (base, name) in names.items():
            if base != module:
                continue
            if key.startswith("*"):
                local |= changed
            elif name in changed:
                local.add(key)
        return local, {key for key, found in modules.items() if module in found}

    @functools.cache
    def compiles(self, path: str) -> bool:
        """Whether the module at ``path`` defines numba-compiled code (numba_cache_key's test)."""
        name, text = _module_name(path), self.read(path)
        if name is None or not path.startswith("src/") or text is None:
            return False
        modules = {key: self.root / value for key, value in self.paths.items()}
        try:
            return _load("numba_cache_key").scan(name, self.root / path, modules, text)[0]
        except SyntaxError:
            return True  # unreadable: assume the worst


# -- Selection ---------------------------------------------------------------------------------


@dataclass
class Selection:
    """The test files to run, each with why, or the reason to run the whole quick suite."""

    files: dict[str, str] = field(default_factory=dict)
    full: str | None = None
    seconds: float = 0.0
    total_seconds: float = 0.0
    map_commit: str = ""
    map_tests: int = 0

    def add(self, files: Iterable[str], reason: str) -> None:
        for name in files:
            self.files.setdefault(name, reason)


def _git_reader(commit: str, root: Path) -> Callable[[str], str | None]:
    @functools.cache
    def read(path: str) -> str | None:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True, check=False
        )
        return result.stdout.decode("utf-8", "replace") if result.returncode == 0 else None

    return read


def _tree_reader(root: Path) -> Callable[[str], str | None]:
    @functools.cache
    def read(path: str) -> str | None:
        try:
            return (root / path).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return None

    return read


class _Selector:
    """One selection: the map, the two versions of the tree, and what has been chosen."""

    def __init__(self, data: dict, root: Path):
        self.data = data
        self.root = root
        self.old = _git_reader(data["commit"], root)
        self.new = _tree_reader(root)
        self.modules, self.old_modules = Modules(root, self.new), Modules(root, self.old)
        self.selection = Selection()
        self.propagated: dict[str, tuple[set[str], set[str]]] = {}

    def ran(self, path: str, qualnames: Iterable[str] | None = None) -> set[str]:
        """The test files that ran ``qualnames`` of ``path`` (any of its functions for ``None``)."""
        by_name = self.data["functions"].get(path, {})
        wanted = by_name.keys() if qualnames is None else set(qualnames) & by_name.keys()
        files = self.data["files"]
        return {files[index] for qualname in wanted for index in by_name[qualname]}

    def run_functions(
        self, path: str, qualnames: Iterable[str], reason: str, session_counts: bool = True
    ) -> None:
        """Select who ran ``qualnames`` of ``path``; one that runs before any test, outside a
        reader of docstrings or names, runs everything."""
        qualnames = set(qualnames)
        session = qualnames & set(self.data["session"].get(path, ())) if session_counts else ()
        if session:
            self.selection.full = f"{path}: {min(session)} runs before any test, and it changed"
        self.selection.add(self.ran(path, qualnames), reason)

    def _imported(self, importer: str, path: str, names: set[str], shape: Shape) -> set[str]:
        """The names ``importer`` binds to ``names`` of ``path``, or reads as ``m.name``."""
        local, aliases = self.modules.local_names(importer, path, names)
        if aliases:
            every = [reads for _, reads, _, _ in shape.statements] + list(shape.reads.values())
            local |= {name for reads in every if reads & aliases for name in names & reads}
        return local

    def propagate(
        self,
        path: str,
        names: set[str],
        shape: Shape | None,
        callables: set[str] = frozenset(),
        origin: str | None = None,
    ) -> None:
        """Rule 4: ``names`` of ``path`` may hold new values, and ``callables`` (functions)
        may return new ones; follow them to the code that reads them.

        A function that reads a changed name becomes a changed callable, and a module-level
        statement that reads either binds changed names (``CAP = limit()`` runs at import). The
        test files that ran a function reading a changed name are selected. In a module that
        compiles with numba a changed callable is a changed name too: its compiled callers never
        show in the map, so their readers are followed instead.
        """
        origin = origin or path
        seen_names, seen_callables = self.propagated.setdefault(path, (set(), set()))
        names, callables = set(names) - seen_names, set(callables) - seen_callables
        if not (names or callables) or shape is None:
            return
        seen_names |= names
        seen_callables |= callables
        compiles = self.modules.compiles(path)
        while True:
            functions = {
                q.split(".")[0] for q, reads in shape.reads.items() if reads & names
            } | callables
            more = {
                bound
                for binds, reads, calls, _ in shape.statements
                if reads & names or calls & functions
                for bound in binds
            }
            if compiles:
                more |= functions
            if more <= names and functions <= callables:
                break
            names |= more
            callables |= functions
        reading = {q for q, reads in shape.reads.items() if reads & names}
        reading |= {q for q in shape.functions if q.split(".")[0] in names and "." in q}
        if reading:
            reason = f"runs code of {path} that reads {min(names)}, from a change to {origin}"
            self.run_functions(path, reading, reason)
        if _module_name(path) is None:
            return
        for importer in sorted(self.modules.importers(path)):
            importer_shape = self.modules.shape(importer)
            if importer_shape is None:
                continue
            local = self._imported(importer, path, names, importer_shape)
            local_callables = self._imported(importer, path, callables, importer_shape)
            if local and _is_conftest(importer):
                self.selection.full = f"{importer} reads {min(local)}, which {origin} changed"
                return
            if local and is_test_file(importer):
                self.selection.add([importer], f"imports {min(local)}, which {origin} changed")
            self.propagate(importer, local, importer_shape, local_callables, origin)

    def python(self, status: str, path: str) -> None:
        before = None if status == "A" else self.old_modules.shape(path)
        after = None if status == "D" else self.modules.shape(path)
        if before is not None and after is not None and before.code == after.code:
            return  # comments and formatting only
        if before is None or after is None:
            present = before or after
            if status == "D":
                self.selection.add(self.ran(path), f"runs {path}, which was deleted")
            if present is not None:
                every = {name for binds, _, _, _ in present.statements for name in binds}
                every |= {q.split(".")[0] for q in present.functions}
                self.propagate(path, every, present)
            if _module_name(path) is None and present is not None:
                self.named(path)
            return
        modified = {
            q
            for q in before.functions.keys() | after.functions.keys()
            if before.functions.get(q) != after.functions.get(q)
        }
        added = after.functions.keys() - before.functions.keys()
        self.run_functions(path, modified - added, f"runs a changed function of {path}")
        extended = (
            (modified & (before.extensions | after.extensions)) if self.compiles(path) else ()
        )
        if extended:
            self.selection.full = f"{path}: {min(extended)} extends numba, and it changed"
            return
        if before.docstrings != after.docstrings:
            self.readers(DOCSTRING_READERS, path, "reads docstrings", tests_only=True)
        top_before = {q for q in before.functions if "." not in q} | set(before.bindings())
        top_after = {q for q in after.functions if "." not in q} | set(after.bindings())
        if top_before != top_after:
            # A lister has to reach the module: by its name, its package's, or sys.modules.
            module = (_module_name(path) or "").split(".")
            reach = {*module, "modules", "import_module", "__path__", "walk_packages"}
            self.readers(ENUMERATORS, path, "lists a module's names", True, frozenset(reach))
        for method in sorted(added):
            owner, _, _ = method.rpartition(".")
            if owner and "<locals>" not in method:
                prefix = f"{owner}."
                ran = [q for q in before.functions if q.startswith(prefix)]
                self.run_functions(path, ran, f"runs {owner}, which gained {method}")
            if method in after.decorated:
                self.selection.add(self.ran(path), f"runs {path}, which gained decorated {method}")
        old_bindings, new_bindings = before.bindings(), after.bindings()
        names = {
            name
            for name in old_bindings.keys() | new_bindings.keys()
            if old_bindings.get(name) != new_bindings.get(name)
        }
        if "" in names:  # a statement that binds nothing changed: what it reads may now differ
            unbound = set(old_bindings.get("", [])) ^ set(new_bindings.get("", []))
            names.discard("")
            # It runs for its effect (a registration): every test that ran this module, and every
            # one that ran a module whose names it calls.
            self.selection.add(self.ran(path), f"runs {path}, whose import-time effects changed")
            imported, _ = self.modules.bindings(path)
            for _, reads, calls, dump in [*before.statements, *after.statements]:
                if dump in unbound:
                    names |= reads
                    for module in {imported[c][0] for c in calls if c in imported}:
                        if module in self.modules.paths:
                            target = self.modules.paths[module]
                            reason = f"runs {target}, which {path} calls at import"
                            self.selection.add(self.ran(target), reason)
        attributes = {
            key.split(".", 1)[1]
            for key in before.attributes.keys() | after.attributes.keys()
            if before.attributes.get(key) != after.attributes.get(key)
        }
        if attributes:
            self.attribute_readers(attributes, path)
        timed = modified & set(self.data["import_time"].get(path, ()))
        names |= {q.split(".")[0] for q in timed}
        if self.compiles(path):
            names |= {q.split(".")[0] for q in modified | added}
        if _module_name(path) is None and names:
            self.named(path)
        self.propagate(path, names, after, {q.split(".")[0] for q in modified})

    def compiles(self, path: str) -> bool:
        return self.modules.compiles(path) or self.old_modules.compiles(path)

    def readers(
        self,
        identifiers: frozenset[str],
        origin: str,
        what: str,
        tests_only: bool = False,
        also: frozenset[str] | None = None,
    ) -> None:
        """The test files that ran a function reading any of ``identifiers`` (and, given
        ``also``, one of those too), in ``tests/`` alone when ``tests_only``."""
        for source in sorted(self.texts):
            if tests_only and not source.startswith("tests/"):
                continue
            shape = self.modules.shape(source)
            if shape is not None:
                reading = {
                    q
                    for q, reads in shape.reads.items()
                    if reads & identifiers and (also is None or reads & also)
                }
                reason = f"runs code of {source} that {what}"
                self.run_functions(source, reading, reason, session_counts=False)

    def attribute_readers(self, attributes: set[str], origin: str) -> None:
        """Rule 4 for class attributes: every function, in any Python file, that reads one by
        name, and every module-level statement that does, followed from there."""
        for source in sorted(self.texts):
            shape = self.modules.shape(source)
            if shape is None:
                continue
            reading = {q for q, reads in shape.reads.items() if reads & attributes}
            reason = (
                f"runs code of {source} that reads {min(attributes)}, from a change to {origin}"
            )
            self.run_functions(source, reading, reason)
            if any(reads & attributes for _, reads, _, _ in shape.statements):
                self.propagate(source, set(attributes), shape, origin=origin)

    @functools.cached_property
    def texts(self) -> dict[str, str]:
        tracked = git(self.root, "ls-files", "--", "*.py").splitlines()
        return {path: text for path in tracked if (text := self.new(path)) is not None}

    def named(self, path: str) -> None:
        """Rule 5: the files that name ``path``, and the test files that ran them."""
        basename = PurePosixPath(path).name
        for source, text in self.texts.items():
            if source != path and basename in text:
                self.selection.add([source] if is_test_file(source) else (), f"names {path}")
                self.selection.add(self.ran(source), f"runs {source}, which names {path}")

    def select(self, changed: list[tuple[str, str]]) -> Selection:
        selection, data = self.selection, self.data
        outside_tests = False
        for status, path in changed:
            if path in FULL_FILES or _is_conftest(path):
                selection.full = f"{path} changed, and it configures every test"
                break
            selection.add(data["opened"].get(path, ()), f"reads {path}")
            # A test that lists a directory may hand its files to a process the recorder cannot
            # see (a linter, node): it counts for any change there, not only an addition.
            parent = str(PurePosixPath(path).parent)
            selection.add(data["listed"].get(parent, ()), f"lists {parent}, where {path} is")
            if is_test_file(path) and status != "D":
                selection.add([path], "changed")
            if path.endswith(".py"):
                outside_tests |= not path.startswith("tests/")
                self.python(status, path)
            elif path not in data["opened"] and status != "D":
                self.named(path)  # rule 5: nothing recorded reads it
            if selection.full:
                break
        if outside_tests:
            selection.add(data["untracked"], "starts a Python the recorder cannot follow")
        selection.add((name for name in ALWAYS if (self.root / name).exists()), "contract test")
        tests = data["tests"]
        selection.map_commit = data["commit"]
        selection.map_tests = sum(count for count, _ in tests.values())
        selection.total_seconds = sum(seconds for _, seconds in tests.values())
        selection.seconds = sum(tests.get(name, (0, 0.0))[1] for name in selection.files)
        if selection.full is None and selection.seconds >= FULL_SHARE * selection.total_seconds:
            selection.full = (
                f"the selection is {selection.seconds:.0f} of the suite's "
                f"{selection.total_seconds:.0f} recorded seconds"
            )
        return selection


def select(data: dict, changed: list[tuple[str, str]], root: Path = ROOT) -> Selection:
    """The test files ``changed`` (``(status, path)`` since the map's commit) reaches."""
    return _Selector(data, root).select(changed)


def choose(root: Path = ROOT) -> Selection:
    """The selection for the working tree against the best stored map, or a full run without one."""
    found = best_map(root)
    if found is None:
        return Selection(full="no test map is stored; this run records one")
    data, changed = found
    return select(data, changed, root)


# -- Command line ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="record a map of the quick suite at a clean HEAD")
    commands.add_parser("select", help="print the selection for the working tree")
    args = parser.parse_args(argv)
    if args.command == "select":
        selection = choose()
        if selection.full:
            print(f"the whole quick suite: {selection.full}")
        for name, reason in sorted(selection.files.items()):
            print(f"{name}\t{reason}")
        print(
            f"{len(selection.files)} files, {selection.seconds:.0f} of "
            f"{selection.total_seconds:.0f} recorded seconds",
            file=sys.stderr,
        )
        return 0
    if git(ROOT, "status", "--porcelain", "--untracked-files=no"):
        sys.exit("test_map: the tree has uncommitted changes; a map records a commit")
    commit = git(ROOT, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory() as out:
        environment = {**os.environ, **recording_environment(Path(out))}
        command = [
            *[sys.executable, "scripts/run_test_suite.py", "-m", QUICK, "-q"],
            *RECORDING_ARGUMENTS,
        ]
        code = subprocess.call(command, cwd=ROOT, env=environment)
        if code != 0:  # a failing test reached less than it would have: no map from it
            sys.exit(f"test_map: the suite exited {code}; no map recorded")
        data = merge(Path(out))
    target = save(data, commit)
    print(f"test_map: recorded {len(data['tests'])} test files at {commit[:12]} in {target}")
    return code


if __name__ == "__main__":
    sys.exit(main())
