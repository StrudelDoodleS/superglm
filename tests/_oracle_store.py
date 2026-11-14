"""Stored answers of slow, deterministic test oracles.

Several suites compare an engine against an independent high-precision oracle
(``mpmath`` at 40-60 digits, ``sympy`` derivatives), and almost all of their
time goes to the oracle, whose answer depends only on its own code and its
arguments.  An :class:`OracleStore` keeps those answers in
``tests/fixtures/oracle_store/<name>.json``, keyed by the call's arguments, and
the engine-against-oracle comparison still runs in full in every test.

What keeps the stored answers the oracle's:

* the store records a fingerprint of the oracle's source and of everything it
  reaches, transitively: every helper it calls, in ``tests`` or ``src``, and
  every constant it reads, by exact value (:class:`_Fingerprint`); it refuses
  to answer under any other fingerprint;
* :meth:`OracleStore.verify`, called by a test in each module that uses a
  store, checks that fingerprint and recomputes a fixed sample of the stored
  entries live (the cheapest ones); they must agree with what is stored;
* a call whose arguments are not stored fails, naming the command that
  refreshes the store, rather than quietly running the slow oracle.

Each answer is stored with the seconds it took to compute, which picks the
sample.  Values are stored exactly: a float by ``float.hex``, an ``mpmath.mpf`` by its
binary mantissa and exponent, so a 50-digit oracle comes back with all its
digits.

Refreshing.  Run the module with ``SUPERGLM_REFRESH_ORACLES=1``, in parallel
if you like::

    SUPERGLM_REFRESH_ORACLES=1 uv run --with mpmath pytest tests/test_x.py -n 8

Every oracle call is then computed live, and each process merges its answers
into the store when it exits, together with the current fingerprint; entries
the run did not request are kept while the fingerprint is unchanged and
dropped when it changed.
"""

from __future__ import annotations

import ast
import atexit
import fractions
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import textwrap
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

STORE_DIRECTORY = Path(__file__).resolve().parent / "fixtures" / "oracle_store"
REFRESH = os.environ.get("SUPERGLM_REFRESH_ORACLES", "") not in ("", "0")
_REFRESH_COMMAND = "SUPERGLM_REFRESH_ORACLES=1 uv run --with mpmath pytest {module} -n 8"
# A recomputed sample entry must agree with the stored one to this relative
# difference: an exact oracle reproduces its stored float bit for bit, and a
# change to the oracle moves its answers by far more.  The allowance only
# covers a platform's libm in an oracle's float64 pre- or post-processing.
_SAMPLE_RTOL = 1e-13
# The same for an ``mpmath`` answer: an oracle can round a step at the caller's
# working precision outside its own ``workdps`` (seen at 1e-73 of a 70-digit
# answer), which no float64 comparison can see; a changed oracle moves far more.
_SAMPLE_MPF_RTOL = 1e-30


def _encode(value: Any) -> Any:
    """``value`` as JSON data that decodes back to an equal value, floats exact."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return {"i": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"f": float(value).hex()}
    if isinstance(value, np.ndarray):
        return {
            "a": str(value.dtype),
            "s": list(value.shape),
            "v": [_encode(item) for item in value.ravel().tolist()],
        }
    if isinstance(value, tuple):
        return {"t": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {"d": [[_encode(key), _encode(item)] for key, item in value.items()]}
    mpf_value = getattr(value, "_mpf_", None)
    if mpf_value is not None:
        sign, mantissa, exponent, bits = mpf_value
        return {"m": [sign, hex(mantissa), exponent, bits]}
    raise TypeError(f"an oracle store cannot hold {type(value).__name__}: {value!r}")


def _decode(data: Any) -> Any:
    if data is None or isinstance(data, (bool, str)):
        return data
    if isinstance(data, list):
        return [_decode(item) for item in data]
    if set(data) == {"a", "s", "v"}:
        items = [_decode(item) for item in data["v"]]
        return np.array(items, dtype=data["a"]).reshape(data["s"])
    if len(data) != 1:
        raise ValueError(f"unreadable oracle store entry {data!r}")
    ((tag, payload),) = data.items()
    if tag == "i":
        return payload
    if tag == "f":
        return float.fromhex(payload)
    if tag == "t":
        return tuple(_decode(item) for item in payload)
    if tag == "d":
        return {_decode(key): _decode(item) for key, item in payload}
    if tag == "m":
        import mpmath

        sign, mantissa, exponent, bits = payload
        # ``make_mpf`` keeps every stored bit; ``mpf(tuple)`` would round to the working precision.
        return mpmath.mp.make_mpf((sign, int(mantissa, 16), exponent, bits))
    raise ValueError(f"unreadable oracle store entry {data!r}")


def _key(args: tuple, kwargs: dict) -> str:
    return json.dumps(_encode((args, tuple(sorted(kwargs.items())))), separators=(",", ":"))


def dump(header: dict[str, str], entries: dict[str, Any]) -> str:
    """The store file: the header's fields, then one line per entry, sorted by key."""
    lines = [f"{json.dumps(key)}: {json.dumps(header[key])}," for key in header]
    rows = [
        f"{json.dumps(key)}: {json.dumps(value, separators=(',', ':'))}"
        for key, value in sorted(entries.items())
    ]
    return "{\n" + "\n".join(lines) + '\n"entries": {\n' + ",\n".join(rows) + "\n}\n}\n"


def _flatten(value: Any) -> Iterable[Any]:
    """The scalars of a decoded value, in a fixed order."""
    if isinstance(value, np.ndarray):
        yield from value.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            yield from _flatten(value[key])
    else:
        yield value


def _nonfinite(value: Any) -> bool:
    """Infinite or NaN, for a float or an ``mpmath.mpf`` (``x - x`` is 0 only when finite)."""
    return not (value - value == 0)


def _agree(stored: Any, live: Any) -> bool:
    stored_items, live_items = list(_flatten(stored)), list(_flatten(live))
    if len(stored_items) != len(live_items):
        return False
    for a, b in zip(stored_items, live_items, strict=True):
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            a, b, rtol = float(a), float(b), _SAMPLE_RTOL
        elif hasattr(a, "_mpf_") and hasattr(b, "_mpf_"):
            rtol = _SAMPLE_MPF_RTOL
        else:
            if type(a) is not type(b) or a != b:
                return False
            continue
        # A relative tolerance means nothing at an infinity (``inf <= inf``):
        # a nonfinite value agrees only with the same infinity, or NaN with NaN.
        if _nonfinite(a) or _nonfinite(b):
            if not (a == b or (a != a and b != b)):
                return False
        elif not (a == b or abs(a - b) <= rtol * max(abs(a), abs(b))):
            return False
    return True


_REPOSITORY = Path(__file__).resolve().parent.parent
# Code under these directories enters a fingerprint by its source, and so does
# everything it reaches; code outside them (numpy, mpmath, sympy, the standard
# library) enters by its qualified name only.
_SOURCE_ROOTS = (_REPOSITORY / "tests", _REPOSITORY / "src")


def _source_file(item: Any) -> Path | None:
    """The repository file that defines ``item``, or ``None`` for code outside it."""
    try:
        name = inspect.getsourcefile(item)
    except TypeError:
        return None
    if name is None:
        return None
    path = Path(name).resolve()
    return path if any(path.is_relative_to(root) for root in _SOURCE_ROOTS) else None


def _code_source(code: Any) -> str:
    """The source of one code object; for a lambda, the lambda's own expression.

    ``inspect.getsource`` gives a lambda's whole statement, which
    ``f, g = lambda x: x, lambda x: 2 * x`` shares between two lambdas; the
    lambda is the one whose span holds most of its code's source positions.
    """
    lines, start = inspect.getsourcelines(code)  # the code object's own: no ``__wrapped__``
    source = "".join(lines)
    if code.co_name != "<lambda>":
        return source
    indent = len(lines[0]) - len(lines[0].lstrip())
    text = textwrap.dedent(source)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return source
    positions = [
        (line - start + 1, column - indent)
        for line, _, column, _ in code.co_positions()
        if line is not None and column is not None
    ]
    best, best_score = source, None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Lambda):
            continue
        span = ((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset))
        segment = ast.get_source_segment(text, node) or source
        node_score = (sum(span[0] <= at < span[1] for at in positions), -len(segment))
        if best_score is None or node_score > best_score:
            best, best_score = segment, node_score
    return best


def _library_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


_ORACLE_LIBRARIES = tuple((name, _library_version(name)) for name in ("mpmath", "sympy"))


def _is_data(value: Any) -> bool:
    """A value, not code: neither callable nor a module."""
    return not (callable(value) or inspect.ismodule(value))


def _node_names(node: ast.AST) -> set[str]:
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
    return found


def _names(source: str) -> list[str]:
    """Every name and attribute name ``source`` mentions, sorted.

    Read from the syntax tree, not from ``co_names``: what a code object
    records varies by Python version and by who imported the module (pytest's
    assertion rewriting adds names), and it leaves out the names in default
    arguments and decorators, which are evaluated outside the function.
    """
    text = textwrap.dedent(source)
    try:
        tree = ast.parse(text)
    except SyntaxError:  # a lambda segment cut from its statement
        tree = ast.parse(f"({text.strip()})")
    return sorted(_node_names(tree))


@functools.cache
def _module_tree(name: str, source: str) -> ast.Module:
    return ast.parse(source)


def _mentions(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _touches(statement: ast.stmt, name: str) -> bool:
    """Whether a module-level statement binds or changes ``name``.

    An assignment whose target mentions it (``X = ...``, ``X[0] = ...``), an
    import of it, an expression or ``del`` that mentions it
    (``X.append(3.0)``), or a compound statement with such a statement inside.
    A ``def`` or ``class`` is code, walked when something names it, and an
    assignment to another name only reads it.
    """
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return any(
            (alias.asname or alias.name.partition(".")[0]) == name for alias in statement.names
        )
    if isinstance(statement, ast.Assign):
        return any(_mentions(target, name) for target in statement.targets)
    if isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        return _mentions(statement.target, name)
    if isinstance(statement, (ast.Expr, ast.Delete)):
        return _mentions(statement, name)
    inner: list[ast.AST] = []
    for field in ("body", "orelse", "finalbody", "handlers", "cases"):
        for child in getattr(statement, field, ()):
            # An ``except`` handler or a ``match`` case holds its statements in ``body``.
            inner.extend([child] if isinstance(child, ast.stmt) else child.body)
    if isinstance(statement, (ast.For, ast.AsyncFor)) and _mentions(statement.target, name):
        return True
    if isinstance(statement, (ast.With, ast.AsyncWith)) and any(
        item.optional_vars is not None and _mentions(item.optional_vars, name)
        for item in statement.items
    ):
        return True
    return any(isinstance(child, ast.stmt) and _touches(child, name) for child in inner)


def _bindings(module: Any, name: str) -> list[ast.stmt]:
    """The module-level statements that bind or change ``name``, whole.

    A compound statement enters whole, so the ``if`` that chooses between two
    bindings, or the ``with`` around one, is part of the constant's source.
    """
    tree = _module_tree(module.__name__, inspect.getsource(module))
    return [statement for statement in tree.body if _touches(statement, name)]


class _Fingerprint:
    """A sha256 of an oracle and everything its answer can depend on.

    Code defined in the repository enters by its source, and the walk follows
    every name that source mentions: the module globals and closure cells it
    resolves to, and the attributes it names of repository modules.  So a
    helper the oracle calls through another helper, in a test module or in
    ``src``, is covered without being listed.

    A module-level constant enters by the source of the statements that bind
    it, and the walk follows the names those statements mention, never by the
    value it computed at import: ``leggauss(3)`` or ``np.log(2 * np.pi)``
    rounds differently on another platform's LAPACK or libm, and an edit to
    the constant is an edit to that source.  Values enter only where they are
    an oracle's input data rather than code: what a test passes in
    ``depends``, closure cells, and a bound method's receiver.  They enter
    losslessly: a float by ``float.hex``, an array by its full little-endian
    dtype descriptor, shape and bytes, an ``mpmath.mpf`` by its mantissa and
    exponent.

    Each object is hashed once, in the order the walk first reaches it, and
    nothing platform- or version-specific enters (no path, no address, no
    library internals), so the digest is the same on every platform, Python
    version and run.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        # id, or (module, global name) -> the order the walk first reached it
        self._seen: dict[Any, int] = {}
        self._keep: list[Any] = []  # holds walked objects so their ids stay unique

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def _put(self, tag: str, text: str | bytes = b"") -> None:
        data = text.encode() if isinstance(text, str) else text
        self._digest.update(f"{tag}:{len(data)}:".encode())
        self._digest.update(data)

    def _first(self, key: Any) -> bool:
        """Whether the walk reaches ``key`` for the first time; a repeat names it by order."""
        if key in self._seen:
            self._put("seen", str(self._seen[key]))
            return False
        self._seen[key] = len(self._seen)
        return True

    def add(self, item: Any) -> None:
        if isinstance(item, (type(None), bool, int, str, bytes, float, complex)):
            self._data(item)
            return
        if not self._first(id(item)):
            return
        self._keep.append(item)
        if isinstance(item, OracleStore):
            self._put("store", item.fingerprint)
        elif inspect.ismodule(item):
            self._module(item)
        elif inspect.isclass(item):
            self._class(item)
        elif isinstance(item, (staticmethod, classmethod)):
            self.add(item.__func__)
        elif isinstance(item, property):
            for accessor in (item.fget, item.fset, item.fdel):
                self.add(accessor)
        elif isinstance(item, functools.partial):
            self._put("partial")
            self.add(item.func)
            self.add(item.args)
            self.add(item.keywords)
        elif inspect.isfunction(item) or inspect.ismethod(item):
            # A decorator's wrapper is code too: its own source, globals and
            # closure (which usually hold the function it wraps), then that function.
            self._function(getattr(item, "__func__", item))
            if inspect.ismethod(item):  # a bound method: its receiver's state too
                self.add(item.__self__)
            if hasattr(item, "__wrapped__"):
                self.add(item.__wrapped__)
        elif hasattr(item, "__wrapped__"):  # ``functools.lru_cache`` and other C wrappers
            self._put("wrapper", type(item).__qualname__)
            self.add(item.__wrapped__)
        elif hasattr(item, "py_func"):  # a numba dispatcher: its Python source is the code
            self.add(item.py_func)
        else:
            self._data(item)

    def _named(self, item: Any) -> None:
        # Only the name: a library's module layout moves between versions, and
        # its code is not the oracle's.
        name = getattr(item, "__qualname__", None) or getattr(item, "__name__", None)
        self._put("external", name or type(item).__qualname__)

    def _module(self, module: Any) -> None:
        if _source_file(module) is None:
            self._put("external module", module.__name__)
            return
        # A module listed in ``depends``: its source, and everything its own code reaches.
        self._put("module", inspect.getsource(module))
        for name, value in vars(module).items():
            if getattr(value, "__module__", None) == module.__name__ and name != "__annotate__":
                self.add(value)

    def _class(self, cls: type) -> None:
        if _source_file(cls) is None:
            self._named(cls)
            return
        source = inspect.getsource(cls)
        self._put("class", source)
        for base in cls.__bases__:
            self.add(base)
        for name, value in vars(cls).items():
            # Methods written in the class body; not the ones ``dataclass`` or
            # ``typing`` generate, nor ``__annotate__``, which vary by Python version.
            if isinstance(value, (staticmethod, classmethod, property)) or inspect.isfunction(
                value
            ):
                function = getattr(value, "__func__", getattr(value, "fget", value))
                if not name.startswith("__annotate") and _source_file(function) is not None:
                    self._put("member", name)
                    self.add(value)
        # The class body's attribute values enter by their source, above; the
        # names that source mentions resolve in the class's module.
        body = [
            node
            for node in ast.parse(textwrap.dedent(source)).body[0].body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        module = sys.modules.get(cls.__module__)
        if module is not None:
            names = sorted({n for node in body for n in _node_names(node)})
            self._resolve(vars(module), {}, names)

    def _function(self, function: Any) -> None:
        if _source_file(function) is None or not hasattr(function, "__code__"):
            self._named(function)
            return
        source = _code_source(function.__code__)
        self._put("function", source)
        cells = {}
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or (), strict=True
        ):
            try:
                cells[name] = cell.cell_contents
            except ValueError:  # an empty cell
                pass
        self._resolve(function.__globals__, cells, _names(source))

    def _resolve(self, scope: dict[str, Any], cells: dict[str, Any], names: list[str]) -> None:
        """Walk what ``names`` resolve to in a module's globals and a closure's cells."""
        modules: dict[int, Any] = {}
        for name in names:
            if name in cells:
                value = cells[name]
            elif name in scope:
                value = scope[name]
            else:
                continue
            if inspect.ismodule(value):
                # A module the code names: only the attributes it names (below).
                self._put("module name", value.__name__)
                if _source_file(value) is not None:
                    modules[id(value)] = value
            elif name in cells or not _is_data(value):
                self.add(value)  # code, or a closure cell's runtime value
            else:
                self._global(scope.get("__name__"), name, value)
        # ``helpers.name``: the attribute of a repository module the code names.
        pending = list(modules.values())
        while pending:
            module = pending.pop(0)
            for name in names:
                value = getattr(module, name, None)
                if value is None:
                    continue
                if inspect.ismodule(value):
                    self._put("module name", value.__name__)
                    if _source_file(value) is not None and id(value) not in modules:
                        modules[id(value)] = value
                        pending.append(value)
                elif _is_data(value):
                    self._global(module.__name__, name, value)
                else:
                    self.add(value)

    def _global(self, module_name: str | None, name: str, value: Any) -> None:
        """A module-level constant: the source of the statements that bind it."""
        module = sys.modules.get(module_name or "")
        if module is None or _source_file(module) is None:
            self._data(value)  # a library's constant: its value, which it does not compute
            return
        if not self._first((module_name, name)):
            return
        text = inspect.getsource(module)
        statements = _bindings(module, name)
        if not statements:
            # Bound some other way (``globals()``, a loop): its value is all there is.
            self._put("unbound global", f"{module_name}.{name}")
            self.add(value)
            return
        self._put("global", f"{module_name}.{name}")
        names: set[str] = set()
        for statement in statements:
            if (
                isinstance(statement, ast.ImportFrom)
                and statement.module
                and any((a.asname or a.name) == name for a in statement.names)
            ):
                # ``from x import name``: the constant is x's; follow it there.
                alias = next(a for a in statement.names if (a.asname or a.name) == name)
                source = importlib.import_module(
                    statement.module
                    if statement.level == 0
                    else importlib.util.resolve_name(
                        "." * statement.level + statement.module, module.__package__
                    )
                )
                self._put("import", f"{source.__name__}.{alias.name}")
                self._global(source.__name__, alias.name, value)
                continue
            # The statement's exact text: ``ast.unparse`` formats differently by version.
            self._put("binding", ast.get_source_segment(text, statement) or "")
            names.update(_node_names(statement))
        names.discard(name)
        self._resolve(vars(module), {}, sorted(names))

    def _data(self, item: Any) -> None:
        if item is None or isinstance(item, (bool, int, str)):
            self._put(type(item).__name__, repr(item))
        elif isinstance(item, bytes):
            self._put("bytes", item)
        elif isinstance(item, float):
            self._put("float", float(item).hex())
        elif isinstance(item, np.generic) and not item.dtype.hasobject:
            # ``np.bool_``, ``np.float32``, ``np.int64``...: the dtype and the bytes.
            little = np.asarray(item).astype(item.dtype.newbyteorder("<"))
            self._put("numpy scalar", repr(little.dtype.descr))
            self._put("bytes", little.tobytes())
        elif isinstance(item, (complex, np.complexfloating)):
            self._put("complex", f"{item.real.hex()},{item.imag.hex()}")
        elif isinstance(item, np.integer):
            self._put("int", repr(int(item)))
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                self._put("array", f"{item.dtype.descr!r}{item.shape}")
                for element in item.ravel().tolist():
                    self.add(element)
            else:
                # The full little-endian descriptor, which keeps a structured
                # dtype's field names, types and offsets.
                little = np.ascontiguousarray(item).astype(item.dtype.newbyteorder("<"))
                self._put("array", f"{little.dtype.descr!r}{item.shape}")
                self._put("bytes", little.tobytes())
        elif isinstance(item, fractions.Fraction):
            self._put("fraction", f"{item.numerator}/{item.denominator}")
        elif hasattr(item, "_mpf_"):
            sign, mantissa, exponent, bits = item._mpf_
            self._put("mpf", f"{sign},{mantissa:x},{exponent},{bits}")
        elif isinstance(item, (range, slice)):
            self._put(type(item).__name__, f"{item.start},{item.stop},{item.step}")
        elif isinstance(item, (tuple, list)):
            self._put(type(item).__name__, str(len(item)))
            if type(item) not in (tuple, list):  # a named tuple
                self.add(type(item))
            for element in item:
                self.add(element)
        elif isinstance(item, (set, frozenset)):
            parts = []
            for element in item:
                part = _Fingerprint()
                part.add(element)
                parts.append(part.hexdigest())
            self._put(type(item).__name__, ",".join(sorted(parts)))
        elif isinstance(item, dict):
            self._put("dict", str(len(item)))
            for key, value in item.items():
                self.add(key)
                self.add(value)
        elif _source_file(type(item)) is not None:
            # An instance of a repository class: the class and the instance's state.
            self.add(type(item))
            slots = [n for c in type(item).__mro__ for n in getattr(c, "__slots__", ())]
            state = dict(vars(item)) if hasattr(item, "__dict__") else {}
            self.add({**state, **{n: getattr(item, n) for n in slots if hasattr(item, n)}})
        elif type(item).__module__.partition(".")[0] == "sympy":
            import sympy

            self._put("sympy", sympy.srepr(item))
        elif callable(item) and hasattr(item, "__name__"):  # ``np.log``, ``math.exp``
            self._named(item)
        elif type(item).__module__ == "builtins":
            raise TypeError(
                f"an oracle fingerprint cannot hash a {type(item).__name__} constant: {item!r}; "
                "teach _Fingerprint._data its value"
            )
        else:
            # Anything else from outside the repository, such as a module-level
            # mpmath context, whose repr would carry the ambient precision: its
            # type names it, and its value is not part of any oracle's code.
            self._put("opaque", f"{type(item).__module__}.{type(item).__qualname__}")


class OracleStore:
    """The stored answers of one oracle function, read back by its arguments.

    ``name`` names the JSON file; ``oracle`` is the function itself, called
    with plain data (floats, ints, strings, tuples); ``depends`` names what
    else enters the fingerprint, beyond what the walk from ``oracle`` reaches
    (another store's fingerprint, a whole module); ``module`` is the test file whose run refreshes
    the store, quoted in the failure messages.
    """

    def __init__(
        self,
        name: str,
        oracle: Callable[..., Any],
        *,
        depends: Iterable[Any] = (),
        module: str,
        sample: int = 2,
    ) -> None:
        self.name, self.oracle, self.module, self.sample = name, oracle, module, sample
        fingerprint = _Fingerprint()
        # The oracle libraries' versions: an upgrade recomputes every answer
        # rather than leaving it to the sampled check.
        fingerprint.add(_ORACLE_LIBRARIES)
        self._depends = tuple(depends)
        for item in (oracle, *self._depends):
            fingerprint.add(item)
        self.fingerprint = fingerprint.hexdigest()
        self.path = STORE_DIRECTORY / f"{name}.json"
        self._entries: dict[str, Any] = {}
        self._stored_fingerprint = None
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._stored_fingerprint = data["fingerprint"]
            self._entries = data["entries"]
        self._computed: dict[str, Any] = {}
        if REFRESH:
            atexit.register(self._write)
        functools.update_wrapper(self, oracle)

    def _refresh_hint(self) -> str:
        return _REFRESH_COMMAND.format(module=self.module)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = _key(args, kwargs)
        if REFRESH:
            if key not in self._computed:
                start = time.perf_counter()
                value = _encode(self.oracle(*args, **kwargs))
                self._computed[key] = [value, round(time.perf_counter() - start, 3)]
            return _decode(self._computed[key][0])
        if self._stored_fingerprint != self.fingerprint:
            raise AssertionError(
                f"the {self.name} oracle changed since its answers were stored; "
                f"refresh them with: {self._refresh_hint()}"
            )
        if key not in self._entries:
            raise AssertionError(
                f"the {self.name} oracle store has no answer for {args!r} {kwargs!r}; "
                f"refresh it with: {self._refresh_hint()}"
            )
        return _decode(self._entries[key][0])

    def verify(self) -> None:
        """The stored answers are this oracle's: fingerprint, then a live sample."""
        if REFRESH:
            return  # a refresh run recomputes every answer it reads
        assert self.path.exists(), (
            f"no {self.name} oracle store; create it with: {self._refresh_hint()}"
        )
        assert self._stored_fingerprint == self.fingerprint, (
            f"the {self.name} oracle changed since its answers were stored; "
            f"refresh them with: {self._refresh_hint()}"
        )
        assert self._entries, f"the {self.name} oracle store is empty"
        # The cheapest entries by their recorded compute time, ties by key: the
        # same sample on every platform and run, and a check that costs seconds
        # however slow the oracle's dearest answers are.
        keys = sorted(self._entries, key=lambda key: (self._entries[key][1], key))
        for key in keys[: self.sample]:
            args, kwargs = _decode(json.loads(key))
            live = self.oracle(*args, **dict(kwargs))
            assert _agree(_decode(self._entries[key][0]), live), (
                f"the {self.name} oracle no longer reproduces its stored answer for "
                f"{args!r}; refresh it with: {self._refresh_hint()}"
            )

    def _write(self) -> None:
        """Merge this process's answers into the store, under a lock xdist workers share."""
        if not self._computed:
            return
        STORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(".lock")
        for _ in range(6000):
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
                break
            except FileExistsError:
                time.sleep(0.01)
        else:
            raise RuntimeError(f"{lock} is held; remove it if no refresh is running")
        try:
            entries = {}
            if self.path.exists():
                data = json.loads(self.path.read_text())
                # Answers stored under another fingerprint belong to an older oracle.
                if data["fingerprint"] == self.fingerprint:
                    entries = data["entries"]
            entries.update(self._computed)
            header = {
                "oracle": f"{self.oracle.__module__}.{self.oracle.__qualname__}",
                "refresh": self._refresh_hint(),
                "fingerprint": self.fingerprint,
            }
            self.path.write_text(dump(header, entries))
        finally:
            os.remove(lock)
