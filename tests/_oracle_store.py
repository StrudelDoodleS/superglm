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

import atexit
import fractions
import functools
import hashlib
import inspect
import json
import os
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


class _Fingerprint:
    """A sha256 of an oracle and everything its answer can depend on.

    Code defined in the repository enters by its source, and the walk follows
    every name that source reaches: the globals and closure cells its code
    objects load (including nested functions and comprehensions), the
    attributes it names of repository modules, default arguments, class bodies
    and methods.  So a helper the oracle calls through another helper, in a test
    module or in ``src``, is covered without being listed.  Data enters
    losslessly: a float by ``float.hex``, an array by its dtype, shape and
    little-endian bytes, an ``mpmath.mpf`` by its mantissa and exponent.  Each
    object is hashed once, in the order the walk first reaches it, so the
    digest is the same on every platform and run.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._seen: dict[int, int] = {}  # id -> the order the walk first reached it
        self._keep: list[Any] = []  # holds walked objects so their ids stay unique

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def _put(self, tag: str, text: str | bytes = b"") -> None:
        data = text.encode() if isinstance(text, str) else text
        self._digest.update(f"{tag}:{len(data)}:".encode())
        self._digest.update(data)

    def add(self, item: Any) -> None:
        if isinstance(item, (type(None), bool, int, str, bytes, float, complex)):
            self._data(item)
            return
        if id(item) in self._seen:
            # Which object, by when the walk first reached it: the same on every run.
            self._put("seen", str(self._seen[id(item)]))
            return
        self._seen[id(item)] = len(self._seen)
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
        self._put("class", inspect.getsource(cls))
        for base in cls.__bases__:
            self.add(base)
        for name, value in vars(cls).items():
            # Methods written in the class body and its attribute values; not the
            # methods ``dataclass`` or ``typing`` generate, nor dunder data such as
            # ``__static_attributes__`` or ``__annotate__``, which vary by Python version.
            code = isinstance(value, (staticmethod, classmethod, property)) or inspect.isfunction(
                value
            )
            if code:
                function = getattr(value, "__func__", getattr(value, "fget", value))
                if not name.startswith("__annotate") and _source_file(function) is not None:
                    self._put("member", name)
                    self.add(value)
            elif not (name.startswith("__") and name.endswith("__")):
                self._put("member", name)
                self.add(value)

    def _function(self, function: Any) -> None:
        if _source_file(function) is None or not hasattr(function, "__code__"):
            self._named(function)
            return
        # The code object's own source: ``getsource(function)`` follows ``__wrapped__``.
        self._put("function", inspect.getsource(function.__code__))
        self.add(function.__defaults__)
        self.add(function.__kwdefaults__)
        scope = dict(function.__globals__)
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or (), strict=True
        ):
            try:
                scope[name] = cell.cell_contents
            except ValueError:  # an empty cell
                scope.pop(name, None)
        names: list[str] = [*function.__code__.co_freevars]
        codes = [function.__code__]
        while codes:
            code = codes.pop(0)
            names.extend(code.co_names)
            # Not the ``__annotate__`` functions Python 3.14 compiles for annotations.
            codes.extend(
                c for c in code.co_consts if inspect.iscode(c) and c.co_name != "__annotate__"
            )
        # Sorted: the order of ``co_names``, and whether a comprehension has its
        # own code object, are the compiler's business and vary by Python version.
        # pytest's assertion rewriting adds ``@pytest_ar`` and ``@py_builtins``,
        # which are not identifiers: the walk must not depend on who imported the module.
        names = sorted({name for name in names if name.isidentifier()})
        modules = {
            id(scope[n]): scope[n]
            for n in names
            if inspect.ismodule(scope.get(n)) and _source_file(scope[n]) is not None
        }
        for name in names:
            if inspect.ismodule(scope.get(name)):
                # A module the code names: only the attributes it names (below).
                self._put("module name", scope[name].__name__)
            elif name in scope:
                self.add(scope[name])
            # ``helpers.name``: the attribute of a repository module the code names.
            for module in modules.values():
                if hasattr(module, name):
                    self.add(getattr(module, name))

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
                self._put("array", f"{item.dtype.str[1:]}{item.shape}")
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
        for item in (oracle, *depends):
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
