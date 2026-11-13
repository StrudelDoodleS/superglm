"""Stored answers of slow, deterministic test oracles.

Several suites compare an engine against an independent high-precision oracle
(``mpmath`` at 40-60 digits, ``sympy`` derivatives), and almost all of their
time goes to the oracle, whose answer depends only on its own code and its
arguments.  An :class:`OracleStore` keeps those answers in
``tests/fixtures/oracle_store/<name>.json``, keyed by the call's arguments, and
the engine-against-oracle comparison still runs in full in every test.

What keeps the stored answers the oracle's:

* the store records a fingerprint of the oracle's source (the function and
  every helper or constant it names, listed by the test) and refuses to answer
  under any other fingerprint;
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


def _agree(stored: Any, live: Any) -> bool:
    stored_items, live_items = list(_flatten(stored)), list(_flatten(live))
    if len(stored_items) != len(live_items):
        return False
    for a, b in zip(stored_items, live_items, strict=True):
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            if np.isnan(a) and np.isnan(b):
                continue
            if not (a == b or abs(a - b) <= _SAMPLE_RTOL * max(abs(a), abs(b))):
                return False
        elif type(a) is not type(b) or a != b:
            return False
    return True


def _source(item: Any) -> str:
    if callable(item) or inspect.ismodule(item):
        return inspect.getsource(item)
    return repr(item)


class OracleStore:
    """The stored answers of one oracle function, read back by its arguments.

    ``name`` names the JSON file; ``oracle`` is the function itself, called
    with plain data (floats, ints, strings, tuples); ``depends`` lists every
    helper, module or constant its answer depends on beyond its own source,
    and enters the fingerprint; ``module`` is the test file whose run refreshes
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
        digest = hashlib.sha256()
        for item in (oracle, *depends):
            digest.update(_source(item).encode())
            digest.update(b"\0")
        self.fingerprint = digest.hexdigest()
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
            payload = {
                "oracle": f"{self.oracle.__module__}.{self.oracle.__qualname__}",
                "refresh": self._refresh_hint(),
                "fingerprint": self.fingerprint,
                "entries": dict(sorted(entries.items())),
            }
            self.path.write_text(json.dumps(payload, indent=0) + "\n")
        finally:
            os.remove(lock)
