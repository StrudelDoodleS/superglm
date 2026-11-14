"""The oracle store answers exactly what its oracle computed, and nothing else."""

from __future__ import annotations

import importlib
import json
import linecache
import sys
import textwrap

import numpy as np
import pytest

from tests import _oracle_store
from tests._oracle_store import OracleStore, _decode, _encode

mp = pytest.importorskip("mpmath")


def _oracle(x, scale=1.0):
    with mp.workdps(50):
        root = mp.sqrt(mp.mpf(x))
    return {"root": root, "pair": (float(root) * scale, np.array([x, -x]))}


def _other_oracle(x, scale=1.0):
    return {"root": mp.mpf(0), "pair": (0.0, np.array([x, -x]))}


@pytest.fixture
def store_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(_oracle_store, "STORE_DIRECTORY", tmp_path)
    return tmp_path


def _refreshed(monkeypatch, oracle, calls):
    """A store of ``oracle`` filled by a refresh run over ``calls``, then read normally."""
    monkeypatch.setattr(_oracle_store, "REFRESH", True)
    monkeypatch.setattr(_oracle_store.atexit, "register", lambda function: None)
    writer = OracleStore("sample", oracle, module="tests/test_oracle_store.py")
    for args, kwargs in calls:
        writer(*args, **kwargs)
    writer._write()
    monkeypatch.setattr(_oracle_store, "REFRESH", False)
    return OracleStore("sample", oracle, module="tests/test_oracle_store.py")


def test_values_round_trip_exactly():
    with mp.workdps(60):
        third = mp.mpf(1) / 3
    value = (
        0.1,
        -0.0,
        float("inf"),
        5e-324,
        third,
        np.array([[0.1, np.nan], [1e308, -2.5]]),
        {(0, 1): [1, "a", None, True]},
    )
    back = _decode(json.loads(json.dumps(_encode(value))))
    assert back[0] == 0.1 and str(back[1]) == "-0.0" and back[2] == float("inf")
    assert back[3] == 5e-324
    assert back[4] == third and back[4]._mpf_ == third._mpf_
    np.testing.assert_array_equal(back[5], value[5])
    assert back[5].shape == (2, 2) and back[5].dtype == np.float64
    assert back[6] == {(0, 1): [1, "a", None, True]}


def test_a_stored_answer_is_the_oracles_and_a_missing_one_refuses(store_directory, monkeypatch):
    store = _refreshed(monkeypatch, _oracle, [((2.0,), {}), ((3.0,), {"scale": 2.0})])
    assert store(2.0)["root"] == _oracle(2.0)["root"]
    assert store(3.0, scale=2.0)["pair"][0] == _oracle(3.0, scale=2.0)["pair"][0]
    store.verify()
    with pytest.raises(AssertionError, match="no answer"):
        store(5.0)


def test_a_changed_oracle_refuses_its_old_answers(store_directory, monkeypatch):
    _refreshed(monkeypatch, _oracle, [((2.0,), {})])
    changed = OracleStore("sample", _other_oracle, module="tests/test_oracle_store.py")
    with pytest.raises(AssertionError, match="changed"):
        changed(2.0)
    with pytest.raises(AssertionError, match="changed"):
        changed.verify()
    # A dependency is part of the fingerprint too.
    depending = OracleStore("sample", _oracle, depends=(7,), module="tests/test_oracle_store.py")
    with pytest.raises(AssertionError, match="changed"):
        depending.verify()


def test_verify_recomputes_the_sample_and_catches_a_wrong_answer(store_directory, monkeypatch):
    _refreshed(monkeypatch, _oracle, [((2.0,), {})])
    path = store_directory / "sample.json"
    data = json.loads(path.read_text())
    ((key, (value, seconds)),) = data["entries"].items()
    value["d"][1][1]["t"][0] = _encode(1.4142135623730951 * (1 + 2**-40))
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="no longer reproduces"):
        OracleStore("sample", _oracle, module="tests/test_oracle_store.py").verify()


def test_an_mpmath_answer_may_differ_below_its_working_precision_only():
    """An oracle step left at the caller's precision moves a 70-digit answer by ~1e-73."""
    from tests._oracle_store import _agree

    with mp.workdps(70):
        exact = mp.mpf(2) / 3
        assert _agree({"a": exact}, {"a": exact * (1 + mp.mpf(10) ** -60)})
        assert not _agree({"a": exact}, {"a": exact * (1 + mp.mpf(10) ** -20)})


def _fingerprint(oracle, depends=()):
    return OracleStore(
        "sample", oracle, depends=depends, module="tests/test_oracle_store.py"
    ).fingerprint


def test_a_constant_enters_the_fingerprint_by_its_exact_value():
    """An array's repr rounds its elements; its bytes, dtype and shape do not."""
    weights = np.array([1.0, 1.0, 2.0])
    nudged = weights.copy()
    nudged[0] = 1.000000001
    assert repr(weights) == repr(nudged)  # the lossy view the fingerprint must not use
    assert _fingerprint(_oracle, (weights,)) != _fingerprint(_oracle, (nudged,))
    assert _fingerprint(_oracle, (weights,)) != _fingerprint(_oracle, (weights.astype(np.float32),))
    assert _fingerprint(_oracle, (weights,)) != _fingerprint(_oracle, (weights.reshape(3, 1),))
    assert _fingerprint(_oracle, (0.1,)) != _fingerprint(_oracle, (np.nextafter(0.1, 1.0),))
    assert _fingerprint(_oracle, (weights,)) == _fingerprint(_oracle, (weights.copy(),))


_SCALE = 1.0


def _inner(x):
    return mp.mpf(x) * _SCALE


def _outer(x):
    """An oracle whose answer depends on a helper it calls and a constant that helper reads."""
    return _inner(x) + 1


def test_the_fingerprint_follows_every_helper_the_oracle_reaches(monkeypatch):
    before = _fingerprint(_outer)
    monkeypatch.setattr(f"{__name__}._inner", lambda x: mp.mpf(x) * 2)
    assert _fingerprint(_outer) != before
    monkeypatch.undo()
    assert _fingerprint(_outer) == before


def test_the_fingerprint_follows_package_code_through_its_callees(monkeypatch):
    """``location_of_mean`` picks a two-piece row's piece through ``log_mean_loading``."""
    from superglm.distributional.kernels import two_piece

    location_of_mean = two_piece.location_of_mean

    def oracle(theta):
        return float(location_of_mean(*(np.array([v]) for v in theta))[0])

    before = _fingerprint(oracle)
    original = two_piece.log_mean_loading

    def shifted(*args, **kwargs):
        loadings = original(*args, **kwargs)
        return (loadings[0] - 100.0, *loadings[1:])

    monkeypatch.setattr(two_piece, "log_mean_loading", shifted)
    assert _fingerprint(oracle) != before


def test_a_stored_store_refuses_to_answer_after_a_helper_changes(store_directory, monkeypatch):
    _refreshed(monkeypatch, _outer, [((2.0,), {})])
    monkeypatch.setattr(f"{__name__}._inner", lambda x: mp.mpf(x) * 2)
    store = OracleStore("sample", _outer, module="tests/test_oracle_store.py")
    with pytest.raises(AssertionError, match="changed"):
        store(2.0)
    with pytest.raises(AssertionError, match="changed"):
        store.verify()


def test_a_nonfinite_value_agrees_only_with_itself():
    """A relative tolerance at an infinity would accept anything (``inf <= inf``)."""
    from tests._oracle_store import _agree

    inf, nan = float("inf"), float("nan")
    for finite in (1.0, 1e308, -0.0):
        assert not _agree(inf, finite) and not _agree(finite, inf)
        assert not _agree(nan, finite) and not _agree(finite, nan)
        assert not _agree(mp.inf, mp.mpf(finite)) and not _agree(mp.mpf(finite), mp.inf)
        assert not _agree(mp.nan, mp.mpf(finite))
    assert not _agree(inf, -inf) and not _agree(mp.inf, -mp.inf)
    assert not _agree(inf, nan) and not _agree(mp.inf, mp.nan)
    assert _agree(inf, inf) and _agree(-inf, -inf) and _agree(nan, nan)
    assert _agree(np.float64(inf), inf) and _agree(mp.inf, mp.inf) and _agree(mp.nan, mp.nan)


def test_verify_catches_a_stored_answer_replaced_by_an_infinity(store_directory, monkeypatch):
    _refreshed(monkeypatch, _oracle, [((2.0,), {})])
    path = store_directory / "sample.json"
    data = json.loads(path.read_text())
    ((key, (value, seconds)),) = data["entries"].items()
    value["d"][0][1] = _encode(mp.inf)
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="no longer reproduces"):
        OracleStore("sample", _oracle, module="tests/test_oracle_store.py").verify()


def test_a_module_level_mpmath_constant_does_not_take_the_ambient_precision():
    """Importing a module that raises ``mp.dps`` first must not change a store's fingerprint."""
    import importlib

    from tests import _negative_binomial_crps_oracle as module

    tail = module._RELATIVE_TAIL
    try:
        with mp.workdps(50):
            assert importlib.reload(module)._RELATIVE_TAIL._mpf_ == tail._mpf_
    finally:
        importlib.reload(module)


class _Scaled:
    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, x):
        return self.scale * x


def test_a_bound_method_enters_with_its_receiver():
    assert _fingerprint(_Scaled(1.0).evaluate) != _fingerprint(_Scaled(2.0).evaluate)
    assert _fingerprint(_Scaled(1.0).evaluate) == _fingerprint(_Scaled(1.0).evaluate)


def test_a_builtin_constant_the_fingerprint_cannot_read_is_refused():
    with pytest.raises(TypeError, match="cannot hash"):
        _fingerprint(_oracle, (bytearray(b"x"),))


@pytest.fixture
def fingerprint_of(tmp_path, monkeypatch):
    """The fingerprint of ``oracle`` in a repository module written from ``source``.

    A constant enters a fingerprint by the source that binds it, so an edit is
    an edit to a file: each call writes the module ``_oracle_store_case`` into
    a fresh directory and imports it afresh, under the same name.
    """
    monkeypatch.setattr(_oracle_store, "_SOURCE_ROOTS", (*_oracle_store._SOURCE_ROOTS, tmp_path))
    calls = []

    def fingerprint(source, helper=""):
        directory = tmp_path / str(len(calls))
        directory.mkdir()
        calls.append(directory)
        (directory / "_oracle_store_case.py").write_text(textwrap.dedent(source))
        (directory / "_oracle_store_helper.py").write_text(textwrap.dedent(helper))
        sys.modules.pop("_oracle_store_case", None)
        sys.modules.pop("_oracle_store_helper", None)
        linecache.clearcache()
        importlib.invalidate_caches()
        sys.path.insert(0, str(directory))
        try:
            return _fingerprint(importlib.import_module("_oracle_store_case").oracle)
        finally:
            sys.path.remove(str(directory))

    yield fingerprint
    sys.modules.pop("_oracle_store_case", None)
    sys.modules.pop("_oracle_store_helper", None)


_EDITS = {
    "array-element": (
        "import numpy as np\nWEIGHTS = np.array([1.0, 2.0])\n"
        "def oracle(x):\n    return WEIGHTS[0] * x\n",
        "1.0, 2.0",
        "1.000000001, 2.0",
    ),
    "constant-through-a-helper": (
        "SCALE = 1.0\ndef inner(x):\n    return x * SCALE\ndef oracle(x):\n    return inner(x)\n",
        "SCALE = 1.0",
        "SCALE = 2.0",
    ),
    "constant-in-a-default": (
        "SCALE = 1.0\ndef oracle(x, scale=SCALE):\n    return x * scale\n",
        "SCALE = 1.0",
        "SCALE = 2.0",
    ),
    "decorator-wrapper": (
        "import functools\nSCALE = 1.0\n"
        "def scaled(f):\n    @functools.wraps(f)\n    def wrapper(x):\n"
        "        return f(x) * SCALE\n    return wrapper\n"
        "@scaled\ndef oracle(x):\n    return x\n",
        "SCALE = 1.0",
        "SCALE = 2.0",
    ),
    "class-attribute": (
        "SCALE = 1.0\nclass Law:\n    scale = SCALE\ndef oracle(x):\n    return x * Law.scale\n",
        "SCALE = 1.0",
        "SCALE = 2.0",
    ),
    "repeated-object": (
        "A, B = [1.0], [2.0]\nCONFIGURATION = (A, B, A)\n"
        "def oracle(x):\n    return CONFIGURATION[2][0] * x\n",
        "(A, B, A)",
        "(A, B, B)",
    ),
    "numpy-bool": (
        "import numpy as np\nENABLED = np.bool_(True)\n"
        "def oracle(x):\n    return x if ENABLED else -x\n",
        "True",
        "False",
    ),
    "structured-dtype": (
        "import numpy as np\n"
        "TABLE = np.array([(1.0, 2.0)], dtype=[('x', 'f8'), ('y', 'f8')])\n"
        "def oracle(x):\n    return TABLE['x'][0] * x\n",
        "[('x', 'f8'), ('y', 'f8')]",
        "[('y', 'f8'), ('x', 'f8')]",
    ),
    "lambda-sharing-a-statement": (
        "IDENTITY, DOUBLED = lambda x: x, lambda x: x if x <= 2 else 2 * x\n"
        "HELPER = IDENTITY\ndef oracle(x):\n    return HELPER(x)\n",
        "HELPER = IDENTITY",
        "HELPER = DOUBLED",
    ),
    "guarded-binding": (
        "ENABLED = True\nif ENABLED:\n    SCALE = 1.0\nelse:\n    SCALE = 2.0\n"
        "def oracle(x):\n    return x * SCALE\n",
        "ENABLED = True",
        "ENABLED = False",
    ),
    "mutation-after-binding": (
        "WEIGHTS = [1.0, 2.0]\nWEIGHTS.append(3.0)\ndef oracle(x):\n    return WEIGHTS[-1] * x\n",
        "append(3.0)",
        "append(4.0)",
    ),
    "callable-selection": (
        "import mpmath as mp\nF = mp.exp\ndef oracle(x):\n    return F(x)\n",
        "F = mp.exp",
        "F = mp.sin",
    ),
    "mutation-in-another-assignment": (
        "COEFFICIENTS = [1.0, 2.0, 3.0]\nHEAD = COEFFICIENTS.pop(0)\n"
        "def oracle(x):\n    return sum(COEFFICIENTS) * x\n",
        "pop(0)",
        "pop(1)",
    ),
    "range": (
        "TERMS = range(2)\ndef oracle(x):\n    return sum(x**k for k in TERMS)\n",
        "range(2)",
        "range(3)",
    ),
}


@pytest.mark.parametrize("edit", sorted(_EDITS))
def test_an_edit_to_what_the_oracle_reads_changes_its_fingerprint(fingerprint_of, edit):
    """A helper, a constant behind it, a default, a decorator's wrapper, a class
    attribute, which object a repeat names, a numpy scalar, a structured dtype,
    one of two lambdas on a line and a range each change the oracle's answer."""
    source, old, new = _EDITS[edit]
    assert old in source
    before = fingerprint_of(source)
    assert fingerprint_of(source) == before
    assert fingerprint_of(source.replace(old, new)) != before


def test_a_value_computed_at_import_does_not_enter_the_fingerprint(fingerprint_of, monkeypatch):
    """``leggauss`` is an eigenvalue solve: another platform's LAPACK rounds its
    nodes differently, and a fingerprint must not see that."""
    source = (
        "import numpy as np\nNODES, WEIGHTS = np.polynomial.legendre.leggauss(3)\n"
        "def oracle(x):\n    return sum(w * x ** t for t, w in zip(NODES, WEIGHTS))\n"
    )
    before = fingerprint_of(source)
    module = sys.modules["_oracle_store_case"]
    monkeypatch.setattr(module, "NODES", np.nextafter(module.NODES, np.inf))
    assert _fingerprint(module.oracle) == before


def test_no_store_fingerprints_the_platforms_legendre_rounding(monkeypatch):
    """The LSS bounds store reaches ``_response_tails``' Gauss-Legendre nodes."""
    from superglm.distributional.kernels import _response_tails
    from tests import test_lss_response_bounds

    store = test_lss_response_bounds.ROW_PARTIALS
    rebuilt = OracleStore(store.name, store.oracle, depends=store._depends, module=store.module)
    assert rebuilt.fingerprint == store.fingerprint
    for name in ("_LEGENDRE_NODES", "_LEGENDRE_WEIGHTS"):
        monkeypatch.setattr(
            _response_tails, name, np.nextafter(getattr(_response_tails, name), np.inf)
        )
    nudged = OracleStore(store.name, store.oracle, depends=store._depends, module=store.module)
    assert nudged.fingerprint == store.fingerprint


def test_every_fingerprint_names_the_oracle_libraries_versions(monkeypatch):
    before = _fingerprint(_oracle)
    monkeypatch.setattr(
        _oracle_store, "_ORACLE_LIBRARIES", (("mpmath", "9.9"), ("sympy", "1.14.0"))
    )
    assert _fingerprint(_oracle) != before


def test_dependencies_passed_as_a_generator_enter_the_fingerprint():
    assert _fingerprint(_oracle, (v for v in (1.0,))) != _fingerprint(_oracle, (v for v in (2.0,)))


def test_the_window_store_neither_hashes_nor_looks_up_the_platforms_legendre_rounding(monkeypatch):
    """The narrow-window store's node fractions must not depend on ``leggauss``:
    neither its fingerprint, nor the fractions the engine asks for."""
    from superglm.distributional.kernels import _response_tails
    from tests import test_response_window_jets as jets

    store = jets._NARROW_INPUTS
    nodes = np.polynomial.legendre.leggauss(3)[0]
    assert jets._LEGENDRE_NODES.tolist() == nodes.tolist()  # this platform rounds them exactly
    monkeypatch.setattr(_response_tails, "_LEGENDRE_NODES", np.nextafter(nodes, np.inf))
    rebuilt = OracleStore(store.name, store.oracle, depends=store._depends, module=store.module)
    assert rebuilt.fingerprint == store.fingerprint
    case = jets._NARROW_CASES[0]
    jets.test_quadrature_jets_match_the_oracle_to_order_four(*case[:1], *case[2:])


def test_a_constant_imported_under_a_guard_is_followed_to_its_module(fingerprint_of):
    source = (
        "try:\n    from _oracle_store_helper import SCALE\nexcept ImportError:\n"
        "    SCALE = 1.0\ndef oracle(x):\n    return x * SCALE\n"
    )
    before = fingerprint_of(source, helper="SCALE = 1.0\n")
    assert fingerprint_of(source, helper="SCALE = 1.0\n") == before
    assert fingerprint_of(source, helper="SCALE = 2.0\n") != before


class _OneSlot:
    __slots__ = "scale"

    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, x):
        return self.scale * x


def test_a_receiver_with_one_string_slot_enters_with_its_state():
    assert _fingerprint(_OneSlot(1.0).evaluate) != _fingerprint(_OneSlot(2.0).evaluate)
