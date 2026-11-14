"""The oracle store answers exactly what its oracle computed, and nothing else."""

from __future__ import annotations

import functools
import json

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


def test_the_fingerprint_follows_every_helper_and_constant_the_oracle_reaches(monkeypatch):
    before = _fingerprint(_outer)
    monkeypatch.setattr(f"{__name__}._SCALE", 1.000000001)
    assert _fingerprint(_outer) != before
    monkeypatch.undo()
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


def _scaled(function):
    @functools.wraps(function)
    def wrapper(x):
        return function(x) * _SCALE

    return wrapper


@_scaled
def _decorated(x):
    return mp.mpf(x)


class _Law:
    scale = _SCALE

    @staticmethod
    def evaluate(x):
        return x * _Law.scale


_A, _B = [1.0], [2.0]
_CONFIGURATION = (_A, _B, _A)
_ENABLED = np.bool_(True)
_TABLE = np.array([(1.0, 2.0)], dtype=[("x", "f8"), ("y", "f8")])


def _reads_a_repeat(x):
    return _CONFIGURATION[2][0] * x


def _reads_a_flag(x):
    return x if _ENABLED else -x


def _reads_a_field(x):
    return _TABLE["x"][0] * x


@pytest.mark.parametrize(
    ("oracle", "name", "changed"),
    [
        (_decorated, "_SCALE", 2.0),
        (_Law.evaluate, "_Law.scale", 2.0),
        (_reads_a_repeat, "_CONFIGURATION", (_A, _B, _B)),
        (_reads_a_flag, "_ENABLED", np.bool_(False)),
        (_reads_a_field, "_TABLE", np.array([(1.0, 2.0)], dtype=[("y", "f8"), ("x", "f8")])),
    ],
    ids=[
        "decorator-wrapper",
        "class-attribute",
        "repeated-object",
        "numpy-bool",
        "structured-dtype",
    ],
)
def test_the_fingerprint_sees_every_kind_of_state_an_oracle_reads(
    monkeypatch, oracle, name, changed
):
    """A decorator's wrapper, a class attribute, which object a repeat names, a numpy
    scalar's value and a structured array's fields each change the oracle's answer."""
    before = _fingerprint(oracle)
    monkeypatch.setattr(f"{__name__}.{name}", changed)
    assert _fingerprint(oracle) != before
    monkeypatch.undo()
    assert _fingerprint(oracle) == before


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


_IDENTITY, _DOUBLED = lambda x: x, lambda x: x if x <= 2 else 2 * x
_HELPER = _IDENTITY
_TERMS = range(2)
_OBJECT_TABLE = np.array([(1.0, 2.0)], dtype=[("x", object), ("y", object)])


def _calls_the_helper(x):
    return _HELPER(x)


def _sums_the_terms(x):
    return sum(x**k for k in _TERMS)


def _reads_an_object_field(x):
    return _OBJECT_TABLE["x"][0] * x


@pytest.mark.parametrize(
    ("oracle", "name", "changed"),
    [
        (_calls_the_helper, "_HELPER", _DOUBLED),
        (_sums_the_terms, "_TERMS", range(3)),
        (
            _reads_an_object_field,
            "_OBJECT_TABLE",
            np.array([(1.0, 2.0)], dtype=[("y", object), ("x", object)]),
        ),
    ],
    ids=["lambda-sharing-a-statement", "range", "object-structured-dtype"],
)
def test_the_fingerprint_tells_apart_values_that_look_alike(monkeypatch, oracle, name, changed):
    before = _fingerprint(oracle)
    monkeypatch.setattr(f"{__name__}.{name}", changed)
    assert _fingerprint(oracle) != before


def test_a_bound_method_enters_with_its_receiver():
    assert _fingerprint(_Scaled(1.0).evaluate) != _fingerprint(_Scaled(2.0).evaluate)
    assert _fingerprint(_Scaled(1.0).evaluate) == _fingerprint(_Scaled(1.0).evaluate)


def test_a_builtin_constant_the_fingerprint_cannot_read_is_refused():
    with pytest.raises(TypeError, match="cannot hash"):
        _fingerprint(_oracle, (bytearray(b"x"),))
