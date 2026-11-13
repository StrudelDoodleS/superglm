"""The oracle store answers exactly what its oracle computed, and nothing else."""

from __future__ import annotations

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
