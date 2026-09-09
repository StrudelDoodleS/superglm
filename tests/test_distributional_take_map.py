"""Ordered chunk selection avoids sorting while retaining positional provenance."""

import numpy as np
import pytest

from superglm.distributional.weights import (
    LikelihoodWeightError,
    WeightContract,
    _take_map,
    resolve_likelihood_weights,
)


@pytest.mark.parametrize(
    "indices,length",
    [
        (np.arange(100, 6800, dtype=np.intp), 7000),
        (np.array([2], dtype=np.uint32), 3),
        (np.array([0, 4, 9], dtype=np.intp), 10),
        (np.array([0, np.iinfo(np.intp).max - 1]), np.iinfo(np.intp).max),
    ],
)
def test_ordered_take_map_does_not_sort(indices, length, monkeypatch):
    def forbid_unique(*args, **kwargs):
        raise AssertionError("ordered chunk selection sorted its row indices")

    monkeypatch.setattr(np, "unique", forbid_unique)
    selected = _take_map(indices, length)

    np.testing.assert_array_equal(selected, indices)
    assert selected.dtype == np.dtype(np.intp)
    assert not selected.flags.writeable
    assert not np.shares_memory(selected, indices)


@pytest.mark.parametrize("indices", [[4, 0, 2], [3, 2, 1]])
def test_unordered_take_map_preserves_requested_order(indices):
    selected = _take_map(np.array(indices), 5)
    np.testing.assert_array_equal(selected, indices)
    assert not selected.flags.writeable


@pytest.mark.parametrize("indices", [[1, 1, 2], [2, 0, 2], [3, 3]])
def test_take_map_rejects_duplicate_rows(indices):
    with pytest.raises(LikelihoodWeightError, match="duplicates"):
        _take_map(np.array(indices), 5)


@pytest.mark.parametrize(
    "indices,message",
    [
        (np.array([[0, 1]]), "one-dimensional integer"),
        (np.array([0.0, 1.0]), "one-dimensional integer"),
        (np.array([False, True]), "one-dimensional integer"),
        (np.array([], dtype=np.intp), "at least one row"),
        (np.array([-1, 1]), "out of range"),
        (np.array([0, 5]), "out of range"),
        (np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64), "out of range"),
    ],
)
def test_take_map_retains_input_boundary_checks(indices, message):
    with pytest.raises(LikelihoodWeightError, match=message):
        _take_map(indices, 5)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_ordered_take_preserves_nested_weight_provenance(semantics):
    root = resolve_likelihood_weights(
        np.array([0, 1, 2, 0, 3, 4, 5]),
        n_observations=7,
        contract=WeightContract(semantics=semantics),
    )
    direct = root.take(np.array([1, 3, 4]))
    nested = root.take(np.array([4, 1, 3])).take(np.array([1, 2, 0]))

    assert direct.provenance is root.provenance
    assert nested.provenance is root.provenance
    assert direct.digest == nested.digest
    for field in ("values", "geometry_values", "root_take_map", "input_positions"):
        np.testing.assert_array_equal(getattr(direct, field), getattr(nested, field))
        assert not getattr(direct, field).flags.writeable
    np.testing.assert_array_equal(direct.values, [2, 4, 5])
    np.testing.assert_array_equal(direct.input_positions, [2, 5, 6])
