"""Row placement and allocation checks for ordered bases with special rows."""

from dataclasses import fields

import numpy as np
import pytest
import scipy.sparse as sp

from superglm import OrderedCategorical
from superglm.types import GroupInfo


@pytest.mark.parametrize("storage", [np.asarray, sp.csr_matrix, sp.csc_matrix, sp.coo_matrix])
@pytest.mark.parametrize("mask", [[False, True, False, True, True, False], [True, True, True]])
def test_expansion_preserves_each_basis_row_and_group_metadata(storage, mask):
    values = np.array([[1, 0, 2], [0, 0, 0], [0, -3, 4]], dtype=np.float32)
    compact = storage(values)
    projection = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
    penalty = np.eye(2)
    info = GroupInfo(
        columns=compact,
        n_cols=2,
        projection=projection,
        penalty_matrix=penalty,
        reparametrize=True,
        supports_row_compression=True,
        subgroup_name="spline",
        penalty_components=[("wiggle", penalty)],
    )
    ordered_mask = np.array(mask, dtype=bool)

    expanded = OrderedCategorical._expand_rows(info, ordered_mask)

    expected = np.zeros((len(mask), 3))
    expected[ordered_mask] = values
    assert sp.isspmatrix_csr(expanded.columns)
    assert expanded.columns.dtype == np.float64
    assert expanded.columns.has_canonical_format
    np.testing.assert_array_equal(expanded.columns.toarray(), expected)
    assert info.columns is compact
    for field in fields(info):
        if field.name != "columns":
            assert getattr(expanded, field.name) is getattr(info, field.name)


@pytest.mark.parametrize("storage", [np.asarray, sp.csr_matrix])
@pytest.mark.parametrize("mask, width", [([], 3), ([False] * 4, 3), ([True, False, True], 0)])
def test_expansion_handles_empty_rows_and_columns(storage, mask, width):
    ordered_mask = np.array(mask, dtype=bool)
    info = GroupInfo(columns=storage(np.zeros((sum(mask), width))), n_cols=width)

    expanded = OrderedCategorical._expand_rows(info, ordered_mask).columns

    assert sp.isspmatrix_csr(expanded)
    assert expanded.shape == (len(mask), width)
    assert expanded.nnz == 0


def test_expansion_canonicalizes_sparse_values_without_mutating_input():
    # Duplicate entries cancel exactly; the middle compact row is empty.
    compact = sp.csr_matrix(
        (
            np.array([2, 3, -2, 0, 4, 7, -1, 1], dtype=np.float64),
            np.array([2, 0, 2, 1, 1, 2, 0, 0]),
            np.array([0, 4, 4, 5, 8]),
        ),
        shape=(4, 3),
    )
    original = [array.copy() for array in (compact.data, compact.indices, compact.indptr)]
    original_flags = compact.has_sorted_indices, compact.has_canonical_format
    mask = np.array([False, True, False, True, True, False, True, False])

    expanded = OrderedCategorical._expand_rows(GroupInfo(compact, 3), mask).columns

    expected = np.zeros((8, 3))
    expected[1, 0], expected[4, 1], expected[6, 2] = 3, 4, 7
    np.testing.assert_array_equal(expanded.toarray(), expected)
    assert expanded.has_canonical_format
    assert expanded.nnz == 3
    for actual, before in zip((compact.data, compact.indices, compact.indptr), original):
        np.testing.assert_array_equal(actual, before)
    assert (compact.has_sorted_indices, compact.has_canonical_format) == original_flags


@pytest.mark.parametrize("storage", [np.asarray, sp.csr_matrix])
def test_expansion_avoids_lil_allocation_and_sparse_densification(monkeypatch, storage):
    """Guard allocation separately from the numerical row-placement tests."""
    compact = storage([[1.0, 0.0], [0.0, 2.0]])
    info = GroupInfo(compact, 2)

    def reject_allocation(*args, **kwargs):
        pytest.fail("Row expansion must avoid LIL allocation and sparse densification")

    with monkeypatch.context() as allocation_guard:
        allocation_guard.setattr(sp.lil_matrix, "__init__", reject_allocation)
        allocation_guard.setattr(sp.csr_matrix, "toarray", reject_allocation)
        expanded = OrderedCategorical._expand_rows(info, np.array([True, False, True]))

    assert expanded.columns.shape == (3, 2)
