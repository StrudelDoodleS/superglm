"""Categorical spline row extraction preserves rows with bounded repeated work."""

from __future__ import annotations

import copy
import pickle
import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix._group_matrix_core import SplineCategoricalGroupMatrix
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)

_KINDS = (
    SplineCategoricalGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)


class _CustomSparse(SplineCategoricalGroupMatrix):
    pass


class _CustomDiscrete(DiscretizedSplineCategoricalGroupMatrix):
    pass


class _CustomSupportCompressed(SupportCompressedSplineCategoricalGroupMatrix):
    pass


def _group(kind, n=12, row_idx=None):
    support = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    transform = np.array([[1.0, 2.0], [2.0, -1.0]])
    bins = np.arange(n, dtype=np.intp) % len(support)
    if row_idx is None:
        row_idx = np.arange(n - 2, -1, -2, dtype=np.intp)
    if issubclass(kind, SplineCategoricalGroupMatrix):
        group = kind(sp.csr_matrix(support[bins]), transform, row_idx)
    else:
        group = kind(support, transform, bins, row_idx)
    expected = np.zeros((n, 2))
    expected[row_idx] = (support @ transform)[bins[row_idx]]
    return group, expected


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "rows",
    [
        np.array([10, 1, 0, 10, 4, 2, 1]),
        np.array([], dtype=np.intp),
        np.array([1, 3, 5]),
        np.array([True, False, True, False] * 3),
    ],
    ids=["reordered-repeated", "empty", "absent-level", "boolean-mask"],
)
def test_row_subset_reconstructs_selected_matrix(kind, rows):
    group, expected = _group(kind)
    subset = group.row_subset(rows)

    assert type(subset) is kind
    np.testing.assert_array_equal(subset.toarray(), expected[rows])
    repeated = np.array([subset.shape[0] - 1, 0, 0]) if subset.shape[0] else rows
    np.testing.assert_array_equal(subset.row_subset(repeated).toarray(), expected[rows][repeated])


@pytest.mark.parametrize("kind", _KINDS)
def test_row_lookup_owns_immutable_category_indices(kind):
    rows = np.array([10, 4, 0, 8, 2, 6], dtype=np.intp)
    group, expected = _group(kind, row_idx=rows)
    group.row_subset(np.array([10, 0]))  # Also exercise a populated lookup.
    rows[:] = 1

    assert rows.flags.writeable
    np.testing.assert_array_equal(group.toarray(), expected)
    np.testing.assert_array_equal(group.row_subset(np.arange(12)).toarray(), expected)
    with pytest.raises(ValueError, match="read-only"):
        group.row_idx[0] = 1


@pytest.mark.parametrize("kind", _KINDS)
def test_row_subset_preserves_empty_category(kind):
    group, expected = _group(kind, row_idx=np.array([], dtype=np.intp))
    rows = np.array([10, 0, 10])
    np.testing.assert_array_equal(group.row_subset(rows).toarray(), expected[rows])


@pytest.mark.parametrize("kind", _KINDS)
def test_legacy_pickle_without_lookup_reconstructs_subset(kind):
    group, expected = _group(kind)
    # These slots did not exist in v0.31.0 learned matrices. Removing them
    # reproduces the state that the ordinary slots pickle protocol saved.
    for name in ("_row_order", "_sorted_rows"):
        if hasattr(group, name):
            delattr(group, name)
    restored = pickle.loads(pickle.dumps(group))
    rows = np.array([10, 0, 1, 10])
    np.testing.assert_array_equal(restored.row_subset(rows).toarray(), expected[rows])


@pytest.mark.parametrize("kind", _KINDS)
def test_pickled_lookup_restores_immutable_category_indices(kind):
    group, expected = _group(kind)
    rows = np.array([10, 0, 1, 10])
    group.row_subset(rows)
    restored = pickle.loads(pickle.dumps(group))

    with pytest.raises(ValueError, match="read-only"):
        restored.row_idx[0] = 1
    np.testing.assert_array_equal(restored.row_subset(rows).toarray(), expected[rows])


@pytest.mark.parametrize("kind", [_CustomSparse, _CustomDiscrete, _CustomSupportCompressed])
@pytest.mark.parametrize("method", ["pickle", "deepcopy"])
def test_restored_subclass_preserves_instance_metadata(kind, method):
    group, expected = _group(kind)
    group.label = {"note": "important metadata", "tags": ["category"]}
    group.row_subset(np.array([10, 0]))

    restored = pickle.loads(pickle.dumps(group)) if method == "pickle" else copy.deepcopy(group)

    assert restored.label == group.label
    assert restored.label is not group.label
    assert not restored.row_idx.flags.writeable
    np.testing.assert_array_equal(restored.toarray(), expected)


@pytest.mark.parametrize("kind", _KINDS)
def test_repeated_small_subsets_do_not_reprocess_parent_rows(kind, monkeypatch):
    group, _ = _group(kind, n=32768)
    parent_work = []
    original_argsort = np.argsort
    original_sort = np.sort
    original_isin = np.isin

    def counted_sort(values, *args, **kwargs):
        if np.size(values) >= group.row_idx.size:
            parent_work.append(("sort", np.size(values)))
        return original_sort(values, *args, **kwargs)

    def counted_argsort(values, *args, **kwargs):
        if np.size(values) >= group.row_idx.size:
            parent_work.append(("argsort", np.size(values)))
        return original_argsort(values, *args, **kwargs)

    def counted_isin(elements, test_elements, *args, **kwargs):
        if np.size(test_elements) >= group.row_idx.size:
            parent_work.append(("membership", np.size(test_elements)))
        return original_isin(elements, test_elements, *args, **kwargs)

    monkeypatch.setattr(np, "argsort", counted_argsort)
    monkeypatch.setattr(np, "sort", counted_sort)
    monkeypatch.setattr(np, "isin", counted_isin)
    group.row_subset(np.arange(16))
    parent_work.clear()
    for start in (16, 32, 48, 64):
        group.row_subset(np.arange(start, start + 16))

    assert parent_work == [], f"Repeated parent-sized work: {parent_work}"


@pytest.mark.parametrize("kind", _KINDS)
def test_warm_small_subset_allocation_does_not_scale_with_parent(kind):
    group, _ = _group(kind, n=262144)
    rows = np.arange(16, dtype=np.intp)
    group.row_subset(rows)

    tracemalloc.start()
    try:
        for _ in range(4):
            group.row_subset(rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # A 16-row, two-column subset is tiny; even one full level index costs
    # 1 MiB. Leave ample room for Python/SciPy objects and chunk-local arrays.
    assert peak < 128 * 1024, f"Tiny subsets allocated {peak} bytes"
