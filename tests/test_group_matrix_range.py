"""Bounded group evaluation preserves the stored subset algebra and live state."""

from __future__ import annotations

import copy
import pickle
import tracemalloc
import warnings
import weakref

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix import _group_matrix_range as ranges
from superglm._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    RandomEffectGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)

_CATEGORIES = (
    SplineCategoricalGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)
_ORDINARY = (
    DenseGroupMatrix,
    CategoricalGroupMatrix,
    RandomEffectGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
_KINDS = _ORDINARY + _CATEGORIES


def _group(kind, n=37, rows=None):
    support = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    transform = np.array([[1.0, 2.0], [2.0, -1.0]])
    bins = np.arange(n, dtype=np.intp) % len(support)
    if kind is DenseGroupMatrix:
        return kind(support[bins] @ transform)
    if kind in (CategoricalGroupMatrix, RandomEffectGroupMatrix):
        return kind(bins, 3)
    if kind in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix):
        return kind(support, transform, bins)
    if rows is None:
        rows = np.arange(n - 2, -1, -2, dtype=np.intp)
    if kind is SplineCategoricalGroupMatrix:
        return kind(sp.csr_matrix(support[bins]), transform, rows)
    return kind(support, transform, bins, rows)


def _assert_close(actual, expected):
    # These well-conditioned fixtures have inner dimensions at most three;
    # cover two matrix products and their rounding with the standard gamma_d.
    inner_dimension = 3
    epsilon = np.finfo(float).eps
    gamma = inner_dimension * epsilon / (1 - inner_dimension * epsilon)
    scale = max(1.0, float(np.max(np.abs(expected), initial=0)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4 * gamma * scale)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("start,stop", [(0, 37), (3, 16), (0, 0), (37, 37), (2, 3)])
def test_range_prediction_and_geometry_preserve_subset_algebra(kind, start, stop):
    group = _group(kind)
    beta = np.arange(1, group.shape[1] + 1, dtype=np.float64) / 3
    expected = group.row_subset(np.arange(start, stop))
    helpers = ranges

    values = helpers.group_range_matvec(group, start, stop, beta)
    subset = helpers.group_row_range(group, start, stop)

    assert values is not None
    assert subset is not None
    assert type(subset) is kind
    _assert_close(values, expected.matvec(beta))
    _assert_close(subset.toarray(), expected.toarray())
    assert values.shape == (stop - start,)


@pytest.mark.parametrize("kind", _KINDS)
def test_range_prediction_does_not_construct_a_subset(kind, monkeypatch):
    group = _group(kind)
    beta = np.ones(group.shape[1])
    expected = group.row_subset(np.arange(3, 19)).matvec(beta)

    def refused(*args, **kwargs):
        pytest.fail("Range prediction performed generic row extraction or construction")

    monkeypatch.setattr(kind, "row_subset", refused)
    monkeypatch.setattr(kind, "__init__", refused)
    _assert_close(ranges.group_range_matvec(group, 3, 19, beta), expected)


@pytest.mark.parametrize("kind", _CATEGORIES)
def test_warm_category_range_search_has_only_two_bounds(kind, monkeypatch):
    group = _group(kind, n=32768)
    group.row_subset(np.arange(16))
    searches = []
    searchsorted = np.searchsorted

    def record(values, queries, *args, **kwargs):
        searches.append(np.size(queries))
        return searchsorted(values, queries, *args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", record)
    assert ranges.group_range_matvec(group, 17, 1041, np.ones(2)) is not None
    assert sum(searches) == 2, f"Range lookup searched {sum(searches)} rows"


@pytest.mark.parametrize("kind", _CATEGORIES)
@pytest.mark.parametrize("rows", [np.array([], dtype=np.intp), np.array([2, 2, 8, 14])])
def test_empty_levels_work_and_duplicate_lookup_declines_range(kind, rows):
    group = _group(kind, rows=rows)
    values = ranges.group_range_matvec(group, 0, 16, np.ones(2))
    if rows.size:
        assert values is None
        assert ranges.group_row_range(group, 0, 16) is None
    else:
        np.testing.assert_array_equal(values, np.zeros(16))


@pytest.mark.parametrize("kind", _CATEGORIES)
@pytest.mark.parametrize("mutation", ["replace", "setstate", "reshape", "custom"])
@pytest.mark.parametrize("name", ["_sorted_rows", "_row_order"])
def test_changed_lookup_storage_or_metadata_declines_range(kind, mutation, name):
    group = _group(kind)
    group.row_subset(np.arange(16))
    if not hasattr(group, name):
        pytest.skip("The exact spline category lookup needs no permutation")
    original = getattr(group, name)
    assert ranges.group_range_matvec(group, 3, 19, np.ones(2)) is not None
    if mutation == "replace":
        changed = original.copy()
        changed[:2] = changed[1::-1]
        setattr(group, name, changed)
    elif mutation == "setstate":
        changed = original.copy()
        changed[:2] = changed[1::-1]
        original.__setstate__(changed.__reduce__()[2])
    elif mutation == "reshape":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            original.shape = (original.size, 1)
    else:

        class CustomArray(np.ndarray):
            pass

        setattr(group, name, original.view(CustomArray))
    assert ranges.group_range_matvec(group, 3, 19, np.ones(2)) is None
    assert ranges.group_row_range(group, 3, 19) is None


@pytest.mark.parametrize("kind", _CATEGORIES)
def test_lookup_cannot_be_made_writeable_and_replacement_releases_old_storage(kind):
    group = _group(kind)
    group.row_subset(np.arange(16))
    assert ranges.group_range_matvec(group, 3, 19, np.ones(2)) is not None
    refs = []
    for name in ("_sorted_rows", "_row_order"):
        if not hasattr(group, name):
            continue
        array = getattr(group, name)
        with pytest.raises(ValueError):
            array.flags.writeable = True
        refs.append(weakref.ref(array))
        setattr(group, name, array.copy())
    del array
    assert all(ref() is None for ref in refs)
    assert ranges.group_range_matvec(group, 3, 19, np.ones(2)) is None


@pytest.mark.parametrize("kind", _CATEGORIES)
@pytest.mark.parametrize("method", ["pickle", "deepcopy"])
def test_serialization_rebuilds_range_certificate_with_lookup(kind, method):
    group = _group(kind)
    before = ranges.group_range_matvec(group, 3, 19, np.ones(2))
    restored = pickle.loads(pickle.dumps(group)) if method == "pickle" else copy.deepcopy(group)
    assert restored._sorted_rows is None
    assert restored._row_lookup_certificate is None
    _assert_close(ranges.group_range_matvec(restored, 3, 19, np.ones(2)), before)


@pytest.mark.parametrize("kind", _KINDS)
def test_subclasses_decline_range_dispatch(kind):
    class CustomGroup(kind):
        pass

    group = _group(kind)
    # Reuse the complete stored state without invoking a parent constructor.
    custom = CustomGroup.__new__(CustomGroup)
    for cls in kind.__mro__:
        for name in getattr(cls, "__slots__", ()):
            if hasattr(group, name):
                setattr(custom, name, getattr(group, name))
    assert ranges.group_range_matvec(custom, 3, 19, np.ones(group.shape[1])) is None
    assert ranges.group_row_range(custom, 3, 19) is None


@pytest.mark.parametrize("kind", _KINDS)
def test_custom_numeric_array_declines_before_slicing(kind):
    group = _group(kind)

    class CustomArray(np.ndarray):
        def __getitem__(self, item):
            pytest.fail("Range path sliced a custom array before refusing it")

    name = (
        "M"
        if kind is DenseGroupMatrix
        else "codes"
        if kind in (CategoricalGroupMatrix, RandomEffectGroupMatrix)
        else "R_inv"
    )
    setattr(group, name, getattr(group, name).view(CustomArray))
    assert ranges.group_range_matvec(group, 3, 19, np.ones(group.shape[1])) is None
    assert ranges.group_row_range(group, 3, 19) is None


@pytest.mark.parametrize("kind", _CATEGORIES)
def test_live_source_basis_map_and_transform_changes_match_current_subset(kind):
    group = _group(kind)
    helpers = ranges
    assert helpers.group_range_matvec(group, 3, 19, np.ones(2)) is not None
    group.R_inv[0, 0] += 0.5
    if kind is SplineCategoricalGroupMatrix:
        # Generic extraction reads B; B_level is intentionally not the oracle.
        group.B.data[:] *= 1.25
        group.B_level.data[:] *= 4
    else:
        group.B_unique[0, 0] += 0.75
        group.bin_idx_level[:] = (group.bin_idx_level + 1) % group.n_bins
    # Existing cached lookup remains the generic semantics on row_idx mutation.
    group.row_idx = group.row_idx[::-1].copy()
    expected = group.row_subset(np.arange(3, 19))
    _assert_close(helpers.group_range_matvec(group, 3, 19, np.ones(2)), expected.matvec(np.ones(2)))
    _assert_close(helpers.group_row_range(group, 3, 19).toarray(), expected.toarray())


@pytest.mark.parametrize("kind", [DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix])
def test_support_prediction_preserves_matvec_association_and_live_maps(kind):
    # (B @ R) @ beta loses the unit term, whereas B @ (R @ beta) is exactly 1.
    group = kind(np.array([[1e16, 1.0]]), np.array([[1.0, 1.0], [1.0, 0.0]]), np.array([0, 0]))
    beta = np.array([1.0, -1.0])
    np.testing.assert_array_equal(ranges.group_range_matvec(group, 0, 2, beta), [1.0, 1.0])
    group.B_unique = np.array([[1.0, 2.0], [3.0, 5.0]])
    group.R_inv[:] = np.eye(2)
    group.bin_idx[:] = [1, 0]
    np.testing.assert_array_equal(ranges.group_range_matvec(group, 0, 2, beta), [-2.0, -1.0])


@pytest.mark.parametrize("kind", _CATEGORIES)
def test_warm_range_allocation_stays_bounded_by_chunk(kind):
    group = _group(kind, n=262144)
    helper = ranges.group_range_matvec
    helper(group, 0, 16, np.ones(2))
    tracemalloc.start()
    try:
        for _ in range(4):
            result = helper(group, 16, 32, np.ones(2))
            assert result.shape == (16,)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 128 * 1024, f"Tiny range evaluation allocated {peak} bytes"


@pytest.mark.parametrize("kind", _CATEGORIES)
@pytest.mark.parametrize("mutation", ["replace", "setstate"])
def test_replacing_lookup_does_not_retain_old_source_sized_buffers(kind, mutation):
    group = _group(kind, n=262144)
    helpers = ranges
    tracemalloc.start()
    try:
        helpers.group_range_matvec(group, 0, 16, np.ones(2))
        original, _ = tracemalloc.get_traced_memory()
        for name in ("_sorted_rows", "_row_order"):
            if hasattr(group, name):
                if mutation == "replace":
                    setattr(group, name, getattr(group, name).copy())
                else:
                    getattr(group, name).__setstate__(getattr(group, name).copy().__reduce__()[2])
        assert helpers.group_range_matvec(group, 0, 16, np.ones(2)) is None
        current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert current < original + 128 * 1024, f"Replaced lookup retained {current - original} bytes"


@pytest.mark.parametrize("kind", _CATEGORIES)
def test_generic_negative_and_repeated_rows_keep_existing_membership_semantics(kind):
    group = _group(kind)
    ranges.group_range_matvec(group, 0, 16, np.ones(2))
    values = group.row_subset(np.array([-1, 3, 3, 1, 2])).matvec(np.ones(2))
    np.testing.assert_array_equal(values, [0.0, 3.0, 3.0, 2.0, 0.0])


@pytest.mark.parametrize("kind", _ORDINARY + (SplineCategoricalGroupMatrix,))
def test_shortened_source_declines_instead_of_silently_truncating_range(kind):
    group = _group(kind)
    if kind is DenseGroupMatrix:
        group.M = group.M[:3]
    elif kind in (CategoricalGroupMatrix, RandomEffectGroupMatrix):
        group.codes = group.codes[:3]
    elif kind is SplineCategoricalGroupMatrix:
        group.B = group.B[:3]
    else:
        group.bin_idx = group.bin_idx[:3]
    with pytest.raises(IndexError):
        group.row_subset(np.arange(3, 19))
    assert ranges.group_range_matvec(group, 3, 19, np.ones(group.shape[1])) is None
    assert ranges.group_row_range(group, 3, 19) is None


@pytest.mark.parametrize(
    "kind", [DenseGroupMatrix, DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix]
)
@pytest.mark.parametrize("direction", ["parent", "child"])
def test_range_geometry_retains_generic_ownership_of_selected_rows(kind, direction):
    group = _group(kind)
    expected = group.row_subset(np.arange(3, 19))
    subset = ranges.group_row_range(group, 3, 19)
    source = group if direction == "parent" else subset
    untouched = subset if direction == "parent" else group
    expected = expected.toarray() if direction == "parent" else group.toarray().copy()
    if kind is DenseGroupMatrix:
        source.M[:] *= 2
    else:
        source.bin_idx[:] = (source.bin_idx + 1) % source.n_bins
    _assert_close(untouched.toarray(), expected)


@pytest.mark.parametrize(
    "kind", [DenseGroupMatrix, DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix]
)
def test_range_geometry_does_not_retain_the_full_parent_row_array(kind):
    group = _group(kind, n=32768)
    source = group.M if kind is DenseGroupMatrix else group.bin_idx
    reference = weakref.ref(source)
    subset = ranges.group_row_range(group, 3, 19)
    del source, group
    assert reference() is None
    assert subset.toarray().shape == (16, 2)
