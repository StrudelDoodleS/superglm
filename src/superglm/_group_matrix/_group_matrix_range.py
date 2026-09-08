"""Bounded contiguous group operations for exact built-in representations.

``None`` declines dispatch: callers retain their ordinary ``row_subset`` path.
No transformed basis or predictor values survive a call. Sparse category
extraction deliberately reads ``B``, matching ``row_subset``, not ``B_level``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    RandomEffectGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from ._group_matrix_discretized import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from ._row_lookup import certified_row_bounds

_CATEGORICAL = (CategoricalGroupMatrix, RandomEffectGroupMatrix)
_SUPPORT = (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix)
_SPLINE_CATEGORY = (
    SplineCategoricalGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)
_METADATA = ("omega", "projection", "omega_components", "component_types")
_CATEGORY_METADATA = (
    *_METADATA,
    "lambda_policies",
    "spline_cat_level",
    "spline_cat_feature",
)


def _plain_array(array, dtype, ndim):
    return type(array) is np.ndarray and array.dtype == dtype and array.ndim == ndim


def _supported(group, start, stop):
    kind = type(group)
    if kind not in (DenseGroupMatrix, *_CATEGORICAL, *_SUPPORT, *_SPLINE_CATEGORY):
        return False
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= group.shape[0]:
        return False
    if kind is DenseGroupMatrix:
        return _plain_array(group.M, np.float64, 2) and group.M.shape[0] == group.shape[0]
    if kind in _CATEGORICAL:
        return _plain_array(group.codes, np.intp, 1) and group.codes.size == group.shape[0]
    if not _plain_array(group.R_inv, np.float64, 2):
        return False
    if kind is SplineCategoricalGroupMatrix:
        if (
            type(group.B) is not sp.csr_matrix
            or not _plain_array(group.B.data, np.float64, 1)
            or group.B.shape[0] != group.shape[0]
        ):
            return False
        if any(
            type(array) is not np.ndarray or array.dtype.kind != "i" or array.ndim != 1
            for array in (group.B.indices, group.B.indptr)
        ):
            return False
    elif not _plain_array(group.B_unique, np.float64, 2):
        return False
    if kind in _SUPPORT:
        return _plain_array(group.bin_idx, np.intp, 1) and group.bin_idx.size == group.shape[0]
    return _plain_array(group.row_idx, np.intp, 1) and (
        kind is SplineCategoricalGroupMatrix or _plain_array(group.bin_idx_level, np.intp, 1)
    )


def _category_rows(group, start, stop):
    """Return local rows and live level bins, without searching each chunk row."""
    exact = type(group) is SplineCategoricalGroupMatrix
    if not group.row_idx.size or start == stop:
        empty = np.empty(0, dtype=np.intp)
        return empty, None if exact else empty
    bounds = certified_row_bounds(group, start, stop, with_order=not exact)
    if bounds is None:
        return None
    lo, hi = bounds
    local_rows = group._sorted_rows[lo:hi] - start
    bins = None if exact else group.bin_idx_level[group._row_order[lo:hi]]
    return local_rows, bins


def group_range_matvec(group, start: int, stop: int, beta):
    """Evaluate a built-in group's current stored algebra on ``[start, stop)``."""
    if not _plain_array(beta, np.float64, 1) or not _supported(group, start, stop):
        return None
    kind = type(group)
    if kind is DenseGroupMatrix:
        return group.M[start:stop] @ beta
    if kind in _CATEGORICAL:
        codes = group.codes[start:stop]
        if kind is RandomEffectGroupMatrix and np.any((codes < 0) | (codes >= group.n_levels)):
            return None
        extended = np.empty(group.n_levels + 1)
        extended[: group.n_levels] = beta
        extended[group.n_levels] = 0.0
        return extended[codes]
    if kind in _SUPPORT:
        values = group.B_unique @ (group.R_inv @ beta)
        return values[group.bin_idx[start:stop]]
    rows = _category_rows(group, start, stop)
    if rows is None:
        return None
    local_rows, bins = rows
    out = np.zeros(stop - start, dtype=np.float64)
    if local_rows.size:
        raw_beta = group.R_inv @ beta
        if kind is SplineCategoricalGroupMatrix:
            out[local_rows] = np.asarray(group.B[local_rows + start] @ raw_beta).ravel()
        else:
            out[local_rows] = (group.B_unique @ raw_beta)[bins]
    return out


def group_row_range(group, start: int, stop: int):
    """Build owned chunk rows with bounded copies and certified category lookup."""
    if not _supported(group, start, stop):
        return None
    kind = type(group)
    if kind is DenseGroupMatrix:
        return DenseGroupMatrix(group.M[start:stop].copy(order="C"))
    if kind in _CATEGORICAL:
        if kind is RandomEffectGroupMatrix:
            return RandomEffectGroupMatrix(
                group.codes[start:stop], group.n_levels, lambda_policies=group.lambda_policies
            )
        return CategoricalGroupMatrix(group.codes[start:stop], group.n_levels)
    if kind in _SUPPORT:
        sub = kind(group.B_unique, group.R_inv, group.bin_idx[start:stop].copy())
        metadata = _METADATA
    else:
        rows = _category_rows(group, start, stop)
        if rows is None:
            return None
        local_rows, bins = rows
        if kind is SplineCategoricalGroupMatrix:
            sub = kind(group.B[start:stop], group.R_inv, local_rows)
        else:
            sub = kind(
                group.B_unique,
                group.R_inv,
                bins,
                local_rows,
                n_rows=stop - start,
                bin_idx_is_level=True,
            )
        metadata = _CATEGORY_METADATA
    for name in metadata:
        setattr(sub, name, getattr(group, name))
    return sub
