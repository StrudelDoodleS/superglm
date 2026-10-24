"""Exact de-duplicated support for a shape-constrained (SCOP) design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ExactSCOPSupport:
    """Unique SCOP design rows, and the map from each original row to them."""

    B_unique: NDArray
    row_to_support: NDArray

    @property
    def n_unique(self) -> int:
        return self.B_unique.shape[0]


def _distinct_rows(B: NDArray) -> tuple[NDArray, NDArray]:
    """``np.unique(B, axis=0, return_inverse=True)``, without sorting whole rows.

    Rows are grouped by one key, a fixed linear combination of the columns
    formed column by column so that equal rows get equal keys, and the
    grouping is then checked exactly: a key shared by unequal rows (a
    rounding coincidence) falls back to sorting whole rows. The distinct
    rows come back in ``np.unique``'s lexicographic order.
    """
    coefficients = np.linspace(1.0, 2.0, B.shape[1])
    key = np.zeros(B.shape[0])
    for column, coefficient in zip(B.T, coefficients, strict=True):
        key += column * coefficient
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    unique = B[first]
    if not np.array_equal(unique[inverse], B):
        return np.unique(B, axis=0, return_inverse=True)
    order = np.lexsort(unique.T[::-1])
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return unique[order], rank[inverse]


def build_exact_scop_support(B_scop: NDArray) -> ExactSCOPSupport | None:
    """Return the de-duplicated SCOP support, or ``None`` when rows are unique."""

    B = np.asarray(B_scop, dtype=np.float64)
    B_unique, row_to_support = _distinct_rows(B)
    if B_unique.shape[0] == B.shape[0]:
        return None
    return ExactSCOPSupport(B_unique=B_unique, row_to_support=row_to_support)
