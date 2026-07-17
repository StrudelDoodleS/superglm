"""Private tabmat construction helpers for group matrices."""

from __future__ import annotations

import numpy as np
import tabmat  # type: ignore[import-untyped]


def _dense_float64(values):
    """Return a solver-compatible dense Tabmat block."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return tabmat.DenseMatrix(array)


def _tabmat_vector(values):
    """Return the writable contiguous float64 buffer Tabmat kernels require."""
    array = np.asarray(values, dtype=np.float64)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=np.float64, order="C", copy=True)
    return array


def _is_tabmat_centering_candidate(gms) -> bool:
    """Return whether centering can use a native categorical Tabmat block."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )

    unsupported = (
        SparseSSPGroupMatrix
        | SplineCategoricalGroupMatrix
        | DiscretizedSplineCategoricalGroupMatrix
        | DiscretizedSSPGroupMatrix
        | DiscretizedSCOPGroupMatrix
    )
    return (
        not any(isinstance(gm, unsupported) for gm in gms)
        and any(not isinstance(gm, CategoricalGroupMatrix) for gm in gms)
        and any(isinstance(gm, CategoricalGroupMatrix) and gm.n_levels > 100 for gm in gms)
    )


def _build_tabmat_split(gms):
    """Build a tabmat SplitMatrix from non-discrete group matrices."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )

    if any(
        isinstance(
            gm,
            SparseSSPGroupMatrix
            | SplineCategoricalGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | DiscretizedSSPGroupMatrix
            | DiscretizedSCOPGroupMatrix,
        )
        for gm in gms
    ):
        return None

    if all(isinstance(gm, CategoricalGroupMatrix) for gm in gms) and all(
        gm.n_levels <= 100 for gm in gms if isinstance(gm, CategoricalGroupMatrix)
    ):
        return None

    matrices = []
    for gm in gms:
        if isinstance(gm, CategoricalGroupMatrix):
            if gm.n_levels > 100:
                codes = gm.codes.copy().astype(np.int32)
                base_mask = codes == gm.n_levels
                codes[~base_mask] += 1
                codes[base_mask] = 0
                categories = np.arange(gm.n_levels + 1)
                matrices.append(
                    tabmat.CategoricalMatrix(
                        codes,
                        categories=categories,
                        drop_first=True,
                        dtype=np.float64,
                    )
                )
            else:
                matrices.append(_dense_float64(gm.toarray()))
        elif isinstance(gm, SparseGroupMatrix):
            matrices.append(tabmat.SparseMatrix(gm.M.astype(np.float64, copy=False)))
        elif isinstance(gm, SparseSSPGroupMatrix):
            matrices.append(_dense_float64(gm.toarray()))
        elif isinstance(gm, DenseGroupMatrix):
            matrices.append(_dense_float64(gm.toarray()))
        else:
            matrices.append(_dense_float64(gm.toarray()))
    return tabmat.SplitMatrix(matrices)
