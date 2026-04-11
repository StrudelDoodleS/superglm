"""Private tabmat construction helpers for group matrices."""

from __future__ import annotations

import numpy as np
import tabmat  # type: ignore[import-untyped]


def _build_tabmat_split(gms):
    """Build a tabmat SplitMatrix from non-discrete group matrices."""
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
    )

    if any(isinstance(gm, DiscretizedSSPGroupMatrix | DiscretizedSCOPGroupMatrix) for gm in gms):
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
                    tabmat.CategoricalMatrix(codes, categories=categories, drop_first=True)
                )
            else:
                matrices.append(tabmat.DenseMatrix(gm.toarray()))
        elif isinstance(gm, SparseGroupMatrix):
            matrices.append(tabmat.SparseMatrix(gm.M))
        elif isinstance(gm, SparseSSPGroupMatrix):
            matrices.append(tabmat.DenseMatrix(gm.toarray()))
        elif isinstance(gm, DenseGroupMatrix):
            arr = gm.toarray()
            if arr.ndim == 1:
                arr = arr[:, None]
            matrices.append(tabmat.DenseMatrix(arr))
        else:
            arr = gm.toarray()
            if arr.ndim == 1:
                arr = arr[:, None]
            matrices.append(tabmat.DenseMatrix(arr))
    return tabmat.SplitMatrix(matrices)
