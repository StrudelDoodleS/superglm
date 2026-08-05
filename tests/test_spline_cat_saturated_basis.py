"""A saturated spline-by-category basis must not be stored sparse.

An interaction marginal on a ``Spline(kind="cr")`` parent resolves to a
cardinal basis, whose functions are GLOBALLY supported: the shared block
measures density 1.000 -- every row, every column.  Stored as CSR that costs 12
bytes per entry against 8, and routes the gram through ``_csr_weighted_gram``,
a scalar accumulation over nonzero pairs, where BLAS would run a blocked
kernel.

The sparse route is retained unchanged for every block that is actually sparse;
this only reclaims the case where CSR was storing no zeros at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from superglm import SuperGLM
from superglm._group_matrix._group_matrix_core import (
    SplineCategoricalGroupMatrix,
    _dense_if_saturated,
)
from superglm._group_matrix._group_matrix_kernels import _csr_weighted_gram
from superglm.features import Categorical, Spline


def _block(density: float, n: int = 4000, p: int = 22, seed: int = 3):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, p))
    if density < 1.0:
        values[rng.random((n, p)) > density] = 0.0
    csr = sp.csr_matrix(values)
    gm = SplineCategoricalGroupMatrix(csr, np.eye(p), np.arange(n))
    weights = np.abs(rng.normal(1.0, 0.2, n))
    return csr, gm, weights


def test_a_saturated_block_takes_the_dense_route() -> None:
    csr, gm, weights = _block(1.0)
    gm.gram(weights)

    assert gm._dense_level is not None, "density 1.000 must not be stored sparse"
    assert gm._dense_level.shape == csr.shape


def test_a_sparse_block_keeps_the_csr_kernel_bitwise() -> None:
    """The numba path is the right tool below saturation and must not move."""
    csr, gm, weights = _block(0.30)
    got = gm.gram(weights)

    assert gm._dense_level is None, "a 30% block must stay sparse"
    reference = _csr_weighted_gram(
        csr.data.astype(np.float64), csr.indices, csr.indptr, weights, csr.shape[1]
    )
    assert np.array_equal(got, reference), "sparse route must be bit-identical"


def test_the_dense_route_agrees_with_the_kernel_it_replaces() -> None:
    """Not bitwise -- BLAS sums in a different order -- but within a few ulps."""
    csr, gm, weights = _block(1.0)
    got = gm.gram(weights)

    reference = _csr_weighted_gram(
        csr.data.astype(np.float64), csr.indices, csr.indptr, weights, csr.shape[1]
    )
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    assert np.max(np.abs(got - reference)) / scale < 1e-13


def test_the_dense_copy_is_deferred_until_a_gram_is_asked_for() -> None:
    """A block replaced by its compressed variant must not pay for a copy.

    Building it eagerly cost ~6% on the support-compressed case, where the
    uncompressed block is constructed and then superseded without ever being
    grammed.
    """
    _, gm, weights = _block(1.0)

    assert gm._dense_level is False, "constructing must not densify"
    gm.gram(weights)
    assert gm._dense_level is not None


def test_saturation_threshold_is_a_property_of_the_block_not_the_caller() -> None:
    for density, expect_dense in ((1.0, True), (0.95, True), (0.50, False), (0.15, False)):
        csr, _, _ = _block(density)
        assert (_dense_if_saturated(csr) is not None) is expect_dense, density


def test_a_cr_interaction_fit_is_unchanged() -> None:
    """End to end: the representation moves, the answer does not."""
    rng = np.random.default_rng(0)
    n = 6_000
    frame = pd.DataFrame({"f": rng.integers(0, 4, n).astype(str), "x": rng.uniform(0.0, 1.0, n)})
    y = rng.poisson(2.0, n).astype(float)

    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"f": Categorical(), "x": Spline(kind="cr", n_knots=20)},
        interactions=[("x", "f")],
    )
    model.fit(frame, y)

    # The blocks this change targets are present and saturated.
    saturated = [
        gm for gm in model._dm.group_matrices if isinstance(gm, SplineCategoricalGroupMatrix)
    ]
    assert saturated, "expected spline_cat blocks in this design"
    for gm in saturated:
        rows, cols = gm.B_level.shape
        # Measured 0.9994 on this design: a cardinal basis is globally
        # supported, but an exact zero can still land on a knot, so assert
        # saturation rather than literal fullness.
        assert gm.B_level.nnz / (rows * cols) >= 0.9
        gm.gram(np.ones(gm.n_rows))
        assert gm._dense_level is not None

    assert np.isfinite(model._result.deviance)
    assert model._result.n_iter > 0
