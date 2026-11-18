"""A penalty's null space stays unpenalised at any smoothing parameter (#468).

At ``lambda = 1e14`` the formed penalty ``lambda S`` rounds by ``u lambda ||S||``
in every direction its coordinates do not align with its null space, which is
as large as the data's own curvature there.  The fit must still be the least
squares fit on the penalty's null space: ``1, x`` for the derivative-penalty
bases, and ``1`` with the coefficient-linear spline ``B j`` for the P-spline's
second-difference penalty (Eilers and Marx 1996).  Both references are
ordinary least squares on the raw basis, without the fit's penalty or rank
code.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import LambdaPolicy, Numeric, Spline, SuperGLM
from superglm.dm_builder import compute_R_inv
from superglm.penalties import GroupElasticNet, GroupLasso, Ridge, SparseGroupLasso
from superglm.reml.penalty_algebra import ssp_penalty_matrix
from superglm.types import GroupInfo

U = 2.0**-53
X_GRID = np.linspace(0.0, 1.0, 200)
Y = 1.0 + 2.0 * X_GRID + 0.1 * np.random.default_rng(468).standard_normal(X_GRID.size)


def _fit(kind: str, penalty: str, route: str, lam: float, discrete: bool = False) -> SuperGLM:
    if route in ("fit", "mapping"):
        smoothing = {"x": lam} if route == "mapping" else lam
        spline, extra = Spline(kind=kind, n_knots=8, penalty=penalty), {"spline_penalty": smoothing}
    else:
        policy = LambdaPolicy.fixed(lam)
        spline, extra = Spline(kind=kind, n_knots=8, penalty=penalty, lambda_policy=policy), {}
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": spline},
        discrete=discrete,
        **extra,
    )
    frame = pd.DataFrame({"x": X_GRID})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, Y) if route == "fit_reml" else model.fit(frame, Y)
    return model


def _null_space_fit(model: SuperGLM, kind: str) -> np.ndarray:
    if kind == "ps":
        basis = model._specs["x"]._raw_basis_matrix(X_GRID)
        null = basis @ np.arange(basis.shape[1], dtype=np.float64)
    else:
        null = X_GRID
    design = np.column_stack((np.ones_like(X_GRID), null))
    return design @ np.linalg.lstsq(design, Y, rcond=None)[0]


# The 200 grid values sit under the 256 bins, so the binned fit is the exact one.
@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("lam", [1e14, 1e18])
@pytest.mark.parametrize("route", ["fit", "mapping", "fit_reml"])
@pytest.mark.parametrize("penalty", ["none", "ssp"])
@pytest.mark.parametrize("kind", ["cr", "bs", "ps", "cr_cardinal"])
def test_a_large_penalty_fits_its_null_space(kind, penalty, route, lam, discrete):
    model = _fit(kind, penalty, route, lam, discrete)
    reference = _null_space_fit(model, kind)
    predictions = model.predict(pd.DataFrame({"x": X_GRID}))
    # The solve runs in coordinates conditioned up to 1/sqrt(u) (an SSP map
    # built at lambda), so its forward error is some thousands of u at the
    # response's scale; a penalty leaking into its null space misses by 1e-9
    # and more.
    assert np.max(np.abs(predictions - reference)) <= 2**16 * U * np.max(np.abs(Y))
    # Two null directions (with the intercept); the range adds at most
    # sum(mu / (mu + lambda)) over generalized eigenvalues mu below 1e4 here,
    # and the trace rounds within some thousands of u.
    assert abs(model.result.effective_df - 2.0) <= 1e5 / lam + 2**16 * U
    assert model.result.converged


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("penalty", ["none", "ssp"])
def test_a_mapping_protects_each_term_at_its_own_penalty(penalty, discrete):
    """``{"x": 1e14, "z": 1e-3}`` fits ``x`` on its null space and ``z`` as at 1e-3 alone.

    The reference gives ``x`` its null space ``1, x`` as a linear term and ``z``
    its small penalty as a scalar, so no term borrows another's.
    """
    z = np.random.default_rng(0).permutation(X_GRID)
    frame = pd.DataFrame({"x": X_GRID, "z": z})
    y = Y + np.sin(6.0 * z)

    def fit(x_term, smoothing):
        features = {"x": x_term, "z": Spline(kind="cr", n_knots=8, penalty=penalty)}
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features=features,
            discrete=discrete,
            spline_penalty=smoothing,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(frame, y)

    mapped = fit(Spline(kind="cr", n_knots=8, penalty=penalty), {"x": 1e14, "z": 1e-3})
    reference = fit(Numeric(), 1e-3)
    # The single-term test's bounds: x's range adds at most 1e5 / lambda edf.
    tolerance = 2**16 * U * np.max(np.abs(y))
    assert np.max(np.abs(mapped.predict(frame) - reference.predict(frame))) <= tolerance
    edf_gap = mapped.result.effective_df - reference.result.effective_df
    assert abs(edf_gap) <= 1e5 / 1e14 + 2**16 * U


def test_the_ssp_map_factors_a_penalty_whose_formed_null_is_negative():
    """``G + lambda Omega`` formed at 1e14 is indefinite when Omega rounds below zero on its null."""
    p, n = 10, 200
    rng = np.random.default_rng(0)
    basis = np.abs(rng.standard_normal((n, p)))
    basis /= basis.sum(axis=1, keepdims=True)  # a partition of unity, as B-splines are
    difference = np.diff(np.eye(p), n=2, axis=0)  # exact integers: null space {1, j}
    omega = difference.T @ difference
    ones = np.full(p, p**-0.5)
    # A rounding-sized dip below zero on the exact null, forced rather than left to BLAS.
    perturbed = omega - 8 * U * np.max(np.abs(omega)) * np.outer(ones, ones)
    weights = np.ones(n)
    lam = 1e14
    with pytest.raises(np.linalg.LinAlgError):
        gram = basis.T @ basis / n
        np.linalg.cholesky(gram + lam * perturbed + 1e-8 * np.eye(p))
    R_inv = compute_R_inv(basis, perturbed, weights, lam)
    # The map whitens the declared penalty, checked through its exact integer
    # root, whose product with R_inv keeps the null direction's rounding second order.
    rooted = difference @ R_inv
    whitened = R_inv.T @ (basis.T @ basis / n + 1e-8 * np.eye(p)) @ R_inv + lam * rooted.T @ rooted
    # Inverting a triangular factor of condition kappa errs by a small multiple
    # of u kappa (Higham 2002, sec. 14.2).
    assert np.max(np.abs(whitened - np.eye(p))) <= 2**8 * U * np.linalg.cond(R_inv)


@pytest.mark.parametrize(
    "selection, turned",
    [
        (lambda: SparseGroupLasso(lambda1=1e-3, alpha=0.5), False),
        (lambda: SparseGroupLasso(lambda1=1e-3, alpha=0.0), True),
        (lambda: GroupLasso(lambda1=1e-3), True),
        (lambda: GroupElasticNet(lambda1=1e-3, alpha=0.5), True),
        (lambda: Ridge(lambda1=1e-3), True),
    ],
    ids=["sgl-l1", "sgl-group", "group-lasso", "group-elastic-net", "ridge"],
)
def test_the_eigenbasis_turn_waits_for_a_rotation_invariant_selection_penalty(selection, turned):
    """The turn is orthogonal within the block, which an elementwise L1 part is not invariant under.

    Under ``SparseGroupLasso(alpha > 0)`` the block keeps its declared
    (projected) coordinates, so the L1 part acts on the coefficients it was
    declared on; penalties reading each group through its norm keep the turn.
    """
    spline = Spline(kind="cr", n_knots=8, penalty="none")
    model = SuperGLM(
        family="gaussian", penalty=selection(), features={"x": spline}, spline_penalty=1e14
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame({"x": X_GRID}), Y)
    block = model._dm.group_matrices[0]
    assert np.array_equal(block.R_inv, block.projection) is not turned


class _NearRangePenalty:
    """``[sin x, cos x]`` with the finite penalty ``1e308 I``, whose root's sums overflow unscaled."""

    def raw(self, x):
        return np.column_stack([np.sin(x), np.cos(x)])

    def build(self, x, sample_weight=None):
        basis = sp.csr_matrix(self.raw(x))
        return GroupInfo(basis, 2, penalty_matrix=1e308 * np.eye(2), reparametrize=True)

    def set_reparametrisation(self, R_inv):
        self.R_inv = R_inv

    def transform(self, x):
        return self.raw(x) @ self.R_inv

    def reconstruct(self, beta):
        return {"beta": self.R_inv @ beta}


def test_a_penalty_near_the_float64_range_fits_its_null_space():
    """A finite penalty above ``2**1023`` penalises the whole term away, as before #468."""
    x = np.linspace(0.0, 10.0, 100)
    frame, y = pd.DataFrame({"x": x}), 1.0 + np.sin(x)
    model = SuperGLM(
        family="gaussian", selection_penalty=0, features={"x": _NearRangePenalty()}
    ).fit_reml(frame, y)
    assert model.result.converged
    assert model.result.effective_df == pytest.approx(1.0, abs=1e-9)
    # The intercept-only least squares fit: the penalty has no null space.
    np.testing.assert_allclose(model.predict(frame), y.mean(), rtol=1e-12)


def test_a_huge_ssp_penalty_is_the_ordinary_one_scaled_by_powers_of_two():
    """Rooting ``2**1021 Omega`` against ``2**-511 C`` gives half of ``C' Omega C``."""
    difference = np.diff(np.eye(6), n=2, axis=0)
    omega = difference.T @ difference  # entries up to 6: 6 * 2**1021 overflows P + P'
    coordinate_map = np.random.default_rng(468).standard_normal((6, 6))
    ordinary = ssp_penalty_matrix(coordinate_map, omega)
    huge = ssp_penalty_matrix(2.0**-511 * coordinate_map, 2.0**1021 * omega)
    assert np.all(np.isfinite(huge))
    np.testing.assert_allclose(huge, 0.5 * ordinary, rtol=0, atol=8 * U * np.max(np.abs(ordinary)))
    # The reviewer's case: the old congruence ``C' Omega C`` gave the identity.
    np.testing.assert_allclose(
        ssp_penalty_matrix(1e-154 * np.eye(2), 1e308 * np.eye(2)), np.eye(2), rtol=8 * U, atol=0
    )
