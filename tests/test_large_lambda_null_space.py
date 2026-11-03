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

from superglm import LambdaPolicy, Spline, SuperGLM
from superglm.dm_builder import compute_R_inv

U = 2.0**-53
X_GRID = np.linspace(0.0, 1.0, 200)
Y = 1.0 + 2.0 * X_GRID + 0.1 * np.random.default_rng(468).standard_normal(X_GRID.size)


def _fit(kind: str, penalty: str, route: str, lam: float) -> SuperGLM:
    if route == "fit":
        spline, extra = Spline(kind=kind, n_knots=8, penalty=penalty), {"spline_penalty": lam}
    else:
        policy = LambdaPolicy.fixed(lam)
        spline, extra = Spline(kind=kind, n_knots=8, penalty=penalty, lambda_policy=policy), {}
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": spline}, **extra)
    frame = pd.DataFrame({"x": X_GRID})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, Y) if route == "fit" else model.fit_reml(frame, Y)
    return model


def _null_space_fit(model: SuperGLM, kind: str) -> np.ndarray:
    if kind == "ps":
        basis = model._dm.group_matrices[0].B.toarray()
        null = basis @ np.arange(basis.shape[1], dtype=np.float64)
    else:
        null = X_GRID
    design = np.column_stack((np.ones_like(X_GRID), null))
    return design @ np.linalg.lstsq(design, Y, rcond=None)[0]


@pytest.mark.parametrize("lam", [1e14, 1e18])
@pytest.mark.parametrize("route", ["fit", "fit_reml"])
@pytest.mark.parametrize("penalty", ["none", "ssp"])
@pytest.mark.parametrize("kind", ["cr", "bs", "ps", "cr_cardinal"])
def test_a_large_penalty_fits_its_null_space(kind, penalty, route, lam):
    model = _fit(kind, penalty, route, lam)
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
