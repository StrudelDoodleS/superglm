"""Small dense REML oracles for fully penalized factor smooths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, LambdaPolicy, SuperGLM
from superglm.reml.penalty_algebra import build_penalty_matrix


def _gaussian_data() -> tuple[pd.DataFrame, np.ndarray]:
    x_level = np.linspace(-1.0, 1.0, 24)
    levels = np.array(["a", "b", "c"], dtype=object)
    x = np.tile(x_level, len(levels))
    group = np.repeat(levels, len(x_level))
    effects = {
        "a": 0.8 * np.sin(2.2 * x_level) + 0.2 * x_level,
        "b": -0.4 * np.cos(1.7 * x_level) - 0.3 * x_level,
        "c": 0.25 + 0.55 * x_level**2,
    }
    y = 1.1 + np.concatenate([effects[level] for level in levels])
    return pd.DataFrame({"x": x, "group": group}), y


def test_fixed_lambda_gaussian_fit_matches_explicit_penalized_least_squares() -> None:
    X, y = _gaussian_data()
    policies = {
        "wiggle": LambdaPolicy.fixed(1.4),
        "null_0": LambdaPolicy.fixed(0.7),
        "null_1": LambdaPolicy.fixed(2.1),
    }
    model = SuperGLM(
        family="gaussian",
        interactions=[FactorSmooth("x", group="group", k=6, lambda_policy=policies)],
        selection_penalty=0.0,
        direct_solve="gram",
    )

    model.fit_reml(X, y, max_reml_iter=2)

    design = model._dm.toarray()
    augmented = np.column_stack([np.ones(len(X)), design])
    penalty = build_penalty_matrix(
        list(model._dm.group_matrices),
        model._groups,
        model._reml_result.lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    augmented_penalty = np.zeros((model._dm.p + 1, model._dm.p + 1))
    augmented_penalty[1:, 1:] = penalty
    expected = np.linalg.solve(
        augmented.T @ augmented + augmented_penalty,
        augmented.T @ y,
    )

    assert model.result.intercept == pytest.approx(expected[0], abs=2e-9)
    np.testing.assert_allclose(model.result.beta, expected[1:], rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_estimated_lambda_dense_factor_smooth_fit_is_finite(family: str) -> None:
    X, gaussian_y = _gaussian_data()
    if family == "gaussian":
        y = gaussian_y
    else:
        rng = np.random.default_rng(571)
        mean = np.exp(-0.3 + 0.35 * gaussian_y)
        y = rng.poisson(mean).astype(np.float64)
    model = SuperGLM(
        family=family,
        interactions=[FactorSmooth("x", group="group", k=6)],
        selection_penalty=0.0,
        direct_solve="gram",
    )

    model.fit_reml(
        X,
        y,
        max_reml_iter=8,
        reml_tol=1e-5,
    )

    assert np.all(np.isfinite(model.result.beta))
    assert np.isfinite(model.result.intercept)
    assert model._reml_result is not None
    assert set(model._reml_result.lambdas) == {
        "x:group:fs:wiggle",
        "x:group:fs:null_0",
        "x:group:fs:null_1",
    }
    assert all(np.isfinite(value) and value > 0.0 for value in model._reml_result.lambdas.values())


def _polynomial_null_reference(X, y, test, lambdas, *, k, m):
    """Penalized least squares with ``null_j`` on degree ``j``, built without SuperGLM.

    Its own cubic B-spline design on FactorSmooth's knot rule (equal spacing over
    the range widened by 0.1% per side), the m-th difference penalty for the
    wiggle, and ``null_j`` the degree-``j`` coefficient polynomial orthonormalized
    over the rows (Gram-Schmidt in ``B'B / n``): its penalty is ``(n_j' G beta)^2``.
    """
    from scipy.interpolate import BSpline

    x = X["x"].to_numpy(dtype=float)
    lo, hi = x.min(), x.max()
    edge = 0.001 * (hi - lo)
    step = (hi - lo + 2.0 * edge) / (k - 3)
    knots = lo - edge + step * np.arange(-3, k + 1)
    basis = BSpline.design_matrix(x, knots, 3).toarray()
    G = basis.T @ basis / len(x)
    frame = np.vander(np.linspace(-1.0, 1.0, k), m, increasing=True)
    null = frame @ np.linalg.inv(np.linalg.cholesky(frame.T @ G @ frame)).T
    D = np.diff(np.eye(k), n=m, axis=0)
    block = lambdas[0] * D.T @ D
    for lam, direction in zip(lambdas[1:], (G @ null).T, strict=True):
        block += lam * np.outer(direction, direction)
    levels = sorted(X["group"].unique())

    def design(frame_rows):
        raw = BSpline.design_matrix(frame_rows["x"].to_numpy(dtype=float), knots, 3).toarray()
        groups = frame_rows["group"].to_numpy()
        blocks = [raw * (groups == level)[:, None] for level in levels]
        return np.column_stack([np.ones(len(frame_rows)), *blocks])

    A = design(X)
    penalty = np.zeros((A.shape[1], A.shape[1]))
    penalty[1:, 1:] = np.kron(np.eye(len(levels)), block)
    hessian = A.T @ A + penalty
    beta = np.linalg.solve(hessian, A.T @ y)
    return design(test) @ beta, np.linalg.cond(hessian)


def _smooth_case() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    X, y = _gaussian_data()
    test = pd.DataFrame(
        {"x": np.tile(np.linspace(-0.95, 0.95, 7), 3), "group": np.repeat(["a", "b", "c"], 7)}
    )
    return X, y + 0.05 * np.cos(9.0 * X["x"].to_numpy()), test


def _two_band_case() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Three levels on two bands of x, with the knot spans between them empty of rows."""
    support = np.r_[0:21, 60:71] / 70.0
    code = np.repeat(np.arange(3), len(support))
    x = np.tile(support, 3)
    X = pd.DataFrame({"x": x, "group": np.array(["a", "b", "c"], dtype=object)[code]})
    y = np.cos(3.0 * x) * (1 + code) + 0.8 * code + 0.1 * np.cos(50.0 * x)
    return X, y, X


@pytest.mark.parametrize(
    ("case", "k", "m", "lambdas"),
    [
        (_smooth_case, 6, 3, (10.0, 1.0, 5.0e4, 1.0e3)),
        (_two_band_case, 10, 2, (2.0, 1.0, 50.0)),
    ],
    ids=["m3", "two_bands"],
)
def test_fixed_null_lambdas_penalize_each_polynomial_degree_in_any_row_order(
    case, k, m, lambdas
) -> None:
    """Each ``null_j`` is the level's degree-j curve, orthonormal over the rows, in any row order.

    Unequal fixed null lambdas give one model: the reference's.  Before, ``null_j``
    was the j-th vector an eigensolver returned for the penalty's repeated zero
    eigenvalue, so its lambda shrank whatever mix of constant, line and quadratic
    that was, and the same rows in another order or on another platform priced
    differently.  With two bands of rows the whitened penalty's eigenvectors also
    mixed the null space into the wiggle columns, so the wiggle is built in the
    complement of the null axes over the rows.
    """
    X, y, test = case()
    names = ("wiggle", *(f"null_{j}" for j in range(m)))
    policies = {name: LambdaPolicy.fixed(value) for name, value in zip(names, lambdas, strict=True)}
    expected, condition = _polynomial_null_reference(X, y, test, lambdas, k=k, m=m)
    # The reference solve and the fit each round row sums of n terms and a
    # small solve with this conditioning (Higham 2002, sections 3.1 and 7.2).
    n = len(X)
    u = np.finfo(np.float64).eps / 2
    atol = 64.0 * (n * u / (1.0 - n * u)) * condition * np.max(np.abs(expected))
    for order in (np.arange(n), np.random.default_rng(630).permutation(n)):
        model = SuperGLM(
            family="gaussian",
            interactions=[FactorSmooth("x", group="group", k=k, m=m, lambda_policy=policies)],
            selection_penalty=0.0,
            tol=1e-12,
        )
        model.fit_reml(X.iloc[order], y[order])
        np.testing.assert_allclose(model.predict(test), expected, rtol=0.0, atol=atol)


def test_high_order_null_axes_leave_out_the_penalized_contrast() -> None:
    """At ``m = k - 1`` the null space is every direction but the binomial contrast.

    Equal null lambdas make the model independent of the null axes' basis, so it is
    the reference's: unit-penalty least squares on the complement, over the rows, of
    the one difference contrast, whose own penalty (above 1e35) leaves it no weight.
    A null frame sampled from Legendre polynomials at the coefficient index loses its
    high degrees as ``m`` nears ``k`` and leaks that contrast into ``null_j``.
    """
    import math

    from scipy.interpolate import BSpline
    from scipy.linalg import null_space, solve_triangular

    rng = np.random.default_rng(316)
    n, k = 960, 60
    x = rng.uniform(-1.0, 1.0, n)
    group = np.resize(np.array(["a", "b", "c"], dtype=object), n)
    y = np.sin(2.0 * x) + 0.25 * (group == "a") + 0.25 * rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "group": group})
    test = pd.DataFrame({"x": np.linspace(-0.99, 0.99, 99), "group": np.resize(group, 99)})
    lo, hi = x.min(), x.max()
    edge = 0.001 * (hi - lo)
    knots = lo - edge + (hi - lo + 2.0 * edge) / (k - 3) * np.arange(-3, k + 1)
    R = np.linalg.qr(BSpline.design_matrix(x, knots, 3).toarray() / np.sqrt(n), mode="r")
    contrast = np.array([(-1.0) ** j * math.comb(k - 1, j) for j in range(k)])
    whitened = solve_triangular(R, contrast / np.linalg.norm(contrast), trans="T")
    null = solve_triangular(R, null_space(whitened[None, :]))

    def design(rows: pd.DataFrame) -> np.ndarray:
        curves = BSpline.design_matrix(rows["x"].to_numpy(), knots, 3).toarray() @ null
        levels = rows["group"].to_numpy()
        blocks = [curves * (levels == level)[:, None] for level in ("a", "b", "c")]
        return np.column_stack([np.ones(len(rows)), *blocks])

    A = design(X)
    hessian = A.T @ A + np.diag(np.r_[0.0, np.ones(A.shape[1] - 1)])
    expected = design(test) @ np.linalg.solve(hessian, A.T @ y)
    u = np.finfo(np.float64).eps / 2
    atol = 64.0 * (n * u / (1.0 - n * u)) * np.linalg.cond(hessian) * np.max(np.abs(expected))
    policies = {f"null_{j}": LambdaPolicy.fixed(1.0) for j in range(k - 1)}
    policies["wiggle"] = LambdaPolicy.fixed(10.0)
    for order in (np.arange(n), rng.permutation(n)):
        model = SuperGLM(
            family="gaussian",
            interactions=[FactorSmooth("x", group="group", k=k, m=k - 1, lambda_policy=policies)],
            selection_penalty=0.0,
            tol=1e-11,
        )
        model.fit_reml(X.iloc[order], y[order])
        np.testing.assert_allclose(model.predict(test), expected, rtol=0.0, atol=atol)


def test_estimated_null_lambdas_keep_their_names_in_any_row_order() -> None:
    """REML estimates each named ``null_j`` the same, whatever order the rows arrive in."""
    rng = np.random.default_rng(20261119)
    n = 600
    x = rng.uniform(-1.0, 1.0, n)
    group = rng.choice(np.array(["a", "b", "c", "d"], dtype=object), n)
    level = {"a": 0.0, "b": 0.35, "c": -0.2, "d": 0.1}
    trend = {"a": 0.0, "b": 0.3, "c": -0.25, "d": 0.05}
    curve = {"a": 0.0, "b": 0.2, "c": 0.15, "d": -0.3}
    y = np.sin(2.5 * x) + np.array(
        [level[g] + trend[g] * xi + curve[g] * xi**2 for g, xi in zip(group, x, strict=True)]
    )
    y += rng.normal(scale=0.3, size=n)
    X = pd.DataFrame({"x": x, "group": group})
    test = pd.DataFrame(
        {"x": np.tile(np.linspace(-0.9, 0.9, 5), 4), "group": np.repeat(["a", "b", "c", "d"], 5)}
    )
    fits = []
    for order in (np.arange(n), np.random.default_rng(7).permutation(n)):
        model = SuperGLM(
            family="gaussian",
            interactions=[FactorSmooth("x", group="group", k=6, m=3)],
            selection_penalty=0.0,
            tol=1e-12,
        )
        model.fit_reml(X.iloc[order], y[order], max_reml_iter=100, reml_tol=1e-10)
        assert model.reml_diagnostics()["converged"]
        fits.append(model)
    first, second = (model.factor_smooth("x:group:fs") for model in fits)
    for name in ("wiggle", "null_0", "null_1", "null_2"):
        assert second.lambdas[name] == pytest.approx(first.lambdas[name], rel=1e-5)
    assert fits[1].result.effective_df == pytest.approx(fits[0].result.effective_df, abs=1e-6)
    np.testing.assert_allclose(fits[1].predict(test), fits[0].predict(test), rtol=0.0, atol=1e-7)
