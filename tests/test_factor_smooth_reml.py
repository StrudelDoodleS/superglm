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
    the range widened by 0.1% per side), augmented per level with the rows of the
    m-th difference penalty for the wiggle and, for ``null_j``, the row
    ``q_j' B / sqrt(n)``, ``q_j`` the degree-``j`` coefficient polynomial on the rows
    orthonormalized by QR, so its penalty is ``(n_j' G beta)^2`` with ``n_j``
    orthonormal in ``G = B'B / n``.  Returns the predictions and the augmented
    design's condition number.
    """
    from scipy.interpolate import BSpline

    x = X["x"].to_numpy(dtype=float)
    lo, hi = x.min(), x.max()
    edge = 0.001 * (hi - lo)
    step = (hi - lo + 2.0 * edge) / (k - 3)
    knots = lo - edge + step * np.arange(-3, k + 1)
    basis = BSpline.design_matrix(x, knots, 3).toarray()
    q, _ = np.linalg.qr(basis @ np.vander(np.linspace(-1.0, 1.0, k), m, increasing=True))
    rows = np.vstack(
        (
            np.sqrt(lambdas[0]) * np.diff(np.eye(k), n=m, axis=0),
            np.sqrt(np.asarray(lambdas[1:]))[:, None] * q.T @ basis / np.sqrt(len(x)),
        )
    )
    levels = sorted(X["group"].unique())

    def design(frame_rows):
        raw = BSpline.design_matrix(frame_rows["x"].to_numpy(dtype=float), knots, 3).toarray()
        groups = frame_rows["group"].to_numpy()
        blocks = [raw * (groups == level)[:, None] for level in levels]
        return np.column_stack([np.ones(len(frame_rows)), *blocks])

    augmented = np.vstack((design(X), np.pad(np.kron(np.eye(len(levels)), rows), ((0, 0), (1, 0)))))
    target = np.r_[y, np.zeros(len(augmented) - len(y))]
    beta = np.linalg.lstsq(augmented, target, rcond=None)[0]
    return design(test) @ beta, np.linalg.cond(augmented)


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


def _bunched_case() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Three levels with 20 rows across the range and 300 within 3% of it.

    The rows leave knot spans nearly empty, so the row factor's condition number is
    about 1e13: a wiggle built through its inverse missed the reference by 2e-5.
    """
    u = np.random.default_rng(316).uniform(-1.0, 1.0, 320)
    support = np.r_[u[:20], 0.03 * u[20:]]
    code = np.repeat(np.arange(3), len(support))
    x = np.tile(support, 3)
    X = pd.DataFrame({"x": x, "group": np.array(["a", "b", "c"], dtype=object)[code]})
    y = np.sin(2.0 * x) * (1 + code) + 0.3 * code + 0.1 * np.cos(40.0 * x)
    return X, y, X


def _wide_case() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Two levels on uniform rows for ``k=150, m=12``: 138 wiggle contrasts.

    The difference matrix's singular values span 14 orders of magnitude, so a wiggle
    built from its explicit right inverse missed the reference by 2.9e-4.
    """
    rng = np.random.default_rng(316)
    n = 1800
    x = rng.uniform(-1.0, 1.0, n)
    group = np.resize(np.array(["a", "b"], dtype=object), n)
    a = group == "a"
    y = np.sin(2.0 * x) + 0.25 * a + 0.35 * np.cos(35.0 * x) * a + 0.25 * rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "group": group})
    test = pd.DataFrame({"x": np.linspace(x.min(), x.max(), 301), "group": np.resize(group, 301)})
    return X, y, test


@pytest.mark.parametrize(
    ("case", "k", "m", "lambdas"),
    [
        (_smooth_case, 6, 3, (10.0, 1.0, 5.0e4, 1.0e3)),
        (_two_band_case, 10, 2, (2.0, 1.0, 50.0)),
        (_bunched_case, 20, 2, (3.0, 1.0, 30.0)),
        (_wide_case, 150, 12, (1.0,) * 13),
    ],
    ids=["m3", "two_bands", "bunched", "wide"],
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
    complement of the null axes over the rows, applying the row factor and the
    difference matrix forward only (the bunched and wide cases).
    """
    X, y, test = case()
    names = ("wiggle", *(f"null_{j}" for j in range(m)))
    policies = {name: LambdaPolicy.fixed(value) for name, value in zip(names, lambdas, strict=True)}
    expected, condition = _polynomial_null_reference(X, y, test, lambdas, k=k, m=m)
    # The reference and the fit each round row sums of n terms and a least-squares
    # solve with this conditioning (Higham 2002, section 3.1 and chapter 20).
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


@pytest.mark.parametrize("wiggle_lambda", [1.0e6, 1.0e10])
def test_high_order_coordinates_give_the_raw_penalized_smoother(wiggle_lambda) -> None:
    """``k=60, m=30``: the natural coordinates' fixed-lambda smoother is the raw penalty's.

    The difference matrix's condition number is 5e11, and its SVD leaves the weakly
    penalized wiggle directions, which a large lambda prices, a penalty that moved this
    smoother by up to 1.4e-6 until the triangular factor of their penalty image
    re-whitened them.  The
    reference, in mpmath, is ``R (R'R + lambda D'D + S_null)^-1 R'`` with ``S_null``
    the projection onto the polynomials of degree below m over the rows (equal null
    lambdas of 1, so any basis of them).
    """
    import math

    from scipy.interpolate import BSpline

    mp = pytest.importorskip("mpmath")
    k, m, n = 60, 30, 720
    x = np.random.default_rng(316).uniform(-1.0, 1.0, n)
    spec = FactorSmooth("x", group="group", k=k, m=m)
    spec.build(x, np.resize(np.array(["a", "b"], dtype=object), n), {})
    lo, hi = x.min(), x.max()
    edge = 0.001 * (hi - lo)
    knots = lo - edge + (hi - lo + 2.0 * edge) / (k - 3) * np.arange(-3, k + 1)
    R = np.linalg.qr(BSpline.design_matrix(x, knots, 3).toarray(), mode="r")

    mp.mp.dps = 50
    R_mp = mp.matrix(R.tolist())
    gram = R_mp.T * R_mp
    powers = mp.matrix([[(mp.mpf(2) * i / (k - 1) - 1) ** j for j in range(m)] for i in range(k)])
    projected = gram * powers
    penalty = projected * mp.inverse(powers.T * projected) * projected.T / n
    stencil = [(-1) ** (m - j) * math.comb(m, j) for j in range(m + 1)]
    for row in range(k - m):
        for a, left in enumerate(stencil):
            for b, right in enumerate(stencil):
                penalty[row + a, row + b] += mp.mpf(left * right) * wiggle_lambda
    expected = np.array((R_mp * mp.inverse(gram + penalty) * R_mp.T).tolist(), dtype=float)

    natural = R @ spec._natural_map
    (_, wiggle), *nulls = spec._base_penalty_components
    diagonal = np.diag(wiggle_lambda * wiggle + sum(null for _, null in nulls))
    stacked = np.vstack((natural, np.diag(np.sqrt(diagonal))))
    stacked /= np.linalg.norm(stacked, axis=0)
    rows = np.linalg.qr(stacked)[0][:k]
    u = np.finfo(np.float64).eps / 2
    atol = 64.0 * (2 * k * u / (1.0 - 2 * k * u)) * np.linalg.cond(stacked)
    np.testing.assert_allclose(rows @ rows.T, expected, rtol=0.0, atol=atol)


def _bunched_rows(k: int, seed: int) -> np.ndarray:
    """A jittered row near every knot plus ``2k`` rows bunched in a random sub-interval."""
    rng = np.random.default_rng(seed)
    lo, hi = sorted(rng.uniform(-0.8, 0.8, 2))
    grid = np.linspace(-1.0, 1.0, k)
    spread = np.r_[-1.0, grid[1:-1] + rng.uniform(-0.23, 0.23, k - 2) / (k - 3), 1.0]
    return np.r_[spread, rng.uniform(lo, hi, 2 * k)]


def test_bunched_rows_build_the_wiggle_at_a_high_admitted_order() -> None:
    """``k=120, m=101`` on unweighted bunched rows builds, in any row order.

    Bunching tilts the complement of the null axes away from ``D``'s row space, so
    the difference operator on it is far worse conditioned than ``D`` (lower bound
    2^42): the Gram of the weak directions' penalty image was singular in float64 and
    its Cholesky factorization raised on every OpenBLAS kernel.  The image's QR gives
    the same triangular factor without forming the Gram.  The wiggle columns stay
    orthogonal over the rows to each other and to the null axes, and the whitened
    fixed-lambda smoother does not depend on the row order, each to ``u cond(R)``
    (cond(R) is 1.9e13 here; measured 6e-4 and 3e-8).
    """
    k, m = 120, 101
    x = _bunched_rows(k, 0)
    rank = k - m
    u = np.finfo(np.float64).eps / 2
    smoothers = []
    for order in (np.arange(len(x)), np.random.default_rng(630).permutation(len(x))):
        spec = FactorSmooth("x", group="group", k=k, m=m)
        spec.build(x[order], np.full(len(x), "a", dtype=object), {})
        natural = spec._natural_map
        assert np.all(np.isfinite(natural))
        basis = spec._spline._basis_matrix(x).toarray()
        tolerance = u * np.linalg.cond(np.linalg.qr(basis, mode="r"))
        rows = basis @ natural
        scale = np.linalg.norm(rows, axis=0)
        gram = (rows.T @ rows) / np.outer(scale, scale)
        assert np.max(np.abs(gram[:rank] - np.eye(k)[:rank])) < tolerance
        (_, wiggle), *nulls = spec._base_penalty_components
        penalty = np.diag(2.0 * wiggle + sum(null for _, null in nulls))
        smoothers.append(rows @ np.linalg.solve(rows.T @ rows + np.diag(penalty), rows.T))
    np.testing.assert_allclose(smoothers[1], smoothers[0], rtol=0.0, atol=tolerance)


def test_unpenalized_wiggle_admits_any_order_and_gives_least_squares() -> None:
    """With the wiggle fixed at zero ``D``'s conditioning does not enter, so the refused
    ``k=79, m=35`` fits: one level, ``null_0`` at one and every other component off
    leave ordinary least squares on the spline design (condition 36), as ``m=1`` does.
    """
    from scipy.interpolate import BSpline

    k, m = 79, 35
    x = np.linspace(-1.0, 1.0, 320)
    X = pd.DataFrame({"x": x, "group": "a"})
    basis = BSpline.design_matrix(x, -1.002 + 2.004 / (k - 3) * np.arange(-3, k + 1), 3).toarray()
    y = 2.0 + np.sin(9.0 * x) + 0.2 * np.cos(92.0 * x)
    coefficients, *_ = np.linalg.lstsq(basis, y, rcond=None)
    with pytest.raises(ValueError, match="cannot represent an m=35"):
        FactorSmooth("x", group="group", k=k, m=m)
    policy = {"wiggle": LambdaPolicy.off(), "null_0": LambdaPolicy.fixed(1.0)}
    policy.update({f"null_{j}": LambdaPolicy.off() for j in range(1, m)})
    smooth = FactorSmooth("x", group="group", k=k, m=m, lambda_policy=policy)
    model = SuperGLM(family="gaussian", interactions=[smooth], selection_penalty=0.0)
    u = np.finfo(np.float64).eps / 2
    np.testing.assert_allclose(
        model.fit_reml(X, y).predict(X), basis @ coefficients, rtol=0.0, atol=1e3 * u * np.max(y)
    )


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
