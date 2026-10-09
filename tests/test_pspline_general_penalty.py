"""P-splines on uneven knots take Li and Cao's general difference penalty (arXiv:2201.06808)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM
from superglm.features._spline_penalties import (
    build_difference_penalty,
    build_general_difference_penalty,
)

UNEVEN = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 5.0, 8.0]
u = np.finfo(np.float64).eps / 2


def _built(**kwargs):
    spline = Spline(kind="ps", **kwargs)
    spline.build(np.linspace(0.0, 10.0, 400))
    return spline


def _null_bound(spline, penalty, line) -> float:
    """Round-off bound on ``line' P line`` for a line ``P`` annihilates in exact arithmetic.

    Building ``D`` costs ``gamma_{2m}`` per entry (m differences, m scalings),
    forming ``D'D`` ``gamma_n``, a final scale factor one rounding, and the
    quadratic form ``gamma_{2n}`` (Higham 2002, sections 3.1 and 3.5). The
    difference rows keep their sign pattern, so ``|D|'|D| = |P|`` up to those
    factors, and every error is bounded by ``|line|' |P| |line|``; with
    ``m = 2`` the sum of the counts is at most ``4n`` roundings, and twice that
    covers ``gamma_k <= 2 k u`` for ``k u <= 1/2``.
    """
    scale = np.abs(line) @ np.abs(penalty) @ np.abs(line)
    return 8 * spline._n_basis * u * scale


def _greville(spline) -> np.ndarray:
    """The coefficients of f(x) = x in the spline's B-spline basis."""
    t, d = spline._knots, spline.degree + 1
    return np.array([t[j + 1 : j + d].mean() for j in range(spline._n_basis)])


@pytest.mark.parametrize(
    "kwargs",
    [{"knots": UNEVEN}, {"n_knots": 8, "knot_strategy": "quantile"}],
    ids=["stated", "quantile"],
)
def test_a_straight_line_is_unpenalised_on_uneven_knots(kwargs):
    """The general penalty's null space is the polynomials of degree below m, at any spacing.

    The standard difference penalty puts 5.7 on this line with the stated knots.
    """
    if "knot_strategy" in kwargs:
        spline = Spline(kind="ps", **kwargs)
        x = np.random.default_rng(0).gamma(2.0, 1.5, 3000)
        spline.build(x)
    else:
        spline = _built(**kwargs)
    line = _greville(spline)
    penalty = spline._build_penalty_for_order(2)
    assert abs(line @ penalty @ line) <= _null_bound(spline, penalty, line)


def test_the_uniform_rule_keeps_the_standard_penalty_exactly():
    spline = _built(n_knots=8)
    np.testing.assert_array_equal(
        spline._build_penalty_for_order(2), build_difference_penalty(spline._n_basis, 2)
    )


def test_on_evenly_spaced_knots_the_general_penalty_is_the_standard_one():
    knots = np.arange(-3.0, 15.0)
    for order in (1, 2, 3):
        general = build_general_difference_penalty(knots, 3, order)
        standard = build_difference_penalty(len(knots) - 4, order)
        np.testing.assert_allclose(general, standard, rtol=0, atol=64 * u * np.abs(standard).max())


def test_a_penalty_order_above_the_degree_keeps_the_standard_penalty():
    spline = _built(knots=UNEVEN, m=4, degree=3)
    np.testing.assert_array_equal(
        spline._build_penalty_for_order(4), build_difference_penalty(spline._n_basis, 4)
    )


def test_the_most_smoothed_fit_on_uneven_knots_is_a_straight_line():
    """Li and Cao's figure 4: the standard penalty's limit bends where the knots crowd.

    The fit minimises ``RSS / (2n) + lambda/2 beta' P beta`` (the Hessian
    ``compute_R_inv`` factors), so comparing it with the constant fit gives
    ``lambda beta' P beta <= TSS / n``, and the part of beta outside the
    penalty's null space has norm at most ``sqrt(TSS / (n lambda sigma))``,
    ``sigma`` the smallest nonzero eigenvalue of ``P``. The null-space part is
    a line, whose second differences on an even grid vanish, and B-splines
    are a nonnegative partition of unity, so a second difference of the fit
    is at most four times that norm. The standard penalty leaves 2.6% of the
    range as curvature at this lambda.
    """
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 10.0, 2000)
    y = np.sin(x) + 0.1 * rng.standard_normal(x.size)
    lam = 1e11
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=lam,
        features={"x": Spline(kind="ps", knots=UNEVEN)},
    ).fit(pd.DataFrame({"x": x}), y)
    fitted = model.predict(pd.DataFrame({"x": np.linspace(0.5, 9.5, 50)}))
    spline = Spline(kind="ps", knots=UNEVEN)
    spline.build(x)
    sigma = np.linalg.eigvalsh(spline._build_penalty_for_order(2))[2]
    tss = np.sum((y - y.mean()) ** 2)
    assert np.abs(np.diff(fitted, 2)).max() <= 4 * np.sqrt(tss / (x.size * lam * sigma))


CLUSTER_X = np.r_[np.linspace(0.0, 1.0, 300), np.linspace(0.5, 0.5001, 300)]
CLUSTER_KNOTS = np.linspace(0.5, 0.5001, 8)


def test_knots_clustered_past_float64_keep_the_null_space_and_the_rank():
    """Li and Cao's rows reach (hbar/h)**4 = 2e16 here, and their Gram's null
    direction rounded to -5 (the correctly rounded exact Gram's to -3).

    The equilibrated rows keep the line unpenalised and REML's rank rule
    finds the penalty's true rank of n_basis - 2.
    """
    spline = Spline(kind="ps", knots=CLUSTER_KNOTS)
    spline.build(CLUSTER_X)
    penalty = spline._build_penalty_for_order(2)
    line = _greville(spline)
    assert abs(line @ penalty @ line) <= _null_bound(spline, penalty, line)
    eigenvalues = np.linalg.eigvalsh(penalty)
    ranked = np.count_nonzero(eigenvalues > np.finfo(np.float64).eps ** (2 / 3) * eigenvalues[-1])
    assert ranked == spline._n_basis - 2


@pytest.mark.parametrize("fit", ["fit", "fit_reml"])
def test_a_fit_on_clustered_stated_knots_is_the_line(fit):
    """Both fits raised LinAlgError: the null direction's negative penalty
    broke the reparametrisation's Cholesky factor.

    The line lies in the penalty's null space and the basis's span, so the
    penalised fit is the line; the standard penalty bends it by 0.27 of its
    range at lambda = 100. The normal equations lose at most half the digits,
    so the fit is the line to sqrt(u) of its range.
    """
    y = 2.0 * CLUSTER_X
    frame = pd.DataFrame({"x": CLUSTER_X})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(kind="ps", knots=CLUSTER_KNOTS)},
        **({"spline_penalty": 100.0} if fit == "fit" else {}),
    )
    getattr(model, fit)(frame, y)
    assert np.abs(model.predict(frame) - y).max() <= np.sqrt(u) * np.ptp(y)


def _skewed(sigma: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0.0, sigma, n)
    return x, rng


def _reml_ranks(model) -> dict[str, float]:
    from superglm.model.reml_setup import collect_reml_groups
    from superglm.reml.penalty_algebra import build_penalty_components

    matrices = model._dm.group_matrices
    components = build_penalty_components(matrices, collect_reml_groups(model._groups, matrices))
    return {component.name: component.rank for component in components}


@pytest.mark.parametrize("select", [False, True], ids=["plain", "select"])
def test_reml_ranks_skewed_quantile_knots_at_the_penalty_true_rank(select):
    """quantile_rows on lognormal(0, 1.5): Li and Cao's rows run from 1e-4 to 1e10.

    REML ranks a penalty at eps**(2/3) of its largest eigenvalue, which put the
    two tail directions under the cut (rank 10 of 12) while the fit still
    penalised them. The penalty REML sees has rank n_basis - 2, the
    polynomials of degree below 2 being its null space.
    """
    x, rng = _skewed(1.5, 10_000, 0)
    y = rng.poisson(np.exp(-1.0 + 0.3 * np.sin(np.log(x))))
    spline = Spline(kind="ps", n_knots=10, knot_strategy="quantile_rows", select=select)
    model = SuperGLM(family="poisson", features={"x": spline}).fit_reml(pd.DataFrame({"x": x}), y)
    ranks = _reml_ranks(model)
    wiggle = ranks["x:wiggle"] if select else ranks["x"]
    assert wiggle == model._specs["x"]._n_basis - 2


@pytest.mark.parametrize("fit", ["fit", "fit_reml"])
def test_a_third_order_penalty_fits_on_lognormal_quantile_knots(fit):
    """m = 3 on lognormal(0, 2) quantile knots spans (hbar/span)**6, past float64;
    the Cholesky factor of the reparametrisation raised LinAlgError."""
    x, rng = _skewed(2.0, 5_000, 7)
    y = rng.poisson(np.exp(-1.0 + 0.1 * np.log(x)))
    frame = pd.DataFrame({"x": x})
    spline = Spline(kind="ps", n_knots=20, knot_strategy="quantile", m=3)
    model = SuperGLM(family="poisson", features={"x": spline})
    getattr(model, fit)(frame, y)
    assert np.all(np.isfinite(model.predict(frame)))


@pytest.mark.parametrize("sigma", [1.0, 1.5])
def test_a_decomposed_tensor_with_a_skewed_quantile_margin_splits_off_the_bilinear(sigma):
    """The tensor split counted null eigenvalues under a fixed 1e-8 of the largest,
    which the margin's spread crossed (2 to 24 of them), and on lognormal(0, 1) the
    margin's 1e6 entries broke the component sum's absolute check."""
    from superglm.features.interaction import TensorInteraction

    x1, rng = _skewed(sigma, 4_000, 5)
    x2 = rng.uniform(0.0, 1.0, x1.size)
    margin_1 = Spline(kind="ps", n_knots=5, knot_strategy="quantile")
    margin_2 = Spline(kind="ps", n_knots=5)
    margin_1.build(x1)
    margin_2.build(x2)
    infos = TensorInteraction("a", "b", decompose=True).build(
        x1, x2, {"a": margin_1, "b": margin_2}
    )
    assert [info.subgroup_name for info in infos] == ["bilinear", "wiggly"]


@pytest.mark.slow
def test_a_tensor_with_a_skewed_quantile_margin_fits_by_reml():
    """It raised PenaltyNumericalError: the reference root could not meet its accuracy contract."""
    x1, rng = _skewed(1.5, 4_000, 1)
    x2 = rng.uniform(0.0, 1.0, x1.size)
    y = rng.poisson(np.exp(-1.0 + 0.2 * np.log(x1) * x2))
    frame = pd.DataFrame({"a": x1, "b": x2})
    model = SuperGLM(
        family="poisson",
        features={
            "a": Spline(kind="ps", n_knots=10, knot_strategy="quantile"),
            "b": Spline(kind="ps", n_knots=5),
        },
        interactions=[("a", "b")],
    ).fit_reml(frame, y)
    assert np.all(np.isfinite(model.predict(frame)))


def test_knots_too_close_for_float64_get_a_finite_penalty():
    """hbar / span reaches 1e159 here and its fourth power overflows: the penalty
    had inf and NaN entries. The projected standard factor keeps the line."""
    knots = np.r_[-0.2, np.arange(4) * 1e-160, 0.2]
    x = np.linspace(-0.5, 0.5, 500)
    spline = Spline(kind="ps", knots=knots)
    spline.build(x)
    penalty = spline._build_penalty_for_order(2)
    assert np.isfinite(penalty).all()
    line = _greville(spline)
    assert abs(line @ penalty @ line) <= _null_bound(spline, penalty, line)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"x": Spline(kind="ps", knots=knots)}
    ).fit_reml(frame, np.sin(3.0 * x))
    assert np.all(np.isfinite(model.predict(frame)))


def test_stated_knots_at_the_uniform_rule_positions_reproduce_the_uniform_fit():
    """``fitted_knots`` with ``fitted_boundary`` is the documented way to reproduce a
    fit; the open knot vector's widened ends made the general penalty differ by 2%."""
    x = np.random.default_rng(0).uniform(0.0, 10.0, 5_000)
    placed = Spline(kind="ps", n_knots=8)
    placed.build(x)
    stated = Spline(kind="ps", knots=placed.fitted_knots, boundary=placed.fitted_boundary)
    stated.build(x)
    np.testing.assert_array_equal(stated._build_penalty(), placed._build_penalty())
