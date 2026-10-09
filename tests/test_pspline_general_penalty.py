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
    scale = np.abs(line) @ np.abs(penalty) @ np.abs(line)
    assert abs(line @ penalty @ line) <= 8 * spline._n_basis * u * scale


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

    At lambda the fit sits within O(||X'X|| / lambda) of the null-space fit, a
    line here; the standard penalty leaves 2.6% of the range as curvature.
    """
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 10.0, 2000)
    y = np.sin(x) + 0.1 * rng.standard_normal(x.size)
    lam = 1e9
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=lam,
        features={"x": Spline(kind="ps", knots=UNEVEN)},
    ).fit(pd.DataFrame({"x": x}), y)
    fitted = model.predict(pd.DataFrame({"x": np.linspace(0.5, 9.5, 50)}))
    curvature = np.abs(np.diff(fitted, 2)).max() / np.ptp(fitted)
    assert curvature <= 10 * x.size / lam
