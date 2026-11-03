"""Binomial/identity fits stay inside the binomial mean space.

The identity link maps the line onto itself, so the parameter space is
``{beta : 0 < X beta + offset < 1}`` and the maximum can lie on its boundary
(Donoghoe & Marschner 2014, Comput. Stat. Data Anal. 80).  Step halving keeps
every trial inside, as R's glm and glm2 halve a step with an invalid mean
(Marschner 2011, R Journal 3(2)), and a state holding rows at a clipped
probability is reported as not converged, never refused.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint, minimize
from scipy.special import xlogy

from superglm import (
    BSplineSmooth,
    Categorical,
    Constraint,
    ConvergenceWarning,
    Numeric,
    Spline,
    SuperGLM,
)
from superglm.links import IdentityLink


def _eta(model):
    return model._dm.matvec(model.result.beta) + model.result.intercept


def _constrained_mle(x, y):
    """Fitted probabilities of ``a + b x`` maximising the binomial likelihood inside (0, 1)."""
    design = np.column_stack([np.ones(len(x)), x])

    def negative(beta):
        mu = design @ beta
        return -np.mean(xlogy(y, mu) + xlogy(1.0 - y, 1.0 - mu))

    def gradient(beta):
        mu = design @ beta
        return design.T @ ((1.0 - y) / (1.0 - mu) - y / mu) / len(y)

    result = minimize(
        negative,
        np.array([0.5, 0.0]),
        jac=gradient,
        constraints=LinearConstraint(design, 1e-12, 1.0 - 1e-12),
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert result.success, result.message
    return design @ result.x


@pytest.mark.parametrize(
    ("direct_solve", "selection_penalty"),
    [("auto", 0.0), ("qr", 0.0), ("auto", 0.01)],
    ids=["direct", "qr", "selection"],
)
def test_a_maximum_on_the_boundary_is_feasible_and_not_converged(direct_solve, selection_penalty):
    # The constrained maximum of y = (0, 0, 1, 1) on x = (0, 1/3, 2/3, 1) is
    # p = x: rows 0 and 3 sit on the boundary, where the score does not vanish.
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 4)})
    model = SuperGLM(
        family="binomial",
        link="identity",
        selection_penalty=selection_penalty,
        direct_solve=direct_solve,
        features={"x": Numeric()},
    )
    with pytest.warns(ConvergenceWarning, match="probability of 0 or 1"):
        model.fit(X, np.array([0.0, 0.0, 1.0, 1.0]))
    assert not model.result.converged
    assert model.result.termination_reason == "mean_space_boundary"
    assert model.result.mean_space_boundary_rows == 2
    eta = _eta(model)
    assert np.all((eta > 0.0) & (eta < 1.0))
    np.testing.assert_allclose(model.predict(X), X["x"].to_numpy(), atol=1e-6)


@pytest.mark.parametrize(("seed", "interior"), [(0, False), (2, True)])
def test_a_fit_reaches_the_constrained_maximum_inside_the_space(seed, interior):
    # p = x puts the maximum on the boundary for seed 0 and just inside it for seed 2
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, 2000)
    y = rng.binomial(1, x).astype(float)
    X = pd.DataFrame({"x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SuperGLM(
            family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
        ).fit(X, y)
    assert model.result.converged is interior
    eta = _eta(model)
    assert np.all((eta > 0.0) & (eta < 1.0))
    # the default deviance tolerance, 1e-6, leaves the probabilities about 2e-4 short
    np.testing.assert_allclose(model.predict(X), _constrained_mle(x, y), atol=1e-3)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_reml_returns_a_boundary_maximum_disclosed_rather_than_refused(discrete):
    # Level c's responses are all 0, so every smoothing parameter leaves its
    # lowest row on the boundary.
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, 1500)
    g = rng.choice(["a", "b", "c"], 1500)
    y = rng.binomial(1, 0.3 + 0.3 * x).astype(float)
    y[g == "c"] = 0.0
    model = SuperGLM(
        family="binomial",
        link="identity",
        discrete=discrete,
        features={"x": Spline(n_knots=8), "g": Categorical()},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(pd.DataFrame({"x": x, "g": g}), y)
    assert not model.result.converged
    assert model.result.termination_reason == "mean_space_boundary"
    assert np.all((_eta(model) > 0.0) & (_eta(model) < 1.0))
    assert any(
        issubclass(item.category, ConvergenceWarning) and "boundary" in str(item.message)
        for item in caught
    )


class _ShiftedIdentity(IdentityLink):
    """``link(mu) = mu - 1``, so the probability is ``eta + 1``."""

    def link(self, mu):
        return mu - 1.0

    def inverse(self, eta):
        return eta + 1.0


def test_a_shifted_identity_link_reads_the_boundary_on_its_mean():
    # The optimum predicts 0.5 on every row, an interior maximum, at eta = -0.5.
    X = pd.DataFrame({"x": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]})
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    model = SuperGLM(
        family="binomial", link=_ShiftedIdentity(), selection_penalty=0, features={"x": Numeric()}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X, y)
    np.testing.assert_allclose(model.predict(X), 0.5, atol=1e-8)
    assert model.result.converged
    assert model.result.mean_space_boundary_rows == 0


def _forced_end_of_fit_qp_failure(monkeypatch, stop):
    """The retained mode fails its shape constraint; the steps before it are untouched."""
    import sys

    from superglm.solvers.irls_direct._qp import _IRLSDirectQPStep

    if stop == "constraint_kkt_incomplete":
        monkeypatch.setattr(
            _IRLSDirectQPStep, "retained_converged", property(lambda self: False, lambda *_: None)
        )
    else:
        feasible = _IRLSDirectQPStep.feasible
        monkeypatch.setattr(
            _IRLSDirectQPStep,
            "feasible",
            lambda self, beta: (
                sys._getframe(1).f_code.co_name != "_finish" and feasible(self, beta)
            ),
        )


@pytest.mark.parametrize("stop", ["constraint_infeasible", "constraint_kkt_incomplete"])
def test_a_constraint_failure_on_the_boundary_keeps_its_own_stop(monkeypatch, stop):
    # Level c's responses are all 0, so the mode holds rows on the boundary;
    # a broken shape constraint there is named as such, not as a boundary maximum.
    from superglm.reml.observed_geometry import ObservedModeNotConvergedError

    _forced_end_of_fit_qp_failure(monkeypatch, stop)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, 1500)
    g = rng.choice(["a", "b", "c"], 1500)
    y = rng.binomial(1, 0.3 + 0.3 * x).astype(float)
    y[g == "c"] = 0.0
    model = SuperGLM(
        family="binomial",
        link="identity",
        features={
            "x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing),
            "g": Categorical(),
        },
    )
    X = pd.DataFrame({"x": x, "g": g})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if stop == "constraint_infeasible":
            with pytest.raises(ObservedModeNotConvergedError, match="infeasible"):
                model.fit_reml(X, y)
            return
        model.fit_reml(X, y)
    assert not model.result.converged
    assert model.result.termination_reason == stop


def test_a_start_spanning_past_one_is_centred_inside_the_space():
    # The default start, the mean response 0.455 plus the offset, spans
    # (0.455, 1.255); aligning its smallest row with the mean leaves it there,
    # while a span of 0.8 fits inside (0, 1).
    rng = np.random.default_rng(5)
    offset = rng.uniform(0.0, 0.8, 2000)
    x = rng.uniform(0.0, 1.0, 2000)
    y = rng.binomial(1, 0.02 + offset + 0.1 * x).astype(float)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X, y, offset=offset)
    assert model.result.converged
    mean = model.predict(X, offset=offset)
    assert np.all((mean > 0.0) & (mean < 1.0))
