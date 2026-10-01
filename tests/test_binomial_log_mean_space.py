"""Binomial/log fits stay inside the binomial mean space.

The log link's inverse ``exp(eta)`` covers ``(0, inf)``, wider than the
binomial probabilities: a row with ``eta >= 0`` has no binomial likelihood, so
the parameter space is ``{beta : X beta + offset < 0}`` and the maximum can lie
on its boundary (Wacholder 1986; Marschner & Gillett 2012).  PIRLS rejects a
trial that leaves the space, as R's glm and glm2 step-halving do (Donoghoe &
Marschner 2018, J. Stat. Softw. 86(9), section 3.2), rather than letting
``clip_mu`` rewrite its means; a state holding rows at the boundary is never a
converged mode; and REML, whose Laplace approximation needs an interior
stationary maximum (Wood, Pya & Saefken 2016, section 3.1), restores an interior
start or says why none exists.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, LambdaPolicy, Numeric, RandomEffect, SuperGLM
from superglm.diagnostics.separation import SeparationWarning
from superglm.distributions import Binomial, Gamma, Poisson
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import CauchitLink, CloglogLink, LogitLink, LogLink, ProbitLink
from superglm.reml.observed_geometry import ObservedModeNotConvergedError
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.irls_state import mean_space_boundary_rows, mean_space_violation
from superglm.solvers.mode_score import mode_certification_bar
from superglm.types import GroupSlice, LinearConstraintSet

_U = 2.0**-53  # unit roundoff, and the spacing of float64 just below one


def _thin_event_levels() -> tuple[pd.DataFrame, np.ndarray]:
    """A random effect beside a slope, plus four one-row levels whose row is an event.

    An event row's log-likelihood ``eta`` is linear, so its level's coefficient
    at a penalty ``lambda`` is ``1 / lambda`` and the row's mode is interior only
    while ``1 / lambda`` is below its distance ``|eta_0| ~ 2.3`` to the boundary:
    the near-unpenalized REML bootstrap (``lambda = 1e-4``) puts it there.
    """
    rng = np.random.default_rng(7)
    n, levels = 1500, 60
    g = rng.integers(0, levels, n)
    x = rng.uniform(size=n)
    eta = -2.3 + 0.4 * x + rng.normal(0.0, 0.3, levels)[g]
    y = (rng.uniform(size=n) < np.exp(eta)).astype(float)
    extra = pd.DataFrame({"x": rng.uniform(size=4), "g": [f"s{j}" for j in range(4)]})
    frame = pd.concat(
        [pd.DataFrame({"x": x, "g": [f"g{c:03d}" for c in g]}), extra], ignore_index=True
    )
    return frame, np.concatenate([y, np.ones(4)])


def _fit(direct_solve: str) -> SuperGLM:
    X, y = _thin_event_levels()
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve=direct_solve,
        features={"x": Numeric(), "g": RandomEffect()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit_reml(X, y)
    return model


def _eta(model: SuperGLM) -> np.ndarray:
    return model._dm.matvec(model.result.beta) + model.result.intercept


def test_only_a_link_that_leaves_the_mean_space_declares_a_boundary() -> None:
    """Declared by family and link: the binomial links that stay in ``(0, 1)`` declare none."""
    contained = [
        (Binomial(), LogitLink()),
        (Binomial(), ProbitLink()),
        (Binomial(), CloglogLink()),
        (Binomial(), CauchitLink()),
        (Poisson(), LogLink()),
        (Gamma(), LogLink()),
    ]
    for family, link in contained:
        assert mean_space_violation(family, link) is None
        assert mean_space_boundary_rows(family, link, np.array([5.0]), np.ones(1)) == 0
    violates = mean_space_violation(Binomial(), LogLink())
    assert violates is not None
    # eta = 0 is mu = 1: outside, unless the row carries no likelihood
    assert violates(np.array([0.0]), np.ones(1))
    assert not violates(np.array([-1e-300, 0.0]), np.array([1.0, 0.0]))
    # the boundary rows are the rows clip_mu caps, exp(eta) >= 1 - 1e-7
    cap = np.log1p(-1e-7)
    eta = np.array([cap, np.nextafter(cap, -np.inf), 0.0])
    assert mean_space_boundary_rows(Binomial(), LogLink(), eta, np.array([1.0, 1.0, 0.0])) == 1


def test_a_near_unpenalized_fit_stops_at_the_boundary_and_says_so() -> None:
    """At the bootstrap's ``lambda = 1e-4`` the event rows' maximum is on the boundary.

    Every accepted state keeps ``eta < 0`` on the rows with weight, and the
    state the fit returns is reported as the constrained point it is, not as a
    converged mode.
    """
    model = _fit("gram")
    X, y = _thin_event_levels()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=np.ones(len(y)),
            family=Binomial(),
            link=LogLink(),
            groups=model._groups,
            lambda2={name: 1e-4 for name in model._reml_result.lambdas},
            reml_penalties=model._reml_penalties,
            direct_solve="gram",
            weight_semantics="prior",
        )
    eta = model._dm.matvec(result.beta) + result.intercept
    assert float(eta.max()) < 0.0
    assert not result.converged
    assert result.termination_reason == "mean_space_boundary"
    assert result.mean_space_boundary_rows == int(np.count_nonzero(eta >= np.log1p(-1e-7)))
    assert result.mean_space_boundary_rows >= 1


def test_reml_reaches_the_interior_maximum_on_both_backends() -> None:
    """The REML optimum's mode is interior: REML reaches it, certified, on structured and gram.

    The two backends stop within the REML stopping rule ``reml_tol (1 + |V|)``
    of the same optimum, so their objectives agree within twice it.
    """
    fits = {direct_solve: _fit(direct_solve) for direct_solve in ("auto", "gram")}
    for model in fits.values():
        assert model._reml_result.converged
        assert model.result.converged
        assert float(_eta(model).max()) < 0.0
    objectives = [float(model._reml_result.objective) for model in fits.values()]
    tolerance = float(fits["auto"]._reml_profile["reml_tol_resolved"])
    assert abs(objectives[0] - objectives[1]) <= 2.0 * tolerance * (1.0 + abs(objectives[0]))


def test_reml_says_why_a_maximum_on_the_boundary_has_no_criterion() -> None:
    """An unpenalized level whose rows are all events is at probability one at every lambda."""
    rng = np.random.default_rng(3)
    n = 3000
    region = rng.choice(["A", "B", "C"], n)
    p = np.where(region == "A", 0.30, np.where(region == "B", 0.10, 1.0))
    y = (rng.random(n) < p).astype(np.float64)
    X = pd.DataFrame(
        {
            "region": region,
            "x": rng.uniform(size=n),
            "g": [f"g{c}" for c in rng.integers(0, 40, n)],
        }
    )
    model = SuperGLM(
        family="binomial",
        link="log",
        features={"region": Categorical(base="first"), "x": Numeric(), "g": RandomEffect()},
    )
    with pytest.raises(ObservedModeNotConvergedError, match="no interior maximum"):
        model.fit_reml(X, y)


def test_a_constraint_failure_keeps_its_reason_at_the_boundary() -> None:
    """The boundary does not overwrite a constrained solve's own failure (#431, P3).

    Ten event rows started ``1e-8`` from the boundary under ``beta >= 0``: the
    one permitted iteration is halved, so the inner QP's KKT certificate is
    incomplete, and the rows end inside the ``1 - 1e-7`` boundary band.  Both
    are reported, the constraint failure as the reason.
    """
    x = np.linspace(0.1, 1.0, 10)[:, None]
    groups = [
        GroupSlice(
            "x",
            0,
            1,
            constraints=LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1)),
            monotone_engine="qp",
        )
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = fit_irls_direct(
            X=DesignMatrix([DenseGroupMatrix(x)], n=len(x), p=1),
            y=np.ones(len(x)),
            weights=np.ones(len(x)),
            family=Binomial(),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            beta_init=np.zeros(1),
            intercept_init=-1e-8,
            max_iter=1,
            direct_solve="gram",
            weight_semantics="prior",
        )
    assert not result.converged
    assert result.termination_reason == "constraint_kkt_incomplete"
    assert result.mean_space_boundary_rows == len(x)


def _offset_fit(offset: float, *, plain: bool, direct_solve: str) -> SuperGLM:
    """Sol's #431 (c) fixture: an interior optimum, probabilities 0.16 to 0.69."""
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(-1.0, 1.0, n)
    g = np.tile(np.arange(30), 10)
    y = (rng.random(n) < 0.4).astype(np.float64)
    X = pd.DataFrame({"x": x, "g": [f"g{c:02d}" for c in g]})
    level = (
        Categorical(base="first") if plain else RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))
    )
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve=direct_solve,
        selection_penalty=0.0,
        features={"x": Numeric(), "g": level},
    )
    if plain:
        model.fit(X, y, offset=np.full(n, offset))
    else:
        model.fit_reml(X, y, offset=np.full(n, offset))
    return model


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_positive_offset_starts_inside_the_mean_space(direct_solve: str) -> None:
    """A constant offset of +2 moves the intercept by -2, not the fit to the boundary (#431 c).

    The default intercept is chosen before the offset, so every row started
    at ``eta >= 0``, outside the space, and the fit ended "no interior
    maximum" (fit_reml) or unconverged at the boundary (fit).  The start is
    now lowered into the space, and the offset fit is a certified mode of the
    same problem: its REML criterion agrees with the no-offset fit's within
    the error the certificate allows each, ``reml_tol (1 + |V|)``, and a plain
    fit's deviance within the deviance stop's own ``tol |D|`` each.
    """
    for plain in (False, True):
        base = _offset_fit(0.0, plain=plain, direct_solve=direct_solve)
        shifted = _offset_fit(2.0, plain=plain, direct_solve=direct_solve)
        assert shifted.result.converged
        assert shifted.result.termination_reason == "converged"
        eta = shifted._dm.matvec(shifted.result.beta) + shifted.result.intercept + 2.0
        assert float(eta.max()) < 0.0
        if plain:
            deviances = [float(base.result.deviance), float(shifted.result.deviance)]
            bound = 2.0 * base._tol * max(abs(d) for d in deviances)
            assert abs(deviances[1] - deviances[0]) <= bound
        else:
            objectives = [float(base._reml_result.objective), float(shifted._reml_result.objective)]
            tolerance = float(base._reml_profile["reml_tol_resolved"])
            assert abs(objectives[1] - objectives[0]) <= 2.0 * tolerance * (
                1.0 + abs(objectives[0])
            )


def _frequency_weighted_events(*, plain: bool, direct_solve: str) -> SuperGLM:
    """Sol's #431 (d) fixture: per level, nine events of weight 1e7 and a non-event of weight 1."""
    n = 20
    g = np.tile(np.arange(2), 10)
    y = np.ones(n)
    y[:2] = 0.0
    weights = np.where(y > 0.0, 1e7, 1.0)
    level = (
        Categorical(base="first") if plain else RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))
    )
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve=direct_solve,
        selection_penalty=0.0,
        weight_semantics="frequency",
        features={"g": level},
    )
    X = pd.DataFrame({"g": [f"g{c}" for c in g]})
    if plain:
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit_reml(X, y, sample_weight=weights)
    return model


@pytest.mark.xfail(
    strict=True,
    reason="#431 (d): the 1 - 1e-7 clip is still the boundary; redone in #431",
)
@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_an_interior_maximum_above_the_old_clip_is_reached(direct_solve: str) -> None:
    """``p = 9e7 / (9e7 + 1)`` is interior (#431 d), and the fit should reach it.

    Not yet: the boundary rule still treats every probability at or above
    ``1 - 1e-7`` as the boundary.  Kept as the target of the #431 redo.

    Both levels hold the same data, so the penalized maximum has a zero
    random effect and an intercept solving the score equation
    ``S(eta) = 18e7 - 2 p / (1 - p) = 0``: ``eta* = -log1p(1 / 9e7)``, above
    the clip's ``log1p(-1e-7)``, which calls it the boundary.  The
    certified mode has ``|S| <= bar sum |s|`` with ``sum |s| = 18e7 + 2 p /
    (1 - p)`` and ``|S'| = 2 p / (1 - p)^2``, so it lies within ``bar sum |s|
    / |S'|`` of ``eta*``.  The score is evaluated at ``p = fl(exp(eta))``,
    whose ``1 - p`` is exact but carries ``p``'s rounding, so the computed
    root can sit two units of ``2^-53`` (``exp`` and the rounding of ``p``)
    from the exact one; four allow for ``exp``'s own error.
    """
    model = _frequency_weighted_events(plain=False, direct_solve=direct_solve)
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    eta_star = -np.log1p(1.0 / 9e7)
    odds = 9e7
    total = 18e7 + 2.0 * odds
    slope = 2.0 * odds * (odds + 1.0)
    bar = mode_certification_bar(float(model._reml_profile["reml_tol_resolved"]))
    assert float(np.max(np.abs(eta - eta_star))) <= bar * total / slope + 4.0 * _U
    # a plain fit stops on its deviance test, not the certificate, and is
    # checked for the verdict alone: an interior mode, not the boundary
    plain = _frequency_weighted_events(plain=True, direct_solve=direct_solve)
    assert plain.result.converged
    assert plain.result.termination_reason == "converged"
    eta = plain._dm.matvec(plain.result.beta) + plain.result.intercept
    assert float(np.max(np.abs(eta - eta_star))) < 0.5 * abs(eta_star)
