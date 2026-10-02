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

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Categorical,
    Constraint,
    LambdaPolicy,
    Numeric,
    PSpline,
    RandomEffect,
    SuperGLM,
)
from superglm.diagnostics.separation import SeparationWarning
from superglm.distributions import Binomial, Gamma, Poisson
from superglm.links import CauchitLink, CloglogLink, LogitLink, LogLink, ProbitLink
from superglm.model.input_validation import FractionalFrequencyWeightWarning
from superglm.reml.observed_geometry import ObservedModeNotConvergedError
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.irls_state import (
    _mean_space_halving_budget,
    mean_space_boundary_rows,
    mean_space_clipped_rows,
    mean_space_deviance_delta,
    mean_space_newton_rows,
    mean_space_score_rows,
    mean_space_violation,
)
from superglm.solvers.mode_score import (
    MODE_CERTIFICATION_BAR,
    mode_certification_bar,
    weighted_column_centring,
)

_U = 2.0**-53  # unit roundoff, and the spacing of float64 just below one
_EPS = 2.0**-52  # machine epsilon, the spacing of float64 above one


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


def _two_level_fit(offset_a: float, offset_b: float, direct_solve: str) -> SuperGLM:
    """Two Categorical levels, one event and one non-event each: both MLE probabilities 1/2."""
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"g": Categorical(base="first")},
    )
    model.fit(
        pd.DataFrame({"g": ["a", "a", "b", "b"]}),
        np.array([0.0, 1.0, 0.0, 1.0]),
        offset=np.array([offset_a, offset_a, offset_b, offset_b]),
        record_diagnostics=True,
    )
    return model


def _assert_at_the_two_level_mle(model: SuperGLM, offset_a: float, offset_b: float) -> None:
    """Converged, with both levels at ``log(1/2)`` within what the deviance stop resolves.

    The deviance stop is ``|D - D_prev| < tol (|D_prev| + 1)``
    (``_irls_objective_relative_change``).  Each level's deviance ``-2 eta - 2
    log(1 - e^eta)`` has curvature 4 at its maximum, where Fisher scoring's
    curvature equals the observed one (``2p/(1-p) = p/(1-p)^2`` at ``p =
    1/2``), so the iteration contracts at least by half near it and the excess
    left after the stop is at most the last change.  A level ``delta`` from its
    maximum carries an excess ``2 delta^2``, so ``delta <= sqrt(tol (|D_prev|
    + 1) / 2)``, within ``sqrt(tol |D|)`` at ``|D| = 8 log 2 >= 1``.
    """
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    offset = np.array([offset_a, offset_a, offset_b, offset_b])
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    bound = np.sqrt(model._tol * abs(float(model.result.deviance)))
    assert float(np.max(np.abs(eta - np.log(0.5)))) <= bound


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_lowered_start_reaches_a_trial_inside_the_space(direct_solve: str) -> None:
    """Sol's #437 fixture: the first step needs more than the ordinary 20 halvings.

    With offsets -16 and +1.3 the lowered start puts level a at eta ~ -18 and
    the Fisher proposal moves it by ~3e7, so only a fraction below ~6e-7 of
    the step stays inside ``eta < 0``: past 2^-20.  Backtracking reaches the
    fraction to the boundary instead of rejecting the step, and the fit
    reaches the maximum.
    """
    model = _two_level_fit(-16.0, 1.3, direct_solve)
    assert model.result.iteration_log[0].step_halvings > 20
    _assert_at_the_two_level_mle(model, -16.0, 1.3)


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
@pytest.mark.parametrize("depth", range(20, 41))
def test_a_level_below_the_clip_floor_is_not_a_converged_fit(direct_solve: str, depth: int) -> None:
    """Sol's #437 fixture over offsets -20 to -40: no stop on the flat clipped deviance.

    The lowered start puts level a below ``clip_mu``'s floor ``1e-7``, where
    the clipped deviance is flat in ``eta``.  A step that leaves the level
    there changed nothing the deviance stop can see, which reported a wrong
    fit as converged at -35 and -40 (probabilities ``1e-7`` and 1/2,
    deviance 35.0088).  A stop is accepted only where the binomial/log score
    certifies it, so the fit goes on to the maximum.  Master returns these
    fits unconverged from an infeasible start, with a "probability" of 1.83.
    """
    model = _two_level_fit(-float(depth), 1.3, direct_solve)
    _assert_at_the_two_level_mle(model, -float(depth), 1.3)


def _three_row_fit(y, offset, weights, direct_solve: str) -> SuperGLM:
    """Sol's #437 fixture: two levels of three rows, the first with its own offset and weight."""
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        weight_semantics="frequency",
        features={"g": Categorical(base="first")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FractionalFrequencyWeightWarning)
        model.fit(
            pd.DataFrame({"g": ["a", "a", "a", "b", "b", "b"]}),
            np.tile(np.asarray(y, dtype=float), 2),
            offset=np.tile(np.asarray(offset, dtype=float), 2),
            sample_weight=np.tile(np.asarray(weights, dtype=float), 2),
        )
    return model


def _common_rows(model: SuperGLM, offset) -> np.ndarray:
    """Each level's probability on its two rows at the common offset."""
    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    return np.exp(eta + np.tile(np.asarray(offset, dtype=float), 2))[[1, 4]]


# Both three-row fixtures have a common-row probability near 1/2 at their
# maximum, where the six rows' scores sum in size to under 4.001 and each
# level's score falls by more than 1.99 per unit of its shift (the common
# non-event's observed curvature p / (1 - p)^2 = 2).  The certificate holds the
# intercept's score within bar sum|s| and level b's centred score within
# bar sum|s| / 2, so each level's score is within 2.5 bar sum|s| and its log
# probability within 2.5 bar 4.001 / 1.99 < 5.1 bar of the maximum.
_THREE_ROW_LOG_P_BOUND = 5.1 * MODE_CERTIFICATION_BAR


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_clipped_non_event_row_does_not_certify_a_stop(direct_solve: str) -> None:
    """Sol's #437 fixture: a non-event of weight 1e14 per level at offset -30, the rest at +10.

    The heavy rows sit far below ``clip_mu``'s floor, where each adds a flat
    ``-2 w log(1 - 1e-7) ~ 2e7`` to the clipped deviance: that constant made
    the relative deviance stop pass after one step, at a common-row
    probability of 0.049.  The maximum solves the level's score equation ``1 -
    p / (1 - p) - w q / (1 - q) = 0``, ``q = p e^-40``: ``p = 0.4999468955724079``
    (Sol's 70-digit oracle).  The stop is accepted only on the binomial/log
    score, so the fit goes on to it.
    """
    offset = [-30.0, 10.0, 10.0]
    model = _three_row_fit([0, 0, 1], offset, [1e14, 1, 1], direct_solve)
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    p_star = 0.4999468955724079
    assert np.max(np.abs(np.log(_common_rows(model, offset) / p_star))) <= _THREE_ROW_LOG_P_BOUND


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_an_optimum_beside_a_negligible_event_row_is_certified(direct_solve: str) -> None:
    """Sol's #437 fixture: an event of weight 1e-12 per level at offset -20, the rest at +1.3.

    The light row sits below ``clip_mu``'s floor, and the stop refused while
    any event row did ran to max_iter at the maximum.  The certificate weighs
    the row's score ``1e-12`` against the whole score instead, and accepts
    the maximum ``p / (1 - p) = 1 + 1e-12``; the same model with every offset
    shifted by -2 starts inside and reaches the same probabilities.
    """
    p_star = (1.0 + 1e-12) / (2.0 + 1e-12)
    for offset in ([-20.0, 1.3, 1.3], [-22.0, -0.7, -0.7]):
        model = _three_row_fit([1, 0, 1], offset, [1e-12, 1, 1], direct_solve)
        assert model.result.converged
        assert model.result.termination_reason == "converged"
        p = _common_rows(model, offset)
        assert np.max(np.abs(np.log(p / p_star))) <= _THREE_ROW_LOG_P_BOUND


def _event_row_below_the_floor(shift: float, direct_solve: str):
    """claude's #437 fixture: 200 rows at offset +2 and one event row at offset -25."""
    rng = np.random.default_rng(0)
    x = np.append(rng.uniform(-1.0, 1.0, 200), 0.3)
    y = np.append((rng.uniform(size=200) < 0.2).astype(np.float64), 1.0)
    offset = np.append(np.full(200, 2.0), -25.0) + shift
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"x": Numeric()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(pd.DataFrame({"x": x}), y, offset=offset)
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    return model, x, y, eta


def _log_binomial_maximum(design: np.ndarray, y: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """The unpenalized log-binomial maximum by Newton's method on the exact score and information.

    The test's own reference: score ``y - (1 - y) e^eta / (1 - e^eta)``,
    observed information ``(1 - y) e^eta / (1 - e^eta)^2``, steps halved
    until every ``eta < 0``, from an intercept inside the space.
    """
    theta = np.zeros(design.shape[1])
    theta[0] = -1.0 - float(np.max(offset))
    for _ in range(200):
        eta = design @ theta + offset
        odds = np.exp(eta) / -np.expm1(eta)
        score = design.T @ (y - (1.0 - y) * odds)
        information = design.T @ (((1.0 - y) * odds / -np.expm1(eta))[:, None] * design)
        step = np.linalg.solve(information, score)
        fraction = 1.0
        while np.any(design @ (theta + fraction * step) + offset >= 0.0):
            fraction /= 2.0
        theta = theta + fraction * step
        if np.max(np.abs(fraction * step)) <= 4.0 * _EPS * (1.0 + np.max(np.abs(theta))):
            break
    return theta


def _certified_eta_bound(design: np.ndarray, y: np.ndarray, offset: np.ndarray, theta) -> float:
    """How far a certified fit's eta can sit from the maximum ``theta``.

    The certificate holds the intercept's score within ``bar sum|s|`` and a
    centred slope's within ``bar`` times its scale, at most ``2 sum|s|`` for a
    column in ``[-1, 1]``; back in raw coordinates each score is within ``4
    bar sum|s|``.  The coefficients then move by at most ``|g| /
    lambda_min`` of the observed information, and each row's eta by ``|(1,
    x)|`` times that: ``sqrt(k) sqrt(k) 4 bar sum|s| / lambda_min`` for ``k``
    coefficients, doubled for the curvature's change along the way.
    """
    eta = design @ theta + offset
    odds = np.exp(eta) / -np.expm1(eta)
    score_size = float(np.sum(np.abs(y - (1.0 - y) * odds)))
    information = design.T @ (((1.0 - y) * odds / -np.expm1(eta))[:, None] * design)
    smallest = float(np.linalg.eigvalsh(information)[0])
    k = design.shape[1]
    return 2.0 * k * 4.0 * MODE_CERTIFICATION_BAR * score_size / smallest


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_an_event_row_below_the_floor_reaches_the_models_maximum(direct_solve: str) -> None:
    """claude's #437 fixture: the event row's score is ``w y = 1`` wherever its mean is.

    Below ``clip_mu``'s floor Fisher scoring weights the row by ``mu'^2 /
    V(1e-7)`` and credits it with ``exp(eta) / 1e-7`` (about 4e-6) of its
    score, so scoring settles on the maximum without the row, 0.13 standard
    errors from the model's.  Master certified that state on both offsets;
    dbeaf5a3 refused both and ended ``score_stagnated``.  The refused,
    stalled stop now hands the iteration to Newton's method on the observed
    information and the true score, its steps judged on the true deviance,
    and both fits reach the model's maximum: certified, and within the
    certificate's bound of a direct Newton solve of the exact score.
    """
    rng = np.random.default_rng(0)
    x = np.append(rng.uniform(-1.0, 1.0, 200), 0.3)
    design = np.column_stack([np.ones_like(x), x])
    etas = []
    bound = 0.0
    for shift in (0.0, -1.0):
        model, _, y, eta = _event_row_below_the_floor(shift, direct_solve)
        assert model.result.converged
        assert model.result.termination_reason == "converged"
        offset = np.append(np.full(200, 2.0), -25.0) + shift
        theta = _log_binomial_maximum(design, y, offset)
        bound = _certified_eta_bound(design, y, offset, theta)
        assert np.max(np.abs(eta - (design @ theta + offset))) <= bound
        etas.append(eta)
    assert np.max(np.abs(etas[0] - etas[1])) <= 2.0 * bound


@pytest.mark.parametrize("constrained", [False, True])
def test_newton_steps_run_on_the_dense_and_constrained_routes(constrained: bool) -> None:
    """claude's event-row fixture through ``fit_irls_direct``, with and without ``beta >= 0``.

    The unconstrained fit takes the dense route's increment step; with a
    hard constraint it takes the QP route, whose Newton system carries the
    score in its right-hand side.  Both must run Newton iterations (the
    profile counts them) and reach the model's maximum, the constraint
    inactive there (the slope is positive).
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice, LinearConstraintSet

    rng = np.random.default_rng(0)
    x = np.append(rng.uniform(-1.0, 1.0, 200), 0.3)
    y = np.append((rng.uniform(size=200) < 0.2).astype(np.float64), 1.0)
    offset = np.append(np.full(200, 2.0), -25.0)
    groups = [
        GroupSlice(
            "x",
            0,
            1,
            constraints=LinearConstraintSet(A=np.eye(1), b=np.zeros(1)) if constrained else None,
            monotone_engine="qp" if constrained else None,
        )
    ]
    profile: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = fit_irls_direct(
            DesignMatrix([DenseGroupMatrix(x[:, None])], n=201, p=1),
            y,
            np.ones(201),
            Binomial(),
            LogLink(),
            groups,
            lambda2=0.0,
            offset=offset,
            weight_semantics="frequency",
            profile=profile,
        )
    assert result.converged
    assert profile["irls_mean_space_newton_iters"] > 0
    design = np.column_stack([np.ones_like(x), x])
    theta = _log_binomial_maximum(design, y, offset)
    eta = x * result.beta[0] + result.intercept + offset
    assert np.max(np.abs(eta - (design @ theta + offset))) <= _certified_eta_bound(
        design, y, offset, theta
    )


def _random_offset_fixture(seed: int):
    """Opus's #437 random sweep (``rand_sweep.py``): probabilities near one, offsets in [-0.5, 1)."""
    rng = np.random.default_rng(seed)
    n = int(rng.choice([60, 150, 400, 1500]))
    k = int(rng.choice([1, 2, 3]))
    x = rng.uniform(-1.0, 1.0, (n, k))
    slopes = rng.normal(0.0, 0.6, k)
    top = rng.choice([-0.3, -0.05, -0.01, -1e-3])
    linear = x @ slopes
    eta = linear - linear.max() + top
    y = (rng.uniform(size=n) < np.exp(eta)).astype(np.float64)
    assert rng.uniform() < 0.3  # the seeds used here carry an offset
    return x, y, rng.uniform(-0.5, 1.0, n)


@pytest.mark.parametrize("convergence", ["deviance", "coefficients"])
def test_a_lowered_fit_near_the_boundary_is_certified_by_newton_steps(convergence: str) -> None:
    """Seed 1285 of the random sweep: an interior maximum with fitted probabilities near one.

    The offset lowers the start.  Near ``eta = 0`` Fisher's information for
    a non-event, ``mu / (1 - mu)``, falls short of the observed ``mu / (1 -
    mu)^2``, and scoring contracts too slowly to reach the certificate's
    bar: dbeaf5a3 ended this fit ``score_stagnated`` after 24 iterations,
    though 3e-7 from the maximum in eta.  Once a refused stop is near the
    mode (every relative score within ``MODE_RESOLVE_CAP``) the iteration
    takes Newton steps on the observed information, and the fit is
    certified at the maximum.
    """
    x, y, offset = _random_offset_fixture(1285)
    columns = {f"x{j}": x[:, j] for j in range(x.shape[1])}
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        convergence=convergence,
        features={name: Numeric() for name in columns},
    )
    model.fit(pd.DataFrame(columns), y, offset=offset)
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    design = np.column_stack([np.ones(len(y)), x])
    theta = _log_binomial_maximum(design, y, offset)
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    assert np.max(np.abs(eta - (design @ theta + offset))) <= _certified_eta_bound(
        design, y, offset, theta
    )


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_subnormal_weights_never_certify_a_wrong_maximum(direct_solve: str) -> None:
    """Sol's #437 fixture: two levels at offsets [-30, -30, 2, 2], equal weights 1e-317 to 1e300.

    The certificate's relative scores are ratios, free of the weights'
    units, but its floors were not: at ``1e-317`` the score fell below the
    smallest normal float, the floor read a relative score of about 1/3 as
    4.5e-10, and dbeaf5a3 certified ``p_a = 6.3e-15`` against a maximum of
    1/2.  Every quantity is now brought to unit scale by one exact power of
    two first.  Across the normal range the verdict and the maximum are the
    same.  Subnormal weights carry fewer than 53 bits and their working
    weights underflow, so those fits cannot reach the maximum; they are
    published as not converged, never as a wrong maximum.
    """
    offset = np.array([-30.0, -30.0, 2.0, 2.0])
    for w in (1e-317, 1e-310, 1e-300, 1.0, 1e300):
        model = SuperGLM(
            family="binomial",
            link="log",
            selection_penalty=0.0,
            direct_solve=direct_solve,
            features={"g": Categorical(base="first")},
        )
        model.fit(
            pd.DataFrame({"g": ["a", "a", "b", "b"]}),
            np.array([0.0, 1.0, 0.0, 1.0]),
            offset=offset,
            sample_weight=np.full(4, w),
        )
        eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
        if w >= np.finfo(np.float64).tiny:
            # certified: each level's score within 2.5 bar sum|s| (sum|s| =
            # 4w) against its curvature 2w, so its eta within 5 bar of log 1/2
            assert model.result.converged
            assert float(np.max(np.abs(eta - np.log(0.5)))) <= 5.1 * MODE_CERTIFICATION_BAR
        else:
            # fewer than 53 bits and underflowing working weights: never a
            # certified wrong maximum
            assert (
                not model.result.converged
                or float(np.max(np.abs(eta - np.log(0.5)))) <= 5.1 * MODE_CERTIFICATION_BAR
            )


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
@pytest.mark.parametrize("weight", [1e-317, 1e-310])
def test_a_subnormal_weight_level_publishes_finite_degrees_of_freedom(
    weight: float, direct_solve: str
) -> None:
    """The extreme sweep's crash: level a at a subnormal weight under offset +10, level b at 1.

    The effective degrees of freedom are ``diag((D + S)^+ D)``, and at a data
    scale of 1e-317 the pseudo-inverse overflowed: d1496789 raised "fit
    candidate scalar results must be finite" on 74 such sweep fits.  The
    trace is invariant under scaling ``D`` and ``S`` jointly, so an
    overflowed trace is formed again in scaled units.  The fit publishes.
    Without a penalty the slope's centred system is 1 x 1 with ``D = H``
    exactly, so its trace is ``h / h`` to the few roundings of a 1 x 1
    pseudo-inverse and product (a square root, a square, a division and a
    product); with the intercept's 1 the total is 2 within ``8u``.  Level b,
    which the certificate pins through its own column's relative score, is
    within the certified ``5 bar`` of log 1/2 whenever the fit says
    converged.
    """
    offset = np.array([10.0, 10.0, -0.5, -0.5])
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        weight_semantics="frequency",
        features={"g": Categorical(base="first")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            pd.DataFrame({"g": ["a", "a", "b", "b"]}),
            np.array([0.0, 1.0, 0.0, 1.0]),
            offset=offset,
            sample_weight=np.array([weight, weight, 1.0, 1.0]),
        )
    edf = float(model.result.effective_df)
    assert np.isfinite(edf)
    assert abs(edf - 2.0) <= 8.0 * _U
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    assert (
        not model.result.converged
        or abs(float(eta[2]) - np.log(0.5)) <= 5.1 * MODE_CERTIFICATION_BAR
    )


def _two_level_log_likelihood(weights: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Each row's ``w [y eta + (1 - y) log(1 - e^eta)]`` for ``y = (0, 1, 0, 1)``, ``log1mexp`` by branch."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    complement = np.where(
        eta > -math.log(2.0),
        np.log(-np.expm1(np.minimum(eta, -1e-300))),
        np.log1p(-np.exp(np.minimum(eta, 0.0))),
    )
    return weights * (y * eta + (1.0 - y) * complement)


def _certified_two_level_fit(weights: np.ndarray, offset: np.ndarray, direct_solve: str):
    """``(converged, deviance excess, its certified bound)`` of a two-level fit whose maximum is ``p = 1/2``.

    Each level holds one event and one non-event, so at the maximum every
    row's score is ``+-w`` and its Fisher weight ``w``: ``sum |s| = F = sum
    w``, and level a (the intercept) carries ``F_a >= F / 2``.  A certified
    fit holds the intercept's score within ``bar F`` and level b's centred
    score within ``bar zeta sqrt(D_bb)`` (``zeta = sqrt(F)``, ``D_bb = F_a
    F_b / F``), or excludes level b with half its Newton decrement within
    ``gamma_4 sum |l|``.  The floors, of order ``u`` times the offsets, stay
    below the bar here.  Per level the log-likelihood gap is ``S^2 / (2 F)``,
    so the two give at most ``(bar F)^2 ((1 + sqrt 2)^2 / F_a + 2 / F) / 2``
    plus that noise, doubled for the curvature's change along the way and
    again for the deviance; the test's own two sums add ``2 gamma_4`` times
    their sizes.
    """
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        weight_semantics="frequency",
        features={"g": Categorical(base="first")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            pd.DataFrame({"g": ["a", "a", "b", "b"]}),
            np.array([0.0, 1.0, 0.0, 1.0]),
            offset=offset,
            sample_weight=weights,
        )
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    at_maximum = _two_level_log_likelihood(weights, np.full(4, math.log(0.5)))
    published = _two_level_log_likelihood(weights, eta)
    excess = 2.0 * (float(np.sum(at_maximum)) - float(np.sum(published)))
    if np.any(eta >= 0.0):  # outside the mean space: no likelihood
        excess = math.inf
    total, level_a = float(np.sum(weights)), float(np.sum(weights[:2]))
    gamma_4 = 4.0 * _U / (1.0 - 4.0 * _U)
    noise = gamma_4 * float(np.sum(np.abs(at_maximum)))
    gap = (MODE_CERTIFICATION_BAR * total) ** 2 * (
        (1.0 + math.sqrt(2.0)) ** 2 / level_a + 2.0 / total
    ) / 2.0 + noise
    rounding = (
        2.0 * gamma_4 * (float(np.sum(np.abs(at_maximum))) + float(np.sum(np.abs(published))))
    )
    return bool(model.result.converged), excess, 4.0 * gap + rounding


@pytest.mark.parametrize("direct_solve", ["auto", "gram", "qr"])
def test_a_level_weak_only_at_an_unfinished_iterate_is_not_certified(direct_solve: str) -> None:
    """Sol's #437 fixture: level a of weight 1e4 at offset +1.3, level b of weight 1e-8 at -30.

    6b2f2bed certified it after one iteration at ``p_b = 7.07e-7`` (maximum
    1/2): level b's Fisher curvature is tiny there only because ``p_b`` is
    far off, so the weak test excluded it and the intercept alone passed.
    Its half Newton decrement there, ``w_b / (4 p_b)`` (about 3.5e-3) against
    noise of ``gamma_4 sum |l|`` (about 6e-12), now keeps it in, and the fit
    runs on.
    """
    converged, excess, bound = _certified_two_level_fit(
        np.array([1e4, 1e4, 1e-8, 1e-8]), np.array([1.3, 1.3, -30.0, -30.0]), direct_solve
    )
    assert not converged or excess <= bound


@pytest.mark.parametrize("direct_solve", ["auto", "gram", "qr"])
def test_a_weight_ratio_never_certifies_a_wrong_two_level_maximum(direct_solve: str) -> None:
    """Level b at ``1 / ratio`` of level a's weight, ratio 1 to 1e12, at offsets -5 to -40."""
    for ratio in (1.0, 1e3, 1e6, 1e9, 1e12):
        for level_b_offset in (-5.0, -10.0, -20.0, -30.0, -40.0):
            converged, excess, bound = _certified_two_level_fit(
                np.array([1e4, 1e4, 1e4 / ratio, 1e4 / ratio]),
                np.array([1.3, 1.3, level_b_offset, level_b_offset]),
                direct_solve,
            )
            assert not converged or excess <= bound, (ratio, level_b_offset, excess, bound)


def test_the_underflow_allowance_is_representable() -> None:
    """Half the subnormal spacing, ``2.0**-1075``, rounds to 0; the allowance counts the whole spacing."""
    from superglm.solvers.irls_direct import _SUBNORMAL_SPACING, _underflow_allowance

    assert 2.0**-1075 == 0.0
    assert _SUBNORMAL_SPACING == np.nextafter(0.0, 1.0) > 0.0
    for rows in (0, 1, 4, 10**6):
        assert _underflow_allowance(rows) == (rows + 2) * _SUBNORMAL_SPACING > 0.0


def test_a_lowered_scop_fit_is_certified_in_its_latent_coordinates() -> None:
    """Sol's #437 fixture: an increasing PSpline (SCOP) with a constant offset of +2.

    dbeaf5a3 formed no latent score and so never certified a SCOP fit under
    the check: it ended this one ``score_stagnated``, where master converges
    in 10 iterations.  The certificate now forms the shape-constrained
    group's latent score ``J' B' s - lambda S theta`` and certifies the fit,
    every fitted mean inside the space and above the clip floor.
    """
    rng = np.random.default_rng(8)
    x = rng.uniform(0.0, 1.0, 60)
    y = (rng.uniform(size=60) < 0.1 + 0.3 * x).astype(np.float64)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=5, constraint=Constraint.fit.increasing)},
    )
    model.fit(pd.DataFrame({"x": x}), y, offset=np.full(60, 2.0))
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + 2.0
    assert float(eta.max()) < 0.0
    assert float(eta.min()) > np.log(1e-7)


@pytest.mark.parametrize("scale", [100.0, 1.0, 1e-100])
def test_a_constraints_units_do_not_change_the_certificate(scale: float) -> None:
    """Sol's #437 fixture: ``scale * beta >= 0`` at weights 1e-307 and offset +2.

    The constrained maximum is ``beta = 0`` with intercept ``log(1/2) - 2``,
    whatever positive multiple of the row states the constraint.  dbeaf5a3
    divided the active row by Jacobi entries near 2e-307, so ``[[100]]``
    overflowed and raised.  Each row is now normalised to unit length and
    every unit brought to scale first, so the three give one answer: the
    intercept within the few roundings of ``log(1/2) - 2``.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice, LinearConstraintSet

    design = DesignMatrix([DenseGroupMatrix(np.array([[0.0], [1.0], [0.0], [1.0]]))], n=4, p=1)
    groups = [
        GroupSlice(
            "x",
            0,
            1,
            constraints=LinearConstraintSet(A=np.array([[scale]]), b=np.zeros(1)),
            monotone_engine="qp",
        )
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FractionalFrequencyWeightWarning)
        result, _ = fit_irls_direct(
            design,
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.full(4, 1e-307),
            Binomial(),
            LogLink(),
            groups,
            lambda2=0.0,
            offset=np.full(4, 2.0),
            weight_semantics="frequency",
        )
    assert result.converged
    assert result.beta[0] == 0.0
    # certified: with beta = 0 every row shares eta, the intercept's score is
    # c (1 - p / (1 - p)) for c events and c non-events, sum|s| = 2c and its
    # slope in eta 2c / (1 - p) = 4c at p = 1/2, so eta is within bar / 2 of
    # log(1/2), doubled for the curvature's change along the way, plus the
    # intercept's own rounding, u |intercept|
    bound = MODE_CERTIFICATION_BAR + _U * abs(np.log(0.5) - 2.0)
    assert abs(result.intercept - (np.log(0.5) - 2.0)) <= bound


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_constrained_lowered_fit_is_certified_with_its_multipliers(direct_solve: str) -> None:
    """Non-negative step increments on a probability that rises then falls, offset by +1.5.

    Four step columns ``x > 0.2, 0.4, 0.6, 0.8`` with ``beta >= 0``
    (``A = I``) and no penalty: the falling half pushes the last two
    increments against their bound, so two constraints are active at the
    maximum.  The lowered start brings the fit under the binomial/log
    certificate, which tests ``G + A' m`` with ``m >= 0`` there; without the
    multipliers it refused this fit.  Checked against the KKT conditions
    directly, from the true score ``s`` and Fisher weights ``W`` of the
    returned eta, each to the bar the certificate holds it to: the
    intercept's score within ``bar sum|s|``; an inactive increment's centred
    score within ``bar zeta sqrt(D_jj)`` (``zeta = sum|s| / sqrt(sum W)``,
    ``D_jj = sum W (x_j - mean_j)^2``); an active one's no more than that
    above zero, its multiplier ``-G_j`` non-negative.  The no-offset fit
    starts inside, stops on master's rule, and holds the same active set.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice, LinearConstraintSet

    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < 0.1 + 0.3 * np.sin(np.pi * x)).astype(np.float64)
    steps = (x[:, None] > np.array([0.2, 0.4, 0.6, 0.8])[None, :]).astype(np.float64)
    k = steps.shape[1]
    actives = []
    for shift in (0.0, 1.5):
        groups = [
            GroupSlice(
                "steps",
                0,
                k,
                constraints=LinearConstraintSet(A=np.eye(k), b=np.zeros(k)),
                monotone_engine="qp",
            )
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, _ = fit_irls_direct(
                DesignMatrix([DenseGroupMatrix(steps)], n=n, p=k),
                y,
                np.ones(n),
                Binomial(),
                LogLink(),
                groups,
                lambda2=0.0,
                offset=np.full(n, shift),
                direct_solve=direct_solve,
                weight_semantics="frequency",
            )
        assert result.converged
        assert result.termination_reason == "converged"
        beta = np.asarray(result.beta)
        assert np.all(beta >= 0.0)
        # at its bound to the rounding of the iterate the QP solved for
        active = np.flatnonzero(beta <= 4.0 * _EPS * float(np.sum(np.abs(beta))))
        actives.append(active.tolist())
        if shift == 0.0:
            continue
        assert active.size >= 1
        eta = steps @ beta + result.intercept + shift
        score, fisher = mean_space_score_rows(y, np.ones(n), eta)
        size = float(np.sum(np.abs(score)))
        mean = steps.T @ fisher / float(np.sum(fisher))
        centred = steps - mean
        gradient = centred.T @ score
        scale = size / np.sqrt(float(np.sum(fisher))) * np.sqrt(fisher @ centred**2)
        bar = MODE_CERTIFICATION_BAR
        assert abs(float(np.sum(score))) <= bar * size
        inactive = np.setdiff1d(np.arange(k), active)
        assert np.all(np.abs(gradient[inactive]) <= bar * scale[inactive])
        assert np.all(gradient[active] <= bar * scale[active])
    assert actives[0] == actives[1]


def test_the_weighted_centring_matches_its_definition_on_each_block_type() -> None:
    """``weighted_column_centring`` against ``sum w (x - mean)^2`` formed column by column.

    The function forms the diagonal one pass per block (a dense block in
    chunks, a spline block from its weighted Gram, a one-hot block in closed
    form); the reference forms every column through a design product and
    sums with ``math.fsum``.  A dense column carries an offset of ``1e6``,
    which the chunked pass centres before squaring.  Tolerances: ``n``
    roundings of each summed term, ``8 n eps sum w x~^2``, plus for a dense
    column the mean's own rounding, ``eps max|x|`` on each of its ``sum w
    |x~|`` terms; a raw-moment block's terms are ``sum w x^2``.
    """
    rng = np.random.default_rng(11)
    n = 300
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": 1e6 + rng.uniform(size=n),
            "g": rng.choice(["p", "q", "r"], n),
            "s": rng.uniform(size=n),
        }
    )
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "a": Numeric(),
            "b": Numeric(),
            "g": Categorical(),
            "s": BSplineSmooth(n_knots=6),
        },
    )
    model.fit(frame, rng.poisson(1.0, n).astype(np.float64))
    dm = model._dm
    weights = rng.uniform(0.1, 2.0, n)
    mean_x, sum_w, diagonal = weighted_column_centring(dm, weights, weights > 0.0)
    dense = np.concatenate(
        [np.full(m.shape[1], type(m).__name__ == "DenseGroupMatrix") for m in dm.group_matrices]
    )
    for j in range(dm.p):
        unit = np.zeros(dm.p)
        unit[j] = 1.0
        column = np.asarray(dm.matvec(unit), dtype=np.float64)
        mean = math.fsum(weights * column) / math.fsum(weights)
        centred = column - mean
        expected = math.fsum(weights * centred**2)
        terms = 8.0 * n * _EPS * math.fsum(weights * (centred**2 if dense[j] else column**2))
        if dense[j]:
            terms += (
                4.0 * _EPS * float(np.max(np.abs(column))) * math.fsum(weights * np.abs(centred))
            )
        assert abs(diagonal[j] - expected) <= terms
        assert abs(mean_x[j] - mean) <= 4.0 * n * _EPS * float(np.max(np.abs(column)))
    assert sum_w == pytest.approx(float(np.sum(weights)), rel=4.0 * n * _EPS)


def test_weight_and_penalty_scales_never_certify_a_wrong_maximum() -> None:
    """Sol's #437 fixture under every weight scale times penalty scale, subnormal weights too.

    Intercept and slope ``x = (-1, -1, 1, 1)``, ``y = (0, 1, 0, 1)``, offsets
    ``(2, 2, 3, 3)`` (a lowered start), weights ``w`` and penalty ``s`` on
    the slope.  The maximum depends only on ``r = s / w``; the reference
    solves it by Newton's method on the exact score, with ``r`` taken as 0
    below 1e-30 and as 1e30 above 1e30, where the slope's maximum no longer
    moves at float64 resolution.  d1496789 certified a wrong point at ``w =
    1e-100, s = 1e300`` (one global power of two had sent the data to zero
    beside the penalty).  Every certified fit must sit within the
    certificate's bound of the reference, doubled because the penalty's own
    term in a slope's scale is bounded at the maximum by the data score it
    balances.  At normal weights every fit is certified; at the smallest
    subnormal weight, whose rows carry one bit, the fit may only refuse.
    """
    x = np.array([-1.0, -1.0, 1.0, 1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    offset = np.array([2.0, 2.0, 3.0, 3.0])
    design = np.column_stack([np.ones(4), x])

    def reference(ratio: float) -> np.ndarray:
        theta = np.array([-4.0, 0.0])
        for _ in range(200):
            eta = design @ theta + offset
            odds = np.exp(eta) / -np.expm1(eta)
            score = design.T @ (y - (1.0 - y) * odds) - np.array([0.0, ratio * theta[1]])
            information = design.T @ (((1.0 - y) * odds / -np.expm1(eta))[:, None] * design)
            step = np.linalg.solve(information + np.diag([0.0, ratio]), score)
            fraction = 1.0
            while np.any(design @ (theta + fraction * step) + offset >= 0.0):
                fraction /= 2.0
            theta = theta + fraction * step
            if np.max(np.abs(fraction * step)) <= 4.0 * _EPS * (1.0 + np.max(np.abs(theta))):
                break
        return theta

    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice

    scales = (1e-300, 1e-100, 1.0, 1e100, 1e300)
    for w in (float(np.nextafter(0.0, 1.0)), *scales):
        for s in scales:
            log_ratio = math.log10(s) - math.log10(w)
            ratio = 0.0 if log_ratio < -30.0 else (1e30 if log_ratio > 30.0 else 10.0**log_ratio)
            theta = reference(ratio)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FractionalFrequencyWeightWarning)
                result, _ = fit_irls_direct(
                    DesignMatrix([DenseGroupMatrix(x[:, None])], n=4, p=1),
                    y,
                    np.full(4, w),
                    Binomial(),
                    LogLink(),
                    [GroupSlice("x", 0, 1)],
                    lambda2=0.0,
                    offset=offset,
                    S_override=np.array([[s]]),
                    weight_semantics="frequency",
                )
            if w >= np.finfo(np.float64).tiny:
                assert result.converged, (w, s)
            if result.converged:
                eta = x * result.beta[0] + result.intercept + offset
                bound = 2.0 * _certified_eta_bound(design, y, offset, theta)
                assert np.max(np.abs(eta - (design @ theta + offset))) <= bound, (w, s)


def test_an_intercept_only_fit_at_the_smallest_weight_is_never_certified_wrong() -> None:
    """Sol's #437 fixture: an intercept-only fit with every weight ``nextafter(0, 1)``.

    The row products ``w odds`` underflowed before d1496789's power of two
    could lift them, the rounded scores cancelled exactly, and it certified
    an intercept of -3.5 (probabilities 0.223 and 0.607; the maximum, 0.232
    and 0.629, has a relative score of 0.045 there).  The weights are now
    normalised before any product.  This fit cannot reach the maximum on one
    bit of weight, so it ends not converged, never certified elsewhere.
    Master returns it not converged at an infeasible state.
    """
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        weight_semantics="frequency",
        features={},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FractionalFrequencyWeightWarning)
        model.fit(
            pd.DataFrame(index=range(4)),
            np.array([0.0, 1.0, 0.0, 1.0]),
            offset=np.array([2.0, 2.0, 3.0, 3.0]),
            sample_weight=np.full(4, np.nextafter(0.0, 1.0)),
        )
    eta = model.result.intercept + np.array([2.0, 2.0, 3.0, 3.0])
    reference = _log_binomial_maximum(
        np.ones((4, 1)), np.array([0.0, 1.0, 0.0, 1.0]), np.array([2.0, 2.0, 3.0, 3.0])
    )
    at_maximum = abs(float(eta[0] - (reference[0] + 2.0))) <= 1e-6
    assert not model.result.converged or at_maximum


@pytest.mark.parametrize("direct_solve", ["gram", "structured"])
def test_reml_newton_steps_are_certified_by_the_models_score(direct_solve: str) -> None:
    """Sol's #437 fixture: ``fit_reml``, a fixed-lambda random effect, an event row at -25 per level.

    REML's inner solves stop on ``mode_score``'s clipped residual, which
    cannot pass at the model's mode: the clip all but removes each
    low-probability event's unit score.  d1496789 reached the mode with
    Newton steps but, its stop gated on that residual, ran to max_iter.
    Under Newton steps the model's own certificate is now the stop.  The
    mode has zero random effects and intercept ``log(2/3) - 2``; master
    certifies ``log(1/2) - 2`` instead, the maximum without the event rows.
    """
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"g": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(
            pd.DataFrame({"g": ["a", "a", "a", "b", "b", "b"]}),
            np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            offset=np.array([-25.0, 2.0, 2.0, -25.0, 2.0, 2.0]),
        )
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    assert model._reml_profile["direct_backend"] == direct_solve
    assert model._reml_profile["irls_mean_space_newton_iters"] > 0
    # certified: the intercept's score within bar sum|s| (sum|s| = 8 at the
    # mode: per level two events of score 1 and a non-event of 2) against the
    # non-events' observed curvature p / (1 - p)^2 = 6 per level, 12 in all,
    # so within 2/3 bar of the mode, doubled for the curvature's change on
    # the way; the random effects' penalty only adds curvature
    bound = 4.0 / 3.0 * MODE_CERTIFICATION_BAR
    assert abs(model.result.intercept - (np.log(2.0 / 3.0) - 2.0)) <= bound
    assert np.all(np.abs(model.result.beta) <= bound)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_a_constraint_row_s_norm_does_not_depend_on_its_units(scale: float) -> None:
    """Sol's #437 fixture: ``scale * beta >= 0`` at unit weights and offset +2.

    Eight rows, ``x`` 0 then 1, ``y = (1, 1, 1, 0, 0, 0, 0, 1)``: every
    positive scale has the maximum ``beta = 0``, intercept ``log(1/2) - 2``.
    d1496789 squared the row before its norm, so ``[[1e200]]`` had norm
    ``inf`` and ``[[1e-200]]`` norm 0, and both ended ``score_stagnated``.
    The row is now brought to a largest entry in ``[1/2, 1)`` by a power of
    two first.  Master returns all three not converged at an infeasible
    state.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice, LinearConstraintSet

    x = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    groups = [
        GroupSlice(
            "x",
            0,
            1,
            constraints=LinearConstraintSet(A=np.array([[scale]]), b=np.zeros(1)),
            monotone_engine="qp",
        )
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = fit_irls_direct(
            DesignMatrix([DenseGroupMatrix(x[:, None])], n=8, p=1),
            np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            np.ones(8),
            Binomial(),
            LogLink(),
            groups,
            lambda2=0.0,
            offset=np.full(8, 2.0),
            weight_semantics="frequency",
        )
    assert result.converged
    assert result.beta[0] == 0.0
    # certified: with beta = 0 every row shares eta, the intercept's score is
    # c (1 - p / (1 - p)) for c events and c non-events, sum|s| = 2c and its
    # slope in eta 2c / (1 - p) = 4c at p = 1/2, so eta is within bar / 2 of
    # log(1/2), doubled for the curvature's change along the way, plus the
    # intercept's own rounding, u |intercept|
    bound = MODE_CERTIFICATION_BAR + _U * abs(np.log(0.5) - 2.0)
    assert abs(result.intercept - (np.log(0.5) - 2.0)) <= bound


def test_the_clip_and_the_true_score_are_read_off_the_unclipped_eta() -> None:
    """``mean_space_clipped_rows`` and ``mean_space_score_rows`` on each of their branches."""
    family, link = Binomial(), LogLink()
    # clip_mu holds 1e-7 <= mu <= 1 - 1e-7: rows below and above it count,
    # events or not, when they carry weight, and so does exp's underflow
    eta = np.array([-17.0, -17.0, -16.0, -1e-8, -1e-6, -800.0, -17.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    assert mean_space_clipped_rows(family, link, eta, weights) == 4
    assert mean_space_clipped_rows(family, LogitLink(), eta, weights) == 0
    # the score w [y - (1 - y) mu / (1 - mu)] and Fisher weight w mu / (1 - mu)
    # against 400-digit references (eta = -1e-300 needs 1 - mu to 300 digits):
    # exp, expm1, the division and the product by w round once each, within 4
    # eps of the reference; exp's underflow below eta ~ -745 gives the exact
    # limits w y and 0
    from decimal import Decimal, localcontext

    eta = np.array([-800.0, -40.0, -1.0, -1e-12, -1e-300, -40.0, -1e-12])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    weights = np.array([3.0, 1e14, 0.5, 2.0, 1.0, 1e-12, 7.0])
    score, fisher = mean_space_score_rows(y, weights, eta)
    with localcontext() as context:
        context.prec = 400
        for k in range(len(eta)):
            mu = Decimal(float(eta[k])).exp()
            odds = mu / (1 - mu)
            w = Decimal(float(weights[k]))
            response = Decimal(float(y[k]))
            expected_score = float(w * (response - (1 - response) * odds))
            expected_fisher = float(w * odds)
            assert abs(score[k] - expected_score) <= 4.0 * _EPS * abs(expected_score)
            assert abs(fisher[k] - expected_fisher) <= 4.0 * _EPS * abs(expected_fisher)
    # outside the space: no likelihood with weight, nothing without
    score, fisher = mean_space_score_rows(np.zeros(2), np.array([1.0, 0.0]), np.zeros(2))
    assert not np.isfinite(score[0])
    assert score[1] == 0.0 and fisher[1] == 0.0
    # Newton rows: observed curvature w (1 - y) mu / (1 - mu)^2 (zero on an
    # event row) with the same score; none where no row has curvature
    eta = np.array([-40.0, -1.0, -1e-12, -1.0])
    y = np.array([0.0, 0.0, 0.0, 1.0])
    weights = np.array([1e14, 0.5, 2.0, 7.0])
    rows = mean_space_newton_rows(y, weights, eta)
    assert rows is not None
    curvature, score = rows
    assert np.array_equal(score, mean_space_score_rows(y, weights, eta)[0])
    with localcontext() as context:
        context.prec = 60
        for k in range(len(eta)):
            mu = Decimal(float(eta[k])).exp()
            expected = float(Decimal(float(weights[k])) * (1 - Decimal(float(y[k]))) * mu)
            expected = float(Decimal(expected) / (1 - mu) ** 2) if expected else 0.0
            assert abs(curvature[k] - expected) <= 5.0 * _EPS * abs(expected)
    assert mean_space_newton_rows(np.ones(2), np.ones(2), np.array([-1.0, -2.0])) is None
    # the true deviance difference, -2 sum w [y d_eta + (1 - y) d log(1 - e^eta)],
    # below the clip floor too, where the clipped deviance is flat
    before, after = np.array([-40.0, -1.0, -1e-9]), np.array([-39.0, -0.5, -2e-9])
    y = np.array([0.0, 1.0, 0.0])
    weights = np.array([1e14, 1.0, 3.0])
    with localcontext() as context:
        context.prec = 60
        expected = Decimal(0)
        for k in range(3):
            w, yk = Decimal(float(weights[k])), Decimal(float(y[k]))
            a, b = Decimal(float(after[k])), Decimal(float(before[k]))
            expected += (
                -2 * w * (yk * (a - b) + (1 - yk) * ((1 - a.exp()).ln() - (1 - b.exp()).ln()))
            )
    delta = mean_space_deviance_delta(y, weights, after, before)
    assert abs(delta - float(expected)) <= 8.0 * _EPS * abs(float(expected))


def test_the_halving_budget_reaches_the_fraction_to_the_boundary() -> None:
    """``_mean_space_halving_budget`` on each of its branches."""
    family, link = Binomial(), LogLink()
    weights = np.ones(2)

    def budget(eta, proposal, default=20, link=link):
        return _mean_space_halving_budget(
            committed=SimpleNamespace(eta_unclipped=np.asarray(eta, dtype=float)),
            proposal=SimpleNamespace(eta_unclipped=np.asarray(proposal, dtype=float)),
            weights=weights,
            family=family,
            link=link,
            default=default,
        )

    # a link that stays in (0, 1), a start outside the space, no row moving up,
    # and a fraction the ordinary budget reaches all keep the default
    assert budget([-1.0, -1.0], [5.0, -1.0], link=LogitLink()) == 20
    assert budget([0.0, -1.0], [5.0, -1.0]) == 20
    assert budget([-1.0, -1.0], [-2.0, -1.5]) == 20
    assert budget([-1.0, -1.0], [1.0, -1.0]) == 20
    # t = 1 / 3e7: the first halving below t, then the ordinary budget
    eta, step = -1.0, 3e7
    depth = budget([eta, -1.0], [eta + step, -1.0]) - 20
    fraction = -eta / step
    assert 2.0**-depth < fraction <= 2.0 ** -(depth - 1)
    # float64's halving depth caps it
    assert budget([-1e-320, -1.0], [1.0, -1.0]) == 1074


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
    raises=ObservedModeNotConvergedError,
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
