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
    mean_space_log_likelihood_rows,
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
    ``"auto"`` is asserted to take the structured solver, so a change of the
    crossover or of the fixture cannot quietly compare gram with gram, and
    the bootstrap's boundary start is asserted to run the restoration loop
    (#425, r4154948497 and r4154950836).
    """
    fits = {direct_solve: _fit(direct_solve) for direct_solve in ("auto", "gram")}
    for model in fits.values():
        assert model._reml_result.converged
        assert model.result.converged
        assert float(_eta(model).max()) < 0.0
    assert fits["auto"].result.direct_backend == "structured"
    assert fits["gram"].result.direct_backend == "gram"
    assert fits["auto"]._reml_profile["reml_mean_space_restorations"] >= 1
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
            record_diagnostics=True,
        )
    assert result.converged
    assert result.mean_space_true_mode
    assert profile["irls_mean_space_newton_iters"] > 0
    # the stop rule that decided is the one published: the true-score
    # certificate's score against its bar, on the iteration log and profile
    last = result.iteration_log[-1]
    assert last.convergence_criterion == "mean_space_mode_score"
    assert last.convergence_tolerance == MODE_CERTIFICATION_BAR
    assert last.convergence_value <= last.convergence_tolerance
    assert profile["irls_mode_rule"] == "mean_space_mode_score"
    assert profile["irls_mode_score"] <= profile["irls_mode_bar"] == MODE_CERTIFICATION_BAR
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


# How far a certified level (or cell) of the two-level fixtures can sit from
# log(1/2), where every row's score is +-w and a level's own score
# w (1 - p / (1 - p)) moves by -2w per unit of eta at p = 1/2:
# - certified on its own rows (``mode_score.row_set_residual``), the level's
#   score is within the bar of its own sum |s| = 2w, so its eta is within bar;
# - excluded as weak, half its decrement 2w deta^2 / 2 is within the noise
#   of its own rows, gamma_6 2w log 2, so deta <= sqrt(2 gamma_6 log 2);
# each doubled for the curvature's change along the way.  The floors, of
# order u times the offsets, stay far below either.
_GAMMA_6 = 6.0 * 2.0**-53 / (1.0 - 6.0 * 2.0**-53)
_TWO_LEVEL_BOUND = 2.0 * max(MODE_CERTIFICATION_BAR, math.sqrt(2.0 * _GAMMA_6 * math.log(2.0)))


def _two_level_levels(
    weights: np.ndarray, offset: np.ndarray, direct_solve: str, base: str = "first"
):
    """``(converged, largest |eta_level - log(1/2)|)`` of a two-level fit whose maxima are ``p = 1/2``."""
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        weight_semantics="frequency",
        features={"g": Categorical(base=base)},
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
    return bool(model.result.converged), float(np.max(np.abs(eta[[0, 2]] - math.log(0.5))))


@pytest.mark.parametrize("direct_solve", ["auto", "gram", "qr"])
@pytest.mark.parametrize(
    ("weights", "offsets", "base", "must_converge"),
    [
        # Sol, on 6b2f2bed: p_b = 7.07e-7
        ((1e4, 1e4, 1e-8, 1e-8), (1.3, 1.3, -30.0, -30.0), "first", True),
        # Sol, on 2357a43a: p_b = 0.0676
        ((1e8, 1e8, 1e-8, 1e-8), (1.3, 1.3, -20.0, -20.0), "first", True),
        # Sol, on 3520abec: the light level is the reference, p_a = 0.334
        ((1e-8, 1e-8, 1e8, 1e8), (-20.0, -20.0, 1.3, 1.3), "a", False),
        # the same fit with the heavy level as the reference reaches both maxima
        ((1e-8, 1e-8, 1e8, 1e8), (-20.0, -20.0, 1.3, 1.3), "b", True),
    ],
)
def test_a_light_level_is_never_certified_away_from_its_own_maximum(
    weights: tuple, offsets: tuple, base: str, must_converge: bool, direct_solve: str
) -> None:
    """Sol's #437 fixtures: a heavy level near offset +1.3 and a light one far below the clip floor.

    Both levels' maxima are ``p = 1/2``.
    - 6b2f2bed excluded the light level as weak on its curvature at an
      unfinished iterate.
    - 2357a43a measured its block decrement against the noise of every row,
      which the heavy level dominates, and certified ``p_b = 0.0676``.
    - 3520abec scaled each non-reference level by its own rows, but a light
      *reference* level has no column: it was read only through the intercept,
      whose scale the heavy level sets, and ``p_a = 0.334`` was certified.
    Every level, the reference included, is now certified on its own rows
    (``mode_score.row_set_residual``).  The assertion is per level.  With the
    light level as the reference at ratio 1e16 the gram route cannot resolve
    the direction that moves it alone, and that fit ends not converged; the
    QR route reaches it.
    """
    converged, error = _two_level_levels(
        np.array(weights), np.array(offsets), direct_solve, base=base
    )
    assert not converged or error <= _TWO_LEVEL_BOUND, error
    if must_converge or direct_solve == "qr":
        assert converged


@pytest.mark.parametrize("light_base", [False, True])
@pytest.mark.parametrize("direct_solve", ["auto", "gram", "qr"])
def test_a_weight_ratio_never_certifies_a_level_away_from_its_maximum(
    direct_solve: str, light_base: bool
) -> None:
    """Level weights ``sqrt(ratio)`` and ``1 / sqrt(ratio)``, ratio 1 to 1e16, the light level at -5 to -40.

    The light level is the second level, or the reference.  Every fit is per
    level within ``_TWO_LEVEL_BOUND`` of its own maximum, or is not converged;
    2357a43a certified levels up to 0.98 away in eta, and 3520abec a light
    reference level 0.40 away.  Every fit converges except a light reference
    at ratio 1e12 and above on the gram route (auto takes gram here), which
    cannot resolve the direction that moves that level alone.
    """
    for ratio in (1.0, 1e4, 1e8, 1e12, 1e16):
        heavy, light = math.sqrt(ratio), 1.0 / math.sqrt(ratio)
        for light_offset in (-5.0, -10.0, -20.0, -30.0, -40.0):
            if light_base:
                weights = np.array([light, light, heavy, heavy])
                offsets = np.array([light_offset, light_offset, 1.3, 1.3])
            else:
                weights = np.array([heavy, heavy, light, light])
                offsets = np.array([1.3, 1.3, light_offset, light_offset])
            converged, error = _two_level_levels(
                weights, offsets, direct_solve, base="a" if light_base else "first"
            )
            case = (ratio, light_offset, converged, error)
            assert not converged or error <= _TWO_LEVEL_BOUND, case
            if not light_base or ratio <= 1e8 or direct_solve == "qr":
                assert converged, case


@pytest.mark.parametrize("direct_solve", ["auto", "gram", "qr"])
@pytest.mark.parametrize(("ratio", "light_offset"), [(1e8, -5.0), (1e8, -20.0), (1e16, -30.0)])
def test_a_light_base_cell_of_an_interaction_is_never_certified_away_from_its_maximum(
    ratio: float, light_offset: float, direct_solve: str
) -> None:
    """``A + B + A:B`` on 2 x 2 cells, the base cell ``(a0, b0)`` light and far below the clip floor.

    Each cell holds one event and one non-event, so the saturated model's
    maxima are ``p = 1/2`` in every cell.  The base cell has no column of
    its own in any block, and the reference rows of ``A`` and of ``B`` pool
    it with a heavy cell, so only the joint cells of the one-hot blocks read
    it on its own rows.  3520abec certified it at ``p = 0.064`` (offset -5)
    and below ``1e-8`` (offsets -20, -30).  The assertion is per cell; the
    mildest case must converge.
    """
    heavy, light = math.sqrt(ratio), 1.0 / math.sqrt(ratio)
    a, b, y, w, o = [], [], [], [], []
    for cell_a, cell_b in (("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")):
        base_cell = (cell_a, cell_b) == ("a0", "b0")
        for event in (0.0, 1.0):
            a.append(cell_a)
            b.append(cell_b)
            y.append(event)
            w.append(light if base_cell else heavy)
            o.append(light_offset if base_cell else 1.3)
    offsets = np.array(o)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        weight_semantics="frequency",
        features={"A": Categorical(base="a0"), "B": Categorical(base="b0")},
        interactions=[("A", "B")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            pd.DataFrame({"A": a, "B": b}), np.array(y), offset=offsets, sample_weight=np.array(w)
        )
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offsets
    error = float(np.max(np.abs(eta[::2] - math.log(0.5))))
    assert not model.result.converged or error <= _TWO_LEVEL_BOUND, error
    if (ratio, light_offset) == (1e8, -5.0):
        assert model.result.converged


@pytest.mark.parametrize(
    ("low_scores", "second", "excluded", "row_noise"),
    [
        ((1e-16, 0.0), 1e-3, True, None),  # at the mode along the pair: excluded
        ((1e-12, 0.0), 1e-3, False, None),  # far from it: refused
        ((1e-16, -2e-13), 1e-3, False, None),  # each slope passes alone, the pair does not
        ((1e-16, 0.0), 0.0, True, None),  # identical columns: H_BB singular, g in its range
        # noise by row: the heavy rows would admit anything, the block's own rows
        # admit the aligned pair and not far from the mode
        ((1e-16, 0.0), 1e-3, True, (1.0, 5e-11)),
        ((1e-12, 0.0), 1e-3, False, (1.0, 5e-11)),
    ],
)
def test_the_weak_exclusion_reads_the_block_newton_decrement(
    low_scores: tuple[float, float], second: float, excluded: bool, row_noise
) -> None:
    """``penalized_mode_residual``'s weak exclusion, by table: allowed, refused, refused as a block.

    Two dense columns vary only on two rows of Fisher weight 1e-20 beside
    eight of weight 1: ``x1 = e_8`` and ``x2 = e_8 + 1e-3 e_9``.  Both are
    weak (curvature about 1e-20 against ``gamma_10`` times the mass), and
    nearly collinear: ``H_BB = 1e-20 [[1, 1], [1, 1 + 1e-6]]``, eigenvalues
    about 2e-20 and 5e-27.  The rows' scores ``(s_8, s_9)`` set ``g``; the
    noise is 1e-10 and the bar ``MODE_CERTIFICATION_BAR``, which each slope
    misses (relative score about ``|g| / 2.8e-10``).
    - ``g = (a, a)``, ``a = 1e-16``, along the large eigenvector: half the
      block decrement is about ``a^2 / 2e-20 = 5e-13``, within the noise, so
      both slopes are excluded and the intercept certifies alone.
    - ``a = 1e-12``: about 5e-5, refused.
    - ``g = (a, -a)``, along the small eigenvector: each slope's own term is
      ``a^2 / 2e-20``, summing to 1e-12 (within the noise, so 430a423a
      excluded both), but the block's half decrement is about ``a^2 / 5e-27 = 2e-6``,
      refused.
    - Identical columns (``x2 = e_8``): ``H_BB`` is singular and ``g = (a,
      a)`` lies in its range.  Its null eigenvalue comes back from LAPACK at
      the rounding, of either sign; decided in ``u``, it is null, ``g``'s
      projection on it is within rounding, and the exclusion is allowed.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.solvers.mode_score import penalized_mode_residual

    columns = np.zeros((10, 2))
    columns[8] = 1.0
    columns[9, 1] = second
    dm = DesignMatrix([DenseGroupMatrix(columns)], n=10, p=2)
    fisher = np.append(np.ones(8), [1e-20, 1e-20])
    score = np.append(np.tile([1.0, -1.0], 4), low_scores)
    sum_w = float(np.sum(fisher))
    mean_x = fisher @ columns / sum_w
    centred = columns - mean_x
    diagonal = fisher @ centred**2
    residual = penalized_mode_residual(
        dm=dm,
        row_score=score,
        fisher_weights=fisher,
        positive_prior=np.ones(10, dtype=bool),
        mean_x=mean_x,
        centered_scale=np.sqrt(diagonal / sum_w),
        alpha=0.0,
        eta_tilde=np.zeros(10),
        penalty_score=np.zeros(2),
        penalty_magnitude=np.zeros(2),
        penalty_curvature=np.zeros(2),
        sum_w=sum_w,
        bar=MODE_CERTIFICATION_BAR,
        decrement_noise=(
            (lambda support: 1e-10)
            if row_noise is None
            else (
                lambda support: float(
                    np.sum(np.where(np.arange(10) < 8, row_noise[0], row_noise[1])[support])
                )
            )
        ),
    )
    assert np.all(residual.relative[1:] > MODE_CERTIFICATION_BAR)
    assert bool(np.all(residual.excluded)) is excluded
    assert (residual.ratio() <= 1.0) is excluded


def test_the_underflow_allowance_is_representable() -> None:
    """Half the subnormal spacing, ``2.0**-1075``, rounds to 0; the allowance counts the whole spacing."""
    from superglm.solvers.irls_direct import _SUBNORMAL_SPACING, _underflow_allowance

    assert 2.0**-1075 == 0.0
    assert _SUBNORMAL_SPACING == np.nextafter(0.0, 1.0) > 0.0
    for rows in (0, 1, 4, 10**6):
        assert _underflow_allowance(rows) == (rows + 2) * _SUBNORMAL_SPACING > 0.0


@pytest.mark.parametrize(
    ("light_scores", "light_y", "separated"),
    [
        ((0.0, 1e-300), (0.0, 1.0), False),  # a non-event's score underflowed: still mixed
        ((-1e-300, 0.0), (0.0, 0.0), True),  # no events: no interior maximum
    ],
)
def test_a_reference_level_is_separated_only_by_its_responses(
    light_scores: tuple, light_y: tuple, separated: bool
) -> None:
    """``row_set_residual`` on a light reference level beside a balanced heavy level.

    The reference rows (no column of the block) are summed directly.  A level
    holding an event and a non-event has an interior maximum, even where the
    non-event's score ``-w odds`` has underflowed to zero at weight 1e-300:
    its own score, 1e-300 of 1e-300, is then refused.  A level without
    events has no interior maximum and is left to the weak test.  Reading
    separation off the scores' signs instead certified such a light level
    on the 10,800-fit sweep at weights near 1e-300.
    """
    from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix
    from superglm.solvers.mode_score import row_set_residual, row_sets

    dm = DesignMatrix([CategoricalGroupMatrix(np.array([-1, -1, 0, 0]), 1)], n=4, p=1)
    ratio = row_set_residual(
        sets=row_sets(dm),
        row_score=np.array([*light_scores, -1.0, 1.0]),
        response=np.array([*light_y, 0.0, 1.0]),
        fisher_weights=np.array([1e-300, 1e-300, 1.0, 1.0]),
        positive_prior=np.ones(4, dtype=bool),
        eta=np.full(4, math.log(0.5)),
        column_penalty=np.zeros(1),
        column_penalty_size=np.zeros(1),
        column_curvature=np.zeros(1),
        set_curvature=np.zeros(1),
        bar=MODE_CERTIFICATION_BAR,
        underflow=0.0,
    )
    assert (ratio <= 1.0) is separated


def test_a_bridge_between_two_crossed_cycles_is_the_only_joint_cell_set() -> None:
    """``mode_score.row_sets`` on two crossed blocks whose cells are two 4-cycles joined by one cell.

    Every column holds two or more cells, so elimination leaves all nine to
    the core's SVD.  A cell is a set exactly when it is a bridge of the
    levels' graph (leverage 1); a cycle's cell has ``1 - h = 1 / (1 + 3)``,
    far outside the derived resolution.  The bridge's direction moves its
    rows by one and every other row by the same amount, which the intercept
    returns to zero, to within the SVD's backward error times the
    incidence's condition number.
    """
    from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix
    from superglm.solvers.mode_score import row_sets

    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3), (1, 2)]
    rows = [pair for pair in pairs for _ in range(2)]
    first = np.array([a - 1 for a, _ in rows])  # level 0 is each block's reference (-1)
    second = np.array([b - 1 for _, b in rows])
    blocks = [CategoricalGroupMatrix(first, 3), CategoricalGroupMatrix(second, 3)]
    dm = DesignMatrix(blocks, n=len(rows), p=6)
    sets = row_sets(dm)
    assert sets.cell_of_row is not None
    kept = sets.cell_of_row >= 0
    assert np.array_equal(kept, np.array([pair == (1, 2) for pair in rows]))
    incidence = np.column_stack(
        [np.ones(len(pairs))]
        + [np.array([a == level for a, _ in pairs], dtype=float) for level in (1, 2, 3)]
        + [np.array([b == level for _, b in pairs], dtype=float) for level in (1, 2, 3)]
    )
    singular = np.linalg.svd(incidence, compute_uv=False)
    rank = int(np.sum(singular > max(incidence.shape) * np.finfo(float).eps * singular[0]))
    tolerance = 4 * max(incidence.shape) * np.finfo(float).eps * singular[0] / singular[rank - 1]
    moved = dm.matvec(sets.cell_directions[[0]].toarray().ravel())
    np.testing.assert_allclose(moved - moved[~kept][0], kept.astype(float), rtol=0, atol=tolerance)


def test_the_row_sets_are_formed_once_per_design(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cells, their elimination and any core SVD read the one-hot codes alone (``mode_score.row_sets``).

    They are held in each design's layout cache.  50ff4b7d formed the cells'
    ``np.unique`` and an SVD of their incidence at every certificate
    evaluation: 90 SVDs in this nested REML fit.  The fit evaluates the
    certificate on two designs, its REML search's and the final model's,
    and forms each one's sets once.  Each nested cell is one district's
    rows, already judged as its level, so no cell is kept and no SVD runs.
    """
    import superglm.solvers.irls_direct as irls_direct
    import superglm.solvers.mode_score as mode_score

    formed: list[int] = []
    decided: list[int] = []
    designs: list[int] = []
    form, decide, sets_of = (
        mode_score._form_row_sets,
        mode_score._decide_core,
        irls_direct.row_sets,
    )
    monkeypatch.setattr(
        mode_score, "_form_row_sets", lambda *a, **k: formed.append(1) or form(*a, **k)
    )
    monkeypatch.setattr(
        mode_score, "_decide_core", lambda *a, **k: decided.append(1) or decide(*a, **k)
    )
    monkeypatch.setattr(irls_direct, "row_sets", lambda dm: designs.append(id(dm)) or sets_of(dm))
    frame, y = _districts_in_regions(2)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        features={"c": Categorical(base="first"), "g": RandomEffect()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert len(designs) > len(set(designs))
    assert len(formed) == len(set(designs))
    assert decided == []
    assert mode_score.row_sets(model._dm).cell_of_row is None


@pytest.mark.parametrize(("slopes", "excluded"), [(32, True), (33, False)])
def test_a_weak_set_at_the_block_limit_is_formed_and_one_past_it_is_refused(
    slopes: int, excluded: bool
) -> None:
    """``_WEAK_BLOCK_LIMIT`` (32): a weak set of 32 slopes is formed and excluded, of 33 refused.

    Each slope's column is one row of Fisher weight 1e-20 beside eight of
    weight 1, its score 1e-16 there, so it misses the bar (relative score
    about 3.5e-7) and is weak; half the block decrement, ``k a^2 / 2e-20``,
    is at most 1.6e-11 against a noise of 1e-10.  Above the limit the
    exclusion is refused, the safe direction, and the slopes stay in the
    ratio.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.solvers.mode_score import _WEAK_BLOCK_LIMIT, penalized_mode_residual

    assert _WEAK_BLOCK_LIMIT == 32
    n = 8 + slopes
    columns = np.zeros((n, slopes))
    columns[8 + np.arange(slopes), np.arange(slopes)] = 1.0
    dm = DesignMatrix([DenseGroupMatrix(columns)], n=n, p=slopes)
    fisher = np.append(np.ones(8), np.full(slopes, 1e-20))
    score = np.append(np.tile([1.0, -1.0], 4), np.full(slopes, 1e-16))
    sum_w = float(np.sum(fisher))
    mean_x = fisher @ columns / sum_w
    diagonal = fisher @ (columns - mean_x) ** 2
    residual = penalized_mode_residual(
        dm=dm,
        row_score=score,
        fisher_weights=fisher,
        positive_prior=np.ones(n, dtype=bool),
        mean_x=mean_x,
        centered_scale=np.sqrt(diagonal / sum_w),
        alpha=0.0,
        eta_tilde=np.zeros(n),
        penalty_score=np.zeros(slopes),
        penalty_magnitude=np.zeros(slopes),
        penalty_curvature=np.zeros(slopes),
        sum_w=sum_w,
        bar=MODE_CERTIFICATION_BAR,
        decrement_noise=lambda support: 1e-10,
    )
    assert np.all(residual.relative[1:] > MODE_CERTIFICATION_BAR)
    assert bool(np.all(residual.excluded)) is excluded
    assert (residual.ratio() <= 1.0) is excluded


@pytest.mark.parametrize("rows", [8191, 8192, 8193, 16385])
def test_the_weak_block_gram_is_whole_across_its_row_chunks(rows: int) -> None:
    """``_centred_block_gram`` at and across the 8192-row chunk boundary: every row counted once.

    Against the same corrected two-pass cross products formed in one pass
    with ``math.fsum``, to ``gamma_{rows + 4}`` of ``sum w |x~| |x~|'``; the
    support is the rows where either column is nonzero, the last row
    included.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.solvers.mode_score import _centred_block_gram

    rng = np.random.default_rng(rows)
    columns = rng.normal(size=(rows, 2))
    columns[rows // 3 : rows // 2] = 0.0
    columns[-1] = (2.0, -3.0)
    weights = rng.uniform(0.5, 2.0, rows)
    dm = DesignMatrix([DenseGroupMatrix(columns)], n=rows, p=2)
    mean_x = weights @ columns / float(np.sum(weights))
    gram, support = _centred_block_gram(dm, np.arange(2), mean_x, weights)
    centred = columns - mean_x
    total = math.fsum(weights)
    gamma = (rows + 4) * _U / (1.0 - (rows + 4) * _U)
    for j in range(2):
        for k in range(2):
            first_j = math.fsum(weights * centred[:, j])
            first_k = math.fsum(weights * centred[:, k])
            expected = (
                math.fsum(weights * centred[:, j] * centred[:, k]) - first_j * first_k / total
            )
            size = math.fsum(weights * np.abs(centred[:, j] * centred[:, k]))
            assert abs(gram[j, k] - expected) <= gamma * size, (j, k)
    assert np.array_equal(support, np.any(columns != 0.0, axis=1))


@pytest.mark.parametrize(("row", "refused"), [(-(2.0**-52), False), (-1e-3, True)])
def test_a_scop_observed_row_is_signed_only_beyond_its_rounding(row: float, refused: bool) -> None:
    """SCOP's observed rows: a negative row within its rounding is zero, one beyond it is refused.

    A binomial/log event row's observed information is exactly zero, and its
    terms cancel, so the computed row is the platform's rounding: -1.1e-16
    and -2.2e-16 on the macOS, Windows and ARM64 runners, where 2357a43a's
    ``test_a_lowered_scop_fit_is_certified_in_its_latent_coordinates`` raised
    "signed observed-information rows are not supported".  Rows are within
    ``gamma_22`` of their terms' scale (``observed_row_error_scale``, at
    least 1 here: the Fisher part of a row at ``p = 1/2``).
    """
    from superglm.reml.observed_geometry import compute_scop_observed_information_weights

    class FixedRows(Binomial):
        def scop_observed_information_weights(self, link, y, mu, eta, sample_weight):
            return np.array([row, 1.0])

    eta = np.full(2, math.log(0.5))
    arguments = (FixedRows(), LogLink(), np.array([1.0, 0.0]), np.exp(eta), eta, np.ones(2))
    if refused:
        with pytest.raises(ValueError, match="signed observed-information rows"):
            compute_scop_observed_information_weights(*arguments)
    else:
        assert np.array_equal(compute_scop_observed_information_weights(*arguments), [0.0, 1.0])


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

    The function forms the diagonal one pass per block (a one-hot block in
    closed form, every other block over its rows in chunks by the corrected
    two-pass algorithm about the rounded mean); the reference forms every
    column through a design product and sums with ``math.fsum``.  A dense
    column carries an offset of ``1e6``.  A second set of weights puts all
    but 1e-15 of the mass on one row, so a spline column is nearly constant
    over it: its ``Var_w / E_w x^2`` is far below ``n u``, where the raw
    moments of eec69403 (``sum w x^2 - sum_w mean^2``) carry an error of
    ``u sum w x^2``.  Tolerances: ``n`` roundings of each summed term, ``8 n
    eps sum w x~^2``, plus the mean's own rounding, ``eps max|x|`` on each of
    the ``sum w |x~|`` terms; the one-hot closed form's terms are ``sum w
    x^2``.
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
    one_hot = np.concatenate(
        [
            np.full(m.shape[1], type(m).__name__ == "CategoricalGroupMatrix")
            for m in dm.group_matrices
        ]
    )
    concentrated = np.full(n, 1e-15)
    concentrated[int(np.argmax(frame["s"].to_numpy() > 0.5))] = 1.0
    for weights in (rng.uniform(0.1, 2.0, n), concentrated):
        mean_x, sum_w, diagonal = weighted_column_centring(dm, weights, weights > 0.0)
        for j in range(dm.p):
            unit = np.zeros(dm.p)
            unit[j] = 1.0
            column = np.asarray(dm.matvec(unit), dtype=np.float64)
            mean = math.fsum(weights * column) / math.fsum(weights)
            centred = column - mean
            expected = math.fsum(weights * centred**2)
            terms = 8.0 * n * _EPS * math.fsum(weights * (column**2 if one_hot[j] else centred**2))
            terms += (
                4.0 * _EPS * float(np.max(np.abs(column))) * math.fsum(weights * np.abs(centred))
            )
            assert abs(diagonal[j] - expected) <= terms, (j, diagonal[j], expected, terms)
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
    # the certificate is relative, so its bound does not depend on the
    # weights' scale: unit weights give the same value as nextafter(0, 1)
    bound = _certified_eta_bound(
        np.ones((4, 1)), np.array([0.0, 1.0, 0.0, 1.0]), np.array([2.0, 2.0, 3.0, 3.0]), reference
    )
    at_maximum = abs(float(eta[0] - (reference[0] + 2.0))) <= bound
    assert not model.result.converged or at_maximum


def _true_laml(
    design: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    rho: float,
    theta: np.ndarray | None = None,
    weights: np.ndarray | None = None,
):
    """``(V, size, condition)``: the model's own LAML at ``rho``, its terms' sizes and ``H``'s condition.

    ``V = -l(theta) + (lambda |b|^2 + log|H| - q rho) / 2`` with ``l`` the
    exact weighted log-likelihood by ``log1mexp`` and ``H`` the weighted
    observed information plus ``lambda`` on the ``q`` random effects.
    ``theta = (alpha, b)`` is evaluated as given, or found as the penalized
    mode by Newton's method on the exact score, with halved steps that stay
    inside the mean space.
    """
    lam = math.exp(rho)
    q = design.shape[1] - 1
    penalty = np.diag([0.0] + [lam] * q)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    carried = w > 0.0

    def log_likelihood(eta: np.ndarray) -> np.ndarray:
        rows = np.zeros_like(eta)
        e = eta[carried]
        complement = np.where(e > -math.log(2.0), np.log(-np.expm1(e)), np.log1p(-np.exp(e)))
        rows[carried] = w[carried] * (y[carried] * e + (1.0 - y[carried]) * complement)
        return rows

    def information_at(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        odds = np.zeros_like(eta)
        curvature = np.zeros_like(eta)
        e = eta[carried]
        odds[carried] = np.exp(e) / -np.expm1(e)
        curvature[carried] = w[carried] * (1.0 - y[carried]) * odds[carried] / -np.expm1(e)
        score = design.T @ (w * (y - (1.0 - y) * odds))
        return score, design.T @ (curvature[:, None] * design) + penalty

    if theta is None:
        theta = np.zeros(design.shape[1])
        theta[0] = -1.0 - float(np.max(offset[carried]))
        for _ in range(200):
            score, information = information_at(design @ theta + offset)
            step = np.linalg.solve(information, score - penalty @ theta)
            fraction = 1.0
            while np.any((design @ (theta + fraction * step) + offset)[carried] >= 0.0):
                fraction /= 2.0
            theta = theta + fraction * step
            if np.max(np.abs(fraction * step)) <= 4.0 * _EPS * (1.0 + np.max(np.abs(theta))):
                break
    eta = design @ theta + offset
    information = information_at(eta)[1]
    rows = log_likelihood(eta)
    quadratic = lam * float(theta[1:] @ theta[1:])
    log_det = float(np.linalg.slogdet(information)[1])
    value = -float(np.sum(rows)) + 0.5 * (quadratic + log_det - q * rho)
    size = float(np.sum(np.abs(rows))) + quadratic + abs(log_det) + q * abs(rho)
    return value, size, float(np.linalg.cond(information))


def _deep_event_levels(deep_y: float = 1.0, deep_offset: float = -25.0, deep_levels: int = 8):
    """Eight levels of 20 rows (1 to 12 events), plus one row ``deep_y`` at ``deep_offset`` in the first ``deep_levels``."""
    events = [1, 3, 6, 10, 2, 8, 4, 12]
    levels, y, offset = [], [], []
    for k, count in enumerate(events):
        for r in range(20):
            levels.append(f"l{k}")
            y.append(1.0 if r < count else 0.0)
            offset.append(0.0)
        if k < deep_levels:
            levels.append(f"l{k}")
            y.append(deep_y)
            offset.append(deep_offset)
    levels_arr = np.array(levels)
    design = np.column_stack(
        [np.ones(len(levels))]
        + [(levels_arr == name).astype(np.float64) for name in sorted(set(levels))]
    )
    return levels_arr, np.array(y), np.array(offset), design


def _laml_noise(design: np.ndarray, size: float, condition: float) -> float:
    """The rounding of ``_true_laml``'s sum: ``gamma_{n+p+4}`` of its sizes, ``p gamma_p cond(H)`` for ``log|H|``."""
    n, p = design.shape

    def gamma(k: int) -> float:
        return k * _U / (1.0 - k * _U)

    return gamma(n + p + 4) * size + p * gamma(p) * condition


def _fixed_lambda_criterion(levels, y, offset, design, rho: float, weights=None):
    """A fixed-lambda ``fit_reml``, its published criterion, and the model's LAML at the fit's own state.

    Evaluated at the fit's own coefficients, not at a reference mode, so the
    inner mode's own error (certified to the bar, entering ``log|H|`` to first
    order) does not enter the comparison: both sides are one expression at
    one state, and agree to the rounding of each, ``_laml_noise``.
    """
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        features={"g": RandomEffect(lambda_policy=LambdaPolicy.fixed(math.exp(rho)))},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(pd.DataFrame({"g": levels}), y, offset=offset, sample_weight=weights)
    result = model._reml_result.pirls_result
    theta = np.concatenate(([result.intercept], result.beta))
    value, size, condition = _true_laml(design, y, offset, rho, theta=theta, weights=weights)
    return model, value, _laml_noise(design, size, condition)


def test_reml_reads_the_models_own_laml_at_a_true_score_mode() -> None:
    """At a fixed lambda, the REML criterion at a Newton-ended true-score mode is the model's own LAML.

    The inner solve ends on the binomial/log certificate under Newton steps
    (``PIRLSResult.mean_space_true_mode``), so the criterion reads the
    model's likelihood, and its observed curvature at the unclipped mean.
    eec69403 read both off the clip: its criterion sat -1.3e-5 from the
    model's here, most of it from the event rows' curvature (about -5e-5 a
    row at the clip, zero in the model).  The two sides share no additive
    constant to remove (the binomial likelihood has none).
    """
    levels, y, offset, design = _deep_event_levels()
    model, value, noise = _fixed_lambda_criterion(levels, y, offset, design, 1.36)
    assert model._reml_profile["irls_mean_space_newton_iters"] > 0
    assert model._reml_result.pirls_result.mean_space_true_mode
    assert abs(model._reml_result.objective - value) <= 2.0 * noise


@pytest.mark.parametrize("convergence", ["coefficients", "mode_score"])
def test_a_fisher_certified_stop_beside_a_clipped_row_is_the_models_mode(convergence: str) -> None:
    """A Fisher stop the true-score certificate confirms is flagged as the model's mode, as a Newton one is.

    One non-event row at offset -20, below the floor, beside 200 rows near
    p = 0.2: Fisher steps credit it with the clip's score, ``-1e-7``
    against ``-e^eta``, within the bar, so the clipped stop passes and the
    true-score certificate confirms it without Newton steps.  The state is
    the model's mode all the same, so ``mean_space_true_mode`` is set and a
    REML criterion read there reads the model's likelihood and curvature
    (``test_reml_reads_the_models_own_laml_at_a_true_score_mode``).
    430a423a set it only under Newton, so its criterion read the clip here,
    about 1e-7 a row off, and stepped between candidates certified one way
    and the other.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice

    rng = np.random.default_rng(3)
    x = np.append(rng.uniform(-1.0, 1.0, 200), 0.0)
    y = np.append((rng.uniform(size=200) < 0.2 + 0.1 * x[:200]).astype(np.float64), 0.0)
    offset = np.append(np.zeros(200), -20.0)
    profile: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = fit_irls_direct(
            DesignMatrix([DenseGroupMatrix(x[:, None])], n=201, p=1),
            y,
            np.ones(201),
            Binomial(),
            LogLink(),
            [GroupSlice("x", 0, 1)],
            lambda2=0.0,
            offset=offset,
            weight_semantics="frequency",
            profile=profile,
            convergence=convergence,
        )
    assert result.converged
    assert profile.get("irls_mean_space_newton_iters", 0) == 0
    assert result.mean_space_true_mode


@pytest.mark.parametrize("direct_solve", ["gram", "structured"])
def test_a_rare_event_random_effect_level_is_certified_at_its_own_mode(direct_solve: str) -> None:
    """claude's #437 fixture: a random-effect level of one event row at offset -40 beside two heavy levels.

    ``RandomEffect`` at a fixed lambda of 1.  The event row's score is its
    weight, 1, wherever its mean is, so the level's own stationarity ``1 -
    lambda beta_c = 0`` puts its mode at ``beta_c = 1`` exactly.  Clipped
    Fisher scoring credits the row with about ``e^eta / 1e-7`` of that score
    and stops near ``beta_c = 0``.  Master and 3520abec certified it there;
    3520abec's level ``zeta`` loosened the level's test by ``mu^(-1/2)``.
    Certified on its own rows, ``|1 - lambda beta_c| <= bar (1 + lambda
    |beta_c|)``, so ``beta_c`` is within ``2 bar`` of 1, doubled.
    """
    rng = np.random.default_rng(5)
    levels = np.array(["a"] * 200 + ["b"] * 200 + ["c"])
    y = np.append((rng.uniform(size=400) < 0.2).astype(np.float64), 1.0)
    weights = np.append(np.full(400, 5e6), 1.0)
    offset = np.append(np.zeros(400), -40.0)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        weight_semantics="frequency",
        features={"g": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(pd.DataFrame({"g": levels}), y, offset=offset, sample_weight=weights)
    assert model.result.converged
    assert abs(float(model.result.beta[2]) - 1.0) <= 4.0 * MODE_CERTIFICATION_BAR


def _districts_in_regions(districts_in_r: int) -> tuple[pd.DataFrame, np.ndarray]:
    """``RandomEffect`` districts nested in ``Categorical`` regions; region ``r`` has no events.

    Regions ``p`` and ``q`` hold four districts of 60 rows each, at event
    probabilities 0.15 and 0.3; region ``r`` holds ``districts_in_r``
    districts of 40 rows, every response zero.
    """
    rng = np.random.default_rng(11)
    region: list[str] = []
    district: list[str] = []
    y: list[float] = []
    for name, probability in (("p", 0.15), ("q", 0.3)):
        for k in range(4):
            region += [name] * 60
            district += [f"{name}{k}"] * 60
            y += (rng.uniform(size=60) < probability).astype(np.float64).tolist()
    for k in range(districts_in_r):
        region += ["r"] * 40
        district += [f"r{k}"] * 40
        y += [0.0] * 40
    return pd.DataFrame({"c": region, "g": district}), np.array(y)


def _random_effect_distances(
    model: SuperGLM, y: np.ndarray, lam: float
) -> list[tuple[float, float]]:
    """Each ``RandomEffect`` level's distance to its own penalized maximum along its column.

    ``(|t*|, bound)`` per level: ``t*`` maximizes ``sum_R l_i(eta_i + t) - lam
    (beta_l + t)^2 / 2`` over the level's rows (Newton's method; the
    function is ``lam``-strongly concave).  The bound is the row-set test's:
    a set it passes by distance is within ``bar``; one it passes by its
    relative score has ``|g| <= bar (sum_R |s| + lam |beta_l|)``, so ``|t*| <=
    |g| / lam`` is within that over ``lam``.  Doubled for this oracle's own
    rounding.
    """
    dm = model._dm
    beta = np.asarray(model.result.beta, dtype=np.float64)
    eta = dm.matvec(beta) + float(model.result.intercept)
    start = sum(matrix.shape[1] for matrix in dm.group_matrices[:-1])
    levels = dm.group_matrices[-1]
    out = []
    for level in range(levels.n_levels):
        rows = levels.codes == level
        events, base = y[rows] > 0.0, eta[rows]
        coefficient = float(beta[start + level])
        with np.errstate(under="ignore"):
            odds = np.exp(base) / -np.expm1(base)
        scale = float(np.sum(np.where(events, 1.0, odds))) + lam * abs(coefficient)
        t = 0.0
        for _ in range(100):
            with np.errstate(under="ignore"):
                shifted = base + t
                odds = np.exp(shifted) / -np.expm1(shifted)
                gradient = float(np.sum(np.where(events, 1.0, -odds))) - lam * (coefficient + t)
                curvature = float(np.sum(np.where(events, 0.0, odds / -np.expm1(shifted)))) + lam
            t += gradient / curvature
        out.append((abs(t), 2.0 * MODE_CERTIFICATION_BAR * max(1.0, scale / lam)))
    return out


@pytest.mark.parametrize("direct_solve", ["gram", "structured"])
@pytest.mark.parametrize("districts_in_r", [1, 2])
def test_a_random_effect_level_inside_a_level_without_events_is_certified_at_its_own_mode(
    districts_in_r: int, direct_solve: str
) -> None:
    """claude's #437 fixture: districts nested in regions, region ``r`` without events, lambda fixed at 1.

    Region ``r``'s free column carries the separation, and each step
    returns the coefficients of the districts inside it to about zero (with
    one district exactly: the step's two stationarity rows leave ``lambda
    (beta_l + delta_l) = 0``).  Such a district's rows score ``-sum w odds``,
    one-signed, against a penalty gradient near zero, so its relative score
    reads about 1 at every iterate however far its mean falls; once ``odds``
    underflows its bar does too.  50ff4b7d ended the fit ``score_stagnated``
    where master converges.  The district's direction carries the penalty,
    so its penalized maximum along it is finite and the set passes by its
    distance to it, ``|g| / lambda`` within the bar.  Every district is then
    within the certificate's bound of its own penalized maximum.
    """
    frame, y = _districts_in_regions(districts_in_r)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={
            "c": Categorical(base="first"),
            "g": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0)),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert model.result.converged
    for distance, bound in _random_effect_distances(model, y, 1.0):
        assert distance <= bound, (distance, bound)


@pytest.mark.parametrize("direct_solve", ["gram", "structured"])
@pytest.mark.parametrize("design", ["crossed", "nested"])
def test_reml_with_a_separated_level_converges_as_on_master(design: str, direct_solve: str) -> None:
    """claude's #437 questions: an unpenalized level without events under ``fit_reml``.

    Crossed with a ``RandomEffect``, or holding two of its levels.  The
    level's maximum is at ``eta -> -infinity``: its rows' responses are all
    zero and its direction carries no penalty, so the row-set test leaves it
    out.  Its global relative score falls like ``sqrt(mu)`` as its mean
    falls, and the stop passes once that is within the bar; the weak test at
    the final mode (``reml.identified.final_mode_weak_slopes``) then
    discloses it.  In the nested design the two ``RandomEffect`` levels
    inside it are penalized sets, each certified by its distance to its own
    penalized maximum; 50ff4b7d ended that fit ``score_stagnated``.  Both
    converge, as on master, and disclose the level.
    """
    if design == "crossed":
        rng = np.random.default_rng(7)
        n = 600
        category = rng.choice(["p", "q", "r"], n)
        group = rng.choice([f"g{k}" for k in range(8)], n)
        probability = np.where(category == "r", 0.0, np.where(category == "p", 0.15, 0.3))
        y = (rng.uniform(size=n) < probability).astype(np.float64)
        frame = pd.DataFrame({"c": category, "g": group})
    else:
        frame, y = _districts_in_regions(2)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"c": Categorical(base="first"), "g": RandomEffect()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert model.result.converged
    assert model.diagnostics()["c"]["weakly_identified"] == [1]  # level r


def _separated_region_districts() -> tuple[pd.DataFrame, np.ndarray]:
    """25 regions of 12 districts of 8 rows; the base region and two more without events."""
    rng = np.random.default_rng(5)
    regions, districts, per = 25, 12, 8
    probability = rng.uniform(0.05, 0.3, regions)
    probability[:3] = 0.0
    region = np.repeat(np.arange(regions), districts * per)
    district = np.repeat(np.arange(regions * districts), per)
    shift = rng.normal(0.0, 0.3, regions * districts)
    p = np.minimum(probability[region] * np.exp(shift[district]), 0.9)
    y = (rng.uniform(size=len(p)) < p).astype(np.float64)
    frame = pd.DataFrame({"c": [f"r{k:02d}" for k in region], "g": [f"d{k:04d}" for k in district]})
    return frame, y


def test_reml_at_a_separated_level_does_not_move_with_a_constant_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constant offset with an intercept does not move lambda-hat, with a level without events.

    The base region and two others have no events, so their directions'
    supremum is at ``eta -> -infinity``.  1ffcacec's true-likelihood REML read
    those directions' vanishing curvature into ``log|H|`` at wherever PIRLS
    stopped its drift.  V(lambda) then moved with the offset by far more than
    its stop rule, the search never converged, and lambda-hat landed between
    2.2 and 5.1 for offsets 0, +1.5, -3 and a lowered start.  The Laplace term
    now leaves those directions out (``reml.identified.separated_directions``).
    And a Newton step on a truncated structured factor now keeps the
    iterate's component along a truncated direction: solving for the iterate
    reset a separated region's coefficient each step, the full step was
    refused and the inner fits stagnated (without it three of the four
    searches end unconverged).

    Each fit converges, the four objectives agree within twice the REML stop
    rule ``reml_tol (1 + |V|)`` (as the two backends' do), and the four
    lambda-hats within what that resolution allows on V's curvature
    ``V''`` in ``log lambda``, measured beside the optimum:
    ``|rho_a - rho_b| <= 2 sqrt(2 epsilon / V'')`` with ``epsilon`` twice the
    stop rule.  The criterion at a fixed lambda agrees across offsets within
    the same resolution.  The fourth offset lowers the start into the mean
    space, checked.
    """
    import superglm.solvers.irls_direct as irls_direct

    lowered: list[bool] = []
    interior_start = irls_direct.interior_start_intercept

    def recording(family, link, eta, weights, intercept, **kwargs):
        start = interior_start(family, link, eta, weights, intercept, **kwargs)
        lowered.append(start != intercept)
        return start

    monkeypatch.setattr(irls_direct, "interior_start_intercept", recording)
    frame, y = _separated_region_districts()

    def fit(offset_value: float, policy=None) -> SuperGLM:
        model = SuperGLM(
            family="binomial",
            link="log",
            selection_penalty=0.0,
            features={
                "c": Categorical(base="first"),
                "g": RandomEffect() if policy is None else RandomEffect(lambda_policy=policy),
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y, offset=np.full(len(y), offset_value))
        return model

    lowering = -float(np.log(np.mean(y))) + 0.5
    fits = {}
    for offset_value in (0.0, 1.5, -3.0, lowering):
        lowered.clear()
        fits[offset_value] = fit(offset_value)
        assert any(lowered) is (offset_value == lowering), offset_value
    for model in fits.values():
        assert model._reml_result.converged
        assert model.result.converged
    objectives = np.array([float(model._reml_result.objective) for model in fits.values()])
    tolerance = float(fits[0.0]._reml_profile["reml_tol_resolved"])
    epsilon = 2.0 * tolerance * (1.0 + float(np.max(np.abs(objectives))))
    assert float(np.ptp(objectives)) <= epsilon
    rho = np.log([float(next(iter(model._reml_lambdas.values()))) for model in fits.values()])
    beside = float(np.exp(rho[0] + 0.5))
    away = {
        offset_value: fit(offset_value, LambdaPolicy.fixed(beside)) for offset_value in (0.0, -3.0)
    }
    values = [float(model._reml_result.objective) for model in away.values()]
    assert abs(values[0] - values[1]) <= epsilon
    curvature = 2.0 * (values[0] - float(objectives[0])) / 0.5**2
    assert curvature > 0.0
    assert float(np.ptp(rho)) <= 2.0 * math.sqrt(2.0 * epsilon / curvature)


@pytest.mark.parametrize("direct_solve", ["gram", "qr"])
def test_a_penalized_set_whose_distance_bar_underflows_is_judged_by_its_score(
    direct_solve: str,
) -> None:
    """``S_override = 1e-16 I`` on a Categorical with a reference level, prior weights 1e300, a lowered start.

    In the weights' units the reference direction's curvature ``1' S_B 1``
    is ``2e-16`` times ``2^-998``, about 7e-317, and ``bar`` times it rounds to
    0.  1ffcacec divided by that product and raised ZeroDivisionError.  The
    set is now judged by its relative score alone, never passed on the
    distance.  The fit converges with each level within ``4 bar`` of its own
    maximum ``log(ybar_level)``: a level set's relative score within the bar
    puts its eta within ``2 bar (1 - mu)``, doubled for the curvature's change.
    """
    rng = np.random.default_rng(4)
    n = 300
    level = np.repeat(["a", "b", "c"], n // 3)
    y = (rng.uniform(size=n) < np.repeat([0.15, 0.25, 0.35], n // 3)).astype(np.float64)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        features={"c": Categorical(base="first")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame({"c": level}), y)
        result, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=np.full(n, 1e300),
            family=Binomial(),
            link=LogLink(),
            groups=model._groups,
            lambda2=0.0,
            S_override=1e-16 * np.eye(model._dm.p),
            offset=np.full(n, 2.0),
            direct_solve=direct_solve,
            weight_semantics="frequency",
        )
    assert result.converged
    eta = model._dm.matvec(result.beta) + result.intercept + 2.0
    for name in ("a", "b", "c"):
        rows = level == name
        error = np.abs(eta[rows] - math.log(float(np.mean(y[rows]))))
        assert float(np.max(error)) <= 4.0 * MODE_CERTIFICATION_BAR, name


def test_a_wide_design_keeps_its_joint_cell_sets() -> None:
    """A light base cell of a 2 x 2 interaction beside a 1.5-million-column block is still judged.

    1ffcacec held the kept cells' directions dense, ``(cells, p)``, and dropped
    every cell set past 2^22 entries: three cells beside 1.5 million columns
    left the base cell uncertified, while every level and reference set was
    judged.  The directions are held sparse now.  The scores put every level
    and reference set within the bar, and the light base cell (a0, b0) at a
    third of its own rows' size from stationary: only its own set refuses.
    """
    from scipy import sparse

    from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix, SparseGroupMatrix
    from superglm.solvers.mode_score import row_set_residual, row_sets

    width = 1_500_000
    first = np.array(
        [-1, -1, -1, -1, 0, 0, 0, 0]
    )  # a1's column; cells (a0,b0) (a0,b1) (a1,b0) (a1,b1)
    second = np.array([-1, -1, 0, 0, -1, -1, 0, 0])  # b1's column
    both = np.array([-1, -1, -1, -1, -1, -1, 0, 0])  # a1:b1's column
    blocks = [
        CategoricalGroupMatrix(first, 1),
        CategoricalGroupMatrix(second, 1),
        CategoricalGroupMatrix(both, 1),
        SparseGroupMatrix(sparse.csr_matrix((8, width))),
    ]
    dm = DesignMatrix(blocks, n=8, p=3 + width)
    sets = row_sets(dm)
    assert sets.cell_of_row is not None
    base = int(sets.cell_of_row[0])
    assert base >= 0
    moved = dm.matvec(sets.cell_directions[[base]].toarray().ravel())
    np.testing.assert_array_equal(moved - moved[2], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    score = np.array([-1e-8, 2e-8, -5.0, 5.0 - 1e-8, -5.0, 5.0 - 1e-8, -5.0, 5.0 + 1e-8])
    ratio = row_set_residual(
        sets=sets,
        row_score=score,
        response=np.tile([0.0, 1.0], 4),
        fisher_weights=np.ones(8),
        positive_prior=np.ones(8, dtype=bool),
        eta=np.full(8, math.log(0.5)),
        column_penalty=np.zeros(dm.p),
        column_penalty_size=np.zeros(dm.p),
        column_curvature=np.zeros(dm.p),
        set_curvature=np.zeros(len(sets.blocks) + sets.cell_directions.shape[0]),
        bar=MODE_CERTIFICATION_BAR,
        underflow=0.0,
    )
    assert ratio > 1.0


def test_reml_keeps_a_zero_weight_row_at_the_clip() -> None:
    """A zero-weight row at eta 3.8 beside a true-score mode: the criterion forms, and is the model's.

    PIRLS never holds a zero-weight row's eta inside the mean space, so at a
    true-score mode 430a423a's unclipped mean passed one there, the variance
    floor overflowed its weight derivatives, and ``fit_reml`` raised
    ``ObservedModeNotConvergedError``.  Such a row is no part of the mode and
    keeps the clip; it adds nothing to either side.
    """
    levels, y, offset, design = _deep_event_levels()
    levels = np.append(levels, "l0")
    y = np.append(y, 0.0)
    offset = np.append(offset, 5.0)
    design = np.vstack([design, design[0]])
    weights = np.append(np.ones(len(levels) - 1), 0.0)
    model, value, noise = _fixed_lambda_criterion(levels, y, offset, design, 1.36, weights)
    assert model._reml_result.pirls_result.mean_space_true_mode
    assert abs(model._reml_result.objective - value) <= 2.0 * noise


def test_reml_with_an_estimated_lambda_minimises_the_models_own_laml() -> None:
    """claude's #437 finding: a REML criterion read off the clip at the model's own mode.

    Eight levels of 20 rows (1 to 12 events each) and one event row per level
    at offset -25, below the clip floor; the random effect's lambda is
    estimated.  Under Newton steps the inner solves stop at the model's own
    mode, where the clipped likelihood is not stationary, but eec69403 still
    read the clipped likelihood and the clip's curvature there (an event row
    below the floor, whose observed curvature is zero, read about -5e-5): its
    lambda-hat sat 0.0085 from the model's LAML minimum in log lambda.
    Master certifies the clipped mode and its own criterion, 0.42 away.

    The reference is a brute-force grid in ``rho = log lambda``, refined by a
    golden section, of the model's own LAML (``_true_laml``).  It locates the
    minimum to within ``sqrt(2 eps_V / V'')``, ``eps_V`` the rounding of its
    sum of terms (``gamma_{n+p+4}`` of their sizes, plus ``p gamma_p`` times
    the information's condition for its log-determinant).  The fit's
    stop resolves ``rho`` to ``reml_tol (1 + |V|) / V''``.  Each is doubled
    for ``V''``'s change across the interval.
    """
    levels_arr, y_arr, offset_arr, design = _deep_event_levels()
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        features={"g": RandomEffect()},
    )
    reml_tol = 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(pd.DataFrame({"g": levels_arr}), y_arr, offset=offset_arr, reml_tol=reml_tol)
    assert model.result.converged
    assert model._reml_profile["irls_mean_space_newton_iters"] > 0
    rho_hat = math.log(next(iter(model._reml_lambdas.values())))

    grid = np.arange(-2.0, 4.0 + 1e-12, 0.05)
    values = [_true_laml(design, y_arr, offset_arr, r)[0] for r in grid]
    best = int(np.argmin(values))
    low, high = grid[best - 1], grid[best + 1]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        left, right = high - golden * (high - low), low + golden * (high - low)
        if (
            _true_laml(design, y_arr, offset_arr, left)[0]
            < _true_laml(design, y_arr, offset_arr, right)[0]
        ):
            high = right
        else:
            low = left
    rho_star = 0.5 * (low + high)
    value, size, condition = _true_laml(design, y_arr, offset_arr, rho_star)
    step = 1e-2
    curvature = (
        _true_laml(design, y_arr, offset_arr, rho_star + step)[0]
        - 2.0 * value
        + _true_laml(design, y_arr, offset_arr, rho_star - step)[0]
    ) / step**2
    noise = _laml_noise(design, size, condition)
    resolution = 2.0 * reml_tol * (1.0 + abs(value)) / curvature + 2.0 * math.sqrt(
        2.0 * noise / curvature
    )
    assert abs(rho_hat - rho_star) <= resolution, (rho_hat, rho_star, resolution)


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
    # the profile publishes the rule that stopped the inner solve, not the
    # clipped residual, which cannot pass at this mode
    assert model._reml_profile["irls_mode_rule"] == "mean_space_mode_score"
    assert model._reml_profile["irls_mode_score"] <= model._reml_profile["irls_mode_bar"]
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
    # below the clip floor too, where the clipped deviance is flat: the rows'
    # differences summed pairwise, as the Newton line search forms it from the
    # committed state's cached rows (``irls_direct``'s ``_newton_deviance_delta``)
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
    rows_after = mean_space_log_likelihood_rows(y, weights, after)
    rows_before = mean_space_log_likelihood_rows(y, weights, before)
    delta = -2.0 * float(np.sum(rows_after - rows_before))
    # Each row is within 8u of its own size: log1mexp's library calls are
    # within one ulp (2u) each, log(-expm1) carries expm1's 2u as absolute
    # error beside |log z| >= log 2, log1p(-t) carries exp's 2u times
    # t / (1 - t) <= 2t beside |log1p(-t)| >= t, and the weight's product
    # adds u: at most 7u.  The rows' differences then round once and their
    # three-term sum adds gamma_2, all doubled by the exact factor -2.
    difference = float(np.sum(np.abs(rows_after - rows_before)))
    tolerance = 2.0 * (
        8.0 * _U * float(np.sum(np.abs(rows_after) + np.abs(rows_before)))
        + (_U + 2.0 * _U / (1.0 - 2.0 * _U)) * difference
    )
    assert abs(delta - float(expected)) <= tolerance


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
